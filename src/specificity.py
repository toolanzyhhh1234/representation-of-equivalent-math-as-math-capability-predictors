"""Phase 1: the specificity control -- is EQ math-specific, or a general-quality meter?

The threat (REPORT.md sec 7.1): "EQ_resid tracks capability" could reduce to
"better-trained models embed all text better, and better-trained small models also do
more math." A clean separation of 'well-trained generally' from 'well-trained in math'
is impossible across models -- training mixes are correlated -- so the tests here are
contrasts, not separations:

  T1 (paired, sharpest). Base vs math-tuned at identical size differ ONLY in math
     training, so general quality is controlled by construction. Prediction: math
     tuning raises MELD-EQ but not paraphrase-EQ.
  T2 (2x2 differential). Math metric x math DV vs language metric x language DV.
     Specificity = the diagonal wins: EQ predicts GSM8K better than ARC-Easy, and
     paraphrase-EQ does not prefer GSM8K.
  T3 (incremental). partial Spearman(EQ_resid, GSM8K | PARA_resid): does the math
     metric predict math capability beyond the shared general-quality component?

The paraphrase task mirrors MELD structurally: PAWS (labeled_final, human-verified
test+validation splits) pairs a sentence with either a true paraphrase (label 1) or a
deliberately high-lexical-overlap NON-paraphrase (label 0) -- the same
"equivalent-vs-lookalike" discrimination, in general language. One structural
difference: PAWS pairs are independent, so the score is a pooled AUROC of
cos(s1, s2) over pairs (label 1 vs 0), not MELD's anchor-wise AUROC -- there is no
candidate-set structure to be anchor-wise over. Same corrections, same honest layer
selection, same TF-IDF partialling; the TF-IDF null is computed on the identical
task and is expected to be weak on PAWS by design.

Usage: python -m src.specificity [model ...]   -> results/paws.json
"""
import gc
import json
import sys
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import POOLINGS, RAW, RESULTS
from .extract import extract, load, verify_padding_invariance
from .lexical import bootstrap_ci
from .run_panel import corrections_gpu

N_PER_LABEL = 1000
SEED = 0
VARIANTS = ("para", "para_resid")


def build_paws():
    """Balanced pair sample from the human-verified splits. Deterministic."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/paws", "labeled_final")
    rows = list(ds["test"]) + list(ds["validation"])
    rng = np.random.default_rng(SEED)
    by_label = {0: [], 1: []}
    for r in rows:
        by_label[r["label"]].append((r["sentence1"].strip(), r["sentence2"].strip()))
    pairs, labels = [], []
    for lab in (1, 0):
        idx = rng.permutation(len(by_label[lab]))[:N_PER_LABEL]
        for i in idx:
            pairs.append(by_label[lab][i])
            labels.append(lab)
    texts, index = [], {}
    for a, b in pairs:
        for t in (a, b):
            if t not in index:
                index[t] = len(texts)
                texts.append(t)
    ij = np.array([(index[a], index[b]) for a, b in pairs])
    return texts, ij, np.array(labels)


def tfidf_pair_cos(texts, ij):
    X = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts))
    return np.asarray(X[ij[:, 0]].multiply(X[ij[:, 1]]).sum(1)).ravel()


def _auc(scores, labels):
    """Pooled AUROC via rank statistic."""
    from scipy import stats
    r = stats.rankdata(scores)
    n1, n0 = int(labels.sum()), int((1 - labels).sum())
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_scores(Xn, ij):
    return (Xn[ij[:, 0]] * Xn[ij[:, 1]]).sum(1)


def variants_for(S_model, L_pair, labels, mask=None):
    """{variant: AUROC} for one setting, optionally restricted to a pair mask."""
    m = np.ones(len(labels), bool) if mask is None else mask
    b = np.polyfit(L_pair[m], S_model[m], 1)[0]
    resid = S_model - b * L_pair
    return {"para": _auc(S_model[m], labels[m]),
            "para_resid": _auc(resid[m], labels[m])}


def run_model(model_id, texts, ij, labels, L_pair, mask_a, mask_b):
    path = RAW / f"paws__{model_id.replace('/', '__')}.npz"
    if path.exists():
        z = np.load(path)
        acts = {k: z[k] for k in z.files}
    else:
        tok, model = load(model_id)
        inv = verify_padding_invariance(tok, model, texts)
        print(f"  padding invariance: {inv}", flush=True)
        acts = extract(tok, model, texts,
                       progress=lambda d, n: print(f"\r  extract {d}/{n}", end="", flush=True))
        print(flush=True)
        np.savez(path, **acts)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    n_layers = acts["last"].shape[1]
    curves = {}
    t0 = time.time()
    for pool in POOLINGS:
        A = torch.tensor(acts[pool], device="cuda")
        for layer in range(n_layers):
            Xs, meta = corrections_gpu(A[:, layer, :])
            for name, Xn in Xs.items():
                S = pair_scores(Xn.double().cpu().numpy(), ij)
                rec = {"layer": layer, **variants_for(S, L_pair, labels)}
                for tag, m in (("a", mask_a), ("b", mask_b)):
                    for k, v in variants_for(S, L_pair, labels, m).items():
                        rec[f"{k}_{tag}"] = v
                curves.setdefault(f"{pool}|{name}", []).append(rec)
        del A
        torch.cuda.empty_cache()
    print(f"  metrics done ({time.time() - t0:.1f}s)", flush=True)

    # headline: layer on split A, report split B; bootstrap CI over pairs at that layer.
    # Layer 0 excluded from selection -- it is the token-identity control (see run_panel).
    headlines = {}
    for key, recs in curves.items():
        pool, name = key.split("|")
        A = torch.tensor(acts[pool], device="cuda")
        for v in VARIANTS:
            best = max(recs[1:], key=lambda r: r[f"{v}_a"])
            Xs, _ = corrections_gpu(A[:, best["layer"], :])
            S = pair_scores(Xs[name].double().cpu().numpy(), ij)
            b = np.polyfit(L_pair, S, 1)[0]
            sc = S if v == "para" else S - b * L_pair
            # per-pair contribution bootstrap: resample pairs, recompute pooled AUROC
            rng = np.random.default_rng(0)
            n = len(labels)
            boots = []
            for _ in range(500):
                bi = rng.integers(0, n, n)
                if labels[bi].min() == labels[bi].max():
                    continue
                boots.append(_auc(sc[bi], labels[bi]))
            headlines[f"{key}|{v}"] = {
                "sel_layer": best["layer"], "n_layers": n_layers,
                "heldout": best[f"{v}_b"], "full": best[v],
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
            }
        del A
        torch.cuda.empty_cache()
    return {"model": model_id, "n_layers": n_layers, "curves": curves,
            "headlines": headlines}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    texts, ij, labels = build_paws()
    L_pair = tfidf_pair_cos(texts, ij)
    print(f"PAWS: {len(texts)} unique sentences, {len(labels)} pairs "
          f"({int(labels.sum())} pos / {int((1-labels).sum())} neg)")
    base = {"para": _auc(L_pair, labels), "para_resid": 0.5}
    print(f"TF-IDF char_wb null on PAWS: {base['para']:.4f} "
          f"(MELD's was 0.7715 -- PAWS negatives are lexically adversarial by design)\n",
          flush=True)

    rng = np.random.default_rng(1)
    order = rng.permutation(len(labels))
    mask_a = np.zeros(len(labels), bool)
    mask_a[order[: len(labels) // 2]] = True
    mask_b = ~mask_a

    from .config import PANEL
    models = sys.argv[1:] or [m for m, _ in PANEL]
    path = RESULTS / "paws.json"
    out = json.loads(path.read_text()) if path.exists() else {"baseline": None, "models": []}
    out["baseline"] = base
    for mid in models:
        print(f"=== {mid} ===", flush=True)
        t0 = time.time()
        r = run_model(mid, texts, ij, labels, L_pair, mask_a, mask_b)
        out["models"] = [x for x in out["models"] if x["model"] != mid] + [r]
        path.write_text(json.dumps(out, indent=1))
        h, hr = r["headlines"]["last|k1|para"], r["headlines"]["last|k1|para_resid"]
        print(f"  {time.time()-t0:.0f}s  PARA={h['heldout']:.4f} "
              f"[{h['ci_lo']:.3f},{h['ci_hi']:.3f}]  PARA_resid={hr['heldout']:.4f}\n",
              flush=True)


if __name__ == "__main__":
    main()
