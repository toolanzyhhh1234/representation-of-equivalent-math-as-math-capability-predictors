"""Phase 0 gate (PLAN sec 9): do three models chosen to differ in math training
order correctly on EQ? If not, the instrument is broken and no amount of N fixes it.
"""
import gc
import json
import sys
import time

import numpy as np
import torch

from .config import PC_REMOVAL_K, PILOT_MODELS, POOLINGS, PRIMARY_POOLING, RAW, RESULTS
from .correct import gap_k, spectrum, topk, zca
from .data import load_meld
from .extract import extract, load, verify_padding_invariance
from .metrics import build_gold_map, eq_scores, retrieval


def corrections(X):
    """All anisotropy corrections for one layer. {name: normalised np.ndarray}.

    fp32 GPU throughout; parity vs the fp64 CPU reference is checked in tests/test_parity.
    gap_k is adaptive: it removes only directions that are well separated from the next
    one, because a direction with s_i/s_{i+1} ~ 1 is not identified and removing it
    injects noise. Anisotropy varies from 10% to 99% of variance across layers, so a
    fixed k cannot be right everywhere.
    """
    out = {f"k{k}": v for k, v in topk(X, PC_REMOVAL_K).items()}
    kg = gap_k(X)
    out["gapk"] = out[f"k{kg}"] if f"k{kg}" in out else topk(X, [kg])[kg]
    out["zca"] = zca(X)
    sp = spectrum(X)
    meta = {"gap_k": kg, "s1_s2": float(sp["ratios"][0]), "var_pc1": float(sp["cumvar"][0])}
    return {k: v.cpu().numpy() for k, v in out.items()}, meta


def domain_stratified_split(stim, seed=0):
    """Split pair indices in half, balanced by domain -- for honest layer selection."""
    rng = np.random.default_rng(seed)
    by_dom = {}
    for i, (*_, dom, _pid) in enumerate(stim.pairs):
        by_dom.setdefault(dom, []).append(i)
    a, b = [], []
    for dom, idxs in sorted(by_dom.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        a += idxs[: len(idxs) // 2]
        b += idxs[len(idxs) // 2:]
    return a, b


def run_model(model_id, stim, gold):
    t0 = time.time()
    tok, model = load(model_id)
    t_load = time.time() - t0

    inv = verify_padding_invariance(tok, model, stim.texts)
    print(f"  padding invariance: {inv}  (>=0.999 required)", flush=True)

    t0 = time.time()
    acts = extract(tok, model, stim.texts,
                   progress=lambda d, n: print(f"\r  extract {d}/{n}", end="", flush=True))
    t_ext = time.time() - t0
    print(f"\r  extract {len(stim.texts)}/{len(stim.texts)}  ({t_ext:.1f}s)", flush=True)

    n_layers = acts[PRIMARY_POOLING].shape[1]
    hid = acts[PRIMARY_POOLING].shape[2]
    np.savez(RAW / f"{model_id.replace('/', '__')}.npz", **acts)  # uncompressed: ~1 min/model faster

    del model
    gc.collect()
    torch.cuda.empty_cache()

    split_a, split_b = domain_stratified_split(stim)
    t0 = time.time()
    curves = {}
    for pool in POOLINGS:
        for layer in range(n_layers):
            Xs, meta = corrections(acts[pool][:, layer, :])
            for name, Xn in Xs.items():
                rec = {"layer": layer, **meta}
                rec.update(eq_scores(Xn, stim))
                rec["auroc_a"] = eq_scores(Xn, stim, split_a)["auroc_anchor"]
                rec["auroc_b"] = eq_scores(Xn, stim, split_b)["auroc_anchor"]
                rec.update(retrieval(Xn, stim, gold))
                curves.setdefault(f"{pool}|{name}", []).append(rec)
        print(f"  metrics {pool} done ({time.time() - t0:.1f}s)", flush=True)

    return {
        "model": model_id, "n_layers": n_layers, "hidden": hid,
        "load_s": round(t_load, 1), "extract_s": round(t_ext, 1),
        "padding_invariance": inv, "curves": curves,
    }


def headline(res, pool=PRIMARY_POOLING, k="k1"):
    """Layer picked on split A, EQ reported on split B -- no selection on the reported number."""
    c = res["curves"][f"{pool}|{k}"]
    best = max(c, key=lambda r: r["auroc_a"])
    best_all = max(c, key=lambda r: r["auroc_anchor"])
    return {
        "sel_layer": best["layer"], "n_layers": res["n_layers"],
        "EQ_heldout": round(best["auroc_b"], 4),
        "EQ_bestlayer_all": round(best_all["auroc_anchor"], 4),
        "align_gap": round(best["align_gap"], 4),
        "recall@1": round(best["recall@1"], 4), "mrr": round(best["mrr"], 4),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    stim = load_meld()
    gold = build_gold_map(stim)
    print(f"MELD: {len(stim.texts)} statements, {len(stim.pairs)} pairs, "
          f"{sum(len(v) for v in stim.distractors.values())} hard negatives\n")

    models = sys.argv[1:] or PILOT_MODELS
    path = RESULTS / "pilot.json"
    # Merge, never clobber: running one model must not delete the rest of the panel.
    out = json.loads(path.read_text()) if path.exists() else []
    for mid in models:
        print(f"=== {mid} ===", flush=True)
        r = run_model(mid, stim, gold)
        out = [x for x in out if x["model"] != mid] + [r]
        out.sort(key=lambda x: models.index(x["model"]) if x["model"] in models else 99)
        path.write_text(json.dumps(out, indent=1))
        print(f"  headline: {headline(r)}\n", flush=True)

    print("=== PHASE 0 GATE ===")
    print(f"{'model':<34} {'EQ(held-out)':>13} {'layer':>7} {'R@1':>7} {'MRR':>7}")
    for r in out:
        h = headline(r)
        print(f"{r['model']:<34} {h['EQ_heldout']:>13.4f} "
              f"{h['sel_layer']:>3}/{h['n_layers']:<3} {h['recall@1']:>7.4f} {h['mrr']:>7.4f}")
    print("\nchance EQ = 0.5. Gate: is the ordering sane and the spread non-trivial?")


if __name__ == "__main__":
    main()
