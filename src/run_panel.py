"""Phase 0.5: the dissociation panel with lexically-controlled EQ variants.

Extends run_pilot in two ways demanded by results/PHASE0.md sec 5:
  1. every metric is computed alongside its lexical counterpart (src/lexical.py), so
     each model's number can be read as a margin over -- or a residual free of -- the
     TF-IDF channel;
  2. per-anchor arrays at the selected layer are kept, so bootstrap CIs resample pairs
     (PLAN sec 4.4) instead of being skipped.

Layer selection stays honest: chosen on stimulus split A, reported on split B,
independently for each metric variant (the best layer for eq_resid need not be the
best layer for eq). Activations are cached to results/raw and reused.
"""
import gc
import json
import sys
import time

import numpy as np
import torch

from .config import PANEL, PC_REMOVAL_K, POOLINGS, RAW, RESULTS
from .correct import gap_k, spectrum, topk, zca
from .data import load_meld
from .extract import extract, load, verify_padding_invariance
from .lexical import anchor_jobs, bootstrap_ci, eq_variants, lexical_cos, summarize
from .run_pilot import domain_stratified_split

VARIANTS = ("eq", "eq_resid", "eq_hard", "eq_lextop")


def corrections_gpu(X):
    """{name: normalised torch tensor on GPU} for one layer."""
    out = {f"k{k}": v for k, v in topk(X, PC_REMOVAL_K).items()}
    kg = gap_k(X)
    out["gapk"] = out[f"k{kg}"] if f"k{kg}" in out else topk(X, [kg])[kg]
    out["zca"] = zca(X)
    sp = spectrum(X)
    meta = {"gap_k": kg, "s1_s2": float(sp["ratios"][0]), "var_pc1": float(sp["cumvar"][0])}
    return out, meta


def split_means(var, mask_a, mask_b):
    """Per-variant anchor means restricted to each stimulus split."""
    out = {}
    for k, v in var["per_anchor"].items():
        for tag, m in (("a", mask_a), ("b", mask_b)):
            sel = v[m]
            sel = sel[~np.isnan(sel)]
            out[f"{k}_{tag}"] = float(sel.mean()) if sel.size else float("nan")
    return out


def get_acts(model_id, stim):
    """Cached activations, or extract (with the padding-invariance guard) and cache."""
    path = RAW / f"{model_id.replace('/', '__')}.npz"
    if path.exists():
        z = np.load(path)
        pf = RAW / f"{model_id.replace('/', '__')}.params"
        info = {"cached": True}
        if pf.exists():
            info["n_params"] = int(pf.read_text())
        return {k: z[k] for k in z.files}, info
    tok, model = load(model_id)
    n_params = sum(p.numel() for p in model.parameters())
    inv = verify_padding_invariance(tok, model, stim.texts)
    print(f"  padding invariance: {inv}   params: {n_params/1e9:.3f}B", flush=True)
    t0 = time.time()
    acts = extract(tok, model, stim.texts,
                   progress=lambda d, n: print(f"\r  extract {d}/{n}", end="", flush=True))
    print(f"\r  extract done ({time.time() - t0:.1f}s)", flush=True)
    np.savez(path, **acts)
    (RAW / f"{model_id.replace('/', '__')}.params").write_text(str(n_params))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return acts, {"cached": False, "padding_invariance": inv, "n_params": n_params}


def run_model(model_id, stim, jobs, L, mask_a, mask_b):
    acts, info = get_acts(model_id, stim)
    n_layers = acts["last"].shape[1]

    t0 = time.time()
    curves = {}       # "pool|corr" -> [per-layer dict]
    grams = {}        # (pool, corr, layer) kept only for selected layers, filled later
    for pool in POOLINGS:
        A = torch.tensor(acts[pool], device="cuda")          # [N, L+1, H] fp32
        for layer in range(n_layers):
            Xs, meta = corrections_gpu(A[:, layer, :])
            for name, Xn in Xs.items():
                S = (Xn @ Xn.T).double().cpu().numpy()
                var = eq_variants(S, jobs, L)
                rec = {"layer": layer, **meta, **summarize(var),
                       **split_means(var, mask_a, mask_b)}
                curves.setdefault(f"{pool}|{name}", []).append(rec)
        del A
        torch.cuda.empty_cache()
        print(f"  metrics {pool} done ({time.time() - t0:.1f}s)", flush=True)

    # headline per (setting, variant): layer argmax on split A, report split B,
    # bootstrap CI over pairs at that layer. Layer 0 is excluded from selection: it is
    # the token-identity CONTROL layer (METRICS.md sec 6 step 3), and deepseek-coder
    # showed a tokenizer-channel lexical leak can make it the split-A argmax -- selecting
    # the control as the readout contradicts the construct being measured.
    headlines = {}
    for key, recs in curves.items():
        pool, name = key.split("|")
        A = torch.tensor(acts[pool], device="cuda")
        for v in VARIANTS:
            best = max(recs[1:], key=lambda r: r[f"{v}_a"])
            Xs, _ = corrections_gpu(A[:, best["layer"], :])
            S = (Xs[name] @ Xs[name].T).double().cpu().numpy()
            var = eq_variants(S, jobs, L)
            lo, hi = bootstrap_ci(var["per_anchor"][v], var["pair_index"])
            headlines[f"{key}|{v}"] = {
                "sel_layer": best["layer"], "n_layers": n_layers,
                "heldout": best[f"{v}_b"], "full": best[v],
                "ci_lo": lo, "ci_hi": hi,
                **({"coverage": best["eq_hard_coverage"],
                    "mean_negs": best["eq_hard_mean_negs"]} if v == "eq_hard" else {}),
            }
        del A
        torch.cuda.empty_cache()

    return {"model": model_id, "n_layers": n_layers, **info,
            "curves": curves, "headlines": headlines}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    stim = load_meld()
    jobs = anchor_jobs(stim)
    L = lexical_cos(stim.texts)

    split_a, split_b = domain_stratified_split(stim)
    pair_index = np.array([pi for pi, *_ in jobs])
    mask_a = np.isin(pair_index, split_a)
    mask_b = np.isin(pair_index, split_b)

    base = summarize(eq_variants(None, jobs, L))
    print("TF-IDF (char_wb 3-5) baseline per variant:",
          {k: round(v, 4) for k, v in base.items()}, "\n", flush=True)

    models = sys.argv[1:] or [m for m, _ in PANEL]
    path = RESULTS / "panel.json"
    out = json.loads(path.read_text()) if path.exists() else {"baseline": None, "models": []}
    out["baseline"] = base
    for mid in models:
        print(f"=== {mid} ===", flush=True)
        t0 = time.time()
        r = run_model(mid, stim, jobs, L, mask_a, mask_b)
        out["models"] = [x for x in out["models"] if x["model"] != mid] + [r]
        path.write_text(json.dumps(out, indent=1))
        h = r["headlines"]["last|k1|eq"]
        hr = r["headlines"]["last|k1|eq_resid"]
        print(f"  {time.time()-t0:.0f}s  EQ(heldout)={h['heldout']:.4f} "
              f"[{h['ci_lo']:.3f},{h['ci_hi']:.3f}]  EQ_resid={hr['heldout']:.4f} "
              f"[{hr['ci_lo']:.3f},{hr['ci_hi']:.3f}]\n", flush=True)


if __name__ == "__main__":
    main()
