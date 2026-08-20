"""Specificity analysis: T1 (paired contrast), T2 (2x2 differential), T3 (incremental).

Reads results/panel.json (EQ on MELD), results/paws.json (PARA on PAWS),
results/gsm8k.json (math DV), results/arc_easy.json (non-math DV).

Interpretation guide, fixed in advance:
- A clean separation of "well-trained generally" vs "well-trained in math" across
  models is impossible (training mixes are correlated), so EQ and PARA WILL correlate
  and both WILL correlate with both DVs. The evidence lives in the contrasts:
    T1: math-tuning (identical size, near-identical general training) should move EQ,
        not PARA. The sharpest test; immune to the cross-model confound.
    T2: the metric x DV interaction. Specificity = EQ prefers GSM8K over ARC, and
        PARA does not prefer GSM8K. Contrasts get model-bootstrap CIs; N=10, so wide.
    T3: partial Spearman(EQ_resid, GSM8K | PARA_resid) -- does the math metric carry
        predictive information beyond the shared general-quality component?
- Failure is informative: if T1 moves PARA as much as EQ and T3 collapses to ~0, the
  honest conclusion is that EQ is a general-representation-quality meter that happens
  to be read on math stimuli, and the report's claims must be weakened accordingly.

Usage: python -m src.analyze_spec
"""
import json

import numpy as np
from scipy import stats

from .config import PANEL, RESULTS
from .analyze import PRIMARY, partial_spearman, perm_spearman

PAIRS = [("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Math-1.5B"),
         ("Qwen/Qwen2-1.5B", "Qwen/Qwen2-Math-1.5B")]


def load_all():
    panel = json.loads((RESULTS / "panel.json").read_text())
    paws = json.loads((RESULTS / "paws.json").read_text())
    gsm = json.loads((RESULTS / "gsm8k.json").read_text())
    arc = json.loads((RESULTS / "arc_easy.json").read_text())
    eqh = {r["model"]: r["headlines"] for r in panel["models"]}
    pah = {r["model"]: r["headlines"] for r in paws["models"]}
    models = [m for m, _ in PANEL if m in eqh and m in pah and m in gsm and m in arc]
    d = {
        "eq":        [eqh[m][f"{PRIMARY}|eq"]["heldout"] for m in models],
        "eq_resid":  [eqh[m][f"{PRIMARY}|eq_resid"]["heldout"] for m in models],
        "eq_hard":   [eqh[m][f"{PRIMARY}|eq_hard"]["heldout"] for m in models],
        "para":      [pah[m][f"{PRIMARY}|para"]["heldout"] for m in models],
        "para_resid":[pah[m][f"{PRIMARY}|para_resid"]["heldout"] for m in models],
        "gsm8k":     [gsm[m]["acc"] for m in models],
        "arc":       [arc[m]["acc_norm"] for m in models],
    }
    return models, {k: np.array(v) for k, v in d.items()}, eqh, pah


def t1_paired(eqh, pah):
    print("\n### T1 -- math-tuning contrast at identical size "
          "(prediction: moves EQ, not PARA)")
    print(f"{'pair':<38} {'d_eq':>8} {'d_eq_resid':>11} {'d_para':>8} {'d_para_resid':>13}")
    for a, b in PAIRS:
        if a not in eqh or b not in eqh or a not in pah or b not in pah:
            continue
        de = eqh[b][f"{PRIMARY}|eq"]["heldout"] - eqh[a][f"{PRIMARY}|eq"]["heldout"]
        der = eqh[b][f"{PRIMARY}|eq_resid"]["heldout"] - eqh[a][f"{PRIMARY}|eq_resid"]["heldout"]
        dp = pah[b][f"{PRIMARY}|para"]["heldout"] - pah[a][f"{PRIMARY}|para"]["heldout"]
        dpr = pah[b][f"{PRIMARY}|para_resid"]["heldout"] - pah[a][f"{PRIMARY}|para_resid"]["heldout"]
        print(f"{a.split('/')[-1]+' -> '+b.split('/')[-1]:<38} "
              f"{de:>+8.4f} {der:>+11.4f} {dp:>+8.4f} {dpr:>+13.4f}")


def boot_contrast(x1, y, x2, n=4000, seed=0):
    """Model-bootstrap CI for rho(x1,y) - rho(x2,y) (dependent correlations)."""
    rng = np.random.default_rng(seed)
    n_m = len(y)
    ds = []
    for _ in range(n):
        i = rng.integers(0, n_m, n_m)
        if len(set(y[i])) < 3:
            continue
        ds.append(stats.spearmanr(x1[i], y[i]).statistic
                  - stats.spearmanr(x2[i], y[i]).statistic)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def boot_contrast_dv(x, y1, y2, n=4000, seed=0):
    """Model-bootstrap CI for rho(x,y1) - rho(x,y2)."""
    return boot_contrast(y1, x, y2, n, seed)


def t2_matrix(d):
    print("\n### T2 -- the 2x2: Spearman(metric, DV), permutation p")
    print(f"{'metric':<11} {'GSM8K':>16} {'ARC-Easy':>16}")
    for m in ("eq", "eq_resid", "eq_hard", "para", "para_resid"):
        row = f"{m:<11}"
        for dv in ("gsm8k", "arc"):
            r, p = perm_spearman(d[m], d[dv])
            row += f"  {r:+.3f} (p={p:.3f})"
        print(row)
    print("\ncontrasts (model-bootstrap 95% CI; positive = prefers its own side):")
    lo, hi = boot_contrast_dv(d["eq_resid"], d["gsm8k"], d["arc"])
    print(f"  rho(EQ_resid,GSM8K) - rho(EQ_resid,ARC)     = "
          f"{stats.spearmanr(d['eq_resid'], d['gsm8k']).statistic - stats.spearmanr(d['eq_resid'], d['arc']).statistic:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = boot_contrast_dv(d["para_resid"], d["arc"], d["gsm8k"])
    print(f"  rho(PARA_resid,ARC) - rho(PARA_resid,GSM8K) = "
          f"{stats.spearmanr(d['para_resid'], d['arc']).statistic - stats.spearmanr(d['para_resid'], d['gsm8k']).statistic:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = boot_contrast(d["eq_resid"], d["gsm8k"], d["para_resid"])
    print(f"  rho(EQ_resid,GSM8K) - rho(PARA_resid,GSM8K) = "
          f"{stats.spearmanr(d['eq_resid'], d['gsm8k']).statistic - stats.spearmanr(d['para_resid'], d['gsm8k']).statistic:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")


def t3_incremental(d):
    print("\n### T3 -- incremental prediction (partial Spearman)")
    print(f"  rho(EQ_resid, PARA_resid)                = "
          f"{stats.spearmanr(d['eq_resid'], d['para_resid']).statistic:+.3f}   (shared general-quality variance)")
    print(f"  rho(EQ_resid, GSM8K | PARA_resid)        = "
          f"{partial_spearman(d['eq_resid'], d['gsm8k'], d['para_resid']):+.3f}")
    print(f"  rho(PARA_resid, GSM8K | EQ_resid)        = "
          f"{partial_spearman(d['para_resid'], d['gsm8k'], d['eq_resid']):+.3f}")
    print(f"  rho(EQ_resid, ARC | PARA_resid)          = "
          f"{partial_spearman(d['eq_resid'], d['arc'], d['para_resid']):+.3f}")


if __name__ == "__main__":
    models, d, eqh, pah = load_all()
    print(f"n = {len(models)} models with all four measurements")
    print(f"\n{'model':<32} {'EQ_resid':>9} {'PARA_resid':>11} {'GSM8K':>7} {'ARC':>7}")
    for i, m in enumerate(models):
        print(f"{m:<32} {d['eq_resid'][i]:>9.4f} {d['para_resid'][i]:>11.4f} "
              f"{100*d['gsm8k'][i]:>7.1f} {100*d['arc'][i]:>7.1f}")
    print(f"\nDV correlation check: Spearman(GSM8K, ARC) = "
          f"{stats.spearmanr(d['gsm8k'], d['arc']).statistic:+.3f} "
          f"(high = the 2x2 has little room to differentiate)")
    t1_paired(eqh, pah)
    t2_matrix(d)
    t3_incremental(d)
