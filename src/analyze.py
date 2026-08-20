"""Metric selection: which EQ variant is fit to scale?

Reads results/panel.json (+ results/gsm8k.json when present) and scores every metric
variant on the four criteria the repo has already committed to:

  1. null behaviour   -- margin of the TF-IDF baseline over the variant's chance level;
                         a good variant has a null at chance (the lexical channel is dead).
  2. layer-0 control  -- METRICS.md sec 6 step 3: layer 0 is token identity, so a metric
                         reading meaning should leave it near its null.
  3. stability        -- METRICS.md sec 6 step 4: mean pairwise Spearman of the
                         between-model ordering across anisotropy corrections.
  4. tracking         -- Spearman with in-house GSM8K vs with log params, plus the
                         Phase 0 pre-registered SmolLM2-1.7B point prediction.

Usage: python -m src.analyze
"""
import json

import numpy as np
from scipy import stats

from .config import CORRECTIONS, GSM8K_PUBLISHED, PANEL, RESULTS

VARIANTS = ("eq", "eq_resid", "eq_hard", "eq_lextop")
NULLS = {"eq": None, "eq_resid": 0.5, "eq_hard": 0.5, "eq_lextop": None}  # None -> tfidf
PRIMARY = "last|k1"


def load():
    panel = json.loads((RESULTS / "panel.json").read_text())
    g = RESULTS / "gsm8k.json"
    gsm = json.loads(g.read_text()) if g.exists() else {}
    order = [m for m, _ in PANEL]
    models = sorted(panel["models"], key=lambda r: order.index(r["model"])
                    if r["model"] in order else 99)
    return panel["baseline"], models, gsm


def null_of(v, base):
    return base[v] if NULLS[v] is None else NULLS[v]


def headline_table(base, models, gsm):
    params = dict(PANEL)
    for v in VARIANTS:
        n0 = null_of(v, base)
        print(f"\n### {v}  (setting {PRIMARY}; null = {n0:.4f}"
              f"{' [TF-IDF]' if NULLS[v] is None else ' [construction]'})")
        print(f"{'model':<32} {'heldout':>8} {'margin':>8} {'95% CI':>16} "
              f"{'layer':>6} {'GSM8K':>6}")
        for r in models:
            h = r["headlines"][f"{PRIMARY}|{v}"]
            gg = gsm.get(r["model"], {}).get("acc")
            extra = f"  cov {h['coverage']:.2f}" if v == "eq_hard" else ""
            print(f"{r['model']:<32} {h['heldout']:>8.4f} {h['heldout']-n0:>+8.4f} "
                  f"[{h['ci_lo']:.3f},{h['ci_hi']:.3f}] {h['sel_layer']:>4}/{r['n_layers']:<3}"
                  f" {100*gg if gg is not None else float('nan'):>6.1f}{extra}")


def layer0_control(base, models):
    """Layer-0 value of each variant (last pooling, k1): distance from the null."""
    print("\n### layer-0 control (last|k1): |layer0 - null|, mean over models "
          "(small = clean control)")
    for v in VARIANTS:
        n0 = null_of(v, base)
        d = [abs(r["curves"][PRIMARY][0][v] - n0) for r in models]
        vals = [r["curves"][PRIMARY][0][v] for r in models]
        print(f"  {v:<10} mean |d| = {np.mean(d):.4f}   range {min(vals):.3f}-{max(vals):.3f}")


def stability(models, pool="last"):
    """Mean pairwise Spearman of the model ordering across corrections, per variant.

    Uses the held-out headline number under each correction."""
    print(f"\n### ordering stability across corrections ({pool}), mean pairwise Spearman")
    for v in VARIANTS:
        orders = []
        for c in CORRECTIONS:
            vals = [r["headlines"][f"{pool}|{c}|{v}"]["heldout"] for r in models]
            orders.append(vals)
        rhos = [stats.spearmanr(orders[i], orders[j]).statistic
                for i in range(len(orders)) for j in range(i + 1, len(orders))]
        print(f"  {v:<10} mean rho = {np.mean(rhos):+.3f}   min {min(rhos):+.3f}")


def perm_spearman(x, y, n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    obs = stats.spearmanr(x, y).statistic
    perm = np.array([stats.spearmanr(rng.permutation(x), y).statistic
                     for _ in range(min(n, 20000))])
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1)
    return obs, p


def partial_spearman(x, y, z):
    """Spearman of x,y after rank-regressing z out of both."""
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    resid = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    return stats.pearsonr(resid(rx, rz), resid(ry, rz)).statistic


def params_of(r):
    """Measured parameter count when extraction recorded it, else the card value."""
    if r.get("n_params"):
        return r["n_params"] / 1e9
    return dict(PANEL)[r["model"]]


def tracking(base, models, gsm):
    have = [r for r in models if r["model"] in gsm]
    if len(have) < 4:
        print("\n(no in-house GSM8K yet -- run python -m src.eval_gsm8k)")
        return
    cap = [gsm[r["model"]]["acc"] for r in have]
    size = [np.log10(params_of(r)) for r in have]
    print(f"\n### tracking, n={len(have)} models: capability (in-house GSM8K) vs size")
    print(f"  Spearman(GSM8K, log params) = {stats.spearmanr(cap, size).statistic:+.3f} "
          f"(panel dissociation check)")
    print(f"{'variant':<10} {'rho(cap)':>9} {'p':>7} {'rho(size)':>10} {'p':>7} "
          f"{'rho(cap|size)':>14} {'rho(size|cap)':>14}")
    for v in VARIANTS:
        m = [r["headlines"][f"{PRIMARY}|{v}"]["heldout"] for r in have]
        rc, pc = perm_spearman(m, cap)
        rs, ps = perm_spearman(m, size)
        print(f"{v:<10} {rc:>+9.3f} {pc:>7.4f} {rs:>+10.3f} {ps:>7.4f} "
              f"{partial_spearman(m, cap, size):>+14.3f} "
              f"{partial_spearman(m, size, cap):>+14.3f}")


def family_of(mid):
    org = mid.split("/")[0]
    return {"HuggingFaceTB": "SmolLM", "Qwen": "Qwen", "TinyLlama": "TinyLlama",
            "tiiuae": "Falcon"}.get(org, org)


def jackknife(base, models, gsm):
    """PLAN sec 6.4: drop each family; if rho moves > ~0.15 the result is family-driven.
    Qwen contributes 6 of 10 panel models, so this is not optional."""
    have = [r for r in models if r["model"] in gsm]
    if len(have) < 4:
        return
    fams = sorted({family_of(r["model"]) for r in have})
    print("\n### leave-one-family-out jackknife, rho(variant, GSM8K)")
    print(f"{'variant':<10} {'full':>7}" + "".join(f"{'-'+f:>12}" for f in fams))
    for v in VARIANTS:
        row = ""
        m_all = [r["headlines"][f"{PRIMARY}|{v}"]["heldout"] for r in have]
        c_all = [gsm[r["model"]]["acc"] for r in have]
        full = stats.spearmanr(m_all, c_all).statistic
        for f in fams:
            keep = [r for r in have if family_of(r["model"]) != f]
            m = [r["headlines"][f"{PRIMARY}|{v}"]["heldout"] for r in keep]
            c = [gsm[r["model"]]["acc"] for r in keep]
            rho = stats.spearmanr(m, c).statistic if len(keep) >= 4 else float("nan")
            row += f"{rho:>+11.3f} "
        print(f"{v:<10} {full:>+7.3f}" + row)


def calibration(base, models, gsm):
    """Refit the Phase 0 margin ~ GSM8K line (fitted there on 3 imported points) on the
    full panel with the in-house axis. Reports slope, zero-crossing, and residuals."""
    have = [r for r in models if r["model"] in gsm]
    if len(have) < 4:
        return
    print("\n### calibration: margin(eq over TF-IDF) ~ in-house GSM8K "
          "(Phase 0, 3 imported points: slope 0.00124, x0 ~ 32.5)")
    x = np.array([100 * gsm[r["model"]]["acc"] for r in have])
    for v in VARIANTS:
        n0 = null_of(v, base)
        y = np.array([r["headlines"][f"{PRIMARY}|{v}"]["heldout"] - n0 for r in have])
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        r = stats.pearsonr(x, y).statistic
        x0 = -a / b if b != 0 else float("nan")
        print(f"  {v:<10} slope {b:+.5f}  zero-crossing GSM8K ~ {x0:5.1f}  "
              f"Pearson r {r:+.3f}  max|resid| {np.abs(y-pred).max():.4f}")


def prediction(base, models, gsm):
    """The Phase 0 pre-registered point prediction (results/PHASE0.md sec 7)."""
    tgt = next((r for r in models if r["model"] == "HuggingFaceTB/SmolLM2-1.7B"), None)
    if tgt is None:
        return
    h = tgt["headlines"][f"{PRIMARY}|eq"]
    print("\n### pre-registered dissociation point: SmolLM2-1.7B, eq @ last|k1")
    print(f"  predicted if EQ tracks capability: 0.770   if size: 0.808")
    print(f"  observed: {h['heldout']:.4f}  (full-set {h['full']:.4f}, "
          f"CI [{h['ci_lo']:.3f},{h['ci_hi']:.3f}])")


def pairs(base, models):
    """Same-size tuning contrasts. Math pairs are the hypothesis; coder/instruct pairs
    are controls -- non-math post-training at identical size should move EQ less."""
    P = [("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Math-1.5B", "math"),
         ("Qwen/Qwen2-1.5B", "Qwen/Qwen2-Math-1.5B", "math"),
         ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Coder-1.5B", "coder (control)"),
         ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "instruct (control)")]
    print("\n### same-size tuning contrasts (delta on held-out score)")
    byid = {r["model"]: r for r in models}
    for a, b, kind in P:
        if a not in byid or b not in byid:
            continue
        print(f"  {a.split('/')[-1]} -> {b.split('/')[-1]}   [{kind}]")
        for v in VARIANTS:
            ha = byid[a]["headlines"][f"{PRIMARY}|{v}"]
            hb = byid[b]["headlines"][f"{PRIMARY}|{v}"]
            print(f"    {v:<10} {ha['heldout']:.4f} -> {hb['heldout']:.4f}   "
                  f"delta {hb['heldout']-ha['heldout']:+.4f}")


if __name__ == "__main__":
    base, models, gsm = load()
    print(f"panel: {len(models)} models; baseline "
          f"{ {k: round(v, 4) for k, v in base.items()} }")
    headline_table(base, models, gsm)
    layer0_control(base, models)
    stability(models)
    tracking(base, models, gsm)
    jackknife(base, models, gsm)
    calibration(base, models, gsm)
    prediction(base, models, gsm)
    pairs(base, models)
