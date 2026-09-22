"""Predefined measurement diagnostics for the larger H2 follow-up."""
import numpy as np

from .h2_metrics import per_problem, summarize


def crossfit(success):
    """Gate on one half of the samples, measure SD on the other, then swap."""
    y = np.asarray(success, dtype=float)
    if y.ndim != 3: raise ValueError("Expected problem x variant x sample outcomes")
    if y.shape[-1] % 2: raise ValueError("Cross-fitting requires an even sample count")
    a, b = (per_problem(z) for z in np.split(y, 2, axis=2))
    ma, mb = a["eligible"], b["eligible"]
    counts = ma.astype(int) + mb.astype(int)
    contributions = np.where(ma, b["sd"], 0) + np.where(mb, a["sd"], 0)
    return {"inv": float(1-contributions.sum()/counts.sum()) if counts.sum() else None,
            "n_unique_problems": int((counts > 0).sum()), "n_fold_contributions": int(counts.sum()),
            "eligibility_agreement": float((ma == mb).mean())}


def describe(success, seed, n_boot=4000, min_eligible=5):
    y = np.asarray(success)
    result = summarize(y, seed=seed, n_boot=n_boot)
    result["adequate_conditional_coverage"] = result["n_eligible"] >= min_eligible
    result["minimum_eligible_for_reporting"] = min_eligible
    result["crossfit"] = crossfit(y)
    p = y.mean(axis=2)
    # Fixed problem set; this descriptive bootstrap conditions on observed sample rates.
    diffs = p[:, 1:].mean(axis=1)-p[:, 0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    result["canonical_accuracy"] = float(p[:, 0].mean())
    result["other_variant_accuracy"] = float(p[:, 1:].mean())
    result["other_minus_canonical"] = float(diffs.mean())
    result["accuracy_difference_problem_bootstrap_ci95"] = np.quantile(diffs[idx].mean(axis=1), [0.025, 0.975]).tolist()
    # The two halves are a repeatability diagnostic, not independent experiments across problems.
    result["halves"] = [{k: v for k, v in summarize(z, seed=seed, n_boot=100).items()
                          if k in ("accuracy", "inv_raw", "inv_conditional", "n_eligible")}
                         for z in np.split(y, 2, axis=2)]
    return result
