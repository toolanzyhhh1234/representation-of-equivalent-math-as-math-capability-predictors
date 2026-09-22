"""H2 endpoints, with explicit floor/ceiling and sampling-noise diagnostics."""
import numpy as np


def per_problem(success):
    """success: [problem, variant, sample], binary; SD uses ddof=0 (PLAN definition)."""
    y = np.asarray(success, dtype=float)
    if y.ndim != 3 or min(y.shape) < 1 or y.shape[1] < 2 or y.shape[2] < 2:
        raise ValueError("Expected problems x >=2 variants x >=2 samples")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Outcomes must be binary and complete")
    p = y.mean(axis=2)
    avg = p.mean(axis=1)
    variance = p.var(axis=1, ddof=0)
    # p_hat*(1-p_hat)/(K-1) is an unbiased estimator of Var(p_hat).
    # Population variance over V variants contributes a factor (V-1)/V.
    noise = (1 - 1 / y.shape[1]) * (p * (1-p) / (y.shape[2]-1)).mean(axis=1)
    return {"accuracy": avg, "variant_accuracy": p, "sd": np.sqrt(variance),
            "sd_noise_adjusted": np.sqrt(np.maximum(variance-noise, 0)),
            "eligible": (avg >= 0.15) & (avg <= 0.85)}


def summarize(success, seed=0, n_boot=2000):
    stats = per_problem(success)
    eligible = stats["eligible"]
    result = {
        "n_problems": len(eligible), "accuracy": float(stats["accuracy"].mean()),
        "inv_raw": float(1-stats["sd"].mean()),
        "n_eligible": int(eligible.sum()), "eligible_fraction": float(eligible.mean()),
        "n_always_wrong": int((stats["accuracy"] == 0).sum()),
        "n_always_correct": int((stats["accuracy"] == 1).sum()),
        "inv_conditional": None, "inv_noise_adjusted_conditional": None,
        "conditional_ci95": None,
        "adequate_conditional_coverage": bool(eligible.sum() >= 5),
    }
    if eligible.any():
        values = 1-stats["sd"][eligible]
        result["inv_conditional"] = float(values.mean())
        result["inv_noise_adjusted_conditional"] = float(
            1-stats["sd_noise_adjusted"][eligible].mean())
        rng = np.random.default_rng(seed)
        boots = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
        result["conditional_ci95"] = np.quantile(boots, [0.025, 0.975]).tolist()
    return result
