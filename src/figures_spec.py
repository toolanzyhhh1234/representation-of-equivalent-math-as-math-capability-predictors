"""Specificity figure: the 2x2 metric-by-DV grid (results/specificity.png).

Rows: math metric (EQ_resid on MELD), language metric (PARA_resid on PAWS).
Cols: math DV (GSM8K), language DV (ARC-Easy).
Specificity shows up as the diagonal: only the top-left cell should carry a strong
positive relationship."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .analyze_spec import load_all
from .config import RESULTS
from .figures import color_of


def cell(ax, x, y, models, xlabel, ylabel, title):
    cols = [color_of(m) for m in models]
    ax.scatter(100 * x, y, c=cols, s=34, zorder=3)
    rho = stats.spearmanr(x, y).statistic
    ax.set_title(f"{title}   rho = {rho:+.2f}", fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def main():
    models, d, *_ = load_all()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0))
    cell(axes[0, 0], d["gsm8k"], d["eq_resid"], models,
         "GSM8K (%)", "EQ_resid (MELD, math)", "math metric x math DV")
    cell(axes[0, 1], d["arc"], d["eq_resid"], models,
         "ARC-Easy (%)", "EQ_resid (MELD, math)", "math metric x language DV")
    cell(axes[1, 0], d["gsm8k"], d["para_resid"], models,
         "GSM8K (%)", "PARA_resid (PAWS, language)", "language metric x math DV")
    cell(axes[1, 1], d["arc"], d["para_resid"], models,
         "ARC-Easy (%)", "PARA_resid (PAWS, language)", "language metric x language DV")
    from .figures import GROUPS
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lab)
               for lab, c, _ in GROUPS]
    axes[0, 1].legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
    fig.suptitle("Specificity 2x2: only the math-metric x math-DV cell carries signal",
                 fontsize=11)
    fig.tight_layout()
    path = RESULTS / "specificity.png"
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
