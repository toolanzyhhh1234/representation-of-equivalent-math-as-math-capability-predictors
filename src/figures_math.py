"""Phase 3 figure: EQ_resid vs MATH-500 (results/eq_vs_math500.png).

The construct-matched DV turns the double dissociation into a single, interpretable
wedge: no model with weak formal-math representation solves formal math (necessity),
while strong representation is compatible with anything from 0% to 42% (not
sufficiency -- the exposure-without-skill models sit on the floor at far right).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from .analyze import PRIMARY, load
from .config import RESULTS
from .figures import color_of, GROUPS, LABEL

EXTRA_LABEL = {"meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
               "google/gemma-3-1b-pt": "gemma-3-1b-pt",
               "google/gemma-2-2b": "gemma-2-2b",
               "Qwen/Qwen2.5-Coder-1.5B": "Qwen2.5-Coder-1.5B"}


def main():
    base, models, _ = load()
    m5 = json.loads((RESULTS / "math500.json").read_text())
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    xs, ys = [], []
    labels = {**LABEL, **EXTRA_LABEL}
    for r in models:
        if r["model"] not in m5:
            continue
        h = r["headlines"][f"{PRIMARY}|eq_resid"]
        x, y = h["heldout"], 100 * m5[r["model"]]["acc"]
        xs.append(x); ys.append(y)
        c = color_of(r["model"])
        ax.plot([h["ci_lo"], h["ci_hi"]], [y, y], color=c, lw=1, alpha=0.4)
        ax.scatter([x], [y], c=c, s=42, zorder=3)
        if r["model"] in labels:
            name = labels[r["model"]]
            dx, dy, ha = {  # stagger the y~0 floor cluster
                "pythia-1.4b": (5, 9, "left"),
                "deepseek-coder-1.3b": (5, -14, "left"),
                "Llama-3.2-1B": (8, 1, "left"),
                "SmolLM2-360M": (-6, -14, "right"),
                "gemma-3-1b-pt": (-6, 6, "right"),
                "phi-1.5": (5, -14, "left"),
                "TinyLlama-1.1B": (-6, 6, "right"),
                "OLMo-2-1B": (6, 5, "left"),
                "Qwen2.5-Math-1.5B": (-6, 5, "right"),
                "Qwen2.5-Coder-1.5B": (-6, -14, "right"),
            }.get(name, (6, 5, "left"))
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=7.5, color="#444444", ha=ha)
    ax.axvline(0.65, ls=":", c="#888888", lw=1.2)
    ax.text(0.6495, ax.get_ylim()[1] * 0.97, "EQ_resid = 0.65: below it, max MATH-500 = 4.6%",
            ha="right", va="top", fontsize=7.5, color="#666666", rotation=90)
    rho = stats.spearmanr(xs, ys).statistic
    ax.set_title(f"Formal-math representation vs formal-math skill "
                 f"(23 models)   Spearman rho = {rho:+.2f}", fontsize=10.5)
    ax.set_xlabel("EQ_resid, held-out at last|k1 (95% CI over pairs)")
    ax.set_ylabel("MATH-500 accuracy, in-house 4-shot greedy (%)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lab)
               for lab, c, _ in GROUPS]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = RESULTS / "eq_vs_math500.png"
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
