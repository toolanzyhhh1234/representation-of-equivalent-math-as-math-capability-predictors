"""Phase 0.5 figures: the dissociation scatter (results/eq_vs_gsm8k.png).

Left: the recommended metric (eq_resid) against in-house GSM8K.
Right: raw eq against the same axis, with the TF-IDF null drawn in -- the two panels
together show what lexical control changes and what it does not.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .analyze import PRIMARY, family_of, load

FAM_COLOR = {  # Okabe-Ito, CVD-safe, fixed assignment by family
    "Qwen": "#0072B2", "SmolLM": "#E69F00", "TinyLlama": "#D55E00", "Falcon": "#009E73",
}
SHORT = {"TinyLlama_v1.1": "TinyLlama-1.1B", "Falcon3-1B-Base": "Falcon3-1B"}


def panel(ax, models, gsm, variant, null, title):
    xs, ys, labels, cols = [], [], [], []
    for r in models:
        if r["model"] not in gsm:
            continue
        h = r["headlines"][f"{PRIMARY}|{variant}"]
        xs.append(100 * gsm[r["model"]]["acc"])
        ys.append(h["heldout"])
        name = r["model"].split("/")[-1]
        labels.append(SHORT.get(name, name))
        cols.append(FAM_COLOR[family_of(r["model"])])
        ax.plot([xs[-1]] * 2, [h["ci_lo"], h["ci_hi"]], color=cols[-1], lw=1, alpha=0.45)
    ax.scatter(xs, ys, c=cols, s=42, zorder=3)
    NUDGE = {"Qwen2-Math-1.5B": -14, "Qwen2-1.5B": -14}     # de-collide label stacks
    for x, y, t in zip(xs, ys, labels):
        right = x > 55                      # keep labels inside the frame at high x
        dy = NUDGE.get(t, 4)
        ax.annotate(t, (x, y), textcoords="offset points",
                    xytext=(-7, dy) if right else (7, dy),
                    ha="right" if right else "left", fontsize=7.5, color="#444444")
    ax.axhline(null["y"], ls=":", c="#888888", lw=1.4)
    ax.text(0.02, null["y"], null["label"], ha="left", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color="#666666")
    rho = stats.spearmanr(xs, ys).statistic
    ax.set_title(f"{title}   Spearman rho = {rho:+.2f}", fontsize=10)
    ax.set_xlabel("GSM8K accuracy, in-house 5-shot greedy (%)")
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def main():
    base, models, gsm = load()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    panel(axes[0], models, gsm, "eq_resid",
          {"y": 0.5, "label": "null = 0.5 by construction"},
          "EQ_resid (lexical cosine partialled out)")
    panel(axes[1], models, gsm, "eq",
          {"y": base["eq"], "label": f"TF-IDF char_wb null = {base['eq']:.3f}"},
          "raw EQ (anchor AUROC)")
    axes[0].set_ylabel("held-out score at last|k1 (95% CI over pairs)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f)
               for f, c in FAM_COLOR.items()]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle("Representation of equivalent math tracks capability, not size "
                 "(10 models, 0.36-1.7B)", fontsize=11)
    fig.tight_layout()
    from .config import RESULTS
    path = RESULTS / "eq_vs_gsm8k.png"
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
