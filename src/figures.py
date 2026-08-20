"""Panel figures: the dissociation scatter (results/eq_vs_gsm8k.png).

Left: the recommended metric (eq_resid) against in-house GSM8K.
Right: raw eq against the same axis, with the TF-IDF null drawn in.

With 20 models, points are colored by the grouping the Phase 2 data itself revealed:
the two directions of the representation/skill double dissociation, vs everything else.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .analyze import PRIMARY, load

# Okabe-Ito, CVD-safe. Group assignment is the Phase 2 finding, fixed here.
EXPOSURE = {"EleutherAI/pythia-1.4b", "deepseek-ai/deepseek-coder-1.3b-base",
            "meta-llama/Llama-3.2-1B"}
SKILL = {"microsoft/phi-1_5", "allenai/OLMo-2-0425-1B"}
GROUPS = [("representation without skill", "#D55E00", EXPOSURE),
          ("skill without representation", "#009E73", SKILL),
          ("rest of panel", "#0072B2", None)]

LABEL = {  # annotate only the models that carry the story
    "EleutherAI/pythia-1.4b": "pythia-1.4b",
    "deepseek-ai/deepseek-coder-1.3b-base": "deepseek-coder-1.3b",
    "microsoft/phi-1_5": "phi-1.5",
    "allenai/OLMo-2-0425-1B": "OLMo-2-1B",
    "TinyLlama/TinyLlama_v1.1": "TinyLlama-1.1B",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B",
    "Qwen/Qwen2.5-Math-1.5B": "Qwen2.5-Math-1.5B",
    "ibm-granite/granite-3.3-2b-base": "granite-3.3-2b",
}


def color_of(mid):
    for _, c, members in GROUPS:
        if members is None or mid in members:
            return c


def panel(ax, models, gsm, variant, null, title):
    for r in models:
        if r["model"] not in gsm:
            continue
        h = r["headlines"][f"{PRIMARY}|{variant}"]
        x, y, c = 100 * gsm[r["model"]]["acc"], h["heldout"], color_of(r["model"])
        ax.plot([x, x], [h["ci_lo"], h["ci_hi"]], color=c, lw=1, alpha=0.4)
        ax.scatter([x], [y], c=c, s=40, zorder=3)
        if r["model"] in LABEL:
            right = x > 55
            dy = {"pythia-1.4b": -13, "deepseek-coder-1.3b": 8,
                  "SmolLM2-360M": -13}.get(LABEL[r["model"]], 4)
            ax.annotate(LABEL[r["model"]], (x, y), textcoords="offset points",
                        xytext=(-7, dy) if right else (7, dy),
                        ha="right" if right else "left", fontsize=7.5, color="#444444")
    ax.axhline(null["y"], ls=":", c="#888888", lw=1.4)
    ax.text(0.40, null["y"], null["label"], ha="left", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color="#666666")
    xs = [100 * gsm[r["model"]]["acc"] for r in models if r["model"] in gsm]
    ys = [r["headlines"][f"{PRIMARY}|{variant}"]["heldout"] for r in models if r["model"] in gsm]
    rho = stats.spearmanr(xs, ys).statistic
    ax.set_title(f"{title}   Spearman rho = {rho:+.2f}", fontsize=10)
    ax.set_xlabel("GSM8K accuracy, in-house 5-shot greedy (%)")
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def main():
    base, models, gsm = load()
    n = sum(1 for r in models if r["model"] in gsm)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    panel(axes[0], models, gsm, "eq_resid",
          {"y": 0.5, "label": "null = 0.5 by construction"},
          "EQ_resid (lexical cosine partialled out)")
    panel(axes[1], models, gsm, "eq",
          {"y": base["eq"], "label": f"TF-IDF char_wb null = {base['eq']:.3f}"},
          "raw EQ (anchor AUROC)")
    axes[0].set_ylabel("held-out score at last|k1 (95% CI over pairs)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lab)
               for lab, c, _ in GROUPS]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle(f"Equivalence representation vs math capability ({n} models, 0.36-2.5B) "
                 "-- correlated, with a double dissociation", fontsize=11)
    fig.tight_layout()
    from .config import RESULTS
    path = RESULTS / "eq_vs_gsm8k.png"
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
