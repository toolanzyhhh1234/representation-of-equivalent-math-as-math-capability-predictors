"""Render the Phase 0 gate table + layer curves from results/pilot.json."""
import json
import sys

import numpy as np

from .config import CORRECTIONS, POOLINGS, RESULTS
from .run_pilot import headline


def table(res, pool, k="k1"):
    print(f"\n### pooling={pool}  PC-removal k={k}")
    print(f"{'model':<34} {'EQ(held-out)':>12} {'EQ(best)':>9} {'gap':>7} "
          f"{'layer':>8} {'R@1':>6} {'MRR':>6}")
    for r in res:
        h = headline(r, pool, k)
        print(f"{r['model']:<34} {h['EQ_heldout']:>12.4f} {h['EQ_bestlayer_all']:>9.4f} "
              f"{h['align_gap']:>7.4f} {str(h['sel_layer'])+'/'+str(h['n_layers']):>8} "
              f"{h['recall@1']:>6.3f} {h['mrr']:>6.3f}")


def sensitivity(res, pool="last"):
    """PLAN sec 4.3: if the effect survives only at k=0, it is an anisotropy artifact."""
    print(f"\n### anisotropy sensitivity (pooling={pool}), EQ held-out")
    print(f"{'model':<34}" + "".join(f"{k:>10}" for k in CORRECTIONS))
    for r in res:
        row = "".join(f"{headline(r, pool, k)['EQ_heldout']:>10.4f}" for k in CORRECTIONS)
        print(f"{r['model']:<34}{row}")


def curves(res, pool="last", k="k1"):
    print(f"\n### EQ by relative depth (pooling={pool}, k={k}) -- chance = 0.500")
    for r in res:
        c = r["curves"][f"{pool}|{k}"]
        n = len(c)
        marks = [int(round(f * (n - 1))) for f in (0, .25, .5, .75, 1.0)]
        vals = "  ".join(f"{c[i]['auroc_anchor']:.3f}@{i}" for i in marks)
        peak = max(c, key=lambda x: x["auroc_anchor"])
        print(f"{r['model']:<34} {vals}   peak {peak['auroc_anchor']:.3f}@L{peak['layer']}")


def plot(res, pool="last", k="k1", path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in res:
        c = r["curves"][f"{pool}|{k}"]
        y = [x["auroc_anchor"] for x in c]
        x = np.linspace(0, 1, len(y))
        ax.plot(x, y, marker="o", ms=2.5, lw=1.4, label=r["model"].split("/")[-1])
    ax.axhline(0.5, ls="--", c="0.5", lw=1, label="chance")
    ax.set_xlabel("relative depth (layer / n_layers)")
    ax.set_ylabel("EQ  (anchor AUROC vs framing-matched hard negatives)")
    ax.set_title(f"MELD equivalence discrimination — pooling={pool}, top-{k} PC removed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = path or RESULTS / "eq_layer_curves.png"
    fig.savefig(path, dpi=150)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    res = json.loads((RESULTS / "pilot.json").read_text())
    for pool in POOLINGS:
        for m in CORRECTIONS:
            table(res, pool, m)
    sensitivity(res, "last")
    sensitivity(res, "mean")
    curves(res, "last", "gapk")
    plot(res, "last", "gapk")
