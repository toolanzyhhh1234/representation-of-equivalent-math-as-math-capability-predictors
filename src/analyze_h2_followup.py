"""Recompute the complete follow-up from raw responses and report all declared readouts."""
import json

import numpy as np

from .config import ROOT
from .run_h2_followup import RUN_DIR, load_inputs, load_saved, summarize_records


def fmt(v, percent=False):
    if v is None: return "—"
    return f"{v*100:.1f}%" if percent else f"{v:.3f}"


def main():
    protocol, bundle = load_inputs(check_environment=False)
    summaries = []
    for mid in protocol["models"]:
        _, _, records, _ = load_saved(protocol, bundle, mid)
        value = summarize_records(protocol, bundle, mid, records)
        path = RUN_DIR/mid.replace("/", "__")/"summary.json"
        if json.loads(path.read_text()) != value: raise ValueError(f"Summary differs from raw records: {mid}")
        summaries.append(value)
    lines = ["# H2 expanded measurement follow-up", "",
             "**Outcome-informed follow-up, not a confirmatory H2 result.** The same-answer",
             "arm is primary; GSM-Symbolic is a separate secondary template-robustness arm.",
             "Five models across three families are insufficient for the planned full-panel claim.", "",
             f"Protocol SHA256: `{protocol['sha256']}`; frozen at `{protocol['created_utc']}`.",
             f"Stimulus SHA256: `{bundle['sha256']}`. Review type: `{protocol['reviewer_type']}`.", "",
             "[Protocol](../prereg/H2_FOLLOWUP.md) · [Semantic review](../prereg/h2_followup_review.json) ·",
             "[First-pilot measurement audit](PHASE4_H2_AUDIT.md).", "",
             f"{len(bundle['development_template_ids'])} development problems and {len(bundle['holdout_template_ids'])} new-problem holdouts.",
             "Holdout means not previously evaluated in this project's H2 pilot; it does not mean",
             "absent from model training. The source selection is convenience-based, not random.",
             "Every problem has three variants in each arm. Strict: 32 fresh samples/variant;",
             "symbolic: 8 fresh samples/variant. Existing pilot draws are not reused.", "",
             f"Total generations: {sum(s['n_generations'] for s in summaries):,}. Batch size 64, temperature 0.7,",
             "top-p 0.95, top-k disabled, 256-token budget, five-shot completion, bf16/SDPA.", ""]
    for arm, title in (("strict", "Primary: same-answer rewrites"), ("symbolic", "Secondary: template robustness")):
        lines += [f"## {title}", ""]
        for partition in ("holdout", "development", "all"):
            lines += [f"### {partition.capitalize()} problems", "",
                      "| Model | EQ_resid | Accuracy | Raw INV | Eligible | Conditional INV | Cross-fitted INV |",
                      "|---|---:|---:|---:|---:|---|---:|"]
            for m in summaries:
                s = m["arms"][arm][partition]
                inv = fmt(s["inv_conditional"])
                if s["adequate_conditional_coverage"] and s["conditional_ci95"] is not None:
                    lo, hi = s["conditional_ci95"]
                    inv += f" [{lo:.3f}, {hi:.3f}]"
                else: inv += " (low coverage)"
                lines.append(f"| {m['model']} | {m['eq_resid']:.3f} | {fmt(s['accuracy'], True)} | "
                             f"{s['inv_raw']:.3f} | {s['n_eligible']}/{s['n_problems']} | {inv} | {fmt(s['crossfit']['inv'])} |")
            if arm == "strict":
                lines += ["", "| Model | Canonical | Rewrites | Rewrite − canonical [problem bootstrap interval] |",
                          "|---|---:|---:|---:|"]
                for m in summaries:
                    s = m["arms"][arm][partition]
                    lo, hi = s["accuracy_difference_problem_bootstrap_ci95"]
                    lines.append(f"| {m['model']} | {fmt(s['canonical_accuracy'], True)} | {fmt(s['other_variant_accuracy'], True)} | "
                                 f"{s['other_minus_canonical']*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}] |")
            lines.append("")
        lines += ["### Generation diagnostics", "",
                  "| Model | Parse failures | Physical token limit | Limit without answer marker or next question |",
                  "|---|---:|---:|---:|"]
        for m in summaries:
            s = m["arms"][arm]["diagnostics"]
            lines.append(f"| {m['model']} | {fmt(s['parse_failure_rate'], True)} | {fmt(s['token_limit_rate'], True)} | "
                         f"{fmt(s['cap_without_marker_or_boundary_rate'], True)} |")
        lines.append("")
    by_model = {m["model"]: m for m in summaries}
    base, math = (by_model[mid] for mid in ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Math-1.5B"))
    lines += ["## Matched-size base → math-tuned contrast", "",
              "Descriptive paired comparison; tuning changes more than the measured EQ score.",
              f"Archived measured parameter counts: base {protocol['n_params'][base['model']]:,}; math {protocol['n_params'][math['model']]:,}.",
              f"Prior EQ_resid change: {math['eq_resid']-base['eq_resid']:+.4f}.", ""]
    for partition in ("holdout", "all"):
        allowed = set(bundle["holdout_template_ids"] if partition == "holdout" else bundle["template_ids"])
        bp = {p["template_id"]: p for p in base["arms"]["strict"]["problems"] if p["template_id"] in allowed}
        mp = {p["template_id"]: p for p in math["arms"]["strict"]["problems"] if p["template_id"] in allowed}
        common = sorted(t for t in bp if bp[t]["eligible"] and mp[t]["eligible"])
        b = base["arms"]["strict"][partition]; m = math["arms"]["strict"][partition]
        lines += [f"- {partition}: accuracy change {100*(m['accuracy']-b['accuracy']):+.1f} pp;",
                  f"  {len(common)} common eligible problems, IDs `{common}`."]
        if common:
            diffs = np.asarray([bp[t]["sd"]-mp[t]["sd"] for t in common])
            value = float(diffs.mean())
            if len(common) >= 5:
                rng = np.random.default_rng(protocol["settings"]["seed"])
                boot = diffs[rng.integers(0, len(diffs), size=(4000, len(diffs)))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                lines.append(f"  INV(math) − INV(base) on that common subset: {value:+.3f} [{lo:+.3f}, {hi:+.3f}].")
            else: lines.append(f"  INV difference {value:+.3f}; too few common problems for an interval.")
    lines += ["", "## Limits", "",
              "Conditional INV may select different problems for each model. Cross-fitting selects",
              "eligibility using half the draws and measures SD on the other half, then swaps;",
              "it uses fewer draws per SD and is a sensitivity analysis, not an interchangeable score.",
              "Its unique-problem counts, fold contributions, and eligibility agreement are in the JSON summaries.",
              "Raw INV, noise-adjusted INV, half-sample repeatability and every per-problem outcome",
              "remain available in those files; no score has been selected for a favorable ordering.", "",
              "Intervals resample problems while conditioning on observed rates and eligibility.",
              "They do not capture all Monte Carlo or selection uncertainty. More samples improve",
              "precision but do not remove true floor/ceiling effects or justify an H2 claim at N=5.",
              "Physical generation caps are not equivalent to unfinished answers. The parser remains",
              "the original one, including its last-number fallback; semantic review is not a formal proof.", "",
              "## Reproduction", "", "```bash", "python -m src.run_h2_followup", "python -m src.analyze_h2_followup", "```", "",
              f"Raw records and summaries: `{RUN_DIR.relative_to(ROOT)}/`. The first pilot is unchanged.", ""]
    target = ROOT/"results/PHASE4_H2_FOLLOWUP.md"
    target.write_text("\n".join(lines))
    print(target)


if __name__ == "__main__":
    main()
