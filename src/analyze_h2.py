"""Summarize a completed frozen pilot; no confirmatory inference on three models."""
import json

import numpy as np

from .config import ROOT
from .h2_data import DEST, digest, validate_bundle
from .run_h2 import PROTOCOL, RUN_DIR, SETTINGS, fingerprints, make_jobs, summarize_model, validate_batch


def fmt(x, percent=False):
    if x is None: return "—"
    return f"{100*x:.1f}%" if percent else f"{x:.3f}"


def main():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["sha256"] != digest({k: v for k, v in protocol.items() if k not in ("sha256", "created_utc")}):
        raise ValueError("Protocol hash mismatch")
    if protocol["code_sha256"] != fingerprints() or protocol["settings"] != SETTINGS:
        raise ValueError("The analysis must use the frozen scoring code and settings")
    bundle = json.loads(DEST.read_text())
    validate_bundle(bundle)
    if bundle["sha256"] != protocol["stimuli_sha256"]:
        raise ValueError("Stimulus mismatch")
    jobs = make_jobs(protocol, bundle)
    batch_size = protocol["settings"]["batch_size"]
    summaries = []
    termination = []
    for model_id in protocol["models"]:
        dest = RUN_DIR / model_id.replace("/", "__")
        records = []
        for b, start in enumerate(range(0, len(jobs), batch_size)):
            path = dest / f"batch_{b:04d}.json"
            if not path.exists():
                raise FileNotFoundError(f"Pilot incomplete: missing {path}")
            records.extend(validate_batch(json.loads(path.read_text()), jobs[start:start+batch_size],
                                          protocol["sha256"], model_id))
        recomputed = summarize_model(protocol, bundle, model_id, records)
        if json.loads((dest / "summary.json").read_text()) != recomputed:
            raise ValueError(f"Saved summary differs from raw records: {model_id}")
        summaries.append(recomputed)
        termination.append({"model": model_id,
                            "next_question": np.mean(["Question:" in r["generation"] for r in records]),
                            "answer_marker": np.mean(["####" in r["generation"].split("Question:")[0] for r in records])})
    lines = [
        "# H2 rewrite-invariance feasibility pilot",
        "",
        "**Three-model pilot; H2 remains open.** Results below are descriptive and do not",
        "establish that EQ predicts invariance better than accuracy. The stimulus set is",
        "small and convenience-selected; conditional scores can use different problems.",
        "",
        f"Protocol SHA256: `{protocol['sha256']}`.",
        f"Locally frozen before inference: `{protocol['created_utc']}`.",
        f"Stimulus SHA256: `{protocol['stimuli_sha256']}`.",
        "",
        "[Frozen protocol](../prereg/H2_PILOT.md) and [machine-readable snapshot](../prereg/h2_pilot.json).",
        "Candidate templates 0–19; ambiguous templates 2 and 16 excluded before inference.",
        "18 retained problems × 3 variants × 8 samples in each of two separate arms:",
        "864 generations/model, 2,592 total. Five-shot plain completion, bf16/SDPA,",
        "temperature 0.7, top-p 0.95, top-k disabled, max 256 new tokens, batch size 32.",
        "The first launch failed on an unsupported API keyword before any responses;",
        "its snapshot is retained separately. These results use the corrected v2 runner.",
        "",
        "Gold arithmetic was independently checked with exact rational expressions.",
        "Strict paraphrases were AI-reviewed for preservation of intended meaning;",
        "there is no formal natural-language equivalence certificate.",
        "",
    ]
    for arm, title in (("strict", "Same-answer rewrites"), ("symbolic", "GSM-Symbolic template robustness")):
        lines.extend([f"## {title}", ""])
        if arm == "symbolic":
            lines.extend(["Numbers and answers change here. These are not strictly equivalent questions.", ""])
        lines.extend([
            "| Model | Prior EQ_resid | Accuracy | Raw INV | Eligible problems | Conditional INV [95% CI] | Noise-adjusted conditional INV |",
            "|---|---:|---:|---:|---:|---|---:|",
        ])
        for summary in summaries:
            s = summary["arms"][arm]
            ci = s["conditional_ci95"]
            inv = fmt(s["inv_conditional"])
            if ci is not None and s["adequate_conditional_coverage"]:
                inv += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
            if not s["adequate_conditional_coverage"]: inv += " (insufficient coverage)"
            lines.append(f"| {summary['model']} | {summary['eq_resid']:.3f} | {fmt(s['accuracy'], True)} | "
                         f"{s['inv_raw']:.3f} | {s['n_eligible']}/{s['n_problems']} | {inv} | "
                         f"{fmt(s['inv_noise_adjusted_conditional'])} |")
        eligible = [{p["template_id"] for p in s["arms"][arm]["problems"] if p["eligible"]} for s in summaries]
        common = sorted(set.intersection(*eligible))
        unit = "problem" if len(common) == 1 else "problems"
        lines.extend(["", f"Common eligible problem IDs across all models: `{common}` ({len(common)} {unit})."])
        if common:
            lines.extend(["", "| Model | INV on common eligible problems |", "|---|---:|"])
            for summary in summaries:
                sd = [p["sd"] for p in summary["arms"][arm]["problems"] if p["template_id"] in common]
                lines.append(f"| {summary['model']} | {1-np.mean(sd):.3f} |")
        if len(common) < 5:
            lines.extend(["", "The common subset is too small for a stable matched-problem comparison."])
        lines.extend(["", "| Model | Uniformly wrong problems | Uniformly correct problems | Parse failures | Token-limit rate |",
                      "|---|---:|---:|---:|---:|"])
        for summary in summaries:
            s = summary["arms"][arm]
            lines.append(f"| {summary['model']} | {s['n_always_wrong']} | {s['n_always_correct']} | "
                         f"{fmt(s['parse_failure_rate'], True)} | {fmt(s['token_limit_rate'], True)} |")
        lines.append("")
    lines.extend(["## Canonical versus rewritten accuracy", "",
                  "Stochastic success rates on the same problems; these are not the archived greedy GSM8K scores.", "",
                  "| Model | Canonical | Rewritten (two versions) | Rewrite minus canonical |",
                  "|---|---:|---:|---:|"])
    for summary in summaries:
        s = summary["arms"]["strict"]
        lines.append(f"| {summary['model']} | {fmt(s['canonical_accuracy'], True)} | "
                     f"{fmt(s['rewrite_accuracy'], True)} | {100*s['rewrite_minus_canonical']:+.1f} pp |")
    lines.extend(["", "## Interpretation limits and next step", "",
                  "Raw INV can be high for uniformly wrong answers. Conditional INV restricts each",
                  "model to problems with mean success in [0.15, 0.85]; coverage must travel with it.",
                  "Intervals bootstrap eligible problems while conditioning on observed sampling and",
                  "selection. Eight draws per variant are noisy. Noise adjustment is exploratory,",
                  "and clipping a variance estimate at zero is not evidence of perfect invariance.", "",
                  "Intervals are not displayed below five eligible problems: a single problem gives",
                  "degenerate bootstrap quantiles, which do not establish precise uncertainty.", "",
                  "Token-limit rates describe physical generation length. Some base models answer",
                  "and continue producing few-shot examples; a token-limit flag alone does not prove",
                  "that the scored answer was incomplete. The following **post-pilot diagnostic**",
                  "counts text markers over both arms, without changing any predictions or scores.", "",
                  "| Model | Contains a subsequent Question: | Contains #### before the next question |",
                  "|---|---:|---:|"])
    for row in termination:
        lines.append(f"| {row['model']} | {fmt(row['next_question'], True)} | {fmt(row['answer_marker'], True)} |")
    lines.extend(["",
                  "Before a full H2 test, independently review rewrites and audit answer extraction,",
                  "output truncation, eligibility coverage and sampling uncertainty. Register a larger",
                  "stimulus/model panel and compare EQ–INV with EQ–accuracy using appropriate family",
                  "and accuracy controls. Any changes informed by this pilot require a new protocol.", "",
                  "## Reproduction", "", "```bash", "source .venv/bin/activate",
                  "python -m src.run_h2  # resumes only missing batches with identical seeds",
                  "python -m src.analyze_h2", "```", "",
                  f"Raw responses, validated batch manifests and per-problem summaries: `{RUN_DIR.relative_to(ROOT)}/`.",
                  "Archived activation caches and earlier phase results were not modified.", ""])
    destination = ROOT / "results/PHASE4_H2_PILOT.md"
    destination.write_text("\n".join(lines))
    print(destination)


if __name__ == "__main__":
    main()
