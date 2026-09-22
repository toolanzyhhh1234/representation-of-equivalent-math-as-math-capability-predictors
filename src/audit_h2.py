"""Outcome-informed measurement audit; does not change any first-pilot score."""
from itertools import combinations
import json

import numpy as np

from .config import ROOT
from .h2_metrics import summarize
from .run_h2 import PROTOCOL, RUN_DIR, make_jobs, validate_batch


def main():
    protocol = json.loads(PROTOCOL.read_text())
    bundle = json.loads((ROOT/"data/h2/pilot.json").read_text())
    jobs = make_jobs(protocol, bundle)
    output = {"protocol_sha256": protocol["sha256"], "models": {}}
    rows = []
    splits = [(0, *c) for c in combinations(range(1, 8), 3)]
    for model_id in protocol["models"]:
        folder = RUN_DIR/model_id.replace("/", "__")
        records = []
        for b, start in enumerate(range(0, len(jobs), 32)):
            records.extend(validate_batch(json.loads((folder/f"batch_{b:04d}.json").read_text()),
                                          jobs[start:start+32], protocol["sha256"], model_id))
        by_key = {(r["item_id"], r["sample"]): r for r in records}
        model = {}
        for arm in ("strict", "symbolic"):
            items = [r for r in bundle["items"] if r["arm"] == arm]
            y = np.asarray([[[by_key[r["item_id"], k]["correct"] for k in range(8)]
                             for r in items if r["template_id"] == tid] for tid in bundle["template_ids"]])
            halves = []
            for split in splits:
                other = [k for k in range(8) if k not in split]
                halves.append([summarize(y[:, :, idx], n_boot=20) for idx in (list(split), other)])
            estimates = [h["inv_conditional"] for pair in halves for h in pair if h["inv_conditional"] is not None]
            counts = [h["n_eligible"] for pair in halves for h in pair]
            rr = [r for r in records if r["item_id"].startswith(arm+"/")]
            capped = [r for r in rr if r["hit_token_limit"]]
            unresolved = [r for r in capped if "Question:" not in r["generation"] and "####" not in r["generation"]]
            d = {"n_responses": len(rr), "token_limit_count": len(capped),
                 "cap_without_boundary_or_marker": len(unresolved),
                 "no_answer_marker_count": sum("####" not in r["generation"].split("Question:")[0] for r in rr),
                 "half_sample_eligible_range": [min(counts), max(counts)],
                 "half_sample_inv_range": [min(estimates), max(estimates)] if estimates else None,
                 "complementary_half_splits": halves}
            model[arm] = d
            rows.append((model_id, arm, d))
        output["models"][model_id] = model
    folder = ROOT/"results/h2/audit"
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"measurement_audit.json").write_text(json.dumps(output, indent=2)+"\n")
    lines = ["# H2 first-pilot measurement audit", "",
             "Outcome-informed diagnostics; original predictions and scores are unchanged.", "",
             "| Model | Arm | Capped generations | Capped, no next question or answer marker | Eligible problems with 4 draws | Conditional INV range with 4 draws |",
             "|---|---|---:|---:|---|---|"]
    for mid, arm, d in rows:
        lo, hi = d["half_sample_inv_range"] or (float('nan'), float('nan'))
        lines.append(f"| {mid} | {arm} | {d['token_limit_count']}/{d['n_responses']} | "
                     f"{d['cap_without_boundary_or_marker']}/{d['n_responses']} | "
                     f"{d['half_sample_eligible_range']} | {lo:.3f}–{hi:.3f} |")
    lines += ["", "The 35 complementary splits enumerate the ways to divide eight sample indices",
              "into two groups of four. Their 70 estimates are dependent diagnostics, not a confidence",
              "interval or 70 independent experiments. Each half re-estimates eligibility.", "",
              "Physical token-budget hits greatly overstate clearly unresolved completions. Even the",
              "absence of a marker is only a review flag; it does not prove the answer was truncated.",
              "The registered parser's fallback to the last number can score an unfinished calculation.", "",
              "At p=0.5, the binomial standard error of a success rate is 0.177 for K=8 and 0.088",
              "for K=32. Increasing samples reduces this noise; it cannot fix a model's true accuracy floor.", "",
              "The follow-up should therefore preserve the original parser/256-token budget for",
              "comparability, increase primary-arm sampling, report canonical-vs-rewrite accuracy",
              "on all fixed problems, and expose eligibility and cross-fitted sensitivity estimates.",
              "New problems must remain separate from pilot development problems. Five models",
              "still do not establish the across-family H2 claim.", "",
              "## Independent semantic review", "",
              "The blind reviewer audited 72 paraphrases for the expanded set without model scores.",
              "The first pass flagged two: development template 19's query-first rewrite strengthened",
              "a collective half into half of each category; new template 27 changed incoming emails",
              "responded to into outgoing replies sent. Both were corrected in the follow-up fixture",
              "and passed targeted re-review. The frozen first pilot, including its original template",
              "19 wording and scores, remains unchanged.", "",
              "[First review](../prereg/h2_followup_review_independent.json) and",
              "[final review](../prereg/h2_followup_review_independent_final.json) bind their exact packets",
              "by SHA256. Final status: 72 pass, no remaining required revisions. This is independent",
              "AI review, not human certification. Shared source conventions and narrative simplification",
              "caveats remain; exact arithmetic checks alone do not establish English equivalence.", ""]
    (ROOT/"results/PHASE4_H2_AUDIT.md").write_text("\n".join(lines))
    print(ROOT/"results/PHASE4_H2_AUDIT.md")


if __name__ == "__main__":
    main()
