# H2 first-pilot measurement audit

Outcome-informed diagnostics; original predictions and scores are unchanged.

| Model | Arm | Capped generations | Capped, no next question or answer marker | Eligible problems with 4 draws | Conditional INV range with 4 draws |
|---|---|---:|---:|---|---|
| HuggingFaceTB/SmolLM2-360M | strict | 430/432 | 56/432 | [1, 2] | 0.575–1.000 |
| HuggingFaceTB/SmolLM2-360M | symbolic | 432/432 | 55/432 | [0, 1] | 0.764–1.000 |
| Qwen/Qwen2.5-0.5B | strict | 153/432 | 29/432 | [8, 13] | 0.711–0.813 |
| Qwen/Qwen2.5-0.5B | symbolic | 172/432 | 36/432 | [5, 11] | 0.775–0.886 |
| microsoft/phi-1_5 | strict | 280/432 | 2/432 | [9, 14] | 0.670–0.842 |
| microsoft/phi-1_5 | symbolic | 301/432 | 7/432 | [1, 5] | 0.646–0.843 |

The 35 complementary splits enumerate the ways to divide eight sample indices
into two groups of four. Their 70 estimates are dependent diagnostics, not a confidence
interval or 70 independent experiments. Each half re-estimates eligibility.

Physical token-budget hits greatly overstate clearly unresolved completions. Even the
absence of a marker is only a review flag; it does not prove the answer was truncated.
The registered parser's fallback to the last number can score an unfinished calculation.

At p=0.5, the binomial standard error of a success rate is 0.177 for K=8 and 0.088
for K=32. Increasing samples reduces this noise; it cannot fix a model's true accuracy floor.

The follow-up should therefore preserve the original parser/256-token budget for
comparability, increase primary-arm sampling, report canonical-vs-rewrite accuracy
on all fixed problems, and expose eligibility and cross-fitted sensitivity estimates.
New problems must remain separate from pilot development problems. Five models
still do not establish the across-family H2 claim.

## Independent semantic review

The blind reviewer audited 72 paraphrases for the expanded set without model scores.
The first pass flagged two: development template 19's query-first rewrite strengthened
a collective half into half of each category; new template 27 changed incoming emails
responded to into outgoing replies sent. Both were corrected in the follow-up fixture
and passed targeted re-review. The frozen first pilot, including its original template
19 wording and scores, remains unchanged.

[First review](../prereg/h2_followup_review_independent.json) and
[final review](../prereg/h2_followup_review_independent_final.json) bind their exact packets
by SHA256. Final status: 72 pass, no remaining required revisions. This is independent
AI review, not human certification. Shared source conventions and narrative simplification
caveats remain; exact arithmetic checks alone do not establish English equivalence.
