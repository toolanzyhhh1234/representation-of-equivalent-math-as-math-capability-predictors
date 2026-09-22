# H2 expanded measurement follow-up

This study is informed by the first pilot and its measurement audit. It is not a
confirmatory test of the full across-model H2 claim. Its purpose is to improve sample
precision and problem coverage, check transfer to new questions, and add a matched-size
base/math-tuned contrast before extending to the full model panel.

## Review gate and data

The same-answer rewrite arm is **primary**. GSM-Symbolic is a separate **secondary**
test of template robustness; changing numbers and answers can change arithmetic
difficulty. This is an explicit clarification of the original broad H2 plan.

Target input: 36 problems, three variants per arm. Eighteen are development problems
from the first pilot. Eighteen additional problems come from candidate template IDs
20–39 and have not been evaluated in our H2 experiment. Report the new-problem
holdout first, development separately, and all problems as a supplementary summary.
This is not a claim of absence from model training, nor a random sample of GSM8K.

New pre-inference exclusions: template 22 says spoons were used without clearly saying
that they no longer count toward the stated total; template 28 leaves hospital
discharges unspecified. Original exclusions 2 and 16 remain. Symbolic instance 26/2
uses an unspecified month while its gold assumes 30 days; replace it with the next
ascending unambiguous instance, 26/3. These exclusions are based on text/labels before
new-question model outcomes. Preserve the reasons in the fixture and stimulus bundle.

An independent AI reviewing agent receives only original questions and rewrites,
without model scores or experiment hypotheses. It audits mathematical quantities,
units, assumptions, relationships and requested quantities. Resolve substantive
findings, send changed wording back for re-review, and record the exact packet hash
and verdicts in `h2_followup_review.json`. The preparation command refuses a missing
approval or a review that does not match the current packet. This is independent
AI review, not human certification or a formal natural-language equivalence proof.

Rewrites are intended to preserve the mathematical problem, not every irrelevant
narrative detail. Each source's original ID, text and gold must match pinned GSM8K.
Exact rational arithmetic independently checks every original and symbolic gold.
That check validates arithmetic labels, not an automatic translation of all English
into equations. Reviewer findings and author adjudication remain available.

Changes to development rewrites are explicit overrides in the new fixture; the old
fixture, first-pilot stimulus bundle, frozen code and results remain unchanged.

## Models and generation

Five fixed models: SmolLM2-360M, Qwen2.5-0.5B, Phi-1.5, Qwen2.5-1.5B and
Qwen2.5-Math-1.5B. The two additions give a matched-size base/math-tuned contrast.
Models were selected from prior repository results and this measurement plan,
not from favorable follow-up outcomes. Preserve the archived held-out
`last|k1|eq_resid`; do not re-extract activations or select new layers.
Checkpoint/tokenizer SHAs and measured parameter counts are frozen in the JSON.
Verify counts against the newly loaded weights, never model-card size labels.

Primary arm: **32 fresh stochastic samples per variant**. Secondary arm: **8 fresh
samples per variant**. At p=0.5, increasing K from 8 to 32 halves the standard error
of a success-rate estimate from 0.177 to 0.088. This cannot eliminate a true accuracy
floor. Unequal sample counts also mean INV values across the two arms do not have
identical Monte Carlo noise; do not pool them or interpret their difference as a
pure construct effect.

For 36 problems, total budget is 4,320 generations per model, **21,600 overall**.
Use batch size 64 on the same RTX 5090, bf16, SDPA, no quantization, five-shot plain
completion with the first five pinned GSM8K train examples. Decoding: temperature
0.7, top-p 0.95, top-k disabled, one beam, no repetition penalty, max 256 new tokens.
Set the registered config as both explicit config and model fallback config. Retain
the original parser, including cutting at a subsequent `Question:` and falling back
to the last number when no `####` answer is found. Do not regrade after seeing results.

The audit found that many physical token-budget hits happen after an answer and a
subsequent few-shot question. Keep the budget comparable; report both the physical
cap and the narrower flag of reaching it without an answer marker or next question.
Neither flag is a semantic adjudication of answer completeness.

Sort jobs by prompt character length, item ID and sample index. Each batch receives
seed 20260923 plus its batch index. Save atomically, validate record hashes and scores
on resumption, and refuse mixed manifests. The final short batch is retained. All
draws are new; old pilot draws are not pooled into the larger sample. Batch size,
sample count, and any reviewed wording changes mean that differences from the old
pilot cannot be attributed solely to increased K.

## Frozen readouts

1. Preserve `INV_raw = 1 - mean_problem(population SD of variant success rates)`.
2. Main descriptive readout: conditional INV on problems whose observed mean success
   lies in [0.15, 0.85]. Always report accuracy and eligible counts. Flag fewer than
   5 eligible problems in an 18-problem partition or fewer than 10 in the full set.
   These are reporting rules, not statistical power guarantees.
3. Report canonical accuracy, mean rewritten accuracy and their paired difference
   on **all fixed problems** in each partition. This avoids changing the problem
   set between the two accuracy measurements.
4. Sensitivity: split samples into two fixed halves. Select eligibility using one
   half and measure variant SD using the other; swap and combine eligible
   problem/fold contributions. Report unique problems and eligibility agreement.
   This separates selection from measurement but uses fewer draws per SD; it is
   not numerically interchangeable with the main full-sample score.
5. Preserve the first pilot's sampling-noise adjustment and half-sample repeatability
   diagnostics in the JSON, together with every per-problem outcome. Do not choose
   whichever metric produces the most favorable EQ ordering.

Use 4,000 problem bootstrap resamples, seed 20260923, for conditional INV and the
paired canonical/rewrite accuracy difference. Intervals condition on observed
sample rates and eligibility; they do not represent all Monte Carlo/selection
uncertainty. Do not display degenerate low-coverage intervals as precise inference.

For the base/math pair, show accuracy differences on the same fixed problems and
the INV difference on their **common eligible subset**, with IDs and count. Bootstrap
paired per-problem differences only when at least five common problems exist. Tuning
changes many properties; a positive contrast is not a causal effect of EQ itself.

No confirmatory cross-model p-values or claim that H2 is established at N=5/three
families. A later full-panel study must compare EQ–INV with EQ–accuracy on a common
model set, account for ability/size and families, and retain an independently chosen
primary readout. Do not select a larger panel based on this follow-up's effect sign.

## Execution and integrity

```bash
python -m src.h2_followup_data
python -m tests.test_h2_followup
python -m tests.test_h2
# Complete and record the blind review before preparation.
python -m src.run_h2_followup --prepare
python -m src.run_h2_followup
python -m src.analyze_h2_followup
```

The protocol JSON records exact stimulus/review/code hashes, model revisions,
environment, shot text and settings before inference. A local freeze is not an
independently timestamped public preregistration. Finish all five registered models
without effect-based stopping. Technical repairs that change scientific code or
settings require a separately versioned protocol and result directory. Preserve
failed attempts and their reason; never silently merge them with corrected results.
