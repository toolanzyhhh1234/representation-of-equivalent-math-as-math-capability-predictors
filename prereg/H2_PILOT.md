# H2 feasibility pilot — protocol frozen before model inference

This operationalizes PLAN.md §5 and starts the rewrite-invariance arm. It is a
three-model feasibility pilot, not the confirmatory model-panel test. These choices
were made during setup of this pilot; do not describe all of them as previously
registered. `h2_pilot.json` records a local creation time and hashes of the protocol,
code, inputs, original EQ scores, dependencies, and model revisions. A local hash is
an integrity check, not independently timestamped public preregistration.

Technical amendment before any responses: the first launch failed validation of a
removed Transformers keyword (`use_model_defaults`) and produced zero generations.
Its protocol is preserved as `h2_pilot_v1_failed_before_inference.json`. The corrected
runner uses the explicit registered config as the model's fallback config too, and
writes to `results/h2/pilot_v2`. Scientific settings and stimuli are unchanged.

## Arms and input audit

- **Strict intended-equivalence arm:** a GSM8K original plus two AI-authored
  paraphrases, retaining the same quantities, mathematical relationships, and answer.
  One paraphrase is a compact retelling and one leads with the query. No intermediate
  results, solution program, gold answer, or reference solution appear in the target
  prompt. Original questions come from pinned `openai/gsm8k`, not from newly generated
  GSM-Symbolic instances. Text equivalence is AI-reviewed, not formally certified.
- **Template-robustness arm:** instances 0, 1 and 2 of `apple/GSM-Symbolic`, `main`.
  This dataset changes names and numbers and can change arithmetic difficulty and
  the answer. It must not be labeled same-problem semantic invariance or pooled with
  the strict arm. P1/P2, which add clauses, are excluded.

Candidate template IDs are 0–19, selected before any pilot outcomes. IDs 2 and 16 are
excluded from both arms before inference: percentage-point versus relative-probability
ambiguity in 2, and ambiguous attachment of a sales comparison in 16. This leaves
18 problems, 3 variants per arm, and 108 target questions. This is a convenience
pilot subset, not a random or representative sample of all GSM8K problems.

The source `original_id`, original text, and original gold must match pinned GSM8K.
Independent arithmetic expressions were transcribed from the givens for each retained
original and each of the 54 symbolic instances. Exact rational evaluation must equal
the source gold. Each strict paraphrase is reviewed against its original's expression;
the program check verifies the arithmetic label, not an automatic parsing of English.
All groups must have three distinct question strings; strict golds must be identical.
The source and audit are in `data/h2_rewrites.json`; the reproducible assembled input
and its SHA256 are in local `data/h2/pilot.json`.

Sources: [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
[GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic), and its
[data documentation](https://github.com/apple-aiml-research/ml-gsm-symbolic).
Keep upstream dataset licenses and canary information with the downloaded sources;
only the question field is passed to the model.

## Models and decoding

Models: SmolLM2-360M (higher representation but weak skill), Qwen2.5-0.5B (small,
moderate skill), and Phi-1.5 (lower representation but moderate GSM8K skill).
Exact checkpoint/tokenizer SHAs are frozen in the JSON. Selection uses prior repo
results, not H2 outcomes. The IV remains the existing held-out `last|k1|eq_resid`;
there is no new layer selection or activation extraction in this pilot.

Use the existing GSM8K five-shot plain-completion prompt, with the first five pinned
train examples. Unlike the older greedy capability run, estimate success probabilities
with **8 stochastic samples** per target: temperature 0.7, top-p 0.95, top-k disabled,
one beam, no repetition penalty, and at most 256 new tokens. Each target is sampled
eight times; do not count eight deterministic greedy repeats as independent samples.
All models use bf16 and SDPA on the same RTX 5090, no quantization or chat templates.

Batch size is 32, job order is deterministic by prompt character length/item/sample,
and the seed is 20260922 plus the batch index, set before every batch. Restarting uses
atomic, validated batch files and the original batch seeds. Identical seed labels do
not imply coupled or identical random draws across different models/tokenizers.
Decode token settings are saved per model; prompt + output lengths must fit the model
context without truncation. Use a fresh default GenerationConfig with the registered
sampling settings and the model's EOS IDs, rather than model-specific sampling defaults.

Answer extraction and numeric equality reuse `src/eval_gsm8k.py`: cut at a subsequent
`Question:`, use the `####` field if present, otherwise the last number. Store every
generation, prediction, correctness label, parse failure, and token-limit indicator.
This intentionally retains the old parser's limitations; do not change it after
viewing outcomes. Raw generations permit a separately labeled later scoring audit.

## Endpoints and interpretation

For each problem, let p_v be the fraction of the eight samples correct for variant v.
Preserve the planned endpoint: `INV_raw = 1 - mean_problem(std_v(p_v, ddof=0))`.
Higher values mean less dispersion; the scale lies in [0.5, 1], not [0, 1].

The main interpretable pilot readout restricts each model/arm to problems whose mean
success across variants is in **[0.15, 0.85]**, as contemplated in PLAN.md §5. Report
eligible problem counts and coverage. No eligible problems yields null, never a
perfect conditional-invariance score. Fewer than five is flagged as insufficient
coverage. This five-problem threshold is a pilot reporting rule, not a power claim.
Models can have different eligible subsets; report any common eligible intersection
separately and do not silently interpret these as matched-item comparisons.

Always accompany invariance with mean accuracy, counts of uniformly wrong/correct
problems, and (strict arm) canonical and rewritten accuracy plus their difference.
A consistently wrong model is not robust in the sense of successful reasoning.
Report per-problem/per-variant values, parse failures, and token-limit rates.

Sampling noise inflates dispersion with only eight draws. An exploratory diagnostic
subtracts `(V-1)/V * mean_v[p_v*(1-p_v)/(K-1)]` from the population variance of p_v,
clips at zero, and then takes the square root before averaging. This estimates a
variance component; clipping and square roots introduce bias. It is not a replacement
for the planned endpoint, nor proof of perfect invariance when the estimate clips.

Conditional INV intervals use 2,000 bootstrap resamples of eligible **problems**,
seed 20260922. They condition on the observed eligibility and samples; they do not
include full uncertainty from selection or repeated sampling. With 18 problems and
three models, all comparisons are descriptive. Do not report a confirmatory H2
p-value or a claim that EQ predicts invariance better than accuracy from this pilot.

The full panel should test EQ–INV against EQ–accuracy with paired/family-aware
uncertainty, size and accuracy controls, leave-family-out checks, adequate eligible
coverage, and a larger frozen stimulus set. Monte Carlo permutation sampling is not
an exact permutation test merely because it uses 100,000 shuffles. H2 remains open
until that adequately sized comparison runs.

## Run and stop rules

```bash
python -m tests.test_h2
python -m src.run_h2 --prepare
python -m src.run_h2
python -m src.analyze_h2
```

Finish the registered three models once, without outcome-based additions or exclusions.
On technical failure, keep partial records and repair under a new protocol version
if stimuli/scoring/decoding/code change. Exact resumption of interrupted batches is
allowed. Refuse modified protocol hashes, mixed manifests, changed runtime versions,
or changed stimuli. Existing archive and main panel results are preserved.

Before scaling, inspect coverage, sample uncertainty, token-limit rates, parser
failures, and rewrite quality. Any outcome-informed changes belong to a new,
explicitly exploratory or separately preregistered study.
