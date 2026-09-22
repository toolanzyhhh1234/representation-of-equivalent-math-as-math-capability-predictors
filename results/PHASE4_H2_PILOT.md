# H2 rewrite-invariance feasibility pilot

**Three-model pilot; H2 remains open.** Results below are descriptive and do not
establish that EQ predicts invariance better than accuracy. The stimulus set is
small and convenience-selected; conditional scores can use different problems.

Protocol SHA256: `e878f29efe46998af7ff950b40ae59b1c38354faa550fbec9ceedba348c6eb09`.
Locally frozen before inference: `2026-09-22T07:17:52.411816+00:00`.
Stimulus SHA256: `6f0d697ed18e24de4f3b201e8c0737b794aea6957ab02686a098f1eba4e5cdbb`.

[Frozen protocol](../prereg/H2_PILOT.md) and [machine-readable snapshot](../prereg/h2_pilot.json).
Candidate templates 0–19; ambiguous templates 2 and 16 excluded before inference.
18 retained problems × 3 variants × 8 samples in each of two separate arms:
864 generations/model, 2,592 total. Five-shot plain completion, bf16/SDPA,
temperature 0.7, top-p 0.95, top-k disabled, max 256 new tokens, batch size 32.
The first launch failed on an unsupported API keyword before any responses;
its snapshot is retained separately. These results use the corrected v2 runner.

Gold arithmetic was independently checked with exact rational expressions.
Strict paraphrases were AI-reviewed for preservation of intended meaning;
there is no formal natural-language equivalence certificate.

## Same-answer rewrites

| Model | Prior EQ_resid | Accuracy | Raw INV | Eligible problems | Conditional INV [95% CI] | Noise-adjusted conditional INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 3.2% | 0.973 | 1/18 | 0.844 (insufficient coverage) | 0.917 |
| Qwen/Qwen2.5-0.5B | 0.732 | 29.9% | 0.854 | 9/18 | 0.765 [0.714, 0.829] | 0.804 |
| microsoft/phi-1_5 | 0.603 | 27.8% | 0.848 | 10/18 | 0.781 [0.723, 0.838] | 0.825 |

Common eligible problem IDs across all models: `[8]` (1 problem).

| Model | INV on common eligible problems |
|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.844 |
| Qwen/Qwen2.5-0.5B | 0.730 |
| microsoft/phi-1_5 | 0.764 |

The common subset is too small for a stable matched-problem comparison.

| Model | Uniformly wrong problems | Uniformly correct problems | Parse failures | Token-limit rate |
|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 12 | 0 | 0.0% | 99.5% |
| Qwen/Qwen2.5-0.5B | 2 | 0 | 0.0% | 35.4% |
| microsoft/phi-1_5 | 1 | 0 | 0.0% | 64.8% |

## GSM-Symbolic template robustness

Numbers and answers change here. These are not strictly equivalent questions.

| Model | Prior EQ_resid | Accuracy | Raw INV | Eligible problems | Conditional INV [95% CI] | Noise-adjusted conditional INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 1.2% | 0.993 | 1/18 | 0.941 (insufficient coverage) | 1.000 |
| Qwen/Qwen2.5-0.5B | 0.732 | 20.8% | 0.917 | 5/18 | 0.840 [0.810, 0.867] | 0.928 |
| microsoft/phi-1_5 | 0.603 | 9.5% | 0.935 | 1/18 | 0.882 (insufficient coverage) | 0.955 |

Common eligible problem IDs across all models: `[]` (0 problems).

The common subset is too small for a stable matched-problem comparison.

| Model | Uniformly wrong problems | Uniformly correct problems | Parse failures | Token-limit rate |
|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 16 | 0 | 0.2% | 100.0% |
| Qwen/Qwen2.5-0.5B | 3 | 0 | 0.0% | 39.8% |
| microsoft/phi-1_5 | 7 | 0 | 0.0% | 69.7% |

## Canonical versus rewritten accuracy

Stochastic success rates on the same problems; these are not the archived greedy GSM8K scores.

| Model | Canonical | Rewritten (two versions) | Rewrite minus canonical |
|---|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 4.9% | 2.4% | -2.4 pp |
| Qwen/Qwen2.5-0.5B | 33.3% | 28.1% | -5.2 pp |
| microsoft/phi-1_5 | 35.4% | 24.0% | -11.5 pp |

## Interpretation limits and next step

Raw INV can be high for uniformly wrong answers. Conditional INV restricts each
model to problems with mean success in [0.15, 0.85]; coverage must travel with it.
Intervals bootstrap eligible problems while conditioning on observed sampling and
selection. Eight draws per variant are noisy. Noise adjustment is exploratory,
and clipping a variance estimate at zero is not evidence of perfect invariance.

Intervals are not displayed below five eligible problems: a single problem gives
degenerate bootstrap quantiles, which do not establish precise uncertainty.

Token-limit rates describe physical generation length. Some base models answer
and continue producing few-shot examples; a token-limit flag alone does not prove
that the scored answer was incomplete. The following **post-pilot diagnostic**
counts text markers over both arms, without changing any predictions or scores.

| Model | Contains a subsequent Question: | Contains #### before the next question |
|---|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 86.0% | 85.6% |
| Qwen/Qwen2.5-0.5B | 39.2% | 88.5% |
| microsoft/phi-1_5 | 77.5% | 85.9% |

Before a full H2 test, independently review rewrites and audit answer extraction,
output truncation, eligibility coverage and sampling uncertainty. Register a larger
stimulus/model panel and compare EQ–INV with EQ–accuracy using appropriate family
and accuracy controls. Any changes informed by this pilot require a new protocol.

## Reproduction

```bash
source .venv/bin/activate
python -m src.run_h2  # resumes only missing batches with identical seeds
python -m src.analyze_h2
```

Raw responses, validated batch manifests and per-problem summaries: `results/h2/pilot_v2/`.
Archived activation caches and earlier phase results were not modified.
