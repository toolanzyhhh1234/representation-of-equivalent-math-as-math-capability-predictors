# H2 expanded measurement follow-up

**Outcome-informed follow-up, not a confirmatory H2 result.** The same-answer
arm is primary; GSM-Symbolic is a separate secondary template-robustness arm.
Five models across three families are insufficient for the planned full-panel claim.

Protocol SHA256: `a7e1437049c0e3283893e13c9bb621e0fd3c61ef46c641c878fc84b08a048f94`; frozen at `2026-09-22T08:11:07.190640+00:00`.
Stimulus SHA256: `ddeae630544ee7f35615748480c27188898099c0b28dc09e43018d73913d5e63`. Review type: `independent_agent`.

[Protocol](../prereg/H2_FOLLOWUP.md) · [Semantic review](../prereg/h2_followup_review.json) ·
[First-pilot measurement audit](PHASE4_H2_AUDIT.md).

18 development problems and 18 new-problem holdouts.
Holdout means not previously evaluated in this project's H2 pilot; it does not mean
absent from model training. The source selection is convenience-based, not random.
Every problem has three variants in each arm. Strict: 32 fresh samples/variant;
symbolic: 8 fresh samples/variant. Existing pilot draws are not reused.

Total generations: 21,600. Batch size 64, temperature 0.7,
top-p 0.95, top-k disabled, 256-token budget, five-shot completion, bf16/SDPA.

## Primary: same-answer rewrites

### Holdout problems

| Model | EQ_resid | Accuracy | Raw INV | Eligible | Conditional INV | Cross-fitted INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 3.5% | 0.972 | 1/18 | 0.748 (low coverage) | 0.743 |
| Qwen/Qwen2.5-0.5B | 0.732 | 34.0% | 0.919 | 10/18 | 0.893 [0.865, 0.917] | 0.892 |
| microsoft/phi-1_5 | 0.603 | 26.7% | 0.940 | 9/18 | 0.908 [0.875, 0.938] | 0.903 |
| Qwen/Qwen2.5-1.5B | 0.772 | 56.8% | 0.912 | 12/18 | 0.883 [0.853, 0.914] | 0.873 |
| Qwen/Qwen2.5-Math-1.5B | 0.780 | 67.4% | 0.926 | 8/18 | 0.881 [0.825, 0.930] | 0.875 |

| Model | Canonical | Rewrites | Rewrite − canonical [problem bootstrap interval] |
|---|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 2.4% | 4.0% | +1.6 pp [-0.4, +4.5] |
| Qwen/Qwen2.5-0.5B | 35.4% | 33.3% | -2.1 pp [-8.3, +4.5] |
| microsoft/phi-1_5 | 25.9% | 27.1% | +1.2 pp [-5.7, +7.4] |
| Qwen/Qwen2.5-1.5B | 54.3% | 58.1% | +3.7 pp [-3.8, +11.4] |
| Qwen/Qwen2.5-Math-1.5B | 66.0% | 68.1% | +2.2 pp [-2.9, +7.7] |

### Development problems

| Model | EQ_resid | Accuracy | Raw INV | Eligible | Conditional INV | Cross-fitted INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 3.3% | 0.984 | 1/18 | 0.974 (low coverage) | 0.949 |
| Qwen/Qwen2.5-0.5B | 0.732 | 29.2% | 0.904 | 10/18 | 0.850 [0.803, 0.897] | 0.838 |
| microsoft/phi-1_5 | 0.603 | 28.6% | 0.895 | 12/18 | 0.861 [0.813, 0.900] | 0.850 |
| Qwen/Qwen2.5-1.5B | 0.772 | 63.7% | 0.890 | 10/18 | 0.846 [0.782, 0.896] | 0.846 |
| Qwen/Qwen2.5-Math-1.5B | 0.780 | 71.8% | 0.908 | 10/18 | 0.862 [0.792, 0.923] | 0.859 |

| Model | Canonical | Rewrites | Rewrite − canonical [problem bootstrap interval] |
|---|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 2.6% | 3.6% | +1.0 pp [-0.4, +2.4] |
| Qwen/Qwen2.5-0.5B | 30.4% | 28.6% | -1.8 pp [-8.7, +4.6] |
| microsoft/phi-1_5 | 35.1% | 25.4% | -9.6 pp [-17.9, -1.6] |
| Qwen/Qwen2.5-1.5B | 57.1% | 66.9% | +9.8 pp [-0.9, +22.4] |
| Qwen/Qwen2.5-Math-1.5B | 71.2% | 72.1% | +1.0 pp [-8.2, +12.0] |

### All problems

| Model | EQ_resid | Accuracy | Raw INV | Eligible | Conditional INV | Cross-fitted INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 3.4% | 0.978 | 2/36 | 0.861 (low coverage) | 0.846 |
| Qwen/Qwen2.5-0.5B | 0.732 | 31.6% | 0.911 | 20/36 | 0.872 [0.843, 0.900] | 0.865 |
| microsoft/phi-1_5 | 0.603 | 27.7% | 0.918 | 21/36 | 0.881 [0.848, 0.909] | 0.873 |
| Qwen/Qwen2.5-1.5B | 0.772 | 60.2% | 0.901 | 22/36 | 0.866 [0.833, 0.896] | 0.860 |
| Qwen/Qwen2.5-Math-1.5B | 0.780 | 69.6% | 0.917 | 18/36 | 0.871 [0.823, 0.912] | 0.866 |

| Model | Canonical | Rewrites | Rewrite − canonical [problem bootstrap interval] |
|---|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 2.5% | 3.8% | +1.3 pp [-0.0, +2.9] |
| Qwen/Qwen2.5-0.5B | 32.9% | 30.9% | -2.0 pp [-6.6, +2.6] |
| microsoft/phi-1_5 | 30.5% | 26.3% | -4.2 pp [-9.7, +1.2] |
| Qwen/Qwen2.5-1.5B | 55.7% | 62.5% | +6.8 pp [+0.3, +13.9] |
| Qwen/Qwen2.5-Math-1.5B | 68.6% | 70.1% | +1.6 pp [-3.7, +7.4] |

### Generation diagnostics

| Model | Parse failures | Physical token limit | Limit without answer marker or next question |
|---|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 0.1% | 99.8% | 10.1% |
| Qwen/Qwen2.5-0.5B | 0.1% | 34.5% | 5.4% |
| microsoft/phi-1_5 | 0.1% | 63.7% | 1.0% |
| Qwen/Qwen2.5-1.5B | 0.1% | 13.3% | 3.3% |
| Qwen/Qwen2.5-Math-1.5B | 0.0% | 42.9% | 0.5% |

## Secondary: template robustness

### Holdout problems

| Model | EQ_resid | Accuracy | Raw INV | Eligible | Conditional INV | Cross-fitted INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 1.6% | 0.990 | 1/18 | 0.941 (low coverage) | 0.882 |
| Qwen/Qwen2.5-0.5B | 0.732 | 31.0% | 0.894 | 10/18 | 0.847 [0.807, 0.889] | 0.816 |
| microsoft/phi-1_5 | 0.603 | 14.4% | 0.941 | 5/18 | 0.848 [0.810, 0.898] | 0.841 |
| Qwen/Qwen2.5-1.5B | 0.772 | 52.8% | 0.885 | 9/18 | 0.832 [0.795, 0.871] | 0.823 |
| Qwen/Qwen2.5-Math-1.5B | 0.780 | 67.8% | 0.897 | 11/18 | 0.861 [0.832, 0.890] | 0.836 |

### Development problems

| Model | EQ_resid | Accuracy | Raw INV | Eligible | Conditional INV | Cross-fitted INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 2.1% | 0.978 | 0/18 | — (low coverage) | 0.882 |
| Qwen/Qwen2.5-0.5B | 0.732 | 19.7% | 0.922 | 5/18 | 0.883 [0.797, 0.956] | 0.860 |
| microsoft/phi-1_5 | 0.603 | 11.3% | 0.944 | 4/18 | 0.834 (low coverage) | 0.830 |
| Qwen/Qwen2.5-1.5B | 0.772 | 41.7% | 0.879 | 10/18 | 0.827 [0.759, 0.891] | 0.815 |
| Qwen/Qwen2.5-Math-1.5B | 0.780 | 60.4% | 0.896 | 13/18 | 0.864 [0.825, 0.898] | 0.825 |

### All problems

| Model | EQ_resid | Accuracy | Raw INV | Eligible | Conditional INV | Cross-fitted INV |
|---|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 0.683 | 1.9% | 0.984 | 1/36 | 0.941 (low coverage) | 0.882 |
| Qwen/Qwen2.5-0.5B | 0.732 | 25.3% | 0.908 | 15/36 | 0.859 [0.821, 0.899] | 0.835 |
| microsoft/phi-1_5 | 0.603 | 12.8% | 0.942 | 9/36 | 0.842 (low coverage) | 0.835 |
| Qwen/Qwen2.5-1.5B | 0.772 | 47.2% | 0.882 | 19/36 | 0.829 [0.788, 0.868] | 0.818 |
| Qwen/Qwen2.5-Math-1.5B | 0.780 | 64.1% | 0.896 | 24/36 | 0.863 [0.838, 0.886] | 0.830 |

### Generation diagnostics

| Model | Parse failures | Physical token limit | Limit without answer marker or next question |
|---|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 0.1% | 99.7% | 12.2% |
| Qwen/Qwen2.5-0.5B | 0.2% | 36.5% | 6.9% |
| microsoft/phi-1_5 | 0.0% | 68.8% | 1.5% |
| Qwen/Qwen2.5-1.5B | 0.1% | 16.1% | 6.6% |
| Qwen/Qwen2.5-Math-1.5B | 0.1% | 41.9% | 0.7% |

## Matched-size base → math-tuned contrast

Descriptive paired comparison; tuning changes more than the measured EQ score.
Archived measured parameter counts: base 1,543,714,304; math 1,543,714,304.
Prior EQ_resid change: +0.0084.

- holdout: accuracy change +10.6 pp;
  7 common eligible problems, IDs `[21, 25, 29, 32, 33, 36, 39]`.
  INV(math) − INV(base) on that common subset: +0.002 [-0.038, +0.037].
- all: accuracy change +9.4 pp;
  16 common eligible problems, IDs `[0, 3, 4, 5, 6, 7, 13, 15, 19, 21, 25, 29, 32, 33, 36, 39]`.
  INV(math) − INV(base) on that common subset: +0.007 [-0.026, +0.037].

## Limits

Conditional INV may select different problems for each model. Cross-fitting selects
eligibility using half the draws and measures SD on the other half, then swaps;
it uses fewer draws per SD and is a sensitivity analysis, not an interchangeable score.
Its unique-problem counts, fold contributions, and eligibility agreement are in the JSON summaries.
Raw INV, noise-adjusted INV, half-sample repeatability and every per-problem outcome
remain available in those files; no score has been selected for a favorable ordering.

Intervals resample problems while conditioning on observed rates and eligibility.
They do not capture all Monte Carlo or selection uncertainty. More samples improve
precision but do not remove true floor/ceiling effects or justify an H2 claim at N=5.
Physical generation caps are not equivalent to unfinished answers. The parser remains
the original one, including its last-number fallback; semantic review is not a formal proof.

## Reproduction

```bash
python -m src.run_h2_followup
python -m src.analyze_h2_followup
```

Raw records and summaries: `results/h2/followup_v1/`. The first pilot is unchanged.
