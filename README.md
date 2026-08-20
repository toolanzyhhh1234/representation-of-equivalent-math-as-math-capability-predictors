# Representation of Equivalent Math as Math-Capability Predictors

Does the geometry of an LLM's internal representation of **mathematically equivalent
statements** predict its mathematical capability — and, more sharply, its robustness to
equivalent reformulations?

## Status: Phase 0 complete. Do not scale yet.

| document | contents |
|---|---|
| [`PLAN.md`](PLAN.md) | full design: hypotheses, panel, statistics, falsification criteria, compute budget |
| [`METRICS.md`](METRICS.md) | what EQ is, why the statistic is valid, where its validity breaks, and a 6-step procedure for reading a result |
| [`results/PHASE0.md`](results/PHASE0.md) | Phase 0 findings and verdict |

### Headline finding

A character n-gram TF-IDF scores **EQ = 0.7715** on the same task with no neural network.
Measured against that null rather than chance, two of five pilot models show no signal and
the rest clear it by only 0.028–0.059. MELD's hard negatives control for mathematical
*framing* but not for *topic*, so lexical overlap survives the dialect change.

What did survive is size-controlled: **Qwen2.5-Math-1.5B beat Qwen2.5-1.5B — identical
parameter count — in 9/10 settings** (mean Δ = +0.041).

### Open question

EQ ranks the 5 pilot models in the plausible order of their math ability, but
`Spearman(EQ, log params) = +0.975` on this panel, so capability and scale are not
separable here. The next experiment deliberately decorrelates them: Qwen2.5-0.5B scores
41.6 on GSM8K at 0.5B params while SmolLM2-1.7B scores 31.1 at 1.7B, so whichever way EQ
ranks that pair identifies what EQ is actually tracking.

## Running it

```bash
python -m tests.test_extract      # padding-invariance guard
python -m tests.test_parity       # fp32-GPU vs fp64-CPU parity
python -m src.run_pilot           # extract + score the panel
python -m src.report              # tables, sensitivity sweep, layer curves
```

Hardware: developed on a single RTX 3070 (8 GB). Extraction is 2–15 s/model and metrics
13–45 s/model; the dominant cost is HuggingFace download. Activations cache to
`results/raw/` (gitignored, ~1.1 GB for 5 models).

Data: [`uw-math-ai/MELD-dataset`](https://huggingface.co/datasets/uw-math-ai/MELD-dataset)
— 270 equivalent pairs across 18 framings, plus 541 framing-matched hard negatives.
