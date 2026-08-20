# Representation of Equivalent Math as Math-Capability Predictors

Does the geometry of an LLM's internal representation of **mathematically equivalent
statements** predict its mathematical capability — and, more sharply, its robustness to
equivalent reformulations?

## Status: Phase 2 (20-model panel) complete. Replicates, attenuated — and a double dissociation.

| document | contents |
|---|---|
| [`REPORT.md`](REPORT.md) | **preliminary technical report** — the self-contained write-up of everything below |
| [`PLAN.md`](PLAN.md) | full design: hypotheses, panel, statistics, falsification criteria, compute budget |
| [`METRICS.md`](METRICS.md) | what EQ is, why the statistic is valid, where its validity breaks, and how to read a result |
| [`results/PHASE0.md`](results/PHASE0.md) | Phase 0: the lexical-null finding and the dissociation predictions |
| [`results/PHASE05.md`](results/PHASE05.md) | Phase 0.5: dissociated 10-model panel, lexically-controlled metrics, in-house GSM8K, metric-selection verdict |
| [`results/PHASE1_SPECIFICITY.md`](results/PHASE1_SPECIFICITY.md) | Phase 1: the specificity 2×2 — EQ_resid predicts math capability, its paraphrase mirror predicts nothing, the two are uncorrelated |
| [`results/PHASE2_PANEL20.md`](results/PHASE2_PANEL20.md) | Phase 2: 20 models / 10 families — replication (+0.69), the representation/skill double dissociation, control tuning arms, protocol hardening |
| [`RELATED_WORK.md`](RELATED_WORK.md) | prior work, mapped to the design decision each one justifies or threatens |

### Headline finding

On 20 models across 10 families (0.36–2.5B), representation of mathematically
equivalent statements tracks **math capability, not size, and not general
representation quality** — `rho(EQ_resid, in-house GSM8K) = +0.687` (p = 0.001) vs
`rho(size) = +0.33` (n.s.) and `rho(ARC-Easy) = +0.16` (n.s.) — **with a double
dissociation that bounds the claim**: Pile/code-trained models (pythia-1.4b,
deepseek-coder-1.3b) carry strong equivalence representation with ~zero math skill,
while textbook/GSM-trained models (phi-1.5, OLMo-2-1B) show the reverse. EQ_resid is
a *formal-math-representation* meter, coupled to skill only when the training recipe
couples them. The Phase 0 pre-registered SmolLM2-1.7B point resolved for capability
(size prediction excluded by the CI), and at fixed 1.54B: math tuning raises EQ_resid,
coder tuning lowers it, in every variant.

![EQ vs GSM8K](results/eq_vs_gsm8k.png)

**Metric selected for scaling: `EQ_resid`** — anchor AUROC after partialling the
TF-IDF cosine out of the model cosine (null = 0.5 by construction, all anchors kept,
most stable ordering, strongest capability tracking), with `eq_hard` as the strict
audit. Raw EQ must always be read against the lexical baseline **0.7715**
(pinned: TF-IDF `char_wb` 3-5gram, `src/lexical.py`).

Both base→math pairs at identical parameter count point the predicted way on every
metric variant (Qwen2.5 and Qwen2 pairs; the effect *grows* under lexical control).

### What is still open

H2 — that EQ predicts robustness to equivalent reformulations better than raw accuracy
— is untested: the GSM-Symbolic INV arm has not run. H1 remains correlational
(statistical lexical control, not by-construction topic-matched negatives). And the
panel is 6/10 Qwen; at fixed GSM8K, Qwen models score systematically higher EQ, which
the full panel must disentangle with more non-Qwen families.

## Running it

```bash
python -m tests.test_extract      # padding-invariance guard
python -m tests.test_parity       # fp32-GPU vs fp64-CPU parity
python -m src.run_panel           # extract + all metric variants, 10-model panel
python -m src.eval_gsm8k          # in-house capability axis (5-shot greedy GSM8K)
python -m src.analyze             # headline tables, tracking, jackknife, calibration
python -m src.figures             # results/eq_vs_gsm8k.png
python -m src.run_pilot           # (Phase 0 pipeline, kept for reproducibility)
```

Hardware: Phase 0 on a single RTX 3070 (8 GB); Phase 0.5 on a single RTX 5090 (32 GB).
Extraction 1-3 s/model and metrics ~25 s/model on the 5090; GSM8K ~2-4 min/model.
Note bf16 hidden states differ across GPU generations at the third decimal — never mix
EQ numbers extracted on different machines. Activations cache to `results/raw/`
(gitignored, ~2.3 GB for 10 models).

Data: [`uw-math-ai/MELD-dataset`](https://huggingface.co/datasets/uw-math-ai/MELD-dataset)
— 270 equivalent pairs across 18 framings, plus 541 framing-matched hard negatives.

## License

Apache-2.0 (see [`LICENSE`](LICENSE)). The bundled MELD data (`data/meld/`) is
redistributed under its own Apache-2.0 license from
[uw-math-ai/MELD-dataset](https://huggingface.co/datasets/uw-math-ai/MELD-dataset).
