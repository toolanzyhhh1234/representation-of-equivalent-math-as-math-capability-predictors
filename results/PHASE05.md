# Phase 0.5 — metric selection on a dissociated panel

**Question answered here:** which representation metric should scale to the full panel,
and does EQ track capability or just size? Run 2026-08-20 on one RTX 5090; 10 models,
MELD, all metrics from `src/run_panel.py`, capability measured in-house
(`src/eval_gsm8k.py`, 5-shot greedy, full 1319-item test set, one harness for every
model). Analysis: `src/analyze.py`. Figure: `results/eq_vs_gsm8k.png`.

## Verdict

1. **The size hypothesis is rejected.** The panel decorrelates size from capability
   (`Spearman(GSM8K, log params) = +0.23`, vs +0.975 in the Phase 0 panel), and on it
   every EQ variant tracks capability, not parameters:
   `rho(EQ_resid, GSM8K) = +0.818` (permutation p = 0.005), `rho(EQ_resid, log params)
   = +0.250` (p = 0.49); partials `+0.807` vs `+0.109`.
2. **The pre-registered point prediction resolved for capability.** Phase 0 predicted
   SmolLM2-1.7B at EQ 0.770 if EQ tracks capability, 0.808 if size. Observed: **0.748
   held-out, 0.769 full-set, 95% CI [0.742, 0.796]** — the CI excludes the size
   prediction and contains the capability one.
3. **Metric to scale: `EQ_resid`** (anchor AUROC after partialling the TF-IDF cosine
   out of the model cosine). Selection table below.
4. Phase 0's *linear calibration* story does not survive the in-house axis: the
   relationship is monotone but not tightly linear, and the "zero-crossing at
   GSM8K ~ 32.5" was a 3-point artifact (see §5).

## 1. Panel and capability axis

In-house GSM8K (5-shot, exemplars = first 5 train items, greedy, max 256 tokens,
answer = `#### n` else last number). Published same-harness numbers are shown only as a
cross-check; they were never used in the analysis.

| model | params (B) | GSM8K in-house | published (SmolLM2 harness) |
|---|---|---|---|
| SmolLM2-360M | 0.36 | 4.6 | 3.2 |
| TinyLlama_v1.1 | 1.10 | 2.4 | — |
| Qwen2.5-0.5B | 0.49 | 35.1 | 33.4 |
| Qwen3-0.6B | 0.60 | 40.1 | — |
| Falcon3-1B-Base | 1.67 | 30.6 | — |
| SmolLM2-1.7B | 1.71 | 29.6 | 31.1 |
| Qwen2-1.5B | 1.54 | 58.3 | — |
| Qwen2-Math-1.5B | 1.54 | 63.2 | — |
| Qwen2.5-1.5B | 1.54 | 59.6 | 61.7 |
| Qwen2.5-Math-1.5B | 1.54 | 72.8 | — |

The three checkable models agree with their published numbers to 1.4-1.7 points, so the
harness is sane. The dissociating rows are **TinyLlama-1.1B** (1.1B params, 2.4%
GSM8K), **SmolLM2-1.7B** (largest model, 29.6%), and **Falcon3/SmolLM2-1.7B vs the
Qwen 1.5Bs** (same size band, 30 vs 60-73%).

Llama-3.2-1B was pre-registered as a dissociator but is gated on HF and this machine
has no token; TinyLlama-1.1B and Falcon3-1B-Base fill the role.

## 2. The metric variants (src/lexical.py)

Phase 0 established that raw EQ contains a topical-lexical term (TF-IDF char_wb 3-5gram
scores 0.7715 with no neural network). Three lexically-controlled variants, computed
per anchor exactly like EQ:

| variant | construction | null |
|---|---|---|
| `eq` | anchor AUROC, unchanged | TF-IDF = **0.7715** |
| `eq_resid` | AUROC of `model_cos - b*tfidf_cos` (b fit per model/layer/correction on all candidate scores) | **0.5 by construction** |
| `eq_hard` | AUROC only vs negatives at least as lexically close to the anchor as the true partner | **0.5**; TF-IDF itself scores **0.0**. Coverage 71% of anchors, ~10 negatives each |
| `eq_lextop` | AUROC vs each anchor's 5 lexically-closest negatives | TF-IDF = **0.4656** |

Reproducibility pin: the Phase 0 baseline is `sklearn TfidfVectorizer(analyzer="char_wb",
ngram_range=(3,5))` — plain `char` gives 0.7608, not 0.7715. The baseline script was
never committed in Phase 0; `src/lexical.py` now is the source of truth. `char_wb` with
`lowercase=False` is stronger still (0.7770) — any future "margin over baseline" claim
should check against it as sensitivity.

## 3. Headline table (last|k1, layer chosen on split A, reported on split B, 95% CI by pair bootstrap)

| model | GSM8K | eq | eq margin | eq_resid | eq_hard | eq_lextop |
|---|---|---|---|---|---|---|
| TinyLlama_v1.1 | 2.4 | 0.616 | **-0.156** | **0.538** | **0.506** | 0.530 |
| SmolLM2-360M | 4.6 | 0.737 | -0.035 | 0.683 | 0.672 | 0.653 |
| SmolLM2-1.7B | 29.6 | 0.748 | -0.024 | 0.704 | 0.670 | 0.665 |
| Falcon3-1B-Base | 30.6 | 0.797 | +0.026 | 0.744 | 0.712 | 0.715 |
| Qwen2.5-0.5B | 35.1 | 0.774 | +0.002 | 0.732 | 0.714 | 0.699 |
| Qwen3-0.6B | 40.1 | 0.801 | +0.029 | 0.764 | 0.716 | 0.742 |
| Qwen2-1.5B | 58.3 | 0.769 | -0.003 | 0.703 | 0.657 | 0.662 |
| Qwen2.5-1.5B | 59.6 | 0.808 | +0.036 | 0.772 | 0.741 | 0.739 |
| Qwen2-Math-1.5B | 63.2 | 0.796 | +0.025 | 0.748 | 0.703 | 0.724 |
| Qwen2.5-Math-1.5B | 72.8 | 0.830 | +0.058 | 0.780 | 0.755 | 0.744 |

Two readings that matter:

- **TinyLlama is the canary.** 1.1B parameters, no math: raw EQ far below even the
  lexical baseline, and every controlled variant at its null (eq_resid CI
  [0.494, 0.571]). A size-driven metric would have put it near 0.79. Its mean-pooled
  *layer-0* raw EQ is 0.768 — a bag of its input embeddings alone nearly matches
  TF-IDF, which is the lexical leak of Phase 0 made vivid; the controlled variants
  collapse that same quantity to 0.47-0.59.
- **Non-lexical signal exists below the capability floor.** SmolLM2-360M (GSM8K 4.6)
  scores eq_resid 0.683 — far above its null — while sitting below the raw-EQ lexical
  baseline. Phase 0's reading "sub-baseline EQ = a model that cannot do arithmetic,
  correctly at the floor" was right about the *margin* but wrong to imply the
  representation carries nothing: what a 360M model lacks is enough capability to beat
  a *strong lexical scorer*, not equivalence signal per se. Representation quality and
  execution come apart at the bottom of the scale — TinyLlama (2.4%) genuinely has
  neither, SmolLM2-360M (4.6%) has representation without execution. This is new
  information that only the residual metric exposes, and it is the first hint that the
  IV may dissociate from raw accuracy in the direction H2 needs.

## 4. Which metric scales? The selection criteria

| criterion | eq | eq_resid | eq_hard | eq_lextop |
|---|---|---|---|---|
| null is lexically dead | no (0.7715) | **yes (0.5 exact)** | yes (0.5; TF-IDF -> 0.0) | partial (0.4656) |
| layer-0 control near null (last\|k1), mean abs dev | 0.287 | 0.207 (below-null side) | **0.049** | **0.017** |
| ordering stability across corrections, mean pairwise rho | 0.718 | **0.726** | 0.623 | 0.715 |
| rho with in-house GSM8K (permutation p) | +0.806 (0.006) | **+0.818 (0.005)** | +0.648 (0.047) | +0.794 (0.008) |
| rho with log params (partial given cap) | +0.238 (+0.089) | +0.250 (+0.109) | -0.044 (-0.262) | +0.206 (+0.038) |
| leave-one-family-out rho range | +0.62..+1.00 | **+0.74..+1.00** | +0.52..+0.80 | +0.69..+1.00 |
| anchors retained | all | **all** | 71% | all |

**Decision: pre-register `EQ_resid` at `last|k1` as the primary metric for the full
panel.** It keeps every anchor, has the highest capability correlation and the most
stable ordering, survives every family drop, and its null is exactly 0.5 with the
lexical channel removed by construction. Report `eq_hard` alongside as the strict
audit (its TF-IDF score is 0 by construction, so any value above 0.5 is unarguably
non-lexical), with its 71% coverage stated. Report raw `eq` with the TF-IDF margin for
continuity with Phase 0. `eq_lextop` adds nothing over `eq_hard` and can be dropped.

Caveats attached to the decision: `eq_resid`'s layer-0 value sits *below* its null
(0.23-0.39) — the partialling over-corrects where the representation is purely lexical.
That is the acceptable side to fail on (no positive lexical signal can leak through),
but it means eq_resid at layer 0 is not a flat 0.5, and the layer-selection split must
stay. More generally (METRICS.md sec 7): equivalent statements *legitimately* share
vocabulary, so partialling TF-IDF out removes real signal along with the artifact —
**eq_resid is a lower bound**, biased toward false negatives, per the hazard Sahoo et
al. (2606.02907 sec 7) flag for their own residualization. That is precisely why
`eq_hard` ships alongside as the audit: it repairs the comparison set rather than
subtracting from the measurement, so it does not carry the bias — at the price of 71%
coverage and noisier pools (family jackknife 0.52-0.80, leaning hardest on the
TinyLlama anchor point). When the two disagree, trust the restriction.

## 5. What did NOT survive Phase 0: the linear calibration

Phase 0 fit `margin = 0.00124 * GSM8K - 0.0404` on three imported points (r = 1.0000,
residuals ±0.0002) and predicted a zero-crossing at GSM8K ~ 32.5. On ten in-house
points: slope +0.00196, Pearson r = +0.78, max residual 0.079, nominal crossing ~ 42 —
and the crossing is not a real boundary: **Falcon3-1B (GSM8K 30.6) sits at margin
+0.026 while SmolLM2-1.7B (GSM8K 29.6) sits at -0.024.** Same capability, opposite
sides of the baseline. The rank relationship is strong (rho +0.81) but the tight
linearity and the crossing were artifacts of fitting 3 points from one model-family
lineage. The Phase 0 addendum flagged exactly this risk ("three points leave one
residual degree of freedom"); it was right to.

The residual scatter is itself informative: at fixed GSM8K, Qwen models score higher
EQ than non-Qwen models of the same capability (see the jackknife: dropping Qwen
raises rho to 1.0 for the remaining 4). Either family-specific representation style
enters EQ, or GSM8K underestimates the Falcon/Qwen difference in the ability EQ
reflects. The full panel needs enough non-Qwen families to tell.

## 6. The paired arm now has two pairs, and lexical control sharpens it

Base -> math-tuned at identical parameter count, held-out headline:

| pair | eq | eq_resid | eq_hard |
|---|---|---|---|
| Qwen2.5-1.5B -> Qwen2.5-Math-1.5B | +0.022 | +0.008 | +0.014 |
| Qwen2-1.5B -> Qwen2-Math-1.5B | +0.028 | +0.045 | +0.047 |

Both pairs positive on every variant (8/8 including eq_lextop). The Qwen2 pair's
effect *grows* under lexical control — the math-tuning gain is not a lexical artifact.
Still only two pairs; the Wilcoxon arm of the full panel remains necessary.

## 7. Numbers hygiene

- bf16 extraction is hardware-sensitive at the third decimal: SmolLM2-360M EQ(last|k1)
  is 0.7352 on the Phase 0 RTX 3070 vs 0.7366 here. Cross-machine EQ tables must not
  be mixed; re-extract when the machine changes. All Phase 0.5 numbers are from this
  machine.
- GSM8K harness details are in the result file (`results/gsm8k.json`) and pinned in
  `src/eval_gsm8k.py`. In-house levels differ from any published harness by a few
  points (that is why they are in-house).
- All EQ numbers in this file are held-out (layer chosen on split A, value from
  split B); full-set values are in `results/panel.json`.

## 8. What this phase does not establish

- H1 remains correlational: EQ_resid could still track capability because stronger
  models embed math text better in ways beyond n-grams. The statistical control
  removes the TF-IDF channel, not every lexical channel — a frozen sentence encoder
  as the partialled regressor is the next-strongest control, and topic-matched
  negatives remain the by-construction fix.
- H2 (rewrite invariance as the DV) is untouched — GSM-Symbolic INV was not run.
- N = 10 with 6 Qwen models; the capability tracking within the non-Qwen minority is
  perfect (rho = 1.0 on 4 points) but 4 points is 4 points.
- Qwen3-0.6B is a post-trained (hybrid) release scored with the base-model harness;
  its GSM8K may understate it. It is not load-bearing for any conclusion above.

## 9. Gate for the full panel

Scale to N = 25-30 only with: (1) `EQ_resid`@`last|k1` pre-registered as primary IV,
`eq_hard` as audit; (2) capability measured in-house with the pinned harness; (3) at
least 4 model families with ≥ 3 non-Qwen dissociating points; (4) the INV arm
(GSM-Symbolic) budgeted, since H2 is the claim the study exists for. Items the panel
does not need: more MELD-only metric variants — the instrument question is settled.
