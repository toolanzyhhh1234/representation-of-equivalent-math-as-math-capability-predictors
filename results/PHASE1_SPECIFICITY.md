# Phase 1 — the specificity control

**Question:** is `EQ_resid` a *math*-representation metric, or a general
"well-trained model" meter read on math stimuli? This was §7.1 of the report — the
main stated threat. Run 2026-08-20; code `src/specificity.py`, `src/eval_arc.py`,
`src/analyze_spec.py`; figure `results/specificity.png`.

## Design

A clean separation of "well-trained generally" from "well-trained in math" is
impossible across models — training mixes are correlated — so the tests are
**contrasts**, not separations:

- **Language-side mirror of the metric.** PAWS (labeled_final, human-verified splits):
  sentence pairs that are either true paraphrases or *deliberately high-lexical-overlap
  non-paraphrases* — the same equivalent-vs-lookalike discrimination as MELD, in
  general language. 1000+1000 pairs, same extraction, same corrections, same honest
  layer selection, same TF-IDF partialling (`PARA_resid`). The TF-IDF null on PAWS is
  **0.475** — at/below chance by design, the mirror image of MELD's 0.7715 defect.
- **Language-side mirror of the DV.** ARC-Easy in-house, zero-shot per-char-normalized
  logprob (no generation), full test set.
- T1: base→math-tuned pairs (general training held ~constant by construction).
- T2: the 2×2 metric×DV differential. T3: incremental partial correlations.

Panel dissociation on the DV side: Spearman(GSM8K, ARC) = **+0.10** — the two
capability axes genuinely come apart on this panel (math tuning trades ARC for GSM8K:
Qwen2.5-Math loses 8 ARC points against its base).

## Result: the general-quality explanation is substantially refuted

| | GSM8K (math DV) | ARC-Easy (language DV) |
|---|---|---|
| **EQ_resid** (math metric) | **+0.818** (p = 0.005) | +0.164 (p = 0.66) |
| **PARA_resid** (language metric) | −0.515 (p = 0.13) | −0.515 (p = 0.14) |

- `rho(EQ_resid, GSM8K) − rho(PARA_resid, GSM8K) = +1.33`, model-bootstrap 95% CI
  **[+0.33, +1.87]** — excludes zero. The math metric's tracking of math capability is
  not reproduced by its language-side mirror.
- **EQ_resid and PARA_resid are uncorrelated across models (ρ = −0.30).** If both were
  reading one underlying "representation quality," they would covary. They do not.
- T3: `rho(EQ_resid, GSM8K | PARA_resid) = +0.81` (nothing is absorbed);
  `rho(EQ_resid, ARC | PARA_resid) = +0.01` (EQ_resid knows nothing about non-math
  capability). The double null is the point: a general-quality meter would fail both.

Model-level illustrations: TinyLlama-1.1B — bottom of every math measure — has the
**highest** PARA_resid on the panel (0.655); SmolLM2-1.7B pairs the panel's best ARC
(74.4) with mid EQ_resid; the math-tuned Qwens gain GSM8K and EQ while *losing* ARC.

## What did not resolve

1. **T1 (the paired contrast) is inconclusive at current precision.** Math-tuning moved
   PARA_resid too (+0.017, +0.033), comparable to its EQ_resid deltas (+0.008, +0.045).
   The per-model bootstrap CIs are ±0.02-0.03, so these deltas are within noise, and
   math post-training corpora contain plenty of natural-language reasoning text — a
   real PARA gain is plausible, not just noise. The sharp prediction "tuning moves EQ,
   not PARA" is neither confirmed nor refuted. More pairs, or paired-bootstrap deltas
   over stimuli, would settle it.
2. **PARA is a weak instrument, and that cuts both ways.** Its cross-model spread
   (0.59–0.65) is a quarter of EQ_resid's (0.54–0.78), and it fails to track even
   ARC (ρ = −0.52, n.s., wrong sign). Supporting reading: at 0.4–1.7B, models differ
   enormously in math-equivalence geometry while barely differing in paraphrase
   geometry — consistent with representational-convergence on ordinary language
   (RELATED_WORK, 2605.23315) and divergence exactly where specialized training
   differs. Skeptical reading: pooled-AUROC-on-PAWS may be too noisy an instrument,
   in which case its failure to correlate is uninformative about *shared* variance —
   though this does not touch the EQ×ARC null (+0.16/+0.01), which uses no PAWS data.
3. The negative PARA↔DV correlations (−0.5, n.s.) are unexplained; with n = 10 they
   are consistent with noise. Do not interpret their sign.

## Verdict for the report

The main threat in REPORT.md §7.1 is downgraded from "untested" to "tested, with the
general-quality explanation disfavoured on three independent contrasts" — EQ_resid
does not correlate with the language metric, does not predict the language DV, and
loses nothing when the language metric is partialled out. What remains open is the
finer-grained version: whether the *math-tuning* effect specifically is math-specific
(T1 inconclusive), and whether a stronger language-side instrument than PAWS would
find shared variance the current one cannot see.

Answer to the design question this phase raised ("why separate general from math
quality at all, when equivalence is math-native?"): because the proxy's failure mode
was that it might be readable off any well-trained model regardless of math. That is
now tested and rejected — the metric earns the word *math* in its name, at this panel
size. And the deeper point stands: MELD equivalence is theorem-mediated (recognizing
"spans V" ≡ "no proper submodule contains S" requires knowing mathematics), while
paraphrase is meaning-mediated; the 2×2 shows models can be good at one and not the
other, in both directions.
