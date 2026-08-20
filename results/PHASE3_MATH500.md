# Phase 3 — the construct-matched DV resolves the dissociation

Run 2026-08-21. Panel now **23 models / 12 families** (gated tranche landed:
Llama-3.2-1B, gemma-3-1b-pt, gemma-2-2b). New DV: **MATH-500 in-house** (4-shot,
exemplars = first 4 items, greedy, boxed-answer match after LaTeX normalization;
`src/eval_math500.py`). Figure: `results/eq_vs_math500.png`.

## Why this phase existed

Phase 2 ended with a double dissociation against GSM8K and two rival readings:
(a) EQ_resid meters formal-math *representation*, which recipes may not couple to
skill; (b) DV mismatch — GSM8K is word-problem arithmetic, MELD is subfield-dialect
theorem text, and a construct-matched DV would realign the "skill without
representation" models. These are not exclusive, and MATH-500 tests them.

## Result: both readings were right, and together they simplify the picture

| quantity | GSM8K | MATH-500 |
|---|---|---|
| ρ(EQ_resid, DV), n=23 | +0.695 (p=0.0003) | **+0.797 (p<0.0001)** |
| same, non-Qwen only (n=13) | +0.212 | **+0.558** |
| ρ(eq_hard, DV) | +0.687 | +0.796 (non-Qwen +0.608) |

- **The construct-matched DV fits better everywhere**, and the Qwen-dependence that
  was Phase 2's biggest scope limit largely dissolves (+0.21 → +0.56 outside Qwen).
- **The triangulation:** ρ(EQ_resid, MATH-500 | GSM8K) = **+0.556** — the metric
  predicts formal-math skill *beyond* GSM8K — while ρ(EQ_resid, GSM8K | MATH-500) =
  **−0.139** — its entire apparent relation to GSM8K was mediated by formal-math
  ability. EQ_resid is a formal-math instrument, full stop.
- **The "skill without representation" corner dissolves.** phi-1.5: GSM8K 31.3 →
  MATH-500 **0.2%**. OLMo-2-1B: 30.5 → **4.6%**. Their word-problem skill never was
  formal-math skill, and their low EQ_resid had it right. SmolLM2-1.7B (GSM8K 29.6 →
  MATH 3.8) reads the same way.
- **What remains is a single, clean asymmetry — necessity without sufficiency:**
  every model with EQ_resid < 0.65 scores ≤ 4.6% on MATH-500 (4/4), while above 0.65
  MATH-500 ranges 0–41.7%. Representation of formal mathematics appears **necessary
  for formal-math skill; it is not sufficient** — the exposure-without-skill models
  (pythia-1.4b, deepseek-coder-1.3b, and now Llama-3.2-1B) sit on the floor at the
  far right of the wedge.

## The gated tranche, honestly scored against Phase 0's expectations

- **Llama-3.2-1B** (GSM8K 6.0, MATH 0.6, EQ_resid 0.730) joins the
  representation-without-skill group. Under Phase 0's capability-vs-size dichotomy
  this point would have looked "size-like" and muddied that story badly — the
  exposure/necessity account explains it without size (TinyLlama, same size band,
  sits at the representation floor). Worth stating plainly: had Llama been runnable
  in Phase 0.5, the clean capability-vs-size resolution would have been messier; the
  construct that survives all 23 models is the narrowed one, not the original H1.
- **gemma-3-1b-pt** (ARC 72.6, GSM8K 2.6, MATH 0.2, EQ_resid 0.573): high general
  ability with neither math skill nor math representation — a clean specificity
  point (general quality alone does not produce EQ_resid).
- **gemma-2-2b** (ARC 80.6 — panel best, GSM8K 22.0, MATH 8.9, EQ_resid 0.724):
  on-trend.

## Status of the headline claims after 23 models

1. EQ_resid is **lexically controlled** (nulls dead by construction), **math-specific**
   (uncorrelated with its paraphrase mirror; does not predict ARC), **not a size
   effect** (partials ≈ 0), and now **construct-validated**: it predicts the
   formal-math DV better than the word-problem DV, incrementally beyond it, and with
   far better family-robustness.
2. Its proper description: **a meter of formal-mathematics representation, which is
   necessary but not sufficient for formal-math performance.** The sufficiency gap is
   exactly where skill training (or its absence) lives — pythia/deepseek/llama have
   the representation and none of the trained skill.
3. Remaining known limits: correlational; one stimulus set (MELD, 270 pairs); the
   necessity threshold (0.65) is descriptive on 23 points, not a fitted boundary with
   uncertainty; MATH-500's floor compresses 9 models below 2%, so rank correlations
   there lean on the mid-range.

## What this sets up

The H2/INV arm now has a two-sided prediction: models *above* the representation
floor with low-to-mid skill (SmolLM2-1.7B, gemma-2-2b, granite) should be more robust
under equivalent rewrites of problems they can solve than models whose skill exceeds
their representation — and the causal arm (H3) has an obvious target: ablate the
equivalence subspace in a high-representation model and formal-math performance
should degrade disproportionately on reformulated variants.
