# Phase 2 — 20-model panel: replication, attenuation, and a double dissociation

Run 2026-08-21. Panel doubled to 20 models across 10 families (ungated tranche;
Llama-3.2/Gemma pending gated access). All metrics, GSM8K, ARC, and PAWS re-run under
the same pinned harnesses. This phase is the out-of-family test of the Phase 0.5/1
result, and it both **replicates the headline and complicates it in an informative
way**.

## 0. Protocol hardening shipped with this phase

1. **Measured parameter counts** now recorded at extraction (safetensors-header backfill
   for the original panel). One card value was materially wrong: **Qwen3-0.6B is
   0.752B**, not 0.60B. The size axis uses measured values everywhere.
2. **Layer 0 is excluded from headline layer-selection.** deepseek-coder-1.3b exposed
   the need: its *layer-0* eq_resid is 0.775 (every other model: 0.23-0.39) — a
   lexical leak through the tokenizer channel that char-3-5gram TF-IDF cannot
   residualize, and split-A selection then picked layer 0 as the readout. Layer 0 is
   the token-identity *control* (METRICS.md sec 6 step 3); selecting the control as
   the measurement contradicts the construct. With the exclusion, deepseek-coder reads
   from layer 9 (0.709), which is a genuine mid-network signal (its eq_hard = 0.67
   concurs). All models re-scored under the amended rule.

## 1. The headline replicates, attenuated

n = 20, held-out at last|k1, in-house GSM8K:

| variant | ρ(GSM8K) | p | ρ(log params) | ρ(cap∣size) | ρ(size∣cap) |
|---|---|---|---|---|---|
| eq | +0.741 | 0.0002 | +0.383 | +0.694 | +0.138 |
| **eq_resid** | **+0.687** | **0.001** | +0.327 | +0.642 | +0.075 |
| eq_hard | +0.666 | 0.002 | +0.287 | +0.628 | +0.026 |
| eq_lextop | +0.714 | 0.0004 | +0.358 | +0.667 | +0.108 |

Down from +0.82 at n=10 — expected out-of-sample shrinkage plus real structure (§3).
Capability-not-size survives (all size partials ≈ 0). The specificity 2×2 also holds at
n=20: ρ(EQ_resid, ARC) = +0.16; EQ_resid ⊥ PARA_resid (−0.10); metric contrast
ρ(EQ,GSM8K)−ρ(PARA,GSM8K) = +0.82, CI [+0.23, +1.34]; DV contrast +0.53,
CI [+0.00, +1.02].

## 2. The new finding: a double dissociation

Four of the ten new models break the monotone story, in two opposite and
interpretable ways:

| group | model | GSM8K | eq_resid | eq_hard |
|---|---|---|---|---|
| **representation without skill** | pythia-1.4b | 2.2 | **0.705** | 0.630 |
| | deepseek-coder-1.3b | 4.2 | **0.709** | 0.673 |
| **skill without representation** | phi-1.5 | 31.3 | 0.603 | **0.496 ≈ null** |
| | OLMo-2-1B | 30.5 | 0.577 | 0.575 |

These are not artifacts: pythia's signal peaks mid-network (layer 11/25; layer 0 is a
normal 0.36) and survives the restriction-based audit metric, and deepseek-coder's
post-fix readout is likewise mid-network. The natural reading is **training-data
exposure**: Pythia (the Pile: arXiv, math StackExchange) and DeepSeek-Coder (code +
math-adjacent web) saw plenty of *formal mathematical prose* and represent
theorem-statement equivalence, while having no problem-solving skill. Phi-1.5
(synthetic "textbook" data) and OLMo-2 (GSM-style mid-training) took the opposite
route: word-problem skill with little exposure to subfield-dialect theorem text —
phi-1.5's eq_hard is *at chance*.

TinyLlama remains the both-absent corner; the Qwen/Granite/Falcon/StableLM/SmolLM
models occupy the both-present-in-proportion diagonal that produces the correlation.

**Implication for the construct.** EQ_resid is best described as a meter of
*formal-math representation* — closely tied to what the model was exposed to — not a
direct meter of math skill. It correlates with capability exactly insofar as a
training recipe couples theorem-prose exposure with skill training (which frontier
recipes do, and which the panel's dissociated corners deliberately do not). An
alternative, testable reading is DV mismatch: GSM8K is grade-school *word-problem
arithmetic*, while MELD is subfield-dialect *theorem equivalence*; a formal-math DV
(MATH-500 or theorem-level tasks) might realign phi/OLMo. Roadmap item.

## 3. Subgroup structure — the honest decomposition

| subset | ρ(eq_resid, GSM8K) |
|---|---|
| full panel (20) | +0.687 |
| Qwen only (10) | +0.394 |
| non-Qwen (10) | +0.212 |
| non-Qwen minus the two exposure outliers (8; post-hoc) | +0.476 |

The full-panel correlation is carried substantially by the coarse contrast between the
high-exposure-high-skill cluster and the rest; within-cluster range restriction leaves
little resolvable signal at these n's. The leave-one-family-out jackknife is stable
for every family except Qwen (dropping Qwen: +0.21), which is now a **stated scope
limit**, not a footnote: as a capability proxy, EQ_resid ranks models reliably across
coarse capability differences and within recipe-consistent families, but should not be
trusted to rank heterogeneous-recipe models of similar capability.

## 4. Same-size tuning contrasts now have control arms

| contrast (all at 1.54B) | d_eq | d_eq_resid | d_eq_hard |
|---|---|---|---|
| Qwen2.5 → **Math** | +0.022 | +0.008 | +0.014 |
| Qwen2 → **Math** | +0.028 | +0.045 | +0.047 |
| Qwen2.5 → **Coder** (control) | **−0.010** | **−0.018** | **−0.017** |
| Qwen2.5 → **Instruct** (control) | +0.017 | +0.009 | +0.034 |

The coder control is the cleanest specificity evidence yet at fixed size: math tuning
raises equivalence representation, coder tuning *lowers* it, on every variant. The
instruct result is intermediate — consistent with Qwen2.5-Instruct's post-training
containing substantial math (its GSM8K: 55.2).

## 5. Verdict for the scaling question

- The metric survives its validity audits at n=20 (lexical null, layer-0 control,
  specificity 2×2, capability-vs-size).
- Its **construct claim is narrowed**: formal-math representation (exposure-linked),
  not math skill. The proxy is trustworthy for coarse capability ranking and
  within-recipe comparisons; it is *not* a universal capability meter across
  arbitrary training recipes — pythia-class and phi-class models are systematic,
  predictable exceptions.
- The dissociation is itself the most interesting scientific object this project has
  produced: representation and execution of mathematics separate cleanly at small
  scale. The H2/INV arm gains motivation — robustness-to-reformulation is exactly
  where a representation-side quantity should bite, and phi/OLMo (skill without
  representation) are the models on which H2 makes its sharpest prediction:
  **their success should be unusually fragile under equivalent rewrites.**

## 6. Pending

Gated tranche (Llama-3.2-1B — the originally pre-registered dissociator — plus
gemma-3-1b-pt, gemma-2-2b): token is installed but repo access returns 403; the
account must accept each model's license on its HF page, and the fine-grained token
must have "read access to public gated repos" enabled. MATH-500 as the second,
construct-matched math DV. Then the INV arm.
