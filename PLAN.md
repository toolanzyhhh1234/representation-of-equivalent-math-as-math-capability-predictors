# Representation of Equivalent Math as a Predictor of Math Capability

**Status:** draft plan, pre-data
**Unit of analysis:** the *model*, not the problem. This is the constraint everything else bends around.

---

## 1. Hypotheses

**H1 (headline, weak).** Across models, the degree to which internal representations identify
mathematically equivalent statements is positively associated with mathematical capability.

**H2 (primary, sharp).** Equivalence-alignment predicts a model's **robustness to equivalent
reformulations** (spread of success across certified-equivalent variants of the same problem)
*better* than it predicts raw benchmark accuracy.

**H3 (causal, confirmatory).** Ablating the equivalence-carrying subspace degrades performance on
reformulated variants **more** than on canonical forms.

H2 is the primary endpoint. Reasons:

- It is the mechanistically motivated claim. "Geometry predicts general ability" is a scale proxy
  wearing a costume; "geometry predicts invariance" is the thing the geometry would actually cause.
- It sidesteps your benchmark objection. Robustness is measured *within* a benchmark as dispersion
  across equivalent variants, so the choice of benchmark shifts the level but not the construct.
- Prior work leaves it open. Existing results either measure the geometry without capability
  (MELD / arXiv 2606.23959) or measure fragility without opening the model (miniF2F-rw / 2605.22257).

H1 is reported for completeness and is expected to be confounded (see §6.3).

---

## 2. Variables

| Role | Name | Definition |
|---|---|---|
| IV | `EQ` | Equivalence-alignment score from hidden states (§4) |
| DV-primary | `INV` | Rewrite invariance: `1 - mean_problem(std of success across its equivalent variants)` |
| DV-secondary | `ACC_b` | Accuracy on benchmark *b* — reported per benchmark, never pooled into one number silently |
| Control | `log_params` | log10 total parameters |
| Control | `math_tuned` | binary: math/proof-specialized post-training |
| Cluster | `family` | Qwen / Llama / DeepSeek / Mistral / Gemma / OLMo ... |

`ACC_b` stays a **vector**, not a scalar. If the sign or magnitude of ρ differs across GSM8K vs MATH
vs miniF2F vs PutnamBench, that heterogeneity is a finding, not noise to average away. Report the
per-benchmark ρ table as a primary figure.

---

## 3. Model panel

**Hardware-constrained scope: 0.5B–2B parameters, fp16, uniform precision.** See §12.

Target **N = 25–30** open-weight models. This number is set by power (§6.2), not convenience.

Sampling must be **crossed, not a convenience sample**:

- 4–5 size tiers across ~0.5B / 1B / 1.5B / 1.7–2B
- **base / math-tuned pairs at identical size.** Only ~3 clean ones exist at this scale:
  Qwen2.5-1.5B vs Qwen2.5-Math-1.5B (canonical), Qwen2-1.5B vs Qwen2-Math-1.5B,
  Qwen2.5-1.5B-Instruct vs DeepSeek-R1-Distill-Qwen-1.5B. The paired arm is correspondingly weaker
  than originally planned — report effect sizes with CIs, not verdicts.
- **checkpoint arm: OLMo-2-1B** (public intermediate checkpoints, non-trivial math ability).
  Pythia has far more checkpoints but floors on math (Pile-trained, GSM8K ≈ 0), making it useless
  as a DV source. Use it only as an IV-side sanity check.

Candidate pool: Qwen3-0.6B/1.7B, Qwen2.5-0.5B/1.5B(/-Math/-Instruct), Qwen2-0.5B/1.5B,
Llama-3.2-1B(/-Instruct), SmolLM2-360M/1.7B, Gemma-3-1B, OLMo-2-1B, stablelm-2-1.6B,
TinyLlama-1.1B, Falcon3-1B, InternLM2-1.8B, Phi-1.5, MiniCPM-1B, DeepSeek-R1-Distill-Qwen-1.5B.

**Range-restriction caveat.** Models whose GSM8K accuracy floors (< 0.15) must be excluded from the
primary `INV` computation (§5), which may drop effective N to ~18–22 and cut power to detecting
ρ ≥ 0.60 rather than 0.55. Prefer candidates trained with some math data. Record the exclusion
count explicitly.

**Scope statement for write-up:** findings established at 0.5–2B; generalization to frontier scale
untested. The IV is cheap at any scale (§12), so a large-model extension is a rented-GPU day if
warranted — the DV generation is what would not be.

The paired and checkpoint sub-designs are the scientific core; the cross-model correlation is the
headline number but the weaker evidence.

---

## 4. Independent variable: measuring `EQ`

### 4.1 Stimuli

| Source | Content | N pairs |
|---|---|---|
| MELD (2606.23959) | equivalent statements in different subfield vocabularies | 270 |
| miniF2F-rw (2605.22257) | Lean-certified equivalent variants | 5–15 per problem |
| Formal↔informal | theorem statement in Lean 4 vs. its natural-language form | build ~300 |
| Templated | GSM-Symbolic-style surface rewrites of the same computation | generate ~500 |

**Hard negatives are not optional.** For every positive pair, sample a distractor that is
*same-subfield, high-lexical-overlap, and non-equivalent* (e.g. the theorem with a quantifier
flipped or a hypothesis dropped). Without them, `EQ` measures topical clustering — exactly the
failure mode MELD documents — and the whole study measures nothing.

Validate the negatives: a Lean check or a 2-annotator pass on a 100-item sample. Report agreement.

### 4.2 Extraction

For each statement, at each layer ℓ: last-token hidden state **and** mean-pooled hidden state.
Report both; pre-register last-token as primary.

### 4.3 Anisotropy correction — critical

Raw cosine similarity between LLM hidden states is dominated by a shared mean direction and a few
high-variance components. Uncorrected, `EQ` largely measures each model's anisotropy, which
correlates with scale, which correlates with capability — a fully spurious H1.

Per model, per layer, before any similarity is computed:

1. center on the mean activation over the full stimulus set,
2. remove the top-*k* principal components (k ∈ {0, 1, 3}; report sensitivity),
3. optionally whiten.

If the headline result survives only at k=0, it is an anisotropy artifact. Say so.

### 4.4 Score

Primary: **AUROC** of separating positive pairs from hard negatives using cosine similarity, at the
best layer (selected on a held-out *split of the stimuli*, not on the outcome).

**The null is a lexical baseline, not 0.5.** Phase 0 measured TF-IDF over character 3-5
grams at **EQ = 0.7715** on this exact task with no neural network (results/PHASE0.md).
Reporting EQ against chance overstates every model by ~0.27 and put two of five pilot
models at or below a bag-of-characters. Every EQ number must be reported as a margin over
that baseline, and the baseline recomputed for any new stimulus set. Stronger still:
partial the TF-IDF cosine out of the model cosine and take AUROC on the residual.

- Scale-free, threshold-free, no calibration needed, bounded — good properties for a small-N
  rank correlation.
- Secondary: alignment gap `mean cos(equiv) − mean cos(hard-neg)`; RSA between the model's
  representational dissimilarity matrix and the ground-truth equivalence-class structure;
  dimensionality of the linearly-decodable equivalence subspace (FARS-style, 2605.09496).

**Measurement error matters.** With only 270 MELD pairs, each model's `EQ` has real sampling
variance, which attenuates ρ toward zero. Bootstrap `EQ` per model (1000 resamples over *pairs*),
report its CI, and report a disattenuated ρ alongside the raw one.

---

## 5. Dependent variables

- `INV`: run each model on **GSM-Symbolic** (templated equivalent rewrites of GSM8K — the right
  difficulty band for a 0.5–2B panel; miniF2F floors at ~0% here and cannot serve as the DV),
  k samples per variant.
  Per problem, compute success rate per variant, then the SD across that problem's variants.
  `INV = 1 - mean(SD)`. Also record mean success so `INV` can be reported conditional on it —
  a model that fails everything has trivially low variance, and that ceiling/floor artifact must
  be handled explicitly (restrict to problems where mean success ∈ [0.15, 0.85], or model the
  variance against its binomial expectation).
- `ACC_b`: GSM8K, MATH-500, SVAMP, ASDiv. (miniF2F/PutnamBench dropped — floor effects at this
  scale.) Run in-house with fixed decoding — **do not** import
  reported numbers, they differ in prompting and would inject noise straight into the IV–DV link.

---

## 6. Statistics

### 6.1 Primary test

Spearman ρ between `EQ` and `INV` across models, with an **exact permutation test**
(shuffle `EQ` labels, 100k permutations) rather than the asymptotic *t* approximation, which is
unreliable at N < 30.

Kendall's τ-b reported alongside: better-behaved small-sample null distribution and handles ties
more gracefully. If ρ and τ disagree in significance, trust τ and say why.

### 6.2 Power — decide N before collecting

Approximate N for 80% power, α = 0.05 two-tailed (Spearman, ≈1.1× the Pearson requirement):

| true ρ | N needed |
|---|---|
| 0.7 | ~15 |
| 0.6 | ~21 |
| 0.5 | ~32 |
| 0.4 | ~46 |

At N = 25 the study is powered for **ρ ≥ 0.55 and no better**. State this in the paper. A null
result at N = 25 rules out a strong relationship, not a moderate one — and that must be the
stated conclusion, not "no relationship."

### 6.3 Confound control

- **Partial Spearman** controlling `log_params`: rank-transform all three, regress out, correlate
  residuals. Honest risk: `log_params` may absorb most of the signal, leaving nothing. That is a
  real possible outcome, not a bug to engineer around.
- **Paired within-size analysis:** base vs. math-tuned at identical parameter count. Wilcoxon
  signed-rank on (Δ`EQ`, Δ`INV`) across pairs. Immune to the size confound entirely. With ~5 pairs
  this is underpowered as a standalone test but is strong *corroborating* evidence — report the
  effect and its CI, not a p-value verdict.
- **Checkpoint analysis:** within one model's training run, does `EQ` rise before, with, or after
  `INV`? Architecture is held exactly constant. This is the cleanest evidence available.

### 6.4 Non-independence

25 models from 6 families are not 25 independent draws. Report a **leave-one-family-out jackknife**
of ρ: if dropping a single family moves ρ by more than ~0.15, the result is family-driven and must
be presented that way.

### 6.5 Multiple comparisons

Pre-register exactly one primary test (§6.1). Everything else — every layer, pooling method, k for
PC removal, metric variant, benchmark — is exploratory and reported under Benjamini–Hochberg FDR
with the full test count disclosed. Layers × metrics × benchmarks runs to hundreds of tests; an
uncorrected "significant at layer 19" is worthless.

---

## 7. Causal arm (H3)

On 3–5 models where H2 holds:

1. Identify the equivalence subspace: concept-centroid PCA over equivalence classes, or a linear
   probe direction set (FARS-style, 2605.09496 — a ~10-dim subspace sufficed there).
2. Ablate it (project out) and re-run the rewrite benchmark.
3. **Control ablation:** remove a random subspace of equal dimension, and a matched-variance
   subspace. Without these, any ablation effect is just "damage."
4. Predicted: accuracy on reformulated variants drops disproportionately vs. canonical forms.
   Test with the interaction term, not two separate drops.

---

## 8. Falsification — decided in advance

The hypothesis is **rejected** if:

- partial ρ(`EQ`, `INV` | `log_params`) has a 95% bootstrap CI containing 0, **and**
- the paired base-vs-math-tuned Δ shows no consistent sign, **and**
- ablating the equivalence subspace is no worse than the matched random control.

Write this section before running anything. It is the difference between a study and a search.

---

## 9. Phases

| Phase | Work | Gate |
|---|---|---|
| 0. Pilot (~1 wk) | 3 models (1 small base, 1 large base, 1 math-tuned), MELD only, `EQ` + `INV` | Does `EQ` separate the 3 models at all, in the expected order? If not, stop and fix the metric. |
| 1. Stimuli (~2 wk) | Build/validate hard negatives, formal↔informal set, templated set; annotator agreement | Negative set validated, equivalence certified |
| 2. Pre-registration | Freeze §6 primary test, §8 falsification, model panel | Committed to file, hashed |
| 3. Sweep (~3 wk) | Full N=25–30 × all stimuli × all layers; all DVs run in-house | Complete matrix, bootstrap CIs |
| 4. Causal (~2 wk) | H3 on 3–5 models with controls | — |
| 5. Write-up (~2 wk) | Per-benchmark ρ table as primary figure | — |

Phase 0 is a genuine gate. If three models chosen to differ maximally don't order correctly on
`EQ`, the instrument is broken and no amount of N fixes it.

---

## 10. Risks

1. **Probes read format, not reasoning.** arXiv 2606.02907 is a direct methodological attack on this
   instrument. Mitigations: hard negatives (§4.1), the causal arm (§7), and reporting `EQ` for a
   format-only control task (same statement, different LaTeX rendering) which should *not* predict
   `INV`.
2. **Anisotropy artifact.** §4.3. Highest-probability silent failure.
3. **Everything is scale.** §6.3. Possible honest outcome; the paired design is the hedge.
4. **Representational convergence.** 2605.23315 — models may agree representationally while
   differing in reasoning, capping `EQ`'s ceiling and compressing its variance across the panel.
   Check the spread of `EQ` early (Phase 0); if all models score within a narrow band, no rank
   correlation can exist regardless of the underlying truth.
5. **`INV` floor/ceiling artifact.** §5. Handle by restriction or by comparing to binomial
   expectation, and state which.

---

## 11. Repo layout

```
data/          stimuli: meld/, minif2f_rw/, formal_informal/, templated/, negatives/
src/extract/   hidden-state extraction, per-layer, cached to disk
src/metrics/   anisotropy correction, AUROC, RSA, subspace dim
src/eval/      benchmark harness (fixed decoding, in-house)
src/stats/     permutation Spearman, partial rank, family jackknife, BH-FDR
prereg/        frozen pre-registration + hash
results/       per-model json, one file per (model, stimulus set)
notebooks/     exploratory only, never the source of a reported number
```


---

## 12. Compute budget — single RTX 3070 (8 GB, ~6.5 GB usable under WSL2)

### 12.1 Hard rules

1. **Never quantize.** 4-bit alters hidden-state geometry and anisotropy — exactly the quantity
   being measured (§4.3). Mixing precisions injects the confound straight into the IV; uniform
   4-bit still corrupts it. A model that needs quantization to fit does not belong in the panel.
2. **Uniform fp16/bf16 across the entire panel.** Non-negotiable for IV comparability.
3. Cast to **fp32 before** centering / PCA / whitening. fp16 PCA is numerically unreliable and the
   correction in §4.3 is the load-bearing step.

### 12.2 Fits

| params | fp16 weights | verdict on ~6.5 GB usable |
|---|---|---|
| 0.6B | 1.2 GB | trivial |
| 1.5B | 3.0 GB | comfortable, batched generation fine |
| 1.7B | 3.4 GB | comfortable |
| 3B | 6.0 GB | extraction only, batch 1–2, no headroom — avoid |
| 4B+ | 8 GB+ | does not fit |

### 12.3 Cost per stage

**Measured, Phase 0: the bottleneck is network, not GPU.** A 5-model run took ~60 min
wall clock, of which ~55 min was HuggingFace download at ~2 MB/s (xet client throwing
`IncompleteMessage` retries). Metrics were 13-45 s/model and extraction 2-15 s/model.
For a 25-30 model panel at ~2 GB each that is **7-9 hours of pure downloading** --
larger than every other cost combined. Therefore: pre-fetch the whole panel in a
separate overnight loop so the science run never blocks on the network, and A/B
`HF_HUB_DISABLE_XET=1` on one model first (2 MB/s looks like protocol trouble, not a
link ceiling). Cached re-runs of the same panel are ~5 min, so metric iteration is free.

- **IV / extraction (cheap).** Forward passes only. ~1500 statements × ~30 layers per model;
  seconds to low minutes each, **well under an hour for the whole panel**.
  Cache **last-token and mean-pooled states only** — never full sequences. Storage ≈ 7 GB total.
- **DV / generation (the real cost).** GSM-Symbolic at ~100 templates × 10 variants × 8 samples
  ≈ 8000 generations × ~300 tokens = ~2.4M tokens per model. vLLM at 1.5B: ~15–25 min/model.
  **Full panel ≈ one overnight run.**
- vLLM on 6.5 GB: set `gpu_memory_utilization` conservatively and cap `max_model_len`; at 1.5B,
  ~3 GB weights leaves ~3 GB KV cache, enough for useful batching.

### 12.4 Silent-bug watchlist

- **Padding side vs. last-token extraction.** With right-padding, the final position is a pad token.
  Index the true last non-pad token per sequence, or left-pad. This bug produces a plausible-looking
  but meaningless `EQ` and will not announce itself.
- Anisotropy statistics (§4.3) must be fit **per model per layer** on that model's own stimulus
  set — never shared across models.
- Fix decoding params identically across the panel (temperature, top-p, seed, max tokens); any
  drift enters the DV as noise.

### 12.5 When to rent

The IV stays cheap at any scale, so a frontier-scale extension is ~1 day on a rented H100-80GB
(≈ $50–70) for extraction across a few large models. DV generation at that scale is what does not
fit a hobby budget — do not start it without deciding the sample size first.
