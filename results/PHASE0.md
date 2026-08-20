# Phase 0 gate — result

5 models x MELD (270 pairs, 541 framing-matched hard negatives, 1080 statements),
5 anisotropy corrections x 2 poolings x every layer.

Compute: metrics 13-45 s/model, extraction 2-15 s/model on one RTX 3070. The 60-minute
wall clock was ~55 min of HuggingFace download at ~2 MB/s, not compute.

## Verdict: DO NOT SCALE YET -- but the panel is the problem, not the metric.

Two distinct defects, which earlier drafts of this document ran together:

1. **Construct validity (unresolved).** EQ contains a topical-lexical term. It could
   track capability *because* stronger models embed mathematical text better, leaving the
   correlation intact and the mechanistic claim dead. Fixed by topic-matched negatives or
   by partialling TF-IDF cosine out of model cosine.
2. **Panel confounding (unresolved).** Size and capability are near-perfectly
   rank-correlated across the pilot panel, so no cross-model claim is identifiable. Fixed
   by adding models where the two dissociate.

What is **not** a defect, contrary to the original reading: models scoring at or below the
lexical baseline. See sec 1 and the addendum in sec 7 -- those models cannot do arithmetic,
and the metric is right to place them at the floor.

### 1. The decisive finding: chance is the wrong null

A character n-gram TF-IDF, with no neural network at all, scores **EQ = 0.7715** on the
identical anchor-AUROC task.

| baseline | EQ |
|---|---|
| binary bag-of-words | 0.6621 |
| TF-IDF word 1-gram | 0.7099 |
| TF-IDF word 1-2gram | 0.7098 |
| **TF-IDF char 3-5gram** | **0.7715** |

Against that null, the pilot's headline (pooling=last, k=1) shrinks sharply -- but the
shrinkage is not uniform noise, it is ordered by the models' actual maths ability
(GSM8K, SmolLM2 5-shot harness where available; see sec 7):

| model | EQ | vs TF-IDF | GSM8K |
|---|---|---|---|
| SmolLM2-360M | 0.7352 | -0.036 | **3.2** |
| Qwen2.5-0.5B | 0.7724 | +0.001 | 33.4 |
| Qwen3-0.6B | 0.7991 | +0.028 | n/a |
| Qwen2.5-1.5B | 0.8080 | +0.037 | 61.7 |
| Qwen2.5-Math-1.5B | 0.8307 | +0.059 | n/a |

**Read the first two rows with their GSM8K column.** SmolLM2-360M scores 3.2 on GSM8K --
it cannot do arithmetic, so it *should* sit at the lexical floor, and it does. Qwen2.5-0.5B
at 33.4 sits essentially on the baseline, matching the zero-crossing at GSM8K ~ 32.5 fitted
in sec 7. Sub-baseline EQ is the metric being calibrated, **not** the metric failing. An
earlier draft of this section read it the other way and was wrong.

What the baseline finding does establish is that **effect sizes are small** (0.00-0.06,
not the 0.24-0.33 that comparing against chance would suggest) and that reporting against
0.5 would badly overstate them.

**Why**: MELD's distractors are matched to the *target's framing* but not to the
*anchor's topic*. A true pair restates one concept, so it shares topic-specific tokens
across dialects; distractors concern different concepts in the same dialect. Lexical
overlap therefore remains a usable signal, contrary to the dataset card's claim that it
is "misleading". The negatives control for subfield, not for topic.

### 2. Layer 0 is not the clean control I claimed after the 3-model pilot

That claim held only for last-token pooling at k=1. It does not generalise:

| setting | layer-0 EQ range across models |
|---|---|
| last/k1 | 0.277 - 0.531 (near chance; a fair control) |
| mean/k0 | 0.569 - 0.601 |
| mean/k1 | 0.682 - 0.725 |
| mean/zca | **0.771 - 0.790** |

Mean-pooled layer 0 is a bag-of-embeddings, and it is strongly informative. So under mean
pooling, depth buys only **+0.01 to +0.12** over the model's own embedding layer -- nearly
the whole EQ score was present before any transformer block ran.

### 3. Ordering stability: raw EQ looks stable, the margin does not

Cross-pooling rank agreement over the 5 models (Spearman):

| correction | raw EQ | margin over layer 0 |
|---|---|---|
| k0 | -0.400 | -0.600 |
| k1 | **+1.000** | -0.400 |
| k3 | +0.500 | +0.600 |
| gapk | 0.000 | -0.100 |
| zca | +0.900 | +0.100 |

The apparent stability of raw EQ under k1/zca is largely **inherited from the baseline**.
Subtract the baseline and the between-model ordering is unstable under every correction.

**`gap_k` failed.** It was proposed here to remove only well-separated directions; it
collapses to k=0 whenever the spectrum is flat and inherits k=0's behaviour, giving the
worst rank agreement of any method (rho = 0.000). Do not use it. `zca` and `k1` are the
usable corrections; note zca inflates layer-0 EQ to ~0.79 under mean pooling, which is
evidence it amplifies surface features.

### 4. What survived: the size-controlled pair

Qwen2.5-1.5B (base) vs Qwen2.5-Math-1.5B (math-tuned), **identical parameter count**:

- raw EQ: math-tuned higher in **9/10** settings, mean delta **+0.041**
- margin over layer 0: math-tuned higher in **9/10** settings, mean delta **+0.042**
- the single inversion is last/gapk (raw) and last/zca (margin)

This is the one result not confounded with scale, and it points the predicted way. But it
is **one model pair**, and the 10 settings are nested views of the same activations, not
independent draws -- a sign test over them would be pseudo-replication.

### 5. Required changes before the panel runs

1. **Report EQ against a lexical baseline, never against 0.5.** TF-IDF char 3-5gram is
   the minimum; a frozen sentence encoder is a stronger one.
2. **Topic-match the hard negatives.** For each anchor, draw negatives from the same
   *topic* (perturbations of the true partner: quantifier weakened, injective/surjective
   swapped, hypothesis dropped) rather than from the same framing only. MELD ships some
   of these; the pool as used is not anchor-topic-matched.
3. **Or partial out lexical similarity**: regress TF-IDF cosine out of model cosine and
   compute AUROC on the residual, isolating the non-lexical component directly.
4. Pre-register `(pooling, correction)`. On current evidence: **last / k1**, chosen
   because last/k1 is the only setting whose layer-0 control sits near chance -- a
   criterion independent of the between-model ordering. Report zca as sensitivity.
5. Drop R@1 as a discriminator (1.1-8.5%, near floor, does not track EQ: Qwen2.5-1.5B
   scores EQ 0.808 with R@1 0.019).

### 6. Status

The measuring device runs correctly and fast, and its validity checks (padding
invariance >= 0.99989, fp32/fp64 parity dEQ <= 1.2e-4) pass. What it currently measures is
substantially lexical. Fixing item 1-3 above is cheap -- all of it is re-analysis of
activations already on disk, no new GPU time -- and must happen before spending the
~7-9 hours of download that the 25-30 model panel requires.

---

## 7. Addendum: reinterpretation after checking published GSM8K

Section 1 above reported two of five models scoring at or below the lexical baseline as
though it indicted the metric. Checking their actual maths ability inverts that reading.

Using one consistent harness (SmolLM2 model card / paper, GSM8K 5-shot) for the three
panel models it covers:

| model | params | GSM8K | EQ (last/k1) | margin vs 0.7715 |
|---|---|---|---|---|
| SmolLM2-360M | 0.36B | 3.2 | 0.7352 | -0.036 |
| Qwen2.5-0.5B | 0.49B | 33.4 | 0.7724 | +0.001 |
| Qwen2.5-1.5B | 1.54B | 61.7 | 0.8080 | +0.037 |

**SmolLM2-360M scores 3.2 on GSM8K.** A model that cannot do arithmetic *should not*
encode mathematical equivalence above character n-grams. Its sub-baseline EQ is the metric
being calibrated, not broken. Likewise Qwen2.5-0.5B at 33.4 landing on the baseline.

```
margin = 0.00124 * GSM8K - 0.0404          residuals +/- 0.0002
Pearson(GSM8K, margin)      = +1.0000
Pearson(log params, margin) = +0.9457
```

Capability fits better than size. This is the first evidence that discriminates between
the two explanations, and it favours capability. The fit crosses zero at **GSM8K ~ 32.5**:
below roughly 32%, a model's representation carries no more equivalence information than
character n-grams do.

**Weight this carefully.** Three points leave one residual degree of freedom, so r = 1.0000
is far less impressive than it appears; the tight residuals (0.3% of range) are the real
signal. Size and capability remain correlated across these three, so the fit cannot settle
the question. And the numbers are published, from a harness that is not ours.

### The prediction this licenses

SmolLM2-1.7B has capability like Qwen2.5-0.5B (GSM8K 31.1) and size like Qwen2.5-1.5B
(1.71B params), so the two hypotheses separate cleanly:

| if EQ tracks | predicted EQ (last/k1) |
|---|---|
| capability | **0.770** |
| size | **0.808** |

A 0.038 AUROC gap, predicted in advance. Add TinyLlama-1.1B and Falcon3-1B for two more
dissociating points, and run GSM8K in-house so the x-axis is ours rather than imported.

### What this addendum does NOT overturn

- Lexical leakage (sec 5.2 of METRICS.md) is untouched: EQ still contains a topical-lexical
  term. EQ could track capability *because* stronger models embed mathematical text better,
  in which case the correlation survives and the mechanistic claim still fails.
- The verdict stands that the panel must decorrelate size from capability before any
  cross-model claim is made.
