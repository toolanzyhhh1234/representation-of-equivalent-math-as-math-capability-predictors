# Phase 0 gate — result

5 models x MELD (270 pairs, 541 framing-matched hard negatives, 1080 statements),
5 anisotropy corrections x 2 poolings x every layer.

Compute: metrics 13-45 s/model, extraction 2-15 s/model on one RTX 3070. The 60-minute
wall clock was ~55 min of HuggingFace download at ~2 MB/s, not compute.

## Verdict: DO NOT SCALE YET. The metric measures too much lexical overlap.

### 1. The decisive finding: chance is the wrong null

A character n-gram TF-IDF, with no neural network at all, scores **EQ = 0.7715** on the
identical anchor-AUROC task.

| baseline | EQ |
|---|---|
| binary bag-of-words | 0.6621 |
| TF-IDF word 1-gram | 0.7099 |
| TF-IDF word 1-2gram | 0.7098 |
| **TF-IDF char 3-5gram** | **0.7715** |

Against that null, the pilot's headline (pooling=last, k=1) mostly evaporates:

| model | EQ | vs TF-IDF |
|---|---|---|
| SmolLM2-360M | 0.7352 | **below baseline** |
| Qwen2.5-0.5B | 0.7724 | tied |
| Qwen3-0.6B | 0.7991 | +0.028 |
| Qwen2.5-1.5B | 0.8080 | +0.037 |
| Qwen2.5-Math-1.5B | 0.8307 | +0.059 |

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
