# Phase 0 gate — result

Run: 3 models x MELD (270 pairs, 541 framing-matched hard negatives, 1080 statements).
Wall clock: ~4 min total on one RTX 3070. Extraction was **2.1-6.5 s per model**.

## Verdict: PASS with a caveat that changes the metric spec

### 1. The instrument works

EQ is far above chance everywhere (chance = 0.500), so hidden states genuinely carry
equivalence information that survives framing-matched lexical distractors.

Two internal validity checks passed:

- **Padding invariance**: min cos(batched, unbatched) >= 0.99989 on all three models.
  The last-token index is hitting real tokens, not PADs.
- **Layer 0 is at chance** (0.44-0.57). The embedding layer is token identity, so it
  *should* fail on lexically-dissimilar equivalents. It does. This is the strongest
  evidence that EQ is not silently measuring lexical overlap.
- Layer curves peak at ~40-55% relative depth and decline after, matching the usual
  finding that mid-layers carry the most semantic abstraction.

### 2. Headline (pooling=last, k=1, layer chosen on split A, EQ reported on split B)

| model | EQ (held-out) | sel layer | R@1 | MRR |
|---|---|---|---|---|
| SmolLM2-360M   | 0.7353 | 20/33 | 0.019 | 0.049 |
| Qwen2.5-0.5B   | 0.7725 | 11/25 | 0.026 | 0.053 |
| Qwen3-0.6B     | 0.7992 | 11/29 | 0.044 | 0.106 |

Ordering matches the pre-stated prediction (SmolLM2 < Qwen2.5-0.5B < Qwen3-0.6B).

### 3. THE CAVEAT: the between-model ordering is not robust to the anisotropy hyperparameter

EQ held-out, by pooling and number of principal components removed:

| pooling | model | k=0 | k=1 | k=3 |
|---|---|---|---|---|
| last | SmolLM2-360M | 0.7091 | 0.7353 | **0.7663** |
| last | Qwen2.5-0.5B | 0.6984 | 0.7725 | 0.7394 |
| last | Qwen3-0.6B   | 0.7673 | 0.7992 | 0.7834 |
| mean | SmolLM2-360M | 0.6110 | 0.7077 | 0.8215 |
| mean | Qwen2.5-0.5B | 0.6174 | 0.7153 | 0.8427 |
| mean | Qwen3-0.6B   | **0.5946** | 0.7451 | 0.8553 |

- **last pooling**: the middle two models swap at k=0 AND at k=3. Only k=1 gives the
  predicted order. Only Qwen3-0.6B is stably top.
- **mean pooling**: at k=0 the order is fully inverted (Qwen3 is last, and barely above
  chance at 0.595). Removing 3 PCs adds **+0.23 AUROC** — the correction is not a
  correction, it is the dominant term.

This is precisely the failure mode PLAN sec 4.3 was written to catch, and it fired.

### 4. Consequences for the design

1. **(pooling, k) must be pre-registered before the panel runs**, on grounds independent
   of the outcome. Picking mean/k=3 because it gives the highest EQ and the nicest
   ordering is exactly the selection the pre-registration exists to forbid.
2. **The k-curve becomes a primary figure, not a robustness appendix.** Any headline rho
   must be shown across all k, and ordering instability reported honestly.
3. Consider replacing the free parameter with a parameter-free correction (full ZCA
   whitening, or all-but-the-top with k set by an explained-variance threshold fixed
   across models) so no post-hoc tuning knob exists.
4. **Retrieval is near floor** (R@1 1.9-4.4% vs ~25% for tuned embedding models in
   arXiv 2606.23959). Base LMs are not embedders. Keep MRR as secondary; R@1 has too
   little dynamic range at this scale to discriminate.

### 5. What this pilot does NOT show

EQ ordering here is **perfectly confounded with parameter count** (360M < 500M < 600M).
Nothing in Phase 0 separates "tracks math ability" from "tracks size". That is expected
at N=3 and is what the paired base/math-tuned design and the OLMo-2 checkpoint arm
(PLAN sec 3, sec 6.3) exist to resolve. No capability numbers were measured yet.

**Do not read this as evidence for H1 or H2.** It is evidence that the measuring device
works and that its free parameter is dangerous.
