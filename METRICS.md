# Metrics: what we measure, why it is valid, and how to read a result

Companion to `PLAN.md` (design) and `results/PHASE0.md` (the Phase 0 findings).
Implementation: `src/metrics.py`, `src/correct.py`.

---

## 1. The pipeline

For each model, each layer `l`, each pooling, extraction produces

```
X  in  R^(1080 x H)        one row per MELD statement
```

then, in order:

1. **anisotropy correction** — centre, then remove structure (top-k PCs, or ZCA); `src/correct.py`
2. **row-normalise** — so every subsequent dot product is a cosine
3. **score** — the metrics below

The measured quantity is therefore a property of the **pair** (representation, correction),
not of the representation alone. Section 5.3 explains why this matters more than it sounds.

---

## 2. Primary metric: anchor-wise AUROC (`EQ`)

For a MELD pair with statements `a` (framing `f_a`) and `b` (framing `f_b`):

| | |
|---|---|
| positive score | `cos(a, b)` |
| negative scores | `cos(a, d)` for every distractor `d` **in b's framing** |
| per-anchor AUROC | fraction of negatives scoring below the positive (ties count 0.5) |

Both directions are scored (`a -> b` against `f_b`'s distractors, `b -> a` against `f_a`'s),
giving 540 anchors from 270 pairs. `EQ` is their mean. See `eq_scores()` in `src/metrics.py`.

**Interpretation.** EQ is the probability that a statement's true mathematical equivalent is
ranked above a random lexically-plausible impostor drawn from the same subfield.

---

## 3. Why anchor-wise AUROC is the right statistic

**3.1 Scale-free.** Cosine distributions differ enormously across models *and across layers of
one model*: we measured PC1 absorbing 10% of variance at `last/L20` and 98.9% at `mean/L10` in
the same model. Any metric touching absolute similarity would conflate "encodes equivalence"
with "this model's cosines happen to be large." AUROC uses only ordering *within* an anchor.

**3.2 Per-anchor rather than pooled.** A single pooled AUROC over all positives and negatives
would mix anchor-specific similarity scales — hub statements close to everything would dominate.
Averaging within-anchor AUROCs removes that. It is equivalent to the mean normalised rank of the
true partner.

**3.3 Bounded in [0, 1].** The design ends in a Spearman correlation across ~25 models
(`PLAN.md` sec 6). An unbounded metric would let a single outlier model drive the ranking.

**3.4 The negatives hold subfield constant.** Distractors come from the *target's* framing, so a
model cannot win by merely recognising which branch of mathematics the anchor belongs to — the
exact failure mode the MELD paper reports in commercial embedders ("they cluster by subfield,
not by equivalence").

**3.5 The two directions are genuinely distinct.** Cosine is symmetric, but the distractor pool
is not: it is tied to the target's framing.

### Secondary metrics, and why they are secondary

| metric | why not primary |
|---|---|
| `align_gap` = mean cos(pos) - mean cos(neg) | a difference of means, so it **fails 3.1**: a model with a compressed cosine range scores low no matter how well it orders. Useful only as a diagnostic. |
| `recall@1`, `recall@5`, `MRR` | near floor at this scale (R@1 1.1-8.5%) and does not track EQ — Qwen2.5-1.5B scores EQ 0.808 with R@1 0.019. Base LMs are not embedders. Descriptive only. |

---

## 4. Controls that make this a measurement rather than a number

| control | result | what it rules out |
|---|---|---|
| layer chosen on stimulus split A, EQ reported on split B | — | reporting the max over ~30 layers, which is upward-biased |
| padding invariance, batched vs unbatched | min cos >= 0.99989 | last-token index landing on a PAD token — a bug that yields plausible-looking, meaningless numbers |
| fp32 GPU vs fp64 CPU parity, 48 settings | dEQ <= 1.2e-4 | the 86x fast path silently changing results |
| positive/distractor collision check at load | caught pairs 158/174 sharing an `entry_1` | a statement counted as both positive and negative, or a guaranteed retrieval miss |

---

## 5. Where validity breaks

This section matters more than section 3.

### 5.1 A valid statistic does not make 0.5 the right null

AUROC's nominal chance is the null for a *random* scorer, not a *plausible* one. Measured on the
identical task with no neural network:

| baseline | EQ |
|---|---|
| binary bag-of-words | 0.6621 |
| TF-IDF word 1-gram | 0.7099 |
| TF-IDF word 1-2gram | 0.7098 |
| **TF-IDF char 3-5gram** | **0.7715** |

A model scoring 0.77 has demonstrated nothing. **Always report EQ as a margin over the lexical
baseline, recomputed for the stimulus set in use.**

### 5.2 Construct validity: the negatives control framing but not topic

A true pair restates one concept, so it shares topic-specific tokens across dialects; the
distractors concern *different* concepts in the same dialect. So EQ currently measures

```
equivalence detection  +  topical lexical matching
```

and the two terms are not separated. This is the substantive defect, not a caveat.

### 5.3 EQ is preprocessing-dependent

Because cosine is taken after correction, EQ is not a property of the representation. Change the
correction and the between-model ordering changes. Cross-pooling rank agreement over 5 models:

| correction | raw EQ | margin over layer 0 |
|---|---|---|
| k0 | -0.400 | -0.600 |
| k1 | **+1.000** | -0.400 |
| k3 | +0.500 | +0.600 |
| gapk | 0.000 | -0.100 |
| zca | +0.900 | +0.100 |

Note the second column: the apparent stability of raw EQ is largely **inherited from the
baseline**, not produced by the deep representation.

---

## 6. How to read a result

A decision procedure. Do not skip to step 5.

### Step 1 — establish the null for *this* stimulus set

Run the TF-IDF baselines. Never compare to 0.5. For MELD the number is **0.7715**.

### Step 2 — subtract it

Report `EQ - baseline`. Applying this to the Phase 0 headline (`last/k1`):

| model | EQ | margin vs 0.7715 | reading |
|---|---|---|---|
| SmolLM2-360M | 0.7352 | **-0.036** | below a bag of characters; no signal |
| Qwen2.5-0.5B | 0.7724 | +0.001 | indistinguishable from lexical |
| Qwen3-0.6B | 0.7991 | +0.028 | weak |
| Qwen2.5-1.5B | 0.8080 | +0.037 | weak |
| Qwen2.5-Math-1.5B | 0.8307 | +0.059 | weak |

The raw column looks like a clean monotone ordering across a 0.10 range. The margin column shows
the actual effect is 0.00-0.06 and two models have none. **The raw ordering is mostly baseline.**

### Step 3 — check the layer-0 control

Layer 0 is token identity, so a metric reading meaning rather than surface form should leave it
near chance. Measured:

| setting | layer-0 EQ across models | verdict |
|---|---|---|
| last/k1 | 0.277 - 0.531 | scattered about chance; a fair control |
| mean/k0 | 0.569 - 0.601 | informative |
| mean/k1 | 0.682 - 0.725 | informative |
| mean/zca | 0.771 - 0.790 | a bag-of-embeddings baseline in disguise |

This is a criterion for choosing the correction that is **independent of the between-model
ordering** — the property `PLAN.md` sec 6.5 demands. On current evidence it selects `last/k1`.

Consequence: under mean pooling, depth buys only **+0.01 to +0.12** over the model's own
embedding layer. Almost the whole score was present before any transformer block ran.

### Step 4 — check stability across corrections

Recompute under every correction. If the between-model ordering flips, you have measured the
preprocessing, not the model. Use the table in 5.3. `gap_k` (rho = 0.000) is why this step
exists: it was proposed as a principled adaptive criterion and failed outright.

### Step 5 — only now interpret between-model differences, and only size-controlled ones

Cross-model EQ correlates with parameter count, so a monotone ordering over a size-varying panel
is uninformative on its own. What is informative is a comparison at **fixed size**:

> Qwen2.5-1.5B vs Qwen2.5-Math-1.5B, identical parameter count:
> math-tuned higher in **9/10** settings, mean delta **+0.041** raw and **+0.042** on the margin.

That is the only Phase 0 result not confounded with scale.

### Step 6 — state what the number cannot support

For the result above:

- it is **one model pair**;
- the 10 settings are nested views of the same activations, **not independent draws** — a sign
  test over them would be pseudo-replication;
- no capability benchmark has been run, so it is evidence about representation only, and says
  nothing yet about H1 or H2.

---

## 7. Summary

The metric is a well-constructed **discrimination statistic**: scale-free, controlled for
subfield, honestly split for layer selection, and guarded by verified numerical checks. It is
not yet a valid **measure of equivalence representation**, because its null was wrong (5.1) and
its negatives leak topical lexical signal (5.2).

Converting it into one requires either topic-matched negatives, or partialling the TF-IDF cosine
out of the model cosine and scoring AUROC on the residual. Both are re-analysis of activations
already cached on disk — no GPU time, no download.
