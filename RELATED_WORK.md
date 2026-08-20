# Related work

Organised by what each paper does for **this project's design decisions**, not as a flat
bibliography. The load-bearing table in section 6 is the part to read if you are short on
time: it maps each of our methodological choices to the work that justifies or threatens it.

---

## 1. The nearest neighbour: equivalence in embedding space

**[Does My Embedding Reflect That A = B? Evaluating Mathematical Equivalence in Embedding
Models](https://www.alphaxiv.org/abs/2606.23959)** (2606.23959)
Introduces **MELD**, which this repo uses as its stimulus set: 270 mathematically equivalent
pairs across 9 domains / 18 framings (vector spaces vs module theory, set theory vs category
theory, ...) plus 541 framing-matched hard negatives. Finds that six SOTA embedding models
**cluster by mathematical subfield rather than by equivalence**; the best baseline reaches
25.0% Recall@1, and contrastive training on formal/informal pairs lifts it to ~28.9%.

Our relationship to it:
- We reuse its data and its framing of the problem.
- We ask a *different* question. It asks whether **embedding models** encode equivalence;
  we ask whether the degree to which **any LM** does so predicts that model's mathematical
  capability.
- **We disagree with one of its dataset claims.** The MELD card states that lexical overlap
  is a "misleading" signal. We measure a character 3-5-gram TF-IDF at **EQ = 0.7715** on the
  anchor-AUROC task, because the negatives are matched to the target's *framing* but not to
  the anchor's *topic*. See `results/PHASE0.md` sec 1.

---

## 2. Does equivalence live in the geometry? (evidence that it does)

**[Beyond Language: Format-Agnostic Reasoning Subspaces in LLMs](https://www.alphaxiv.org/abs/2605.09496)** (2605.09496)
The strongest positive result in this area, and the closest thing to a mechanism for our H1.
Builds the TriForm benchmark (324 stimuli, 18 concepts, 6 formats) and uses RSA, cross-form
probing, and activation patching over five 1.6-8B models. Finds a **10-dimensional subspace**
that amplifies concept structure ~3x while suppressing surface form to near zero; patching
only those dimensions preserves 90-96% of output across prose/maths/code transfers, versus
44-56% for full activation patching. Notable asymmetry: prose<->maths transfers better than
prose<->code, so the axis is declarative vs procedural rather than natural vs formal.
*Relevance:* this is the causal design our H3 arm imitates, and its subspace dimensionality
(~10) is a prior for what we should expect to find.

**[Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of
Natural-Language and Symbolic Views](https://www.alphaxiv.org/abs/2604.19716)** (2604.19716)
Same shape of claim for NL <-> symbolic alignment, with steering.

**[The Geometry of Reasoning: Flowing Logics in Representation Space](https://www.alphaxiv.org/abs/2510.09782)** (2510.09782)
Models reasoning as flows through representation space. Background framing.

**[LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness
Signals](https://www.alphaxiv.org/abs/2604.05655)** (2604.05655)
Extracts correctness signals from step-wise representation geometry. Adjacent but distinct:
it predicts **per-instance** correctness, where we want **per-model** capability.

---

## 3. Behavioural sensitivity to equivalent reformulations (our dependent variable)

**[What are the Right Symmetries for Formal Theorem Proving?](https://www.alphaxiv.org/abs/2605.22257)** (2605.22257)
Formalises **success invariance** -- for semantically equivalent T and T', the success
probability should satisfy s(T) = s(T'). Builds `miniF2F-rw` by generating 5-15 Lean-certified
equivalent variants per problem, and reports swings as large as **80% -> 10%** across rewrites
of the same theorem. A rewriting ensemble recovers invariance in the sampling limit.
*Relevance:* this is the source of our H2 dependent variable (`INV`). At 0.5-2B, miniF2F
floors near 0%, so we substitute GSM-Symbolic; the construct is theirs.

**[GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs](https://arxiv.org/abs/2410.05229)** (2410.05229, ICLR 2025)
Templated symbolic variants of GSM8K -- literally equivalent rewrites at the right difficulty
band for a small-model panel. **Our actual `INV` source.**

**[Measuring Representation Robustness in LLMs for Geometry](https://www.alphaxiv.org/abs/2604.16421)** (2604.16421)
and **[Characterizing Paraphrase-Induced Failures in Lean 4 Autoformalization](https://www.alphaxiv.org/abs/2604.23135)** (2604.23135)
and **[Evaluating the Robustness of Proof Autoformalization in Lean 4](https://www.alphaxiv.org/abs/2606.14867)** (2606.14867)
Further documentation that equivalent surface forms produce very different outcomes.

---

## 4. Threats to our instrument (read these before believing a result)

**[Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden
States](https://www.alphaxiv.org/abs/2606.02907)** (2606.02907)
**The direct methodological attack on our measurement.** Probing hidden states is widely used
to claim models learn distinct representations for different reasoning types; this paper tests
that on Qwen3-14B and finds the probes are reading **task format**. Our defence is the
hard-negative construction, the layer-0 control, and the causal arm -- and Phase 0 showed the
concern is warranted, since our metric does carry a substantial lexical component.

**[Convergence Without Understanding: When Language Models Agree on Representations but
Disagree on Reasoning](https://www.alphaxiv.org/abs/2605.23315)** (2605.23315)
Models trained differently converge representationally (the Platonic Representation
Hypothesis) while still disagreeing on reasoning. *Implication for us:* EQ may saturate before
capability does, compressing its variance across the panel and capping any rank correlation
regardless of the underlying truth. Check the spread of EQ early.

---

## 5. Anisotropy: the geometry methods we actually implemented

`src/correct.py` is built on this line of work, and Phase 0 is largely a story about it.

**[All-but-the-Top: Simple and Effective Postprocessing for Word Representations](https://arxiv.org/abs/1702.01417)**
Mu & Viswanath, ICLR 2018. Remove the common mean vector and a few top dominating directions.
**This is exactly our `k0/k1/k3` correction.** Phase 0 found its central assumption -- that a
few directions dominate -- holds for mean-pooled states (PC1 = 24-36% of variance) but fails
for last-token states (PC1 = 6-12%, s3/s4 ~ 1.03), where the removed directions are not even
identified. That is why our `gap_k` variant was tried, and why it failed.

**[How Contextual are Contextualized Word Representations?](https://arxiv.org/abs/1909.00512)**
Ethayarajh, EMNLP 2019. Contextual representations are **not isotropic in any layer**, and
self-similarity falls in upper layers. Predicts the layer-dependence we measured: PC1 absorbs
10% of variance at `last/L20` but 98.9% at `mean/L10` **in the same model**, which is why a
fixed `k` cannot be right across layers.

**[Representation Degeneration Problem in Training Natural Language Generation
Models](https://arxiv.org/abs/1907.12009)** Gao et al., ICLR 2019. Embeddings occupy a narrow
cone. The mechanism behind why raw cosine is untrustworthy without correction.

---

## 6. Load-bearing map: which decision rests on which paper

| our decision | rests on / threatened by |
|---|---|
| MELD as stimulus set; subfield-clustering is the failure mode to beat | 2606.23959 |
| anisotropy correction before any cosine | 1702.01417, 1909.00512, 1907.12009 |
| fixed-k is unsafe; correction must be reported as a sweep | 1909.00512 (layer-dependence) + our Phase 0 |
| `INV` (rewrite invariance) as the primary DV | 2605.22257 (construct), 2410.05229 (instrument) |
| causal ablation arm, expect ~10 dims | 2605.09496 |
| hard negatives + layer-0 control are mandatory | **2606.02907** (probes read format) |
| check EQ spread before trusting a rank correlation | 2605.23315 (convergence) |
| lexical baseline, not chance, as the null | our Phase 0 (contra the MELD card) |

---

## 7. The gap this project occupies

Nobody in the above has closed the loop we are aiming at. The literature splits cleanly:

- Section 2 measures the **geometry** without relating it to capability.
- Section 3 measures **capability fragility** without opening the model.
- Section 5 provides the **methods** but on word/sentence embeddings, not on a capability question.

The unoccupied position is: **take a representational quantity measured on equivalence pairs
as the independent variable, and predict a model's mathematical capability -- or better, its
robustness to reformulation -- as the dependent variable, across models.** 2605.09496 comes
closest but uses five models of 1.6-8B, far too few to regress against capability.

Phase 0's contribution so far is negative-but-useful and belongs in any write-up: on MELD, a
character n-gram baseline reaches 0.7715, so results in this area reported against chance are
overstated, and the benchmark's hard negatives do not control for topic.

---

## 8. Models and data cited for benchmark numbers

| source | used for |
|---|---|
| [SmolLM2 paper](https://arxiv.org/abs/2502.02737) + model cards | GSM8K 5-shot: SmolLM2-360M 3.2, SmolLM2-1.7B 31.1, Qwen2.5-0.5B 33.4, Qwen2.5-1.5B 61.7, Llama3.2-1B 7.6 |
| [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5-llm/) / [tech report](https://arxiv.org/abs/2412.15115) | GSM8K 4-shot: Qwen2.5-0.5B 41.6, Qwen2.5-1.5B 68.5 |
| [Qwen2.5-Math report](https://arxiv.org/abs/2409.12122) | Qwen2.5-Math family |
| [uw-math-ai/MELD-dataset](https://huggingface.co/datasets/uw-math-ai/MELD-dataset) | stimuli |

**Note the harness problem.** Qwen2.5-1.5B is reported at **68.5** by Qwen and **61.7** by the
SmolLM2 authors on the same benchmark -- a 7-point spread from shot count and harness alone.
This is why `PLAN.md` sec 5 requires capability to be measured in-house, and why the Phase 0
addendum uses one consistent harness rather than mixing sources.
