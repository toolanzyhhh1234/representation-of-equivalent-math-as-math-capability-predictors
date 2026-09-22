# Logical premises and consequences in hidden representations

Proposal, not an experiment result or a replacement for the existing H2/H3 roadmap.
Here “prerequisite” means a logical premise, not a concept-learning prerequisite.

## Core question

Does a model encode **which statements follow from which premises**, beyond topic,
word overlap, and prompt position? Does that encoding predict correct deduction and
survive equivalent rewrites of the premises?

The current MELD measurement concerns equivalence. The extension concerns a directed
relation. Under a fixed background theory T, P and Q are equivalent when both
T,P entail Q and T,Q entail P. One-way entailment can hold without equivalence.
Cosine similarity is symmetric, so it cannot distinguish P→Q from Q→P by itself.
Entailment is a preorder; quotienting by equivalence gives a partial order. This
provides a conceptual bridge to MELD without assuming the model already realizes
that geometry.

For real x, x > 2 entails x > 0, while x > 0 does not entail x > 2. Test both
directions of the same pair, plus equivalent rewrites, contradictions and unrelated
statements. Keep the domain and background assumptions explicit. For general
theories, use sets of premises: conjunction and alternative proofs make the structure
more complicated than a chain of sentence vectors.

## Separate three labels

| Property | Definition / test |
|---|---|
| Consequence | Γ entails q |
| Premise indispensable within Γ | Γ entails q, but Γ without p does not entail q |
| Necessary condition of q relative to T | T,q entails p |

The last two differ. For Γ = {a, a→q}, removing a prevents deriving q, but q does
not imply a: q may be true for some other reason. Likewise, membership in one proof
does not establish indispensability when another proof omits that premise.
Track useful-but-redundant premises separately from indispensable and irrelevant ones.
An abductive completion is a further question: which additional p would make q
derivable? Do not merge it with the labels above.

## First pilot

1. Generate 300–1,000 small, satisfiable propositional or finite-domain Horn theories
   from an explicit symbolic representation. Compute closures or enumerate truth
   assignments to validate all labels; never use an LLM judge as the sole ground
   truth. Include conjunction, two alternative proofs, irrelevant facts, negation,
   direction reversal, and depths 1–5. For each entailed q, delete each candidate
   premise and recompute entailment to label indispensability.
2. Render each theory in symbolic and controlled-English forms with several
   equivalent paraphrases. Counterbalance predicate names, sentence order, length,
   connective wording and label frequency. Construct matched cases where deleting
   one premise changes entailment and cases where it does not. Include false and
   unknown queries separately under an explicitly chosen open-world semantics.
   Filter inconsistent theories so classical explosion does not trivialize labels.
3. Freeze the stimulus generator, split and primary endpoint before evaluating
   models. Hold out entire theories, their paraphrases, and their deletion variants
   together. Add held-out templates and depth extrapolation tests; random row splits
   would leak nearly identical theories across partitions.
4. Start with SmolLM2-360M, Qwen2.5-0.5B, Qwen2.5-1.5B and Qwen2.5-Math-1.5B.
   This reuses the existing small pilot and a fixed-size tuning contrast. Four models
   are a feasibility pilot, not enough for another capability-correlation claim.
5. Read hidden states before any gold answer or proof is shown. Compare isolated
   statement representations with contextual representations from a full theory
   plus query, using a fixed readout marker. With a causal decoder, an early premise
   token cannot see later premises: a query/readout token after the full theory can.
   Report layerwise results with layer 0 as a control, and choose readout/layer on
   development data only. Check token lengths explicitly: the current MAX_LEN=256
   can silently truncate logic contexts, and sentence pooling can dilute relations.

## Readouts and controls

Begin with a regularized linear classifier over an ordered pair representation
`[h(P), h(Q), h(Q)-h(P)]` and, separately, a classifier over the contextual readout.
Compare against symmetric cosine and symmetric pair features. A small bilinear
score `h(P)^T W h(Q)` with unrestricted W is a secondary directional readout. Tune
capacity and regularization only on the development split. Context-conditioned
representations are required when entailment changes with background assumptions.

An asymmetric score succeeding is evidence of decodable information, not proof that
the model computes deduction using that representation. Trained order embeddings
are relevant geometry baselines, but learning an order projection does not establish
that pretrained hidden states natively have an ordered geometry.
[Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361).

Primary endpoint: held-out discrimination between strictly one-way entailed pairs
and their verified non-entailing reversals, with paired uncertainty over theories.
Do not blindly label every reversed pair false: some pairs are equivalent.
Secondary endpoints: three-way entailment/contradiction/unknown balanced accuracy,
premise-deletion classification, depth generalization, and agreement across rewrites.
Pair those with behavioral reasoning accuracy using the same cases. Consistent wrong
answers are not successful robustness; report correctness and consistency separately.

Controls: ordered TF-IDF pair features, premise-only and query-only classifiers,
length/position controls, label-shuffled probes, equal-capacity probes on random or
layer-0 features, random subspaces, and vocabulary/template holdouts. Probes can learn
the task themselves; selectivity controls and capacity curves help qualify claims.
[Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/).

After a reliable held-out effect, test causal use: patch representations between
matched theories differing in an indispensable premise, and compare answer-logprob
changes with equally sized irrelevant-premise edits. For subspace ablations, include
random rank- and norm-matched ablations, unrelated-task controls, and a rescue/patching
test. Discover subspaces only on training examples. A general performance collapse
would not demonstrate a logic-specific mechanism.

## External data

- **ProofWriter** supplies theories, questions, labels, depth information and proof
  annotations, making it a useful transfer benchmark. The downloaded
  [tasksource/proofwriter](https://huggingface.co/datasets/tasksource/proofwriter)
  is a community repackaging; inspect `config` and proof fields rather than assuming
  every row has the same proof depth or semantics. Keep all questions from one
  theory grouped. For premise interventions, reconstruct and validate theories with
  a solver; a listed proof alone is insufficient to label every necessary premise.
  [Original paper](https://aclanthology.org/2021.findings-acl.317/).
- **FOLIO** adds natural-language premises with first-order-logic annotations. Use it
  after the controlled pilot to test transfer; solver timeouts must be distinct from
  a semantic “unknown” label. Its [HF repository](https://huggingface.co/datasets/yale-nlp/FOLIO)
  is gated. [Original paper](https://arxiv.org/abs/2209.00840).
- **MELD** remains the equivalence comparison. Its distractors are not automatically
  implication or premise-necessity labels; do not relabel them without checking.

## Improvements suggested by the current code review

These are proposals for a subsequent methodology change; setup preserves the existing
evaluation harness and reported results.

1. **Prioritize the registered rewrite-invariance H2 arm.** The logic pilot can share
   its controlled rewriting machinery, while remaining a separately labeled analysis.
2. **Fit preprocessing on training stimuli for a strict inductive claim.** Currently
   `eq_variants` fits the lexical slope over all candidate scores; centering/PCA also
   sees all stimuli. This is transductive preprocessing even though layer choice uses
   split A. Cross-fit preprocessing and test topic/domain holdouts as sensitivity
   analyses. Pair-stratified splits do not establish unseen-domain generalization.
3. **Calibrate the residual null empirically.** The implementation forces the
   degenerate TF-IDF-on-itself residual to 0.5. Removing a linear TF-IDF component
   does not force an arbitrary model's null AUROC to 0.5, remove nonlinear lexical
   information, or guarantee a numerical lower bound on semantic information.
   Use structure-preserving label permutations and matched negatives to test these
   claims. Keep the current lexical baseline pinned at `char_wb` 3–5 grams.
4. **Align uncertainty with the reported endpoint.** `run_panel` reports split-B
   scores but bootstraps all pairs at the chosen layer. Add split-B-only intervals,
   and separately nested resampling that repeats layer selection. Cluster correlated
   models by family for panel inference. Report multiplicity across exploratory
   readouts and endpoints; retain the fixed primary metric.
5. **Record cache provenance.** Add model/tokenizer SHA, ordered-stimulus hash,
   dtype, library versions, device, pooling, truncation counts and prompt template
   to future caches, then reject mismatches. Dataset SHA pins added in this setup
   address input drift, but do not retroactively recover historical model revisions.
6. **Tighten claim wording in a separate report revision.** Observational partial
   correlations do not establish mediation; the observed low-capability region
   does not prove necessity. The agreed confirmatory claim is the correlation, and
   necessity-without-sufficiency remains a generative hypothesis. Update stale
   10-model and 1.7B scope statements and the CPU-only reproduction claim: current
   metric runners explicitly require CUDA, although aggregate analysis runs on CPU.

Success for the pilot would be a directional signal that survives theory/template
holdouts and matched lexical controls, followed by selective causal effects. A probe
that only works within a template, tracks sentence position, or succeeds without
behavioral/causal support would substantially narrow the interpretation.
