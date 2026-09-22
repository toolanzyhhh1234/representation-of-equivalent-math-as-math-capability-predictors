# Logical-consequence and premise-role pilot

This is a new exploratory experiment, distinct from H2. No hypothesis is treated as
established merely because a supervised probe decodes its labels.

## Ground truth and splits

Generate 300 finite positive-Horn theory groups with seed 20260924: 120 training,
30 development, 50 IID test, 50 depth test, and 50 structural test groups. Every
direction reversal, unrelated query, premise-role example and English/symbolic
rendering belonging to a group stays in its split. Atomic proposition names are
opaque, fixed-width identifiers; their whole-name vocabularies are disjoint by split.
Names and rule order are randomized independently of direction labels.

Ground truth is checked by two independent implementations: least-fixed-point
forward chaining and exhaustive Boolean assignment enumeration. Non-entailments
carry countermodels. All theories are satisfiable. There is no negation in this
pilot: “not entailed” means some compatible world falsifies the query, not that its
negation follows. Cyclic equivalence is checked in tests and never mislabeled as a
negative reversal.

Training/development/IID directions use chains of depth 1–3. The depth test uses 4–5.
The structural test uses parallel paths and conjunctions with a stated guard fact.
For a pair P,Q, verify T,P entails Q and T,Q does not entail P. Pair endpoints have
matching local incoming/outgoing rule degrees. Include two directions of a query
between disconnected components as unrelated non-entailments, not contradiction.
Within each reversed pair, the theory, facts, length and bag of named propositions
are identical. Position controls still matter because randomized order is not a
proof that every positional cue has been removed.

The auxiliary premise-role task has two minimal supporting fact sets. One guard is
indispensable, a fact used in one alternative support is redundant, and a disconnected
fact is irrelevant. Enumerate all fact subsets to identify minimal supports, and
independently verify the result of deleting each candidate with truth tables.
Structural holdouts move the guard from the final rule to early rules. This tests
one structural shortcut; it does not exhaust all shortcut explanations.

There are 6,000 prompts: per theory and style, two direction queries, two unrelated
queries, three contextual node representations, and three premise-role queries.
The solver metadata, countermodels and correct labels never enter encoded prompts.

## Models and extraction

Use the same five cached, revision-pinned models as the H2 follow-up. Extract every
layer including embedding layer 0, with bf16 forward computation and float32 saved
last-real-token vectors. Readout occurs at a fixed marker before any answer. Use
right padding with indices from the attention mask, no truncation, batch size 16,
and a batch-versus-single padding check (minimum cosine 0.999). Record input length,
actual parameter count, model revision, array shape, dtype and cache SHA256.

Encode the full theory with ordered premise and conclusion for the primary readout.
Separately encode each query proposition with the same theory and no premise/conclusion
role, allowing ordered-pair and symmetric-cosine comparisons. A decoder alone supplies
hidden states; the model output head applied to the final readout supplies next-token
choice logits. Verify that space-prefixed A/B/C are single continuation tokens for
each tokenizer. Native behavior is a forced-choice next-token readout, not generated
chain-of-thought reasoning or a general logic benchmark score.

## Probes, controls and selection

Train on English training examples only. Fit all scalers on training rows only.
Project linearly to 128 dimensions with a fixed Gaussian projection, seed 20260924,
then scale and fit logistic regression with C=1, max_iter=2000. Projection and scaling
remain linear; this bounds probe capacity across model widths, but may discard some
information. A null result does not prove no other readout can decode it.

Primary: a linear readout of the joint T,P,Q context, scored by direction AUROC.
Select the layer on English development AUROC, excluding layer 0; ties choose the
earliest layer. Evaluate the selected readout once on each test split and both styles.
No test result changes the layer, regularization, seed, dataset or primary endpoint.

Secondary: ordered node features [h(T,P),h(T,Q),h(T,Q)-h(T,P)], independently selected
on development direction AUROC, and a premise-role classifier selected on development
balanced accuracy (three classes). Save the trained projection, scalers, coefficients
and prediction probabilities as arrays, not opaque pickles.

Controls: layer 0; a random orientation label per entire theory at the selected layer;
training-fitted character TF-IDF and word-bigram baselines; a baseline using proposition
occurrence counts and first/last positions; symmetric cosine of the contextual node
vectors. A symmetric score must have AUROC 0.5 on exactly paired reversals by
construction. That is a control, not a newly discovered empirical limitation.

Evaluate unrelated-query false positives alongside the primary task: success at
ordering connected nodes can reflect a ranking and need not imply general entailment
recognition. Report native direction AUROC/balanced accuracy, unrelated false positives,
and native premise-role balanced accuracy. Native class-token bias is not corrected
using test labels; AUROC complements fixed-threshold accuracy.

This first run reports descriptive test metrics on 50 theories per test regime,
not an across-model capability correlation or a causal claim. Structural and
symbolic transfer, lexical/position controls and label controls constrain the
interpretation. Positive Horn logic is a restricted fragment, not full first-order
logic or evidence of a universal implication geometry. Successful decoding does
not show that the model uses the decoded information when answering.

## Integrity and preservation

Freeze dataset, source and library/model revisions before extracting states or
fitting probes. Preserve raw activations, next-token logits, trained probes, all
development-layer scores and complete test metrics. Changed scientific code or
inputs require a new version, not cache reuse. Upload the completed logic snapshot
to the existing HF archive and push the corresponding code/results to Git.

```bash
python -m src.logic_data
python -m tests.test_logic
python -m src.run_logic --prepare
python -m src.run_logic
python -m src.analyze_logic
```
