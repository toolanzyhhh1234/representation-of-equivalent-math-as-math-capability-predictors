# Logical-consequence pilot: interpretation and next design

We now have a reproducible, solver-checked investigation of directed consequence
and premise roles. The results do **not** establish a general implication geometry
or a causal reasoning mechanism. See [the registered tables](LOGIC_PILOT.md) and
[the frozen protocol](../prereg/LOGIC_PILOT.md).

## Directed consequence

The registered primary readout uses the hidden state of a full theory plus ordered
premise/conclusion. Its strongest IID English result is Qwen2.5-1.5B at AUROC 0.645,
versus a position baseline of 0.596. Its depth and structural tests are 0.559 and
0.563, respectively. These are small descriptive samples, not significance claims.

The secondary ordered-node readout is more encouraging within the training grammar:
Qwen2.5-1.5B reaches 0.787 IID English AUROC. It falls to 0.624 on deeper chains,
0.480 on new structures, and 0.510 on symbolic rendering. Its unrelated-query
false-positive rate is 0.26 on IID English theories. This is evidence of a limited
decodable signal, not robust general entailment recognition. Native forced-choice
direction AUROC for this model is 0.505 on IID English: probe decoding and the
model's native response are different measurements.

Symmetric cosine scores exactly 0.5 on reversed pairs by construction. The fact that
an ordered readout can exceed that control does not establish that pretrained hidden
states natively form an implication order. A ranking of connected nodes also does
not solve non-entailment between disconnected components.

## Premise roles: a discovered confound

Qwen2.5-1.5B's premise-role probe scores 0.760 balanced accuracy on IID theories,
0.787 on deeper theories and 0.767 on the early-gate structural holdout, versus chance
1/3. Those numbers look stronger than the consequence results, but they are not
yet evidence of semantic indispensability.

A **post-hoc** audit found a simple text-only classifier that reaches 100% on all
1,800 premise-role prompts, across all splits and both styles:

1. If the candidate appears after `and` / `&`, call it indispensable.
2. Otherwise, if it appears as a rule consequent, call it irrelevant.
3. Otherwise, call it redundant.

The generator consistently puts the guard in the second conjunction slot and gives
the irrelevant fact a distinctive incoming rule. The heuristic does not consult
the facts, the query, proof search, hidden states or the target label. Layer-0 and
random-label controls do not remove this template-level confound. The frozen
experiment remains unchanged; the diagnostic is reproducible with
`python -m src.audit_logic` and saved separately as `premise_shortcut_audit.json`.

The logical labels are valid: forward chaining and truth-table enumeration agree,
and minimal supports/deletions are checked. Correct labels alone do not guarantee
that a representation probe is measuring the intended construct.

## Next version

Use counterfactual theory pairs where the **same candidate and local rule pattern**
change from indispensable to redundant when a remote alternative proof is added.
Match degrees, conjunction participation, operand order, mention counts and positions
across roles; vary graph topology, and randomize conjunction order. Randomizing
wording alone would not fix the current confound. Preserve disconnected queries and
theory/template/depth holdouts for consequence readouts.

Add those syntax baselines before another model run, and reserve new graph families
for evaluation. A non-symmetric bilinear readout is a possible secondary extension
to the ordered linear probe, but should be frozen in a new protocol rather than
chosen after inspecting test results. Causal ablation/patching should wait until a
signal survives these controls. ProofWriter is a later transfer test, not a substitute
for verifying this controlled instrument.

## What is preserved

All 300 theories, labels, countermodels, minimal supports and 6,000 prompts; every
layer's readout for five models; native A/B/C logits; cache provenance and checksums;
selected trained probe arrays and probabilities; all development curves and test
metrics; and the syntax-shortcut audit. Eight logical data tests passed, and all five
padding checks exceeded 0.999. This is a positive-Horn, in-context study, not a claim
about unrestricted first-order logic or a model's stored mathematical knowledge.
