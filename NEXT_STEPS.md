# Next session: geometry and reasoning performance

Recorded 2026-09-22 after the user's review of the equivalence, rewrite-invariance,
and logical-consequence experiments. This is a research handoff, not a frozen
protocol. No new experiment was run for this decision.

## Agreed research question

As the geometry of a model's hidden representations better preserves mathematical
or logical relationships, does its independently measured reasoning performance
improve, within a model or across models?

Make a prespecified geometric score the primary representation measurement.
Trained probes may remain auxiliary diagnostics. The immediate next step is to
define and validate what “better geometry” means, then design the geometry–accuracy
test; do not simply expand the existing logic-probe panel.

The user's key observation: a probe's out-of-distribution degradation could reflect
failure of the probe to generalize, worse model performance on the task, or both.
Probe performance alone cannot distinguish these explanations. Measure the model's
own behavior on the same task conditions independently of the probe. Even stable
behavior with a declining probe score would not uniquely identify the mechanism.

## What the completed experiments establish

| Experiment | Measurement | Relationship to the next question |
|---|---|---|
| Original MELD/H1 | Equivalent-vs-nonequivalent geometric ranking, related to independent math accuracy | Direct cross-model evidence: reported Spearman rho(EQ_resid, MATH-500) = 0.797 on 23 models; correlational |
| H2 rewrites | Archived EQ_resid related to consistency under equivalent rewrites | Tests a different behavioral endpoint; the five-model follow-up found no clear positive H2 signal, which does not refute H1 |
| Logic pilot | Supervised entailment/premise-role readouts from hidden states, plus native choice scores | Tests decodability; does not directly establish a native geometric quality–capability relationship |

The original EQ score asks whether an equivalent statement ranks above matched
nonequivalent statements under hidden-state similarity. It does not train a label
decoder, but it does use geometric preprocessing, a fitted lexical adjustment, and
development-set layer selection. Some preprocessing is transductive. Do not describe
it as entirely parameter-free or as a measurement of untouched raw geometry.

Matched-size base/math model pairs are useful comparisons, but are not a longitudinal
checkpoint study. Comparing layers of one checkpoint also does not by itself show
that the model's behavioral performance improves.

The logic pilot's auxiliary premise-role labels have a perfect post-hoc syntax
shortcut. Its high role-probe scores are not evidence of semantic indispensability.
See [the logic readout](results/LOGIC_READOUT.md) and
[the H2 readout](results/H2_FOLLOWUP_READOUT.md).

## Recommended design to develop next

1. **Define the geometric property before examining its capability correlation.**
   For equivalence, reward proximity of equivalent statements relative to
   meaning-changing controls, rather than tight clustering alone. Collapsing all
   representations must not count as success. Labels can evaluate a fixed geometry
   without fitting a supervised decoder. State every fitted transformation and
   choose settings on development data only.
2. **Resolve the logical-consequence metric.** Entailment is directional; cosine
   similarity is symmetric. A symmetric similarity score cannot distinguish
   P entails Q from Q entails P for the same pair. An asymmetric context/readout
   could potentially support a geometric test, but requires an explicit hypothesis
   and validation. Learning an order embedding or entailment classifier would not
   by itself demonstrate that the original hidden space has that organization.
   No primary consequence-geometry metric has been chosen yet.
3. **Prefer a within-family checkpoint trajectory as the first new test.** Keep
   architecture, tokenizer, parameter count, stimuli, readout convention, and
   evaluation harness fixed. At each checkpoint, measure geometric quality and
   independent reasoning accuracy. Check suitable checkpoint availability before
   choosing the family; do not assume the current five models form a trajectory.
   Account for training progress and dependence between checkpoints rather than
   treating them as independent model samples. Covariation is not causal evidence.
4. **Keep other meanings of “within model” distinct.** A complementary fixed-model
   test asks whether local geometric quality predicts success across problems,
   controlling for difficulty and surface features. It needs its own protocol and
   must not use generated correct answers to define the predictor. Cross-model
   replication should address size/family dependence. Neither is interchangeable
   with a checkpoint trajectory or a layer comparison.
5. **Freeze the evaluation before the main run.** Separate development from held-out
   theories/problems and capability outcomes; keep related rewrites/proof graphs in
   the same split. Prespecify layers, pooling, corrections, baselines, uncertainty,
   and the primary association. Include lexical/position controls and checks for
   representation collapse. Do not choose a metric by maximizing its final
   benchmark correlation. If probes are retained, report probe scores and native
   task performance separately for every distribution shift.
6. **Repair the logic data if used.** Create a new version with counterfactual remote
   alternative proofs that change premise necessity while matching local syntax,
   mentions, degrees, and positions. Randomizing conjunction order alone does not
   remove the discovered shortcut. Retain solver checks and adversarial surface
   baselines before extracting new states.

## Resume in another container

Read this file first, then [CLAUDE.md](CLAUDE.md), [SETUP.md](SETUP.md), and
[ARCHIVES.md](ARCHIVES.md). The dependency lock and pinned HF snapshots preserve the
completed work; model weights and the environment can be reconstructed. Restore
only the archive needed for the next analysis using the checked commands there.

The completed experiment/archive handoff was committed at `fe44d48`; subsequent
documentation records this change in research priority. Frozen scientific files,
data, and caches must retain their hashes. Use new versioned files for new work.
Historical roadmaps in PLAN.md and LOGIC_EXTENSION.md do not override this priority.

Start the next session by proposing candidate geometric definitions and a concrete
checkpoint/behavior evaluation design. Choose and freeze the protocol before new
large extraction runs. No new data need archiving for this documentation-only update.
