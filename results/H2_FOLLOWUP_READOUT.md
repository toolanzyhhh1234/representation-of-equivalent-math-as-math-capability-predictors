# H2 follow-up: what the completed run tells us

The expanded run does not show a clear positive H2 signal. It improves the measurement
and provides a useful negative/uncertain result; it does not establish or conclusively
reject the full hypothesis. The complete tables are in
[PHASE4_H2_FOLLOWUP.md](PHASE4_H2_FOLLOWUP.md).

The run used 36 problems, five models, two separately reported arms and 21,600 fresh
responses. Eighteen problems were development cases and eighteen had not previously
been evaluated in this project's H2 experiment. The strict arm used 32 samples per
variant. The secondary GSM-Symbolic arm used eight. An independent AI reviewer checked
72 paraphrases without model scores; two wording issues were corrected and re-reviewed
before inference. This was not human certification or a formal equivalence proof.

## Main findings

- **The proposed Phi fragility pattern did not generalize to the new questions.**
  On development problems, Phi's rewritten accuracy was 9.6 percentage points below
  canonical accuracy. On the new problems, the difference was +1.2 points, with a
  problem-bootstrap interval of [-5.7, +7.4]. This is not evidence that rewriting helps
  Phi; it shows that the earlier apparent penalty is not stable across these subsets.
- **Math tuning improved accuracy without a clear invariance gain.** The Qwen2.5
  base and math models each have 1,543,714,304 measured parameters. On the new problems,
  mean accuracy across variants increased from 56.8% to 67.4%, while prior EQ_resid
  increased by 0.0084. On their seven common eligible problems, the INV difference
  was +0.002, with a paired problem-bootstrap interval of [-0.038, +0.037]. The
  interval leaves modest effects unresolved; this is not an equivalence test.
- **Reliable conditional point estimates do not follow the predicted positive EQ
  ordering.** Holdout conditional INV was 0.908 for Phi, 0.893 for Qwen2.5-0.5B,
  0.883 for Qwen2.5-1.5B, and 0.881 for its math-tuned counterpart. These use different
  eligible subsets and have substantial overlapping uncertainty. They must not be
  turned into a claim that lower EQ causally improves robustness. The cross-fitted
  sensitivity analysis also provides no clean positive ordering.
- **SmolLM2 remains uninformative for this success-based robustness measure.** Its
  strict-arm accuracy was 3.4% overall, with only two eligible problems out of 36.
  More draws improved precision but did not solve the true ability-floor problem.

All models' canonical-versus-rewrite accuracy intervals on the new problems included
zero. The intervals condition on observed sample rates and problem selection; they
are not complete uncertainty estimates for a randomly sampled population of tasks.
Five models represent only three families, and excluding the low-coverage SmolLM2
leaves three Qwen models and Phi. This is too little independent model diversity for
a confirmatory H2 result. No cross-model significance test was run or selected.

## Recommended next decision

Keep the strict arm primary and preserve this result as uncertain/negative evidence
rather than adjusting the metric to obtain the desired ordering. The next meaningful
extension is a preregistered, more diverse panel of models with enough independently
measured math capability to provide usable coverage, especially additional non-Qwen
families. Set capability/coverage rules before observing their rewrite results and
report excluded or unidentifiable models. Retain the raw, conditional, cross-fitted,
accuracy and common-problem views; do not silently switch the primary readout.

A larger independent problem sample and response-extraction audit are also needed
before strong claims. Further runs should test a fixed claim and quantify remaining
sampling uncertainty, not repeatedly select new wording or readouts until a positive
correlation appears. The earlier H1 capability correlation is a separate result and
was not modified by this experiment.

## Validation and artifacts

Fourteen tests passed (nine original H2 guards plus five follow-up guards). All 340
saved batches were validated; summaries were recomputed from their raw generations.
Completed-run resumption performed no new inference. Frozen scientific code and
input hashes matched, and the original pilot and earlier aggregate results were
preserved.

- [Frozen follow-up protocol](../prereg/H2_FOLLOWUP.md)
- [Protocol snapshot](../prereg/h2_followup.json)
- [Independent review and adjudication](../prereg/h2_followup_review.json)
- [Original-pilot measurement audit](PHASE4_H2_AUDIT.md)
- [Run instructions](../H2_RUNNING.md)
- Local raw responses: `results/h2/followup_v1/`
