# Running the rewrite-invariance pilot

The first experiment is a feasibility pilot with two separate arms: same-answer
paraphrases of GSM8K originals, and GSM-Symbolic variants with changed numbers.
The locally frozen design is in [prereg/H2_PILOT.md](prereg/H2_PILOT.md), with exact
settings and fingerprints in [prereg/h2_pilot.json](prereg/h2_pilot.json).

From the repository root, after [environment setup](SETUP.md):

```bash
source .venv/bin/activate
python -m src.h2_data        # download/check inputs; assemble 108 audited questions
python -m tests.test_h2      # CPU guards, including a tiny local model API check
python -m src.run_h2 --prepare
python -m src.run_h2         # run or resume the frozen three-model pilot
python -m src.analyze_h2     # requires every registered batch to be complete
```

To run just one of the registered models:

```bash
python -m src.run_h2 --models Qwen/Qwen2.5-0.5B
```

The runner downloads weights as needed at the registered revision. It saves 32
responses atomically after each batch, and validates old records and per-batch seeds
on resumption. There are 27 batches (864 responses) per model. Running it again after
completion checks the stored batches and regenerates summaries without model loading.

Current outputs are local and ignored by Git:

```text
data/h2/pilot.json
results/h2/pilot_v2/<model>/manifest.json
results/h2/pilot_v2/<model>/generation_config.json
results/h2/pilot_v2/<model>/batch_0000.json  ... batch_0026.json
results/h2/pilot_v2/<model>/summary.json
```

The report is generated at `results/PHASE4_H2_PILOT.md`. Its tables are recomputed
from raw responses and checked against the saved summaries. The archive and earlier
phase result files are preserved.

The original v1 protocol is retained because a removed Transformers argument caused
the first launch to fail before producing any responses. All scientific settings
and stimuli were preserved for v2; the failed directory contains no response batches.

Do not edit frozen scientific code/settings and resume into an existing run. The
runner rejects changed hashes and runtime versions. A new stimulus set, decoding
budget, checkpoint panel, or parser requires a separately named protocol/run, with
changes informed by this pilot explicitly acknowledged.

Raw INV alone is not a success metric: a model that always fails has INV=1. Always
read accuracy, eligible-problem counts, conditional INV, sampling uncertainty and
token-limit diagnostics together. The token-limit flag refers to physical generation
length; a model can answer and then continue generating new few-shot examples, so
this flag alone does not establish an incomplete answer.

## Expanded follow-up

The next protocol is separately frozen in [prereg/H2_FOLLOWUP.md](prereg/H2_FOLLOWUP.md)
and [h2_followup.json](prereg/h2_followup.json). It uses 36 problems (18 development,
18 new-problem holdouts), five models, and 32 samples per strict variant versus
8 per symbolic variant: 21,600 fresh generations in total. The added Qwen2.5-1.5B
base/math pair supplies a matched-size contrast. Five models still do not constitute
the full-panel H2 test.

An independent AI reviewer audited all 72 paraphrases blind to model scores. Two
wording changes were required and independently re-reviewed before inference.
See the [review manifest](prereg/h2_followup_review.json); both exact blind packets
are preserved in [prereg/review_packets](prereg/review_packets). This is not human
certification or a formal proof of natural-language equivalence.

```bash
source .venv/bin/activate
python -m src.h2_followup_data
python -m tests.test_h2_followup
python -m src.run_h2_followup --prepare
python -m src.run_h2_followup
python -m src.analyze_h2_followup
```

Use `--models MODEL_ID` to resume an individual registered model. Results are in
`results/h2/followup_v1/`, with 68 atomic batch files per model (the final batch has
32 responses). Analysis requires the whole registered run to be complete and
recomputes all summaries from raw responses. It writes `results/PHASE4_H2_FOLLOWUP.md`.

The primary report separates new-problem holdouts from development cases and keeps
accuracy, eligibility, conditional INV, and cross-fitted sensitivity estimates visible.
The [measurement audit](results/PHASE4_H2_AUDIT.md) explains why higher sample counts
and those diagnostics were chosen. Earlier pilot draws and scores are not overwritten
or silently merged into the follow-up.
