# Reproduce the logical-consequence pilot

Read [the findings](results/LOGIC_READOUT.md) before interpreting the probe scores.
The initial premise-role generator has a demonstrated syntax shortcut; do not reuse
that task as a clean measure of semantic indispensability without redesigning it.

```bash
source .venv/bin/activate
python -m src.logic_data
python -m tests.test_logic
python -m src.run_logic --prepare
python -m src.run_logic
python -m src.analyze_logic
python -m src.audit_logic
```

The already frozen protocol is `prereg/logic_pilot.json`; source, data, model and
library fingerprints must match. A changed experiment needs a new version.
`run_logic` verifies completed activation caches and skips their recomputation.
`analyze_logic` fits only training rows, selects layers only on development rows,
then writes all declared test metrics. It can run from restored caches without
model weights or GPU execution. The shortcut audit is explicitly post-hoc and does
not modify the registered results.

Preserved locally and in the HF snapshot:

```text
data/logic/pilot.json                       # theories, prompts, labels, witnesses
results/logic/pilot_v1/<model>/states.npz   # fp32 layer readouts + native choice logits
results/logic/pilot_v1/<model>/manifest.json
results/logic/pilot_v1/<model>/*_probe.npz   # scalers, projections, coefficients, probabilities
results/logic/pilot_v1/<model>/probe_results.json
results/logic/pilot_v1/analysis.json
results/logic/pilot_v1/premise_shortcut_audit.json
```

There are five models and 6,000 prompt rows per model. Array row order is exactly
`data/logic/pilot.json["items"]`; layers start with the embedding control at index 0.
Dataset/model revisions and hashes are in the manifests and protocol. See
[ARCHIVES.md](ARCHIVES.md) for snapshot retrieval and checked restoration.
