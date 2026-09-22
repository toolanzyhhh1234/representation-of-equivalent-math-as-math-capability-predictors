# Preserved experiment data

The public data repository is
[toolazyhhh123/representation-of-equivalent-math-raw](https://huggingface.co/datasets/toolazyhhh123/representation-of-equivalent-math-raw).
Use pinned revisions for reproduction.

| Snapshot | HF revision | Location | Code commit |
|---|---|---|---|
| Original MELD/PAWS activations and capability outputs | `9fc0c488165f29adb6cb0386ae39d2aee3818d16` | Original archive root | Original report code history |
| H2 pilot and expanded follow-up | `2f53217efb77eaaa23db2fdb549ea9995b06d305` | [experiments/h2-2026-09-22](https://huggingface.co/datasets/toolazyhhh123/representation-of-equivalent-math-raw/tree/2f53217efb77eaaa23db2fdb549ea9995b06d305/experiments/h2-2026-09-22) | `62fc4dc4948466701e118eff64d43bcaba40f407` |
| Logical-consequence pilot and shortcut audit | `07f4d0c992dd2e37a650105fa369fe145860668f` | [experiments/logic-pilot-2026-09-22](https://huggingface.co/datasets/toolazyhhh123/representation-of-equivalent-math-raw/tree/07f4d0c992dd2e37a650105fa369fe145860668f/experiments/logic-pilot-2026-09-22) | `90786b0202c31076bca54a25fa490353c75c5da5` |

The H2 snapshot contains 483 data files, including all 24,192 response records from
the two completed runs, inputs, reviews, protocols, summaries, audit and environment
records, and exact upstream data copies with their provenance/license information.
`code.tar.gz` preserves the matching source tree. `manifest.json` records file sizes
and SHA256 hashes. Model caches and the Python installation are reconstructible
dependencies and are excluded; model revisions and the dependency lock are preserved.
Transient download logs and credentials are excluded.

```bash
hf download toolazyhhh123/representation-of-equivalent-math-raw \
  --repo-type dataset --revision 2f53217efb77eaaa23db2fdb549ea9995b06d305 \
  --include 'experiments/h2-2026-09-22/*' --local-dir /tmp/math-archive
python -m src.archive_data verify /tmp/math-archive/experiments/h2-2026-09-22 --restore-to .
python -m src.analyze_h2
python -m src.analyze_h2_followup
```

Restore verifies every archive and member before writing, and refuses to overwrite
different local data. Keep the snapshot's component licenses: third-party data are
not relicensed under the root repository's Apache label. Existing archive revisions
remain available when new experiment snapshots are added.

## Logical-consequence snapshot

The logic snapshot preserves 48 files (about 2.25 GB): all 300 theory groups and
6,000 prompts, five models' layerwise readouts and native choice logits, fitted probe
parameters/probabilities, protocols, complete metrics, environment versions and the
post-hoc syntax-shortcut audit. The raw states are float32 outputs of bf16 forward
passes, not a new quantized representation.

The data archive is transported as 104 ordered binary parts to avoid slow long-lived
connections. `parts.json` contains every part's size/SHA256 and the assembled archive
hash. Concatenation recreates the exact archive checked by `manifest.json`:

```bash
hf download toolazyhhh123/representation-of-equivalent-math-raw \
  --repo-type dataset --revision 07f4d0c992dd2e37a650105fa369fe145860668f \
  --include 'experiments/logic-pilot-2026-09-22/*' --local-dir /tmp/math-archive
cat /tmp/math-archive/experiments/logic-pilot-2026-09-22/data.tar.gz.part-*.bin \
  > /tmp/math-archive/experiments/logic-pilot-2026-09-22/data.tar.gz
python -m src.archive_data verify /tmp/math-archive/experiments/logic-pilot-2026-09-22 --restore-to .
python -m src.analyze_logic
python -m src.audit_logic
```

All remote part and code-object hashes were checked against the local archive;
metadata were freshly downloaded and compared byte for byte. The Git repository
also carries the manifests in [archive_manifests](archive_manifests).
Read [LOGIC_READOUT.md](results/LOGIC_READOUT.md) before treating the premise-role
data as a semantic benchmark: its labels have a demonstrated perfect syntax shortcut.
