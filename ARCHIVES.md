# Preserved experiment data

The public data repository is
[toolazyhhh123/representation-of-equivalent-math-raw](https://huggingface.co/datasets/toolazyhhh123/representation-of-equivalent-math-raw).
Use pinned revisions for reproduction.

| Snapshot | HF revision | Location | Code commit |
|---|---|---|---|
| Original MELD/PAWS activations and capability outputs | `9fc0c488165f29adb6cb0386ae39d2aee3818d16` | Original archive root | Original report code history |
| H2 pilot and expanded follow-up | `2f53217efb77eaaa23db2fdb549ea9995b06d305` | [experiments/h2-2026-09-22](https://huggingface.co/datasets/toolazyhhh123/representation-of-equivalent-math-raw/tree/2f53217efb77eaaa23db2fdb549ea9995b06d305/experiments/h2-2026-09-22) | `62fc4dc4948466701e118eff64d43bcaba40f407` |

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
