# Environment and data

Run commands from the repository root. The supported setup is Linux, Python 3.12,
and an NVIDIA GPU supporting bf16. Extraction and the fast metric pipeline require
CUDA; the aggregate analysis scripts can run on CPU.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then:

```bash
uv sync --locked
source .venv/bin/activate
hf auth login
hf auth whoami
```

`hf` is installed with `huggingface-hub` inside `.venv`. Enter tokens only into the
interactive login prompt. Authentication uses the standard Hugging Face location,
so the CLI, Transformers and Datasets share the same login.

`pyproject.toml` pins torch 2.11.0 (CUDA 12.8 wheel) and transformers 5.15.1 to the
versions documented in the [activation archive](https://huggingface.co/datasets/toolazyhhh123/representation-of-equivalent-math-raw).
`uv.lock` pins all resolved dependencies. The other dependency pins describe this
setup; the archive does not record the complete original environment. New extraction
is therefore not assumed bit-identical solely because these two versions match.
The CUDA runtime comes with the wheels; a working NVIDIA driver is still required.

## Download

```bash
# All four benchmark datasets, upstream MELD check, one parity-test cache,
# and the archived per-item evaluation outputs:
python -m src.setup_data

# Full activation archive (~38 GB decimal / 35.4 GiB), plus ProofWriter:
python -m src.setup_data --archive full --logic

# Optional natural-language first-order logic data, if your account has access:
python -m src.setup_data --archive none --folio
```

Downloads resume after interruption. Allow roughly 55–65 GB free for the complete
archive, Python/CUDA environment, caches, and a pilot checkpoint. Downloading every
panel model requires additional space. Downloads do not run model evaluations.

| Input | Local location / use |
|---|---|
| MELD | Tracked `data/meld/`; checked against pinned upstream JSON |
| GSM8K (`main`) | HF Datasets cache; train exemplars + 1,319 test items |
| MATH-500 | HF Datasets cache; 500 rows, of which the existing harness evaluates 496 |
| ARC-Easy | HF Datasets cache; 2,376 test items |
| PAWS (`labeled_final`) | HF Datasets cache; deterministic 2,000-pair sample |
| Original activations | `data/hf/archive/activations/`, linked as `results/raw/` |
| Per-item eval outputs | `data/hf/archive/outputs/`, linked as `results/raw_outputs/` |
| Archived aggregate results | `data/hf/archive/results/`; tracked `results/*.json` are preserved |
| ProofWriter | `data/hf/proofwriter/`, raw parquet (~43 MB download) |
| FOLIO (optional) | `data/hf/folio/`; HF access gate may require accepting terms |

All HF dataset revisions are recorded in `data/hf_sources.json`; the four existing
benchmark loaders use those revisions. Their prompt, sampling, scoring, and split
rules are unchanged. The pins record the revisions available during setup, not
unrecorded historical revisions from the original run. Cache validation checks the
expected MELD and PAWS layouts. `results/setup/data.json` records downloaded inputs.

ProofWriter is downloaded as parquet without expanding the entire dataset's proof
strings into Arrow. It is an input for the proposed logic extension, not wired into
the equivalence experiment. Use parquet column selection for a first pilot.

The symlinks avoid duplicating the large archive. Setup refuses to replace an
existing `results/raw` or `results/raw_outputs` directory. Keep new extractions in
a separate checkout or deliberately move the existing cache before switching.
Do not mix newly extracted bf16 runs with the archived panel. Existing runners
write their outputs to the configured paths, including through these symlinks.

## Verify

```bash
# GPU bf16 kernel, dependency versions, MELD lexical baseline and dataset layout:
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m src.check_environment --datasets

# Downloads the small SmolLM2 checkpoint if absent, then tests real extraction:
python -m tests.test_extract

# Requires the archived SmolLM2 MELD cache and CUDA:
python -m tests.test_parity

# Read-only reproduction of tables from tracked aggregate results:
python -m src.analyze
python -m src.analyze_spec
```

The environment check records versions and GPU identity in
`results/setup/environment.json`. Model weights are fetched as needed; gated Gemma
and Llama checkpoints additionally require access granted on their model pages.
Use the archived activations for historical reproduction, since the existing model
loaders do not pin model revisions.

The standard `run_panel`, eval, specificity and figure commands write outputs.
Run them deliberately when starting an experiment; environment verification does
not replace the tracked reported results. See [LOGIC_EXTENSION.md](LOGIC_EXTENSION.md)
for the proposed premises/consequences arm and review recommendations.

The rewrite-invariance feasibility pilot has a separate frozen protocol and
resumable runner; see [H2_RUNNING.md](H2_RUNNING.md).
