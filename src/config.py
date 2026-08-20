"""Pilot configuration. Phase 0 gate: see PLAN.md sec 9."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "meld"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"

# Phase 0 pilot panel: three ~0.4-0.6B models chosen to differ in math training.
# Expected EQ ordering (the gate): SmolLM2 < Qwen2.5-0.5B < Qwen3-0.6B.
# To extend, add a line; nothing else changes.
PILOT_MODELS = [
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B",
]

# PLAN sec 12.1: uniform precision across the panel, never quantized.
DTYPE = "bfloat16"
MAX_LEN = 256
BATCH_SIZE = 16

# PLAN sec 4.3: anisotropy correction. k = number of top principal components removed.
PC_REMOVAL_K = [0, 1, 3]
POOLINGS = ["last", "mean"]
PRIMARY_POOLING = "last"
