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

# Phase 0 found fixed-k is not a safe default (results/PHASE0.md). All corrections are
# computed; which is PRIMARY must be pre-registered on conditioning grounds, not on EQ.
CORRECTIONS = ["k0", "k1", "k3", "gapk", "zca"]

# (b): first base / math-tuned pair at identical size -- the earliest point where
# parameter count and math training come apart.
PAIR_MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Math-1.5B"]

# Phase 0.5 panel: the pre-registered dissociation test (results/PHASE0.md sec 7).
# SmolLM2-1.7B has Qwen2.5-0.5B's maths and Qwen2.5-1.5B's size; TinyLlama-1.1B and
# Falcon3-1B add size~1B points whose capability differs; Qwen2/Qwen2-Math-1.5B is a
# second base/math pair at identical parameter count. Llama-3.2-1B is gated (no token
# on this machine) and dropped. params = total parameters, from each model's card.
PANEL = [
    # model_id,                      params(B)
    ("HuggingFaceTB/SmolLM2-360M",   0.36),
    ("Qwen/Qwen2.5-0.5B",            0.49),
    ("Qwen/Qwen3-0.6B",              0.60),
    ("TinyLlama/TinyLlama_v1.1",     1.10),
    ("tiiuae/Falcon3-1B-Base",       1.67),
    ("HuggingFaceTB/SmolLM2-1.7B",   1.71),
    ("Qwen/Qwen2-1.5B",              1.54),
    ("Qwen/Qwen2-Math-1.5B",         1.54),
    ("Qwen/Qwen2.5-1.5B",            1.54),
    ("Qwen/Qwen2.5-Math-1.5B",       1.54),
    # Phase 2 expansion, ungated tranche: 6 new families + within-Qwen contrasts.
    # params(B) are card values; extraction records the measured count as ground truth.
    ("allenai/OLMo-2-0425-1B",       1.48),
    ("microsoft/phi-1_5",            1.42),
    ("stabilityai/stablelm-2-1_6b",  1.64),
    ("EleutherAI/pythia-1.4b",       1.41),
    ("deepseek-ai/deepseek-coder-1.3b-base", 1.35),
    ("ibm-granite/granite-3.3-2b-base", 2.53),
    ("Qwen/Qwen2-0.5B",              0.49),
    ("Qwen/Qwen2.5-Coder-1.5B",      1.54),
    ("Qwen/Qwen2.5-1.5B-Instruct",   1.54),
    ("Qwen/Qwen3-1.7B",              1.72),
    # Gated tranche (HF token + accepted licenses):
    ("google/gemma-3-1b-pt",         1.00),
    ("google/gemma-2-2b",            2.61),
    ("meta-llama/Llama-3.2-1B",      1.24),
]

# Published same-harness GSM8K (SmolLM2 paper / model cards, 5-shot) -- used only as a
# cross-check; the capability axis for any claim is measured in-house (src/eval_gsm8k.py).
GSM8K_PUBLISHED = {
    "HuggingFaceTB/SmolLM2-360M": 3.2,
    "Qwen/Qwen2.5-0.5B": 33.4,
    "Qwen/Qwen2.5-1.5B": 61.7,
    "HuggingFaceTB/SmolLM2-1.7B": 31.1,
}
