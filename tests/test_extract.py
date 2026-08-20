"""The padding-invariance guard, runnable standalone: python -m tests.test_extract"""
import sys
sys.path.insert(0, ".")
from src.config import PILOT_MODELS
from src.data import load_meld
from src.extract import load, verify_padding_invariance

if __name__ == "__main__":
    stim = load_meld()
    mid = sys.argv[1] if len(sys.argv) > 1 else PILOT_MODELS[0]
    tok, model = load(mid)
    print(mid, "padding invariance:", verify_padding_invariance(tok, model, stim.texts))
    print("PASS")
