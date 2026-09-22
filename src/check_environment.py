"""Offline setup checks; add --datasets to check prepared benchmark caches."""
import argparse
import importlib.metadata
import json
import platform

import numpy as np
import torch

from .config import RESULTS
from .data import load_meld
from .lexical import anchor_jobs, eq_variants, lexical_cos, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", action="store_true")
    args = parser.parse_args()
    versions = {p: importlib.metadata.version(p) for p in (
        "torch", "transformers", "datasets", "huggingface-hub", "numpy", "scipy",
        "scikit-learn", "matplotlib",
    )}
    report = {"python": platform.python_version(), "packages": versions}
    if not torch.cuda.is_available():
        raise RuntimeError("The extraction and metric pipelines require a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This study requires bf16 support; do not silently change precision.")
    # Execute a kernel, not just a driver-availability query.
    x = torch.eye(16, device="cuda", dtype=torch.bfloat16)
    if not torch.equal(x @ x, x):
        raise RuntimeError("CUDA bf16 matrix multiply failed.")
    report["gpu"] = torch.cuda.get_device_name(0)
    report["cuda"] = torch.version.cuda
    stim = load_meld()
    if len(stim.pairs) != 270 or len(stim.texts) != 1080:
        raise AssertionError("Unexpected MELD stimulus counts")
    baseline = summarize(eq_variants(None, anchor_jobs(stim), lexical_cos(stim.texts)))
    if not np.isclose(baseline["eq"], 0.7715, atol=5e-5, rtol=0):
        raise AssertionError(f"TF-IDF baseline changed: {baseline['eq']}")
    report["meld"] = {"pairs": len(stim.pairs), "texts": len(stim.texts), "baseline": baseline}
    if args.datasets:
        from .hf_data import load_benchmark
        from .specificity import build_paws
        expected = {"gsm8k": 1319, "math500": 500, "arc_easy": 2376, "paws": 8000}
        report["datasets"] = {}
        for name, count in expected.items():
            ds = load_benchmark(name)
            if len(ds["test"]) != count:
                raise AssertionError(f"Unexpected {name} test size")
            report["datasets"][name] = {k: len(v) for k, v in ds.items()}
        texts, pairs, labels = build_paws()
        if len(texts) != 3566 or len(pairs) != 2000 or int(labels.sum()) != 1000:
            raise AssertionError("PAWS sample no longer matches the archived activation layout")
        report["paws_sample"] = {"texts": len(texts), "pairs": len(pairs)}
    dest = RESULTS / "setup"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Environment checks PASS")


if __name__ == "__main__":
    main()
