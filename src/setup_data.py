"""Fetch pinned inputs without running experiments or replacing tracked results.

Usage: python -m src.setup_data --archive full --logic
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

from .config import DATA, RAW, RESULTS, ROOT
from .hf_data import load_benchmark, sources

DOWNLOADS = ROOT / "data" / "hf"


def fetch(name, patterns=None):
    spec = sources()[name]
    print(f"Downloading {name} at {spec['revision']}", flush=True)
    return Path(snapshot_download(
        spec["repo_id"], repo_type="dataset", revision=spec["revision"],
        local_dir=DOWNLOADS / name, allow_patterns=patterns, max_workers=4,
    ))


def link_directory(target, link):
    """Never merge or replace existing caches from a different extraction run."""
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(
            f"{link} already exists. Keep it separate from the archive at {target}; "
            "move your existing cache before linking the archived one."
        )
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=["none", "pilot", "full"], default="pilot",
                        help="pilot: SmolLM2 MELD cache + all raw outputs; full: ~38 GB")
    parser.add_argument("--logic", action="store_true", help="download ProofWriter parquet files")
    parser.add_argument("--folio", action="store_true", help="also fetch FOLIO (requires Hub access)")
    args = parser.parse_args()
    report = {"sources": sources(), "benchmarks": {}, "archive": args.archive}

    meld = fetch("meld")
    for filename in ("adversarial_theorem_pairs_2.json", "distractors_all.json"):
        upstream = (meld / filename).read_bytes()
        local = (DATA / filename).read_bytes()
        if json.loads(upstream) != json.loads(local):
            raise ValueError(f"Bundled MELD differs from pinned upstream: {filename}")
        report.setdefault("meld_sha256", {})[filename] = hashlib.sha256(local).hexdigest()
    print("Bundled MELD matches pinned upstream.", flush=True)

    for name in ("gsm8k", "math500", "arc_easy", "paws"):
        ds = load_benchmark(name)
        counts = {split: len(rows) for split, rows in ds.items()}
        report["benchmarks"][name] = counts
        print(f"{name}: {counts}", flush=True)

    if args.archive != "none":
        patterns = None if args.archive == "full" else [
            "README.md", "activations/HuggingFaceTB__SmolLM2-360M.*",
            "outputs/**", "results/**",
        ]
        archive = fetch("archive", patterns)
        link_directory(archive / "activations", RAW)
        link_directory(archive / "outputs", RESULTS / "raw_outputs")
    if args.logic:
        fetch("proofwriter")
    if args.folio:
        try:
            fetch("folio")
            report["folio"] = "downloaded"
        except GatedRepoError:
            report["folio"] = "gated: accept access terms on its HF page, then rerun --folio"
            print(report["folio"], flush=True)
    dest = RESULTS / "setup"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Data preparation complete; inventory: {dest / 'data.json'}", flush=True)


if __name__ == "__main__":
    main()
