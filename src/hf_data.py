"""Load benchmark revisions recorded in data/hf_sources.json."""
import json

from datasets import load_dataset

from .config import ROOT


def sources():
    return json.loads((ROOT / "data" / "hf_sources.json").read_text())


def load_benchmark(name):
    spec = sources()[name]
    return load_dataset(spec["repo_id"], spec["config"], revision=spec["revision"])
