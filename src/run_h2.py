"""Frozen, resumable H2 pilot. Prepare first, then run the three registered models."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .config import ROOT
from .eval_gsm8k import build_prompt, norm, pred_of
from .h2_data import DEST, build, digest, validate_bundle
from .h2_metrics import per_problem, summarize
from .hf_data import load_benchmark

PROTOCOL = ROOT / "prereg" / "h2_pilot.json"
RUN_DIR = ROOT / "results" / "h2" / "pilot_v2"
MODELS = {
    "HuggingFaceTB/SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "microsoft/phi-1_5": "77aa61eeac94fbf33d492b9f2744c98b42d5b5eb",
}
SETTINGS = {"samples": 8, "batch_size": 32, "seed": 20260922, "n_shot": 5,
            "max_new_tokens": 256, "temperature": 0.7, "top_p": 0.95, "top_k": 0}
CODE_FILES = ["src/run_h2.py", "src/h2_data.py", "src/h2_metrics.py", "src/eval_gsm8k.py",
              "src/hf_data.py", "src/config.py", "data/hf_sources.json", "data/h2_rewrites.json",
              "pyproject.toml", "uv.lock", "prereg/H2_PILOT.md"]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprints():
    return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in CODE_FILES}


def environment():
    return {"packages": {p: importlib.metadata.version(p) for p in
                          ("torch", "transformers", "numpy", "datasets", "huggingface-hub")},
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}


def prepare():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("H2 requires CUDA and bf16, with no quantization")
    bundle = build()
    ds = load_benchmark("gsm8k")
    shots = [dict(ds["train"][i]) for i in range(SETTINGS["n_shot"])]
    panel_path = ROOT / "results/panel.json"
    panel = json.loads(panel_path.read_text())
    eq = {r["model"]: r["headlines"]["last|k1|eq_resid"]["heldout"]
          for r in panel["models"] if r["model"] in MODELS}
    value = {"schema": 1, "stage": "feasibility pilot, not confirmatory H2 test",
             "models": MODELS, "settings": SETTINGS, "stimuli_sha256": bundle["sha256"],
             "template_ids": bundle["template_ids"], "shots": shots,
             "eq_resid": eq, "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
             "code_sha256": fingerprints(), "environment": environment()}
    value["sha256"] = digest(value)
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text())
        if {k: v for k, v in old.items() if k != "created_utc"} != value:
            raise ValueError("Protocol changed. Preserve this pilot and register a new run/version.")
        print("Existing frozen protocol matches.")
        return old
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(PROTOCOL, value)
    print(f"Frozen protocol SHA256: {value['sha256']}")
    return value


def load_inputs():
    protocol = json.loads(PROTOCOL.read_text())
    unhashed = {k: v for k, v in protocol.items() if k not in ("sha256", "created_utc")}
    if protocol["sha256"] != digest(unhashed):
        raise ValueError("Protocol hash mismatch")
    if protocol["code_sha256"] != fingerprints() or protocol["settings"] != SETTINGS:
        raise ValueError("Code/settings changed since the protocol was frozen")
    if protocol["models"] != MODELS or protocol["environment"] != environment():
        raise ValueError("Models/environment changed since the protocol was frozen")
    bundle = json.loads(DEST.read_text())
    validate_bundle(bundle)
    if bundle["sha256"] != protocol["stimuli_sha256"]:
        raise ValueError("Different stimuli from frozen protocol")
    return protocol, bundle


def make_jobs(protocol, bundle):
    jobs = []
    for item in bundle["items"]:
        # Only the question enters the prompt: no audit expression, gold or reference solution.
        prompt = build_prompt(protocol["shots"], item["question"])
        for sample in range(SETTINGS["samples"]):
            jobs.append({"item_id": item["item_id"], "sample": sample, "prompt": prompt,
                         "gold": item["gold"]})
    return sorted(jobs, key=lambda j: (len(j["prompt"]), j["item_id"], j["sample"]))


def score(generation, gold):
    pred = pred_of(generation)
    p, g = norm(pred), norm(gold)
    correct = p is not None and g is not None and math.isfinite(p) and p == g
    return pred, bool(correct)


def generate_tokens(model, tokenizer, prompts, generation_config):
    """Use the registered config as both explicit config and fallback defaults."""
    model.generation_config = generation_config
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    return model.generate(**enc, generation_config=generation_config)[:, enc["input_ids"].shape[1]:]


def validate_batch(batch, jobs, protocol_hash, model_id):
    if batch["protocol_sha256"] != protocol_hash or batch["model"] != model_id:
        raise ValueError("Batch belongs to a different experiment")
    records = batch["records"]
    if batch["records_sha256"] != digest(records):
        raise ValueError("Batch record hash mismatch")
    expected = [(j["item_id"], j["sample"], j["gold"]) for j in jobs]
    actual = [(r["item_id"], r["sample"], r["gold"]) for r in records]
    if actual != expected:
        raise ValueError("Batch sample order/labels changed")
    for r in records:
        if score(r["generation"], r["gold"]) != (r["pred"], r["correct"]):
            raise ValueError("Stored score does not match generation")
    return records


def summarize_model(protocol, bundle, model_id, records):
    by_key = {(r["item_id"], r["sample"]): r for r in records}
    if len(by_key) != len(bundle["items"]) * SETTINGS["samples"]:
        raise ValueError("Missing or duplicate generations")
    out = {"model": model_id, "model_revision": MODELS[model_id],
           "protocol_sha256": protocol["sha256"], "eq_resid": protocol["eq_resid"][model_id],
           "n_generations": len(records), "arms": {}}
    for arm in ("strict", "symbolic"):
        matrices, problems = [], []
        for tid in bundle["template_ids"]:
            items = [r for r in bundle["items"] if r["arm"] == arm and r["template_id"] == tid]
            rows, variants = [], []
            for item in items:
                recs = [by_key[item["item_id"], k] for k in range(SETTINGS["samples"])]
                outcomes = [r["correct"] for r in recs]
                rows.append(outcomes)
                variants.append({"item_id": item["item_id"], "variant": item["variant"],
                                 "gold": item["gold"], "accuracy": float(np.mean(outcomes))})
            matrices.append(rows)
            problems.append({"template_id": tid, "variants": variants})
        matrix = np.asarray(matrices)
        stats = per_problem(matrix)
        summary = summarize(matrix, seed=SETTINGS["seed"])
        arm_ids = {r["item_id"] for r in bundle["items"] if r["arm"] == arm}
        arm_records = [r for r in records if r["item_id"] in arm_ids]
        summary["parse_failure_rate"] = sum(r["pred"] is None for r in arm_records) / len(arm_records)
        summary["token_limit_rate"] = sum(r["hit_token_limit"] for r in arm_records) / len(arm_records)
        if arm == "strict":
            summary["canonical_accuracy"] = float(matrix[:, 0, :].mean())
            summary["rewrite_accuracy"] = float(matrix[:, 1:, :].mean())
            summary["rewrite_minus_canonical"] = summary["rewrite_accuracy"]-summary["canonical_accuracy"]
        for i, problem in enumerate(problems):
            problem.update({"mean_accuracy": float(stats["accuracy"][i]),
                            "sd": float(stats["sd"][i]), "eligible": bool(stats["eligible"][i])})
        summary["problems"] = problems
        out["arms"][arm] = summary
    return out


@torch.no_grad()
def run_model(protocol, bundle, model_id):
    dest = RUN_DIR / model_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(protocol, bundle)
    chunks = [jobs[i:i+SETTINGS["batch_size"]] for i in range(0, len(jobs), SETTINGS["batch_size"])]
    metadata = {"protocol_sha256": protocol["sha256"], "model": model_id,
                "model_revision": MODELS[model_id], "job_order_sha256": digest(jobs)}
    meta_path = dest / "manifest.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Refusing to mix results from different runs")
    atomic_json(meta_path, metadata)
    records, pending = [], []
    for b, chunk in enumerate(chunks):
        path = dest / f"batch_{b:04d}.json"
        if path.exists():
            records.extend(validate_batch(json.loads(path.read_text()), chunk, protocol["sha256"], model_id))
        else:
            pending.append(b)
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODELS[model_id])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=MODELS[model_id], dtype=torch.bfloat16,
            attn_implementation="sdpa").cuda().eval()
        eos = model.generation_config.eos_token_id
        if eos is None: eos = tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        generation_config = GenerationConfig(
            do_sample=True, temperature=SETTINGS["temperature"], top_p=SETTINGS["top_p"],
            top_k=SETTINGS["top_k"], max_new_tokens=SETTINGS["max_new_tokens"],
            num_beams=1, repetition_penalty=1.0, use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
            bos_token_id=tokenizer.bos_token_id)
        context_limit = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
        all_lengths = [len(tokenizer(j["prompt"])["input_ids"]) for j in jobs[::SETTINGS["samples"]]]
        if max(all_lengths)+SETTINGS["max_new_tokens"] > context_limit:
            raise ValueError("Prompt + output budget exceeds context; refusing silent truncation")
        atomic_json(dest / "generation_config.json", generation_config.to_dict())
        start = time.monotonic()
        for b in pending:
            chunk = chunks[b]
            torch.manual_seed(SETTINGS["seed"]+b)
            tokens = generate_tokens(model, tokenizer, [j["prompt"] for j in chunk], generation_config)
            generations = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            batch_records = []
            for job, text, ids in zip(chunk, generations, tokens.cpu().tolist()):
                pred, correct = score(text, job["gold"])
                stop = next((i for i, t in enumerate(ids) if t in eos_ids), None)
                n_tokens = len(ids) if stop is None else stop+1
                batch_records.append({"item_id": job["item_id"], "sample": job["sample"],
                                      "gold": job["gold"], "generation": text, "pred": pred,
                                      "correct": correct, "n_generated_tokens": n_tokens,
                                      "hit_token_limit": stop is None and n_tokens == SETTINGS["max_new_tokens"]})
            batch = {"protocol_sha256": protocol["sha256"], "model": model_id,
                     "batch_id": b, "seed": SETTINGS["seed"]+b,
                     "records": batch_records, "records_sha256": digest(batch_records)}
            validate_batch(batch, chunk, protocol["sha256"], model_id)
            atomic_json(dest / f"batch_{b:04d}.json", batch)
            records.extend(batch_records)
            print(f"{model_id}: {len(records)}/{len(jobs)} generations saved "
                  f"({time.monotonic()-start:.0f}s)", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    summary = summarize_model(protocol, bundle, model_id, records)
    atomic_json(dest / "summary.json", summary)
    print(json.dumps({arm: {k: v for k, v in data.items() if k != "problems"}
                      for arm, data in summary["arms"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="validate stimuli and freeze protocol; no inference")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    protocol, bundle = load_inputs()
    for model_id in args.models:
        run_model(protocol, bundle, model_id)


if __name__ == "__main__":
    main()
