"""Larger, separately frozen H2 measurement follow-up with two sampling budgets."""
import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .config import ROOT
from .eval_gsm8k import build_prompt
from .h2_data import digest, validate_bundle
from .h2_followup_data import DEST, build
from .h2_followup_metrics import describe
from .h2_metrics import per_problem
from .hf_data import load_benchmark
from .run_h2 import atomic_json, environment, generate_tokens, score, validate_batch

PROTOCOL = ROOT / "prereg/h2_followup.json"
REVIEW = ROOT / "prereg/h2_followup_review.json"
RUN_DIR = ROOT / "results/h2/followup_v1"
MODELS = {
    "HuggingFaceTB/SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "microsoft/phi-1_5": "77aa61eeac94fbf33d492b9f2744c98b42d5b5eb",
    "Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
    "Qwen/Qwen2.5-Math-1.5B": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
}
SETTINGS = {"samples": {"strict": 32, "symbolic": 8}, "batch_size": 64, "seed": 20260923,
            "n_shot": 5, "max_new_tokens": 256, "temperature": 0.7, "top_p": 0.95, "top_k": 0}
CODE_FILES = ["src/run_h2_followup.py", "src/h2_followup_data.py", "src/h2_followup_metrics.py",
              "src/analyze_h2_followup.py", "src/run_h2.py", "src/h2_data.py", "src/h2_metrics.py",
              "src/eval_gsm8k.py", "src/hf_data.py", "src/config.py", "data/hf_sources.json",
              "data/h2_rewrites.json", "data/h2_followup_rewrites.json", "pyproject.toml", "uv.lock",
              "prereg/H2_FOLLOWUP.md", "prereg/h2_followup_review.json"]


def fingerprints():
    return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in CODE_FILES}


def jobs_for(protocol, bundle):
    jobs = []
    for item in bundle["items"]:
        prompt = build_prompt(protocol["shots"], item["question"])
        for sample in range(protocol["settings"]["samples"][item["arm"]]):
            jobs.append({"item_id": item["item_id"], "sample": sample, "gold": item["gold"], "prompt": prompt})
    return sorted(jobs, key=lambda j: (len(j["prompt"]), j["item_id"], j["sample"]))


def prepare():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/bf16 required")
    bundle = build()
    review = json.loads(REVIEW.read_text())
    packet = ROOT / "data/h2/followup_review_packet.json"
    if review["packet_sha256"] != hashlib.sha256(packet.read_bytes()).hexdigest():
        raise ValueError("The review does not cover the current wording")
    if not review["approved_for_exploratory_run"]:
        raise ValueError("Resolve review findings before running")
    ds = load_benchmark("gsm8k")
    panel_path = ROOT / "results/panel.json"
    panel = json.loads(panel_path.read_text())
    eq = {r["model"]: r["headlines"]["last|k1|eq_resid"]["heldout"]
          for r in panel["models"] if r["model"] in MODELS}
    counts = {r["model"]: r["n_params"] for r in panel["models"] if r["model"] in MODELS}
    value = {"schema": 1, "stage": "outcome-informed measurement follow-up; not confirmatory H2",
             "primary_arm": "strict", "secondary_arm": "symbolic", "models": MODELS, "settings": SETTINGS,
             "stimuli_sha256": bundle["sha256"], "reviewer_type": review["reviewer_type"],
             "shots": [dict(ds["train"][i]) for i in range(SETTINGS["n_shot"])],
             "eq_resid": eq, "n_params": counts, "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
             "code_sha256": fingerprints(), "environment": environment()}
    value["sha256"] = digest(value)
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text())
        if {k: v for k, v in old.items() if k != "created_utc"} != value:
            raise ValueError("Changed frozen protocol; use a new version")
        return old
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(PROTOCOL, value)
    print(f"Frozen follow-up: {value['sha256']}")
    print(f"{len(jobs_for(value, bundle))} generations/model × {len(MODELS)} models")
    return value


def load_inputs(check_environment=True):
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["sha256"] != digest({k: v for k, v in protocol.items() if k not in ("sha256", "created_utc")}):
        raise ValueError("Protocol hash mismatch")
    if protocol["code_sha256"] != fingerprints() or protocol["settings"] != SETTINGS or protocol["models"] != MODELS:
        raise ValueError("Frozen scientific code/settings/models changed")
    if check_environment and protocol["environment"] != environment():
        raise ValueError("Runtime changed")
    bundle = json.loads(DEST.read_text())
    validate_bundle(bundle)
    if bundle["sha256"] != protocol["stimuli_sha256"]: raise ValueError("Stimulus mismatch")
    return protocol, bundle


def summarize_records(protocol, bundle, model_id, records):
    expected = sum(protocol["settings"]["samples"][r["arm"]] for r in bundle["items"])
    by_key = {(r["item_id"], r["sample"]): r for r in records}
    if len(by_key) != expected or len(records) != expected: raise ValueError("Incomplete or duplicate responses")
    out = {"model": model_id, "protocol_sha256": protocol["sha256"], "eq_resid": protocol["eq_resid"][model_id],
           "n_generations": len(records), "arms": {}}
    for arm in ("strict", "symbolic"):
        k = protocol["settings"]["samples"][arm]
        items = [r for r in bundle["items"] if r["arm"] == arm]
        rows = np.asarray([[[by_key[r["item_id"], sample]["correct"] for sample in range(k)]
                            for r in items if r["template_id"] == tid] for tid in bundle["template_ids"]])
        stats = per_problem(rows)
        groups = {}
        for name, ids in (("all", bundle["template_ids"]), ("development", bundle["development_template_ids"]),
                          ("holdout", bundle["holdout_template_ids"])):
            mask = np.isin(bundle["template_ids"], ids)
            groups[name] = describe(rows[mask], seed=SETTINGS["seed"], min_eligible=10 if name == "all" else 5)
            if arm == "symbolic":
                # Instance zero is not a canonical original in the symbolic arm.
                for key in ("canonical_accuracy", "other_variant_accuracy", "other_minus_canonical",
                            "accuracy_difference_problem_bootstrap_ci95"):
                    del groups[name][key]
        rr = [r for r in records if r["item_id"].startswith(arm+"/")]
        groups["diagnostics"] = {
            "parse_failure_rate": sum(r["pred"] is None for r in rr)/len(rr),
            "token_limit_rate": sum(r["hit_token_limit"] for r in rr)/len(rr),
            "cap_without_marker_or_boundary_rate": sum(r["hit_token_limit"] and "####" not in r["generation"].split("Question:")[0]
                                                       and "Question:" not in r["generation"] for r in rr)/len(rr),
        }
        groups["problems"] = [{"template_id": tid, "accuracy": float(stats["accuracy"][i]),
                               "sd": float(stats["sd"][i]), "eligible": bool(stats["eligible"][i]),
                               "variant_accuracy": stats["variant_accuracy"][i].tolist(),
                               "outcomes": rows[i].tolist()} for i, tid in enumerate(bundle["template_ids"])]
        out["arms"][arm] = groups
    return out


def load_saved(protocol, bundle, model_id, require_complete=True):
    jobs = jobs_for(protocol, bundle)
    bs = protocol["settings"]["batch_size"]
    folder = RUN_DIR/model_id.replace("/", "__")
    chunks = [jobs[start:start+bs] for start in range(0, len(jobs), bs)]
    records, pending = [], []
    for b, chunk in enumerate(chunks):
        path = folder/f"batch_{b:04d}.json"
        if path.exists():
            records.extend(validate_batch(json.loads(path.read_text()), chunk, protocol["sha256"], model_id))
        else: pending.append(b)
    if require_complete and pending: raise ValueError(f"Incomplete run: {model_id}, {len(pending)} pending batches")
    return jobs, chunks, records, pending


@torch.no_grad()
def run_model(protocol, bundle, model_id):
    folder = RUN_DIR/model_id.replace("/", "__")
    folder.mkdir(parents=True, exist_ok=True)
    jobs, chunks, records, pending = load_saved(protocol, bundle, model_id, require_complete=False)
    metadata = {"protocol_sha256": protocol["sha256"], "model": model_id,
                "revision": MODELS[model_id], "job_order_sha256": digest(jobs)}
    manifest = folder/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Refusing mixed run manifests")
    atomic_json(manifest, metadata)
    if pending:
        tok = AutoTokenizer.from_pretrained(model_id, revision=MODELS[model_id])
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=MODELS[model_id],
                            dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != protocol["n_params"][model_id]:
            raise ValueError("Parameter count differs from archived weight measurement")
        atomic_json(folder/"model_stats.json", {"n_params": n_params, "dtype": str(model.dtype),
                                               "revision": MODELS[model_id]})
        eos = model.generation_config.eos_token_id
        if eos is None: eos = tok.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        cfg = GenerationConfig(do_sample=True, temperature=SETTINGS["temperature"], top_p=SETTINGS["top_p"],
                    top_k=0, max_new_tokens=SETTINGS["max_new_tokens"], num_beams=1, repetition_penalty=1.0,
                    use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=eos, bos_token_id=tok.bos_token_id)
        max_prompt = max(len(tok(prompt)["input_ids"]) for prompt in {j["prompt"] for j in jobs})
        if max_prompt+SETTINGS["max_new_tokens"] > getattr(model.config, "max_position_embeddings", tok.model_max_length):
            raise ValueError("Context exceeded; no silent truncation allowed")
        atomic_json(folder/"generation_config.json", cfg.to_dict())
        atomic_json(folder/"lengths.json", {"max_prompt_tokens": max_prompt, "max_new_tokens": SETTINGS["max_new_tokens"]})
        start = time.monotonic()
        for b in pending:
            torch.manual_seed(SETTINGS["seed"]+b)
            chunk = chunks[b]
            tokens = generate_tokens(model, tok, [j["prompt"] for j in chunk], cfg)
            generations = tok.batch_decode(tokens, skip_special_tokens=True)
            rr = []
            for j, generation, ids in zip(chunk, generations, tokens.cpu().tolist()):
                pred, correct = score(generation, j["gold"])
                stop = next((i for i, t in enumerate(ids) if t in eos_ids), None)
                n = len(ids) if stop is None else stop+1
                rr.append({"item_id": j["item_id"], "sample": j["sample"], "gold": j["gold"],
                           "generation": generation, "pred": pred, "correct": correct,
                           "n_generated_tokens": n, "hit_token_limit": stop is None and n == SETTINGS["max_new_tokens"]})
            value = {"protocol_sha256": protocol["sha256"], "model": model_id, "batch_id": b,
                     "seed": SETTINGS["seed"]+b, "records": rr, "records_sha256": digest(rr)}
            validate_batch(value, chunk, protocol["sha256"], model_id)
            atomic_json(folder/f"batch_{b:04d}.json", value)
            records.extend(rr)
            print(f"{model_id}: {len(records)}/{len(jobs)} saved ({time.monotonic()-start:.0f}s)", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    summary = summarize_records(protocol, bundle, model_id, records)
    atomic_json(folder/"summary.json", summary)
    print(f"Completed {model_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    protocol, bundle = load_inputs()
    for model_id in args.models: run_model(protocol, bundle, model_id)


if __name__ == "__main__":
    main()
