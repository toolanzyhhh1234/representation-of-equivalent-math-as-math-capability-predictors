"""Versioned follow-up stimuli; never alter the frozen first pilot."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import ROOT
from .eval_gsm8k import gold_of
from .h2_data import SYMBOLIC_REPO, SYMBOLIC_REVISION, digest, validate_bundle
from .hf_data import load_benchmark, sources

FIXTURE = ROOT / "data/h2_followup_rewrites.json"
DEST = ROOT / "data/h2/followup.json"


def build():
    old_path = ROOT / "data/h2_rewrites.json"
    old = json.loads(old_path.read_text())
    new = json.loads(FIXTURE.read_text())
    specs = old["items"] + new["items"]
    # Any corrections after independent review must be explicit and frozen before inference.
    for spec in specs:
        if str(spec["template_id"]) in new.get("development_overrides", {}):
            spec.update(new["development_overrides"][str(spec["template_id"])])
    path = hf_hub_download(SYMBOLIC_REPO, "main/test.jsonl", repo_type="dataset",
                           revision=SYMBOLIC_REVISION, local_dir=ROOT / "data/hf/gsm_symbolic")
    upstream = {(r["id"], r["instance"]): r for r in
                (json.loads(line) for line in Path(path).read_text().splitlines())}
    gsm = load_benchmark("gsm8k")["test"]
    development = [s["template_id"] for s in old["items"]]
    holdout = [s["template_id"] for s in new["items"]]
    items, packet = [], []
    for spec in specs:
        tid = spec["template_id"]
        row = upstream[tid, 0]
        original = gsm[row["original_id"]]
        if original["question"].strip() != row["original_question"].strip():
            raise ValueError(f"Original question mismatch: {tid}")
        gold = gold_of(original["answer"])
        if gold != gold_of(row["original_answer"]): raise ValueError(f"Original gold mismatch: {tid}")
        partition = "development" if tid in development else "holdout"
        packet.append({"template_id": tid, "original": original["question"],
                       "rewrites": [{"variant": v, "question": q} for v, q in
                                    zip(("paraphrase", "query_first"), spec["paraphrases"])]})
        for variant, question in zip(("canonical", "paraphrase", "query_first"),
                                     [original["question"], *spec["paraphrases"]]):
            items.append({"item_id": f"strict/{tid}/{variant}", "arm": "strict",
                          "template_id": tid, "original_id": row["original_id"], "partition": partition,
                          "variant": variant, "question": question, "gold": gold,
                          "audit_expression": spec["expression"]})
        for instance, expression in zip(spec.get("symbolic_instances", [0, 1, 2]), spec["symbolic_expressions"]):
            symbolic = upstream[tid, instance]
            if symbolic["original_id"] != row["original_id"]: raise ValueError("Symbolic identity changed")
            items.append({"item_id": f"symbolic/{tid}/{instance}", "arm": "symbolic",
                          "template_id": tid, "original_id": row["original_id"], "partition": partition,
                          "variant": str(instance), "question": symbolic["question"],
                          "gold": gold_of(symbolic["answer"]), "audit_expression": expression})
    bundle = {"schema": 2, "template_ids": development+holdout,
              "development_template_ids": development, "holdout_template_ids": holdout,
              "sources": {"gsm8k": sources()["gsm8k"], "symbolic": {"repo_id": SYMBOLIC_REPO,
                           "revision": SYMBOLIC_REVISION, "config": "main"}},
              "fixture_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (old_path, FIXTURE)},
              "exclusions": {**old["excluded_templates"], **new["excluded_templates"]},
              "excluded_instances": new["excluded_instances"], "items": items}
    bundle["sha256"] = digest(bundle)
    validate_bundle(bundle)
    if set(development) & set(holdout): raise ValueError("Development/holdout overlap")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "prereg/h2_followup.json"
    if frozen.exists() and DEST.exists() and json.loads(DEST.read_text()) != bundle:
        raise ValueError("Follow-up is already frozen; use a new protocol version")
    DEST.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    (DEST.parent / "followup_review_packet.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False)+"\n")
    print(f"Validated {len(items)} questions, {len(development)} development + {len(holdout)} holdout problems")
    print(bundle["sha256"])
    return bundle


if __name__ == "__main__":
    build()
