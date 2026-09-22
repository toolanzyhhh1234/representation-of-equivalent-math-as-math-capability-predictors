"""Build the frozen H2 pilot inputs; keep arithmetic checks out of model prompts."""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import ROOT
from .eval_gsm8k import gold_of
from .hf_data import load_benchmark, sources

SYMBOLIC_REPO = "apple/GSM-Symbolic"
SYMBOLIC_REVISION = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
FIXTURE = ROOT / "data" / "h2_rewrites.json"
DEST = ROOT / "data" / "h2" / "pilot.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def arithmetic(expression):
    """Exact rational arithmetic, with no Python eval or function calls."""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError("Only numeric constants and +, -, *, / are allowed")
    return visit(ast.parse(expression, mode="eval"))


def validate_bundle(bundle):
    if bundle["sha256"] != digest({k: v for k, v in bundle.items() if k != "sha256"}):
        raise ValueError("Stimulus bundle hash mismatch")
    seen = set()
    groups = {}
    for item in bundle["items"]:
        if item["item_id"] in seen:
            raise ValueError("Duplicate item ID")
        seen.add(item["item_id"])
        if arithmetic(item["audit_expression"]) != Fraction(item["gold"]):
            raise ValueError(f"Incorrect arithmetic label: {item['item_id']}")
        if not item["question"].strip():
            raise ValueError("Empty question")
        groups.setdefault((item["arm"], item["template_id"]), []).append(item)
    expected_templates = bundle["template_ids"]
    for arm in ("strict", "symbolic"):
        for tid in expected_templates:
            rows = groups.get((arm, tid), [])
            if len(rows) != 3 or len({r["question"] for r in rows}) != 3:
                raise ValueError(f"Expected three distinct variants: {arm}/{tid}")
            if arm == "strict" and len({r["gold"] for r in rows}) != 1:
                raise ValueError("Strict rewrites must share a gold answer")
    if len(bundle["items"]) != 6 * len(expected_templates):
        raise ValueError("Unexpected stimulus groups")


def build():
    fixture = json.loads(FIXTURE.read_text())
    path = hf_hub_download(SYMBOLIC_REPO, "main/test.jsonl", repo_type="dataset",
                           revision=SYMBOLIC_REVISION, local_dir=ROOT / "data/hf/gsm_symbolic")
    upstream = {(r["id"], r["instance"]): r for r in
                (json.loads(line) for line in Path(path).read_text().splitlines())}
    gsm = load_benchmark("gsm8k")["test"]
    items = []
    for spec in fixture["items"]:
        tid = spec["template_id"]
        source = upstream[tid, 0]
        original = gsm[source["original_id"]]
        if original["question"].strip() != source["original_question"].strip():
            raise ValueError(f"Original question mismatch at template {tid}")
        gold = gold_of(original["answer"])
        if gold != gold_of(source["original_answer"]):
            raise ValueError(f"Original answer mismatch at template {tid}")
        strict = [original["question"], *spec["paraphrases"]]
        for variant, question in zip(("canonical", "paraphrase", "query_first"), strict):
            items.append({"item_id": f"strict/{tid}/{variant}", "arm": "strict",
                          "template_id": tid, "original_id": source["original_id"],
                          "variant": variant, "question": question, "gold": gold,
                          "audit_expression": spec["expression"]})
        for instance, expr in enumerate(spec["symbolic_expressions"]):
            row = upstream[tid, instance]
            if row["original_id"] != source["original_id"]:
                raise ValueError("Template identity changed across variants")
            items.append({"item_id": f"symbolic/{tid}/{instance}", "arm": "symbolic",
                          "template_id": tid, "original_id": source["original_id"],
                          "variant": str(instance), "question": row["question"],
                          "gold": gold_of(row["answer"]), "audit_expression": expr})
    bundle = {
        "schema": 1, "sources": {"gsm8k": sources()["gsm8k"],
                                   "symbolic": {"repo_id": SYMBOLIC_REPO,
                                                "revision": SYMBOLIC_REVISION, "config": "main"}},
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "template_ids": [s["template_id"] for s in fixture["items"]],
        "exclusions": fixture["excluded_templates"], "items": items,
        "review": "AI-reviewed wording and independently checked arithmetic; no formal NL equivalence certificate",
    }
    bundle["sha256"] = digest(bundle)
    validate_bundle(bundle)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and json.loads(DEST.read_text()) != bundle:
        raise ValueError(f"Refusing to replace a different frozen stimulus bundle: {DEST}")
    DEST.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    print(f"Validated {len(items)} items / {len(bundle['template_ids'])} templates")
    print(f"Stimulus SHA256: {bundle['sha256']}")
    print(DEST)
    return bundle


if __name__ == "__main__":
    build()
