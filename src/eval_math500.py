"""In-house MATH-500: the construct-matched second math DV.

GSM8K is grade-school word-problem arithmetic; MELD is subfield-dialect theorem text.
Phase 2's skill-without-representation models (phi-1.5, OLMo-2) might be an artifact
of that mismatch -- MATH-500 (competition mathematics, LaTeX, boxed answers) sits far
closer to MELD's register and tests it.

Harness, pinned and identical for every model: 4-shot with exemplars = the first 4
items of MATH-500 (problem + reference solution, which ends in \\boxed{...}),
evaluated on the remaining 496; greedy, max 512 new tokens; prediction = the last
\\boxed{...} in the generation, string-matched after LaTeX normalization with a
numeric fallback. The normalizer is simple by design; its misses are shared by every
model, so cross-model comparability -- the only thing we use -- survives.

Usage: python -m src.eval_math500 [model ...]  -> results/math500.json
"""
import gzip
import json
import re
import sys
import time

import torch
from datasets import load_dataset

from .config import PANEL, RESULTS

MAX_NEW = 512
BATCH = 32
N_SHOT = 4


def boxed(s):
    """Content of the last \\boxed{...} (brace-balanced)."""
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1] if depth == 0 else None


def norm(a):
    if a is None:
        return None
    a = a.strip().strip("$").strip()
    a = re.sub(r"\\(left|right|!|,|;)", "", a)
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\^\{?\\circ\}?", "", a)
    a = a.replace(" ", "")
    a = re.sub(r"(?<=\d),(?=\d{3}\b)", "", a)      # 1,234 -> 1234
    return a


def num(a):
    try:
        return float(a)
    except (TypeError, ValueError):
        return None


def match(pred, gold):
    p, g = norm(pred), norm(gold)
    if p is None:
        return False
    if p == g:
        return True
    np_, ng = num(p), num(g)
    return np_ is not None and ng is not None and abs(np_ - ng) < 1e-6


@torch.no_grad()
def eval_model(model_id, prompts, golds, problems):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    correct, done, t0 = 0, 0, time.time()
    records = [None] * len(prompts)
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, g in zip(idx, outs):
            gen = g.split("Problem:")[0]
            pred = boxed(gen)
            ok = match(pred, golds[i])
            correct += ok
            records[i] = {"i": i, "problem": problems[i], "generation": g,
                          "pred": pred, "gold": golds[i], "correct": bool(ok)}
        done += len(idx)
        print(f"\r  {done}/{len(prompts)}  acc so far {correct/done:.3f} "
              f"({time.time()-t0:.0f}s)", end="", flush=True)
    print(flush=True)
    del model
    torch.cuda.empty_cache()
    return {"acc": correct / len(prompts), "n": len(prompts),
            "harness": f"{N_SHOT}-shot(first-{N_SHOT}-items) greedy max{MAX_NEW} boxed"}, records


def main():
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    shots = "\n\n".join(f"Problem: {ds[i]['problem']}\nSolution: {ds[i]['solution']}"
                        for i in range(N_SHOT))
    prompts = [f"{shots}\n\nProblem: {ex['problem']}\nSolution:"
               for ex in list(ds)[N_SHOT:]]
    golds = [ex["answer"] for ex in list(ds)[N_SHOT:]]
    problems = [ex["problem"] for ex in list(ds)[N_SHOT:]]
    cap_dir = RESULTS / "raw_outputs" / "math500"
    cap_dir.mkdir(parents=True, exist_ok=True)
    print(f"MATH-500 eval n={len(prompts)}", flush=True)
    path = RESULTS / "math500.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    models = sys.argv[1:] or [m for m, _ in PANEL]
    for mid in models:
        cap = cap_dir / f"{mid.replace('/', '__')}.jsonl.gz"
        if mid in out and cap.exists() and not sys.argv[1:]:
            print(f"=== {mid} === cached acc {out[mid]['acc']:.3f}", flush=True)
            continue
        print(f"=== {mid} ===", flush=True)
        out[mid], records = eval_model(mid, prompts, golds, problems)
        with gzip.open(cap, "wt") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        path.write_text(json.dumps(out, indent=1))
        print(f"  acc {out[mid]['acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
