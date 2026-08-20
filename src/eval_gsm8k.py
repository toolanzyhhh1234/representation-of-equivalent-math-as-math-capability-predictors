"""In-house GSM8K, one fixed harness for the whole panel.

PLAN sec 5: never import published capability numbers into the IV-DV link -- the same
model differs by 7 points across harnesses (RELATED_WORK sec 8). This harness is
deliberately minimal and identical for every model:

  5-shot, exemplars = the first 5 items of the GSM8K train split (deterministic),
  plain completion format (no chat template -- the panel is base models),
  greedy decoding, max 256 new tokens, answer = the '#### n' line, else last number.

Usage: python -m src.eval_gsm8k [model ...] [--limit N]
Results merge into results/gsm8k.json; a model already present is skipped unless
re-listed explicitly.
"""
import gzip
import json
import re
import sys
import time

import torch
from datasets import load_dataset

from .config import PANEL, RESULTS

MAX_NEW = 256
BATCH = 64
N_SHOT = 5


def build_prompt(shots, q):
    parts = [f"Question: {s['question']}\nAnswer: {s['answer']}" for s in shots]
    parts.append(f"Question: {q}\nAnswer:")
    return "\n\n".join(parts)


def gold_of(ans):
    return ans.split("####")[-1].strip().replace(",", "")


_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def pred_of(gen):
    gen = gen.split("Question:")[0]                      # cut at the next few-shot turn
    m = re.search(r"####\s*([^\n]*)", gen)
    cand = m.group(1) if m else gen
    nums = _NUM.findall(cand)
    if not nums and not m:
        nums = _NUM.findall(gen)
    if not nums:
        return None
    return nums[-1].replace(",", "").replace("$", "").rstrip(".")


def norm(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@torch.no_grad()
def eval_model(model_id, prompts, golds, questions):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                            # generation: pad on the left
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    correct, n_done, t0 = 0, 0, time.time()
    preds = [None] * len(prompts)
    records = [None] * len(prompts)
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, g in zip(idx, outs):
            preds[i] = pred_of(g)
            ok = norm(preds[i]) is not None and norm(preds[i]) == norm(golds[i])
            correct += ok
            records[i] = {"i": i, "question": questions[i], "generation": g,
                          "pred": preds[i], "gold": golds[i], "correct": bool(ok)}
        n_done += len(idx)
        print(f"\r  {n_done}/{len(prompts)}  acc so far {correct/n_done:.3f}  "
              f"({time.time()-t0:.0f}s)", end="", flush=True)
    print(flush=True)
    del model
    torch.cuda.empty_cache()
    return {"acc": correct / len(prompts), "n": len(prompts), "correct": correct,
            "harness": f"{N_SHOT}-shot greedy max{MAX_NEW} first-{N_SHOT}-train-exemplars"}, records


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1]) if "=" in a else None
    ds = load_dataset("openai/gsm8k", "main")
    shots = [ds["train"][i] for i in range(N_SHOT)]
    test = ds["test"] if limit is None else ds["test"].select(range(limit))
    prompts = [build_prompt(shots, ex["question"]) for ex in test]
    golds = [gold_of(ex["answer"]) for ex in test]
    questions = [ex["question"] for ex in test]
    cap_dir = RESULTS / "raw_outputs" / "gsm8k"
    cap_dir.mkdir(parents=True, exist_ok=True)
    print(f"GSM8K test n={len(prompts)}", flush=True)

    path = RESULTS / "gsm8k.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    models = args or [m for m, _ in PANEL]
    for mid in models:
        cap = cap_dir / f"{mid.replace('/', '__')}.jsonl.gz"
        if mid in out and cap.exists() and not args:
            print(f"=== {mid} === cached: acc {out[mid]['acc']:.3f}", flush=True)
            continue
        print(f"=== {mid} ===", flush=True)
        out[mid], records = eval_model(mid, prompts, golds, questions)
        with gzip.open(cap, "wt") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        path.write_text(json.dumps(out, indent=1))
        print(f"  acc {out[mid]['acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
