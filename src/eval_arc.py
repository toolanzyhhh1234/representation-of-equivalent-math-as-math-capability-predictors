"""In-house ARC-Easy: the non-math capability axis for the 2x2 specificity test.

Scored by per-character-normalized log-likelihood of each answer option after
"Question: {q}\nAnswer:" (the lm-eval acc_norm convention), zero-shot, no generation.
One pinned harness for every model, mirroring src/eval_gsm8k.py's role on the math side.

Usage: python -m src.eval_arc [model ...]  -> results/arc_easy.json
"""
import json
import sys
import time

import numpy as np
import torch
from datasets import load_dataset

from .config import PANEL, RESULTS

BATCH = 64


def build(ex):
    prompt = f"Question: {ex['question']}\nAnswer:"
    golds = ex["choices"]["label"].index(ex["answerKey"])
    return prompt, [" " + t for t in ex["choices"]["text"]], golds


@torch.no_grad()
def option_logprobs(tok, model, items):
    """items: [(prompt, option)] -> per-char-normalized logprob of option tokens."""
    out = np.empty(len(items))
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        full = [items[i][0] + items[i][1] for i in idx]
        enc = tok(full, return_tensors="pt", padding=True).to("cuda")
        n_prompt = [len(tok(items[i][0])["input_ids"]) for i in idx]
        logits = model(**enc).logits.float().log_softmax(-1)
        for row, i in enumerate(idx):
            ids = enc["input_ids"][row]
            n_tok = int(enc["attention_mask"][row].sum())
            lp = 0.0
            for pos in range(n_prompt[row], n_tok):
                lp += float(logits[row, pos - 1, ids[pos]])
            out[i] = lp / max(1, len(items[i][1]))
    return out


@torch.no_grad()
def eval_model(model_id, examples):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # scoring reads absolute positions, not the tail
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    items, spans, golds = [], [], []
    for ex in examples:
        prompt, opts, gold = build(ex)
        spans.append((len(items), len(items) + len(opts)))
        golds.append(gold)
        items.extend((prompt, o) for o in opts)
    lp = option_logprobs(tok, model, items)
    correct = sum(int(np.argmax(lp[a:b]) == g) for (a, b), g in zip(spans, golds))
    del model
    torch.cuda.empty_cache()
    return {"acc_norm": correct / len(golds), "n": len(golds),
            "harness": "zero-shot per-char-normalized logprob"}


def main():
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")["test"]
    examples = [ex for ex in ds if ex["answerKey"] in ex["choices"]["label"]]
    print(f"ARC-Easy test n={len(examples)}", flush=True)
    path = RESULTS / "arc_easy.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    models = sys.argv[1:] or [m for m, _ in PANEL]
    for mid in models:
        if mid in out and not sys.argv[1:]:
            print(f"=== {mid} === cached acc_norm {out[mid]['acc_norm']:.3f}", flush=True)
            continue
        print(f"=== {mid} ===", flush=True)
        t0 = time.time()
        out[mid] = eval_model(mid, examples)
        path.write_text(json.dumps(out, indent=1))
        print(f"  acc_norm {out[mid]['acc_norm']:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
