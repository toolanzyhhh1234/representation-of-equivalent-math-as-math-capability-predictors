"""Hidden-state extraction.

The single most dangerous bug here is last-token indexing under right padding: the final
position is a PAD token, which yields a plausible-looking but meaningless EQ score that
never announces itself. We derive the index from attention_mask and verify padding
invariance numerically (see verify_padding_invariance / tests).
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BATCH_SIZE, DTYPE, MAX_LEN


def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # indices come from attention_mask, not position -1
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, DTYPE), attn_implementation="sdpa",
    ).cuda().eval()
    return tok, model


def _pool(hidden_states, attn):
    """hidden_states: tuple[L+1] of [B,T,H] -> dict of [B, L+1, H] float32."""
    mask = attn.unsqueeze(-1)                                  # [B,T,1]
    last_idx = attn.sum(1) - 1                                 # true last non-pad token
    stack = torch.stack(hidden_states, dim=1).float()          # [B, L+1, T, H]
    b = torch.arange(stack.shape[0], device=stack.device)
    last = stack[b, :, last_idx, :]                            # [B, L+1, H]
    m = mask.unsqueeze(1).float()                              # [B,1,T,1]
    mean = (stack * m).sum(2) / m.sum(2).clamp(min=1)          # [B, L+1, H]
    return {"last": last, "mean": mean}


@torch.no_grad()
def extract(tok, model, texts, batch_size=BATCH_SIZE, progress=None):
    """Returns {pooling: np.ndarray [N, L+1, H] float32} in the order of `texts`."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # bucket by length
    out = {}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        pooled = _pool(hs, enc["attention_mask"])
        for k, v in pooled.items():
            if k not in out:
                out[k] = np.zeros((len(texts), v.shape[1], v.shape[2]), dtype=np.float32)
            out[k][idx] = v.cpu().numpy()
        if progress:
            progress(min(s + batch_size, len(order)), len(order))
    return out


@torch.no_grad()
def verify_padding_invariance(tok, model, texts, n=8, tol=0.999):
    """Batched (padded) vs one-at-a-time (unpadded) must agree. Guards the PAD-token bug.

    Compared by cosine rather than abs diff: bf16 kernels differ slightly by batch shape,
    but a padding bug moves the vector wholesale, not by an ulp.
    """
    probe = sorted(texts, key=len)[::max(1, len(texts) // n)][:n]
    batched = extract(tok, model, probe, batch_size=n)
    single = extract(tok, model, probe, batch_size=1)
    report = {}
    for k in batched:
        a = torch.tensor(batched[k]).flatten(1)
        b = torch.tensor(single[k]).flatten(1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        report[k] = float(cos.min())
        if report[k] < tol:
            raise AssertionError(
                f"padding invariance FAILED for pooling={k}: min cos={report[k]:.6f} < {tol}. "
                "Last-token index is almost certainly hitting a PAD position."
            )
    return report
