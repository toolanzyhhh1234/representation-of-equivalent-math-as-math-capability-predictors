"""EQ scoring: anisotropy correction, then equivalence-vs-hard-negative discrimination.

PLAN sec 4.3: raw cosine between LLM hidden states is dominated by a shared mean direction
and a few high-variance components. Anisotropy correlates with scale, which correlates with
capability -- so an uncorrected EQ can produce a fully spurious result. Correction is fit
per model, per layer, on that model's own stimulus set, and is unsupervised (no labels).
"""
import numpy as np


def correct(X, k):
    """Center, then project out the top-k principal components. X: [N, H] float32."""
    X = np.asarray(X, dtype=np.float64)          # PLAN sec 12.1: fp32+ before PCA
    X = X - X.mean(0, keepdims=True)
    if k > 0:
        # right singular vectors of the centered matrix == principal directions
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:k]                                # [k, H]
        X = X - (X @ V.T) @ V
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


def _anchor_auroc(pos, negs):
    """One positive vs n negatives -> P(pos ranked above a random negative). Ties = 0.5."""
    negs = np.asarray(negs)
    return float((negs < pos).mean() + 0.5 * (negs == pos).mean())


def eq_scores(Xn, stim, pair_subset=None):
    """Xn: row-normalised [N, H]. Returns EQ metrics for one layer/pooling/k."""
    pairs = stim.pairs if pair_subset is None else [stim.pairs[i] for i in pair_subset]
    per_anchor = []
    pos_all, neg_all = [], []
    for ia, ib, fa, fb, _dom, _pid in pairs:
        for anchor, target, tgt_framing in ((ia, ib, fb), (ib, ia, fa)):
            d = stim.distractors.get(tgt_framing)
            if not d:
                continue
            pos = float(Xn[anchor] @ Xn[target])
            negs = Xn[d] @ Xn[anchor]
            per_anchor.append(_anchor_auroc(pos, negs))
            pos_all.append(pos)
            neg_all.extend(negs.tolist())
    return {
        "auroc_anchor": float(np.mean(per_anchor)),
        "n_anchors": len(per_anchor),
        "mean_cos_pos": float(np.mean(pos_all)),
        "mean_cos_neg": float(np.mean(neg_all)),
        "align_gap": float(np.mean(pos_all) - np.mean(neg_all)),
    }


def retrieval(Xn, stim, gold_map):
    """Full-pool retrieval: rank the true partner among all 1080 statements.

    gold_map handles the multi-gold anchor (MELD pairs 158/174 share entry_1
    '$G$ is connected.'), which would otherwise be scored as a guaranteed miss.
    """
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    r1 = r5 = 0.0
    rr = []
    anchors = sorted(gold_map)
    for a in anchors:
        gold = gold_map[a]
        order = np.argsort(-S[a])
        rank = min(int(np.where(order == g)[0][0]) + 1 for g in gold)
        r1 += rank <= 1
        r5 += rank <= 5
        rr.append(1.0 / rank)
    n = len(anchors)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": float(np.mean(rr)), "n_queries": n}


def build_gold_map(stim):
    gold = {}
    for ia, ib, *_ in stim.pairs:
        gold.setdefault(ia, set()).add(ib)
        gold.setdefault(ib, set()).add(ia)
    return gold


def correct_multi(X, ks):
    """Same as correct(), but one SVD shared across all k. Returns {k: normalised X}."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    out = {}
    kmax = max(ks)
    Vt = None
    if kmax > 0:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    for k in ks:
        Y = Xc if k == 0 else Xc - (Xc @ Vt[:k].T) @ Vt[:k]
        out[k] = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12, None)
    return out
