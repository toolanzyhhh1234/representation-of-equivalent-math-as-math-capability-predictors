"""Lexical similarity and lexically-controlled EQ variants.

Phase 0's decisive finding (results/PHASE0.md sec 1): a character 3-5gram TF-IDF scores
EQ = 0.7715 on the anchor-AUROC task, so EQ as computed contains a large topical-lexical
term. METRICS.md sec 7 names the two fixes -- topic-matched negatives, or partialling the
lexical cosine out of the model cosine. This module implements the second, plus two
restriction-based variants that make the lexical channel uninformative by construction:

  EQ_resid   anchor AUROC on residuals of model-cos after regressing out TF-IDF-cos.
             The regression is fit per (model, layer, correction) on the pooled candidate
             scores, because cosine scales differ across all three.
  EQ_hard    anchor AUROC restricted to negatives at least as lexically close to the
             anchor as the true partner. On this subset TF-IDF scores <= 0.5 by
             construction, so any excess is non-lexical signal. Anchors with no such
             negative drop out -- coverage is reported and must be quoted with the score.
  EQ_lextop  anchor AUROC against each anchor's m lexically-closest negatives only.
             Keeps every anchor (no selection), weakens but does not zero the lexical
             channel -- the TF-IDF baseline must be recomputed on the same restriction.

All three reuse the per-anchor construction of metrics.eq_scores; the lexical similarity
matrix is computed once per stimulus set and shared across models.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def lexical_cos(texts, analyzer="char_wb", ngram_range=(3, 5)):
    """Row-normalised TF-IDF vectors -> dense cosine matrix [N, N].

    analyzer="char_wb", ngram 3-5, sklearn defaults otherwise: verified to reproduce the
    Phase 0 baseline EQ = 0.7715 exactly (and word 1gram/1-2gram give the documented
    0.7099/0.7098). Phase 0 never committed its baseline script; this pins it.
    lowercase=False is slightly stronger still (0.7770) -- report it as sensitivity.
    """
    V = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    X = normalize(V.fit_transform(texts))
    return np.asarray((X @ X.T).todense(), dtype=np.float64)


def anchor_jobs(stim):
    """The anchor structure shared by every EQ variant: one job per scoring direction.

    Returns [(pair_index, anchor, target, distractor_rows)] -- 540 jobs from 270 pairs,
    matching metrics.eq_scores exactly (both directions, distractors in target's framing).
    """
    jobs = []
    for pi, (ia, ib, fa, fb, _dom, _pid) in enumerate(stim.pairs):
        for anchor, target, tgt_framing in ((ia, ib, fb), (ib, ia, fa)):
            d = stim.distractors.get(tgt_framing)
            if d:
                jobs.append((pi, anchor, target, np.asarray(d)))
    return jobs


def _auroc(pos, negs):
    return float((negs < pos).mean() + 0.5 * (negs == pos).mean())


def eq_variants(score, jobs, L, m_top=5):
    """All EQ variants for one (model, layer, correction).

    score: [N, N] similarity matrix (cosine gram of the corrected activations).
    score=None scores the lexical matrix L against itself -- that is how the TF-IDF
    baseline for each variant is produced, guaranteeing identical anchor structure.

    Returns {"per_anchor": {variant: [n_jobs] float array, NaN = anchor dropped},
             "pair_index": [n_jobs], "hard_n": mean negatives per surviving hard anchor}.
    """
    if score is None:
        score = L

    n = len(jobs)
    out = {k: np.full(n, np.nan) for k in ("eq", "eq_resid", "eq_hard", "eq_lextop")}
    pair_index = np.empty(n, dtype=int)
    hard_ns = []

    # global partial fit: model score ~ a + b * lexical score, pooled over all candidates
    ms, ls = [], []
    for _, a, t, d in jobs:
        ms.append(score[a, t]); ls.append(L[a, t])
        ms.extend(score[a, d]); ls.extend(L[a, d])
    ms, ls = np.asarray(ms), np.asarray(ls)
    b, a0 = np.polyfit(ls, ms, 1)
    # Degenerate case (score == L, i.e. the TF-IDF baseline scoring itself): residuals
    # are fp dust and their ordering is noise. The null of eq_resid is 0.5 by
    # construction; report it as exactly that rather than the AUROC of rounding error.
    resid_degenerate = float(np.std(ms - b * ls)) < 1e-12

    for j, (pi, anc, tgt, d) in enumerate(jobs):
        pair_index[j] = pi
        pos, negs = score[anc, tgt], score[anc, d]
        lpos, lnegs = L[anc, tgt], L[anc, d]

        out["eq"][j] = _auroc(pos, negs)
        out["eq_resid"][j] = 0.5 if resid_degenerate else \
            _auroc(pos - b * lpos, negs - b * lnegs)

        hard = negs[lnegs >= lpos]
        if hard.size:
            out["eq_hard"][j] = _auroc(pos, hard)
            hard_ns.append(hard.size)

        top = negs[np.argsort(-lnegs)[:m_top]]
        out["eq_lextop"][j] = _auroc(pos, top)

    return {"per_anchor": out, "pair_index": pair_index,
            "hard_n": float(np.mean(hard_ns)) if hard_ns else 0.0}


def summarize(var):
    """Anchor means + coverage for one eq_variants() result."""
    s = {}
    for k, v in var["per_anchor"].items():
        ok = ~np.isnan(v)
        s[k] = float(v[ok].mean())
        if k == "eq_hard":
            s["eq_hard_coverage"] = float(ok.mean())
            s["eq_hard_mean_negs"] = var["hard_n"]
    return s


def bootstrap_ci(per_anchor, pair_index, n_boot=1000, seed=0):
    """Percentile CI of the anchor-mean, resampling PAIRS (the sampling unit), not
    anchors -- the two directions of one pair are not independent (PLAN sec 4.4)."""
    rng = np.random.default_rng(seed)
    npairs = pair_index.max() + 1
    ok = ~np.isnan(per_anchor)
    sums = np.bincount(pair_index[ok], weights=per_anchor[ok], minlength=npairs)
    cnts = np.bincount(pair_index[ok], minlength=npairs)
    idx = rng.integers(0, npairs, size=(n_boot, npairs))
    tot, num = sums[idx].sum(1), cnts[idx].sum(1)
    means = tot[num > 0] / num[num > 0]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
