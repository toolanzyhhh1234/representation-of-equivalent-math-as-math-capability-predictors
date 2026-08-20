"""Anisotropy corrections, GPU backend.

Phase 0 found fixed-k PC removal is the wrong instrument (results/PHASE0.md): under
last-token pooling the top of the spectrum is near-flat (s1/s2 ~ 1.06-1.15, PC1 explains
6-12% of variance, s3/s4 ~ 1.03), so "remove k PCs" removes near-degenerate directions
whose identity is unstable across models -- and the between-model ordering flips with k.
Under mean pooling PC1 is genuinely dominant (s1/s2 ~ 1.9-2.2, 24-36%) and removal helps.

So the correction must adapt to the spectrum instead of assuming one. Three are provided;
which is primary must be pre-registered on conditioning grounds, never on resulting EQ.

Precision note: this GPU path is fp32 by design. GA104 (RTX 3070) runs fp64 at 1/64 rate,
so a fp64 GPU SVD measured 3x SLOWER than fp64 on CPU. Parity vs the fp64 CPU reference is
verified numerically in tests/test_parity.py rather than assumed.
"""
import numpy as np
import torch


def _center(X, device="cuda"):
    if not torch.is_tensor(X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    return X.float() - X.float().mean(0, keepdim=True)


def _norm(Y):
    return Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-12)


def spectrum(X, device="cuda"):
    """Singular values of the centred matrix, plus the conditioning diagnostics."""
    Xc = _center(X, device)
    s = torch.linalg.svdvals(Xc)
    v = torch.cumsum(s**2, 0) / (s**2).sum()
    return {"s": s, "ratios": (s[:-1] / s[1:]), "cumvar": v}


def topk(X, ks, device="cuda", exact=False, seed=0):
    """Remove the top-k principal directions. {k: normalised tensor}.

    exact=False uses randomized SVD (~100x faster, seeded for reproducibility); we only
    ever need a handful of leading directions, so a full decomposition is wasted work.
    """
    Xc = _center(X, device)
    kmax = max(ks)
    V = None
    if kmax > 0:
        if exact:
            V = torch.linalg.svd(Xc, full_matrices=False)[2][:kmax]
        else:
            torch.manual_seed(seed)
            V = torch.pca_lowrank(Xc, q=min(kmax + 6, min(Xc.shape)), niter=4)[2].T[:kmax]
    return {k: _norm(Xc if k == 0 else Xc - (Xc @ V[:k].T) @ V[:k]) for k in ks}


def zca(X, eps_frac=1e-3, device="cuda"):
    """Full whitening: every direction scaled to unit variance.

    Parameter-free w.r.t. how many directions to remove, so it cannot be tuned toward a
    preferred ordering -- the property fixed-k lacks. eps is set relative to the largest
    eigenvalue so it does not depend on the model's activation scale.
    """
    Xc = _center(X, device)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    eps = eps_frac * S.max()
    return _norm((Xc @ Vt.T) / (S / np.sqrt(Xc.shape[0] - 1) + eps) @ Vt)


def gap_k(X, thresh=1.5, kmax=10, device="cuda"):
    """Remove only directions that are actually well-separated from the next one.

    Rationale: a direction with s_i/s_{i+1} ~ 1 is not identified -- which of the two the
    solver returns is arbitrary, so removing it injects noise rather than removing
    anisotropy. Returns the chosen k (0 if no direction is well-determined).
    """
    s = spectrum(X, device)["s"]
    r = (s[:-1] / s[1:]).cpu().numpy()
    k = 0
    while k < min(kmax, len(r)) and r[k] >= thresh:
        k += 1
    return k
