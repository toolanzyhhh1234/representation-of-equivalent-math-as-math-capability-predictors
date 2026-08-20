"""fp32-GPU fast path must reproduce the fp64-CPU reference. Verified, not assumed.

Run: python -m tests.test_parity
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.correct import gap_k, spectrum, topk, zca
from src.data import load_meld
from src.metrics import correct_multi, eq_scores, build_gold_map

KS = [0, 1, 3]
TOL_EQ = 2e-3
TOL_COS = 1e-4


def eq(Xn, stim):
    Xn = Xn.cpu().numpy() if torch.is_tensor(Xn) else Xn
    return eq_scores(Xn, stim)["auroc_anchor"]


def main():
    stim = load_meld()
    z = np.load("results/raw/HuggingFaceTB__SmolLM2-360M.npz")
    fails = []
    for pool in ("last", "mean"):
        for layer in (1, 10, 20, 30):
            X = z[pool][:, layer, :]
            ref = correct_multi(X, KS)                       # fp64 CPU reference
            gpu_exact = topk(X, KS, exact=True)              # fp32 GPU, full SVD
            gpu_rand = topk(X, KS, exact=False)              # fp32 GPU, randomized

            for k in KS:
                r = np.asarray(ref[k])
                # subspace removal is sign-invariant, so compare row directions
                for name, got in (("exact", gpu_exact[k]), ("lowrank", gpu_rand[k])):
                    g = got.cpu().numpy()
                    cos = np.abs((r * g).sum(1)).min()
                    d_eq = abs(eq(r, stim) - eq(g, stim))
                    ok = (1 - cos) < TOL_COS and d_eq < TOL_EQ
                    if not ok:
                        fails.append((pool, layer, k, name, cos, d_eq))
                    print(f"{pool:<5} L{layer:<3} k={k} {name:<8} "
                          f"min|cos|={cos:.6f}  dEQ={d_eq:.2e}  {'ok' if ok else 'FAIL'}")
    print()
    X = z["last"][:, 20, :]
    for label, fn in (("fp64 CPU full-svd", lambda: correct_multi(X, KS)),
                      ("fp32 GPU full-svd", lambda: topk(X, KS, exact=True)),
                      ("fp32 GPU lowrank ", lambda: topk(X, KS, exact=False)),
                      ("fp32 GPU zca     ", lambda: zca(X))):
        fn(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        print(f"{label}: {(time.perf_counter()-t)/5*1000:7.1f} ms")

    print()
    for pool in ("last", "mean"):
        for layer in (1, 10, 20, 30):
            k = gap_k(z[pool][:, layer, :])
            sp = spectrum(z[pool][:, layer, :])
            print(f"gap_k({pool},L{layer}) = {k}   s1/s2={sp['ratios'][0]:.3f}  "
                  f"var@1={sp['cumvar'][0]:.3f}")

    print("\n" + ("PARITY PASS" if not fails else f"PARITY FAIL: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
