"""
pipeline.py -- the RIQFS distance-based decision procedure (Algorithm 1 of the supplementary material).

Input: a list of kappa-separable RIQFS matrices (T, F), one per panel, each of shape
(alternatives, criteria, sources). The encoding of published statistics into such matrices is in
analysis/case_study.py (Section 3 of the manuscript).
"""
import numpy as np
from riqfs_core import aggregate, dist, cell_dist

def normalise(Ms, ctypes):
    """Step 2: complement on cost criteria."""
    out = []
    for T, F in Ms:
        T2, F2 = T.copy(), F.copy()
        for j, ty in enumerate(ctypes):
            if ty == "C":
                T2[:, j], F2[:, j] = F[:, j].copy(), T[:, j].copy()
        out.append((T2, F2))
    return out


def expert_weights(Ms):
    """Consensus-based panel weights (Step 3 of Algorithm 1)."""
    r = len(Ms)
    D = np.array([[dist(Ms[a], Ms[b]) for b in range(r)] for a in range(r)])
    s = 1 - D.sum(1) / (r - 1)
    return s / s.sum(), D, s


def maxdev_weights(A):
    T, F = A; m, n = T.shape[:2]
    dev = np.array([sum(cell_dist((T[i, j], F[i, j]), (T[l, j], F[l, j])) for i in range(m) for l in range(m))
                    for j in range(n)])
    return dev / dev.sum()


def references(A, kappa, kind="absolute"):
    T, F = A; n = T.shape[1]; k = np.asarray(kappa, float)
    if kind == "absolute":
        return (np.tile(k, (n, 1)), np.zeros((n, len(k)))), (np.zeros((n, len(k))), np.tile(k, (n, 1)))
    return (T.max(0), F.min(0)), (T.min(0), F.max(0))


def closeness(A, beta, refs, p):
    T, F = A; m = T.shape[0]
    (Tp, Fp), (Tn, Fn) = refs
    Dp = np.array([dist((T[i:i+1], F[i:i+1]), (Tp[None], Fp[None]), beta[None], p) for i in range(m)])
    Dn = np.array([dist((T[i:i+1], F[i:i+1]), (Tn[None], Fn[None]), beta[None], p) for i in range(m)])
    den = Dp + Dn
    C = np.where(den > 0, Dn / np.where(den > 0, den, 1), 0.5)
    return Dp, Dn, C


def run(Ms_raw, ctypes, kappa=(0.5, 0.5), gamma=1.0, p=2, w="consensus", beta=None,
        ref="absolute", method="sep"):
    """Full procedure. Ms_raw: encoded (not yet direction-normalised) panel matrices."""
    Ms = normalise(Ms_raw, ctypes)
    if isinstance(w, str):
        if w == "consensus":
            w, _, _ = expert_weights(Ms)
        elif w == "equal":
            w = np.ones(len(Ms)) / len(Ms)
    w = np.asarray(w, float)
    A = aggregate(Ms, w, gamma, kappa, method)
    if beta is None or (isinstance(beta, str) and beta == "maxdev"):
        beta = maxdev_weights(A)
    elif isinstance(beta, str) and beta == "equal":
        beta = np.ones(A[0].shape[1]) / A[0].shape[1]
    beta = np.asarray(beta, float)
    refs = references(A, kappa, ref)
    Dp, Dn, C = closeness(A, beta, refs, p)
    return dict(A=A, w=w, beta=beta, refs=refs, Dp=Dp, Dn=Dn, C=C, Ms=Ms)


def ranks(C):
    o = np.argsort(-np.asarray(C), kind="stable"); r = np.empty(len(C), int); r[o] = np.arange(1, len(C) + 1)
    return r


def order(C, labels):
    return [labels[i] for i in np.argsort(-np.asarray(C), kind="stable")]
