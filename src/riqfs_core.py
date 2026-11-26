"""
riqfs_core.py -- core mathematics of refined intuitionistic Q-fuzzy sets (RIQFS).

A refined value (cell) is a pair (T, F): T in [0,1]^eps (refined memberships),
F in [0,1]^delta (refined non-memberships), with sum(T) + sum(F) <= 1.
An RIQFS on U x P is stored as arrays T[i, j, l], F[i, j, l].

Section references are to the manuscript.
"""
import numpy as np

TOL = 1e-12


# ============================================================== admissibility
def is_valid_cell(T, F, tol=1e-9):
    T = np.asarray(T, float); F = np.asarray(F, float)
    return bool(T.min() >= -tol and F.min() >= -tol and T.max() <= 1 + tol
                and F.max() <= 1 + tol and T.sum() + F.sum() <= 1 + tol)


def is_separable(T, F, kappa, tol=1e-9):
    """kappa-separable value (Def. 3.5): T^l + F^l <= kappa_l for every source l."""
    return bool(np.all(np.asarray(T) + np.asarray(F) <= np.asarray(kappa) + tol))


# ============================================================== design S: linear-order selection
def key(x):
    """Sum-then-lexicographic admissible order (Def. 3.3)."""
    x = np.round(np.asarray(x, float), 12)
    return (round(float(x.sum()), 12),) + tuple(x.tolist())


def vmax(x, y):
    return np.asarray(x, float) if key(x) >= key(y) else np.asarray(y, float)


def vmin(x, y):
    return np.asarray(x, float) if key(x) <= key(y) else np.asarray(y, float)


def union(a, b):
    return (vmax(a[0], b[0]), vmin(a[1], b[1]))


def inter(a, b):
    return (vmin(a[0], b[0]), vmax(a[1], b[1]))


def comp(a):
    return (np.asarray(a[1], float), np.asarray(a[0], float))


def weak_subset(a, b):
    return key(a[0]) <= key(b[0]) and key(b[1]) <= key(a[1])


def strong_subset(a, b):
    return bool(np.all(np.asarray(a[0]) <= np.asarray(b[0]) + TOL)
                and np.all(np.asarray(a[1]) >= np.asarray(b[1]) - TOL))


# ============================================================== design C: componentwise + renormalisation
def runion(a, b):
    """Componentwise max/min followed by renormalisation of T onto the budget (Def. 3.9)."""
    T = np.maximum(a[0], b[0]); F = np.minimum(a[1], b[1])
    t, f = T.sum(), F.sum()
    if t + f > 1 and t > 0:
        T = T * (1 - f) / t
    return (T, F)


def rinter(a, b):
    T = np.minimum(a[0], b[0]); F = np.maximum(a[1], b[1])
    t, f = T.sum(), F.sum()
    if t + f > 1 and f > 0:
        F = F * (1 - t) / f
    return (T, F)


# ============================================================== domain (soft-style) operations
def ext_inter(A, B):
    out = {}
    for k in list(A) + [k for k in B if k not in A]:
        out[k] = inter(A[k], B[k]) if (k in A and k in B) else (A[k] if k in A else B[k])
    return out


def ext_union(A, B):
    out = {}
    for k in list(A) + [k for k in B if k not in A]:
        out[k] = union(A[k], B[k]) if (k in A and k in B) else (A[k] if k in A else B[k])
    return out


def res_union(A, B):
    return {k: union(A[k], B[k]) for k in A if k in B}


def res_inter(A, B):
    return {k: inter(A[k], B[k]) for k in A if k in B}


def res_diff(A, B):
    return {k: inter(A[k], comp(B[k])) for k in A if k in B}


# ============================================================== Hamacher machinery
def ham_tnorm(a, b, g):
    den = g + (1 - g) * (a + b - a * b)
    return 0.0 if den == 0 else a * b / den


def ham_tconorm(a, b, g):
    return (a + b - a * b - (1 - g) * a * b) / (1 - (1 - g) * a * b)


def gen_g(x, g):
    """Additive generator of the Hamacher t-norm: g(x) = ln((g + (1-g)x)/x)."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        return np.log((g + (1 - g) * x) / x)


def gen_g_inv(y, g):
    y = np.asarray(y, float)
    with np.errstate(over="ignore"):
        return g / (np.exp(y) + g - 1)


def gen_h(x, g):
    """Generator of the dual t-conorm: h(x) = g(1-x)."""
    return gen_g(1 - np.asarray(x, float), g)


def gen_h_inv(y, g):
    return 1 - gen_g_inv(y, g)


def M_minus(x, w, g):
    """Hamacher weighted mean for non-membership: g^{-1}(sum w g(x))."""
    x = np.asarray(x, float); w = np.asarray(w, float)
    if np.any((x <= 0) & (w > 0)):
        return 0.0
    num = g * np.prod(x ** w)
    den = np.prod((1 + (g - 1) * (1 - x)) ** w) + (g - 1) * np.prod(x ** w)
    return float(num / den)


def M_plus(x, w, g):
    return 1.0 - M_minus(1.0 - np.asarray(x, float), w, g)


# ---- operational laws on intuitionistic pairs (mu, nu)  (Def. 5.2)
def if_oplus(a, b, g):
    return (ham_tconorm(a[0], b[0], g), ham_tnorm(a[1], b[1], g))


def if_smul(lam, a, g):
    mu, nu = a
    A = (1 + (g - 1) * mu) ** lam; B = (1 - mu) ** lam
    m = (A - B) / (A + (g - 1) * B)
    if nu <= 0:
        v = 0.0
    else:
        C = (1 + (g - 1) * (1 - nu)) ** lam; D = nu ** lam
        v = g * D / (C + (g - 1) * D)
    return (m, v)


# ---- separable RIQFS operations (source-wise, normalised by kappa)
def sep_oplus(a, b, g, kappa):
    kappa = np.asarray(kappa, float)
    T = np.zeros(len(kappa)); F = np.zeros(len(kappa))
    for l, k in enumerate(kappa):
        m, v = if_oplus((a[0][l] / k, a[1][l] / k), (b[0][l] / k, b[1][l] / k), g)
        T[l], F[l] = k * m, k * v
    return (T, F)


def sep_smul(lam, a, g, kappa):
    kappa = np.asarray(kappa, float)
    T = np.zeros(len(kappa)); F = np.zeros(len(kappa))
    for l, k in enumerate(kappa):
        m, v = if_smul(lam, (a[0][l] / k, a[1][l] / k), g)
        T[l], F[l] = k * m, k * v
    return (T, F)


def hwa_sep(cells, w, g, kappa):
    """Source-separable refined Hamacher weighted average (default operator, Thm S3.5)."""
    w = np.asarray(w, float); kappa = np.asarray(kappa, float)
    Ts = np.array([c[0] for c in cells], float); Fs = np.array([c[1] for c in cells], float)
    T = np.array([kappa[l] * M_plus(Ts[:, l] / kappa[l], w, g) for l in range(len(kappa))])
    F = np.array([kappa[l] * M_minus(Fs[:, l] / kappa[l], w, g) for l in range(len(kappa))])
    return (T, F)


def hwa_sep_by_laws(cells, w, g, kappa):
    """Same operator computed as  (+)_k w_k (.) alpha_k  (used to verify Thm S3.5)."""
    acc = sep_smul(w[0], cells[0], g, kappa)
    for c, wk in zip(cells[1:], w[1:]):
        acc = sep_oplus(acc, sep_smul(wk, c, g, kappa), g, kappa)
    return acc


def hwa_general(cells, w, g):
    """Aggregate-then-allocate operator for non-separable data (Remark S3.7)."""
    w = np.asarray(w, float)
    Ts = np.array([c[0] for c in cells], float); Fs = np.array([c[1] for c in cells], float)
    ts = M_plus(Ts.sum(1), w, g); fs = M_minus(Fs.sum(1), w, g)
    pt = (w[:, None] * Ts).sum(0); pf = (w[:, None] * Fs).sum(0)
    T = ts * pt / pt.sum() if pt.sum() > TOL else np.zeros(Ts.shape[1])
    F = fs * pf / pf.sum() if pf.sum() > TOL else np.zeros(Fs.shape[1])
    return (T, F)


def convex_pool(cells, w):
    w = np.asarray(w, float)
    return (sum(wk * c[0] for wk, c in zip(w, cells)), sum(wk * c[1] for wk, c in zip(w, cells)))


def aggregate(Ms, w, g, kappa, method="sep"):
    """Cellwise aggregation of expert RIQFS matrices Ms = [(T,F), ...]."""
    T0, F0 = Ms[0]
    T = np.zeros_like(T0, dtype=float); F = np.zeros_like(F0, dtype=float)
    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            cells = [(M[0][i, j], M[1][i, j]) for M in Ms]
            if method == "sep":
                T[i, j], F[i, j] = hwa_sep(cells, w, g, kappa)
            elif method == "general":
                T[i, j], F[i, j] = hwa_general(cells, w, g)
            elif method == "convex":
                T[i, j], F[i, j] = convex_pool(cells, w)
            else:
                raise ValueError(method)
    return T, F


# ============================================================== distances
def dist(A, B, W=None, p=1):
    """Weighted Minkowski-type distance (Def. 4.1). A,B=(T,F) arrays (m,n,*); W (m,n) sums to 1."""
    TA, FA = A; TB, FB = B
    m, n = TA.shape[:2]
    if W is None:
        W = np.full((m, n), 1.0 / (m * n))
    cell = (np.abs(TA - TB) ** p).sum(-1) + (np.abs(FA - FB) ** p).sum(-1)
    return float((0.5 * (W * cell).sum()) ** (1.0 / p))


def cell_dist(a, b, p=1):
    Ta, Fa = map(np.asarray, a); Tb, Fb = map(np.asarray, b)
    return float((0.5 * ((np.abs(Ta - Tb) ** p).sum() + (np.abs(Fa - Fb) ** p).sum())) ** (1 / p))


def reduce_phi(A):
    """Aggregation map Phi: refined -> intuitionistic (t, f)."""
    return (A[0].sum(-1, keepdims=True), A[1].sum(-1, keepdims=True))


# ============================================================== evidence theory (supplementary Section S1.5)
def source_bpas(T, F, kappa):
    """Per-source BPAs on Theta={Y,N}: m_l(Y)=T^l/k_l, m_l(N)=F^l/k_l, m_l(Theta)=rest."""
    kappa = np.asarray(kappa, float)
    mY = np.asarray(T) / kappa; mN = np.asarray(F) / kappa
    return np.stack([mY, mN, 1 - mY - mN], -1)


def discount(m, r):
    """Shafer discounting with reliability r: masses on Y,N scaled by r, rest to Theta."""
    return np.array([r * m[0], r * m[1], 1 - r * (m[0] + m[1])])


def mixture(ms, wts):
    return sum(w * m for w, m in zip(wts, ms))


def dempster(m1, m2):
    """Dempster's rule on Theta={Y,N}; masses ordered (Y, N, Theta)."""
    K = m1[0] * m2[1] + m1[1] * m2[0]
    if 1 - K <= TOL:
        return np.array([0.0, 0.0, 1.0])
    Y = m1[0] * m2[0] + m1[0] * m2[2] + m1[2] * m2[0]
    N = m1[1] * m2[1] + m1[1] * m2[2] + m1[2] * m2[1]
    Th = m1[2] * m2[2]
    return np.array([Y, N, Th]) / (1 - K)
