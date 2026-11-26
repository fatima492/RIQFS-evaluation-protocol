"""
Randomised verification of every theorem, proposition and counterexample of the supplementary material
(numbers with prefix S refer to the supplementary material).
Run:  python -m pytest -q tests/        (or)   python tests/test_theory.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riqfs_core import *  # noqa

RNG = np.random.default_rng(20261123)
N = 4000
GAMMAS = (0.1, 0.5, 1.0, 2.0, 5.0, 20.0)


def rcell(e=2, d=2):
    x = RNG.dirichlet(np.ones(e + d + 1))
    return (x[:e], x[e:e + d])


def gcell():
    while True:
        v = RNG.integers(0, 6, 4) / 10
        if v.sum() <= 1:
            return (v[:2], v[2:])


def rsep(kappa=(0.5, 0.5)):
    kappa = np.asarray(kappa)
    T = np.zeros(2); F = np.zeros(2)
    for l, k in enumerate(kappa):
        x = RNG.dirichlet(np.ones(3)); T[l], F[l] = k * x[0], k * x[1]
    return (T, F)


def eq(a, b, tol=1e-9):
    return np.allclose(a[0], b[0], atol=tol) and np.allclose(a[1], b[1], atol=tol)


# ------------------------------------------------------------------ Section 3: selection design
def test_selection_lattice_demorgan():
    for gen in (rcell, gcell):
        for _ in range(N):
            a, b, c = gen(), gen(), gen()
            assert is_valid_cell(*union(a, b)) and is_valid_cell(*inter(a, b))
            assert eq(union(a, b), union(b, a))
            assert eq(union(union(a, b), c), union(a, union(b, c)))
            assert eq(union(a, inter(a, b)), a) and eq(inter(a, union(a, b)), a)
            assert eq(union(a, inter(b, c)), inter(union(a, b), union(a, c)))
            assert eq(comp(union(a, b)), inter(comp(a), comp(b)))
            assert eq(comp(comp(a)), a)
            assert (not (weak_subset(a, b) and weak_subset(b, a))) or eq(a, b)
            assert (not strong_subset(a, b)) or weak_subset(a, b)


def test_selection_union_discontinuous():
    """Thm S1.14: a jump of size ~0.30 for an arbitrarily small perturbation."""
    a = (np.array([0.30, 0.0]), np.zeros(2))
    for eta in (1e-3, 1e-6, 1e-9):
        b_lo = (np.array([0.0, 0.30 - eta]), np.zeros(2))
        b_hi = (np.array([0.0, 0.30 + eta]), np.zeros(2))
        jump = cell_dist(union(a, b_lo), union(a, b_hi))
        assert jump > 0.29 and cell_dist(b_lo, b_hi) < 2 * eta


# ------------------------------------------------------------------ Section 3: renormalised design
def test_renormalised_properties():
    assoc_fail = absorb_fail = distr_fail = 0
    for _ in range(N):
        a, b, c = rcell(), rcell(), rcell()
        for op in (runion, rinter):
            assert is_valid_cell(*op(a, b))
            assert eq(op(a, b), op(b, a)) and eq(op(a, a), a)
        assert eq(comp(runion(a, b)), rinter(comp(a), comp(b)))
        # continuity: small perturbation -> small change
        e = 1e-7 * RNG.standard_normal(4)
        a2 = (np.clip(a[0] + e[:2], 0, 1), np.clip(a[1] + e[2:], 0, 1))
        if is_valid_cell(*a2):
            assert cell_dist(runion(a, b), runion(a2, b)) < 1e-5
        assoc_fail += not eq(runion(runion(a, b), c), runion(a, runion(b, c)))
        absorb_fail += not eq(runion(a, rinter(a, b)), a)
        distr_fail += not eq(runion(a, rinter(b, c)), rinter(runion(a, b), runion(a, c)))
    assert assoc_fail > 0 and absorb_fail > 0 and distr_fail > 0


def renormalised_counterexamples():
    a = (np.array([0.1, 0.6]), np.array([0.0, 0.2]))
    b = (np.array([0.5, 0.0]), np.array([0.3, 0.1]))
    c = (np.array([0.6, 0.1]), np.array([0.2, 0.0]))
    x = (np.array([0.3, 0.3]), np.array([0.2, 0.0]))
    y = (np.array([0.2, 0.3]), np.array([0.0, 0.4]))
    return dict(assoc=(runion(runion(a, b), c), runion(a, runion(b, c))),
                absorb=(runion(x, rinter(x, y)), x))


def test_renormalised_counterexamples():
    ce = renormalised_counterexamples()
    assert not eq(*ce["assoc"]) and not eq(*ce["absorb"])


# ------------------------------------------------------------------ Section 4: metric
def test_metric_and_equivalence():
    for _ in range(600):
        m, n = 3, 4
        def rmat():
            T = np.zeros((m, n, 2)); F = np.zeros((m, n, 2))
            for i in range(m):
                for j in range(n):
                    T[i, j], F[i, j] = rcell()
            return T, F
        A, B, C = rmat(), rmat(), rmat()
        W = RNG.dirichlet(np.ones(m * n)).reshape(m, n)
        for p in (1, 2, 3):
            assert dist(A, C, W, p) <= dist(A, B, W, p) + dist(B, C, W, p) + 1e-12
            assert 0 <= dist(A, B, W, p) <= 1
        dh, de = dist(A, B, W, 1), dist(A, B, W, 2)
        assert de ** 2 <= dh + 1e-12 and dh <= np.sqrt(2) * de + 1e-12
        assert dist(reduce_phi(A), reduce_phi(B), W, 1) <= dh + 1e-12


def test_order_metric_compatibility():
    """Prop S2.5: weak inclusion is monotone for the reduced distance; strong for the refined one."""
    for _ in range(N):
        a, b, c = gcell(), gcell(), gcell()
        if weak_subset(a, b) and weak_subset(b, c):
            pa, pb, pc = [([x[0].sum()], [x[1].sum()]) for x in (a, b, c)]
            assert cell_dist(pa, pb) <= cell_dist(pa, pc) + 1e-12
        if strong_subset(a, b) and strong_subset(b, c):
            for p in (1, 2):
                assert cell_dist(a, b, p) <= cell_dist(a, c, p) + 1e-12


# ------------------------------------------------------------------ Section 5: operational laws
def test_hamacher_identities():
    for _ in range(N):
        a, b = RNG.uniform(0.01, 1, 2); g = RNG.uniform(0.05, 30)
        assert abs(ham_tnorm(a, b, g) - gen_g_inv(gen_g(a, g) + gen_g(b, g), g)) < 1e-10
        assert abs(ham_tconorm(a, b, g) - (1 - ham_tnorm(1 - a, 1 - b, g))) < 1e-10


def test_operational_laws_and_operator():
    kappa = (0.5, 0.5)
    for _ in range(1500):
        g = float(RNG.choice(GAMMAS))
        a, b, c = rsep(kappa), rsep(kappa), rsep(kappa)
        s = sep_oplus(a, b, g, kappa)
        assert is_valid_cell(*s) and is_separable(*s, kappa)
        assert eq(s, sep_oplus(b, a, g, kappa), 1e-8)
        assert eq(sep_oplus(s, c, g, kappa), sep_oplus(a, sep_oplus(b, c, g, kappa), g, kappa), 1e-7)
        lam1, lam2 = RNG.uniform(0.1, 2, 2)
        assert is_separable(*sep_smul(lam1, a, g, kappa), kappa)
        assert eq(sep_smul(lam1, s, g, kappa), sep_oplus(sep_smul(lam1, a, g, kappa), sep_smul(lam1, b, g, kappa), g, kappa), 1e-7)
        assert eq(sep_smul(lam1 + lam2, a, g, kappa), sep_oplus(sep_smul(lam1, a, g, kappa), sep_smul(lam2, a, g, kappa), g, kappa), 1e-7)
        cells = [rsep(kappa) for _ in range(3)]; w = RNG.dirichlet(np.ones(3))
        assert eq(hwa_sep(cells, w, g, kappa), hwa_sep_by_laws(cells, w, g, kappa), 1e-7)


def test_sep_operator_componentwise_monotone_and_general_not():
    kappa = (0.5, 0.5); viol_sep = viol_gen = 0
    for _ in range(N):
        g = float(RNG.choice(GAMMAS))
        cells = [rsep(kappa) for _ in range(3)]; w = RNG.dirichlet(np.ones(3))
        k = RNG.integers(3); l = RNG.integers(2)
        slack = kappa[l] - cells[k][0][l] - cells[k][1][l]
        c2 = [(c[0].copy(), c[1].copy()) for c in cells]; c2[k][0][l] += RNG.uniform(0, slack)
        A0, A1 = hwa_sep(cells, w, g, kappa), hwa_sep(c2, w, g, kappa)
        viol_sep += np.any(A1[0] < A0[0] - 1e-12) or np.any(A1[1] > A0[1] + 1e-12)
        G0, G1 = hwa_general(cells, w, g), hwa_general(c2, w, g)
        viol_gen += np.any(G1[0] < G0[0] - 1e-12)
    assert viol_sep == 0 and viol_gen > 0


def test_kolmogorov_nagumo_midpoint():
    """Thm S3.6 (midpoint condition): M(a,b) (+) M(a,b) = a (+) b for the Hamacher mean."""
    for _ in range(N):
        a, b = RNG.uniform(0, 0.999, 2); g = float(RNG.choice(GAMMAS))
        mbar = M_plus([a, b], [0.5, 0.5], g)
        assert abs(ham_tconorm(mbar, mbar, g) - ham_tconorm(a, b, g)) < 1e-9
        nbar = M_minus([a + 1e-3, b + 1e-3], [0.5, 0.5], g)
        assert abs(ham_tnorm(nbar, nbar, g) - ham_tnorm(a + 1e-3, b + 1e-3, g)) < 1e-9


# ------------------------------------------------------------------ Section 6: decision-stage theorems
def rand_sep_matrix(m, n, kappa):
    T = np.zeros((m, n, 2)); F = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            T[i, j], F[i, j] = rsep(kappa)
    return T, F


def closeness(T, F, beta, refs, p):
    (Tp, Fp), (Tn, Fn) = refs
    m = T.shape[0]; Dp = np.zeros(m); Dn = np.zeros(m)
    for i in range(m):
        Dp[i] = dist((T[i:i+1], F[i:i+1]), (Tp[None], Fp[None]), beta[None], p)
        Dn[i] = dist((T[i:i+1], F[i:i+1]), (Tn[None], Fn[None]), beta[None], p)
    return Dp, Dn, Dn / (Dp + Dn)


def test_hamming_collapse():
    """Thm S4.3: with p=1 and dominating references, C is affine increasing in the weighted score."""
    for kappa in [(0.5, 0.5), (0.7, 0.3)]:
        k = np.asarray(kappa)
        for _ in range(300):
            m, n = 6, 4
            T, F = rand_sep_matrix(m, n, kappa); beta = RNG.dirichlet(np.ones(n))
            score = (beta[None] * (T.sum(-1) - F.sum(-1))).sum(1)
            for refs in [((np.tile(k, (n, 1)), np.zeros((n, 2))), (np.zeros((n, 2)), np.tile(k, (n, 1)))),
                         ((T.max(0), F.min(0)), (T.min(0), F.max(0)))]:
                Dp, Dn, C = closeness(T, F, beta, refs, 1)
                K = (Dp + Dn)[0]
                assert np.allclose(Dp + Dn, K)
                assert np.allclose(C, C[0] + (score - score[0]) / (2 * K))


def test_euclidean_disagreement_decomposition():
    """Thm S4.5: with p=2, equal kappa and absolute references, D+^2=P+Lam, D-^2=N+Lam."""
    kappa = np.array([0.5, 0.5]); S = 2
    for _ in range(300):
        m, n = 5, 4
        T, F = rand_sep_matrix(m, n, kappa); beta = RNG.dirichlet(np.ones(n))
        refs = ((np.tile(kappa, (n, 1)), np.zeros((n, 2))), (np.zeros((n, 2)), np.tile(kappa, (n, 1))))
        Dp, Dn, C = closeness(T, F, beta, refs, 2)
        t, f = T.sum(-1), F.sum(-1)
        P = (0.5 / S) * (beta[None] * ((1 - t) ** 2 + f ** 2)).sum(1)
        Nn = (0.5 / S) * (beta[None] * (t ** 2 + (1 - f) ** 2)).sum(1)
        Lam = 0.5 * (beta[None] * (((T - T.mean(-1, keepdims=True)) ** 2).sum(-1)
                                   + ((F - F.mean(-1, keepdims=True)) ** 2).sum(-1))).sum(1)
        assert np.allclose(Dp ** 2, P + Lam) and np.allclose(Dn ** 2, Nn + Lam)
        # threshold formula for reaching closeness c
        for i in range(m):
            c = 0.5 + 0.3 * (C[i] - 0.5)
            if abs(c - 0.5) > 1e-6:
                L = (c ** 2 * P[i] - (1 - c) ** 2 * Nn[i]) / ((1 - c) ** 2 - c ** 2)
                if L >= 0:
                    Cn = np.sqrt(Nn[i] + L) / (np.sqrt(P[i] + L) + np.sqrt(Nn[i] + L))
                    assert abs(Cn - c) < 1e-9


def test_gamma_bound():
    """Thm S4.6: |D(g1)-D(g2)| <= B_i for the separable operator with fixed references/weights."""
    kappa = np.array([0.5, 0.5])
    for _ in range(100):
        m, n, r = 4, 3, 3
        Ms = [rand_sep_matrix(m, n, kappa) for _ in range(r)]
        w = RNG.dirichlet(np.ones(r)); beta = RNG.dirichlet(np.ones(n))
        refs = ((np.tile(kappa, (n, 1)), np.zeros((n, 2))), (np.zeros((n, 2)), np.tile(kappa, (n, 1))))
        mu = np.array([M[0] / kappa for M in Ms]); nu = np.array([M[1] / kappa for M in Ms])
        Rm = mu.max(0) - mu.min(0); Rn = nu.max(0) - nu.min(0)
        for p in (1, 2):
            B = (0.5 * (beta[None] * ((kappa ** p) * (Rm ** p + Rn ** p)).sum(-1)).sum(1)) ** (1 / p)
            D = [closeness(*aggregate(Ms, w, g, kappa, "sep"), beta, refs, p) for g in (0.2, 1.0, 7.0)]
            for x in D:
                for y in D:
                    assert np.all(np.abs(x[0] - y[0]) <= B + 1e-12) and np.all(np.abs(x[1] - y[1]) <= B + 1e-12)


# ------------------------------------------------------------------ Section 3.6: evidence theory
def test_evidence_interpretation():
    for kappa in [(0.5, 0.5), (0.7, 0.3)]:
        k = np.asarray(kappa)
        for _ in range(N):
            T, F = rsep(kappa)
            ms = source_bpas(T, F, k)
            disc = [discount(ms[l], k[l]) for l in range(2)]
            for l in range(2):
                assert abs(disc[l][0] - T[l]) < 1e-12 and abs(disc[l][1] - F[l]) < 1e-12
            mix = mixture(ms, k)
            assert abs(mix[0] - T.sum()) < 1e-12 and abs(mix[1] - F.sum()) < 1e-12
            d = dempster(*disc)
            assert d.min() >= -1e-12 and abs(d.sum() - 1) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        out = fn(); print(f"PASS  {fn.__name__}" + (f"  {out}" if out is not None else ""))
    print(renormalised_counterexamples())
