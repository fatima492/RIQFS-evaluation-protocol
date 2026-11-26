"""epp_encoding.py -- the two encodings (conventional and evidence of harm) and helpers.

Imported by epp_extensions.py and epp_weights.py. Importing it imports case_study.py, which runs the base case study once.
"""
import os, sys, io, contextlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import case_study as cs

ALTS, H = cs.ALTS, cs.H
m, n = cs.m, cs.n
KAPPA, BETA, Z, MB, DEFF = cs.KAPPA, cs.BETA, cs.Z95, cs.MBENCH, cs.DEFF


def neg_share(a, h, p_pos, neg_mult=1.0):
    """Imputed share of significantly negative estimates: the horizon split of CKW Table 2
    (sig. negative / (insignificant + sig. negative)), optionally scaled, applied to the non-positive share."""
    t = cs.T2[h]; ins, neg = float(t["pct_insignificant"]), float(t["pct_sig_negative"])
    return (1 - p_pos) * min(neg_mult * neg / (ins + neg), 1.0)


def grades_alt(a, h, vote="base", effect="base", p_override=None, m_override=None, M=MB, z=Z,
               neg_mult=1.0, se_mult=1.0, pse_flip=False, n_scale=DEFF):
    r = cs.cell(a, h); nn = max(int(r["n_sign"]) * n_scale, 1.0)      # design-effect-adjusted n
    p = float(r["share_sig_positive_pct"]) / 100 if p_override is None else p_override
    L1, U1 = cs.wilson(p, nn, z)
    if vote == "base":
        mu1, nu1 = L1, 1 - U1
    else:                                       # power-aware: only significant negatives count against
        q = neg_share(a, h, p, neg_mult)
        Lq, _ = cs.wilson(q, nn, z)
        mu1, nu1 = L1, Lq
    me = float(r["mean_effect_pp"]) if m_override is None else m_override
    if pse_flip and a == "PSE" and h == "ST" and m_override is None:
        me = -me
    se = cs.se_cell(a, h, se_mult); lo, hi = me - z * se, me + z * se
    g = lambda x: min(max(x / M, 0.0), 1.0)
    gs = lambda x: min(max((x + M) / (2 * M), 0.0), 1.0)
    if effect == "base":                        # conventional: symmetric scale [-M, M]
        mu2, nu2 = gs(lo), 1 - gs(hi)
    elif effect == "clip0":                     # earlier one-sided scale, clipped at zero
        mu2, nu2 = g(lo), 1 - g(hi)
    else:                                       # benefit/harm: lower bounds of the gain and of the harm
        mu2, nu2 = g(lo), g(-hi)
    return mu1, nu1, mu2, nu2


def encode_alt(vote="base", effect="base", draws=None, M=MB, z=Z, kappa=KAPPA, **kw):
    k = np.asarray(kappa); T = np.zeros((m, n, 2)); F = np.zeros_like(T)
    for i, a in enumerate(ALTS):
        for j, h in enumerate(H):
            po, mo = (None, None) if draws is None else draws[i][j]
            mu1, nu1, mu2, nu2 = grades_alt(a, h, vote, effect, po, mo, M, z, **kw)
            T[i, j] = k * [mu1, mu2]; F[i, j] = k * [nu1, nu2]
    return T, F


def draw_cells(rng):
    """One parametric-bootstrap draw of the published statistics (binomial share, normal mean)."""
    out = []
    for a in ALTS:
        row = []
        for h in H:
            r = cs.cell(a, h); p0 = float(r["share_sig_positive_pct"]) / 100
            ne = max(1, int(round(int(r["n_sign"]) * DEFF)))          # design-effect-adjusted number of draws
            row.append((rng.binomial(ne, p0) / ne, rng.normal(float(r["mean_effect_pp"]), cs.se_cell(a, h))))
        out.append(row)
    return out


def closeness_w(A, W, p=1, kappa=KAPPA):
    """Closeness with absolute references for many criterion-weight vectors W (rows) at once."""
    T, F = A; k = np.asarray(kappa)
    cp = 0.5 * ((np.abs(T - k) ** p).sum(-1) + (F ** p).sum(-1))
    cn = 0.5 * ((T ** p).sum(-1) + (np.abs(F - k) ** p).sum(-1))
    a = (np.atleast_2d(W) @ cn.T) ** (1 / p); b = (np.atleast_2d(W) @ cp.T) ** (1 / p)
    return a / (a + b)
