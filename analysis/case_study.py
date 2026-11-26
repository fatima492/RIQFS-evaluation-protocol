"""
case_study.py -- reproduces every number, table and figure of Sections 4-5 and Appendices B-C of the manuscript.

Case study: ranking four types of active labour market programme from the published
meta-analysis of Card, Kluve and Weber (2018, JEEA 16(3):894-931), Tables 2 and 3a.
Two evidence channels per cell: (1) the share of estimates that are significantly positive
(vote count, Wilson interval); (2) the mean effect on the employment probability (effect
size, normal interval). Both are mapped to grades by the interval-to-intuitionistic map
[L, U] -> <L, 1-U>.

Outputs (under outputs/):
  results.json      all numerical results
  tables/*.tex      LaTeX table bodies used in the manuscript
  figures/*.png     case-study figures (600 dpi)
  csv/*.csv         encoded grades and closeness values
"""
import os, sys, json, csv, itertools, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from riqfs_core import *          # noqa
from pipeline import run, ranks, order, references, closeness, maxdev_weights  # noqa

OUT = os.path.join(ROOT, "outputs")
for d in ("tables", "figures", "csv"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

# ===================================================================== data
ROWS = list(csv.DictReader(open(os.path.join(ROOT, "data", "ckw2018_table3a.csv"))))
T2 = {r["horizon"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "ckw2018_table2.csv")))}
H = ["ST", "MT", "LT"]
HLAB = ["short term", "medium term", "long term"]
NAMES = {r["prog_short"]: r["programme"] for r in ROWS}
ALTS = ["TRN", "JSA", "PRS", "PSE"]           # main set
EXTRA = "OTH"                                  # used in the rank-reversal test
m, n, S = len(ALTS), len(H), 2

KAPPA = (0.5, 0.5); MBENCH = 20.0; Z95 = 1.959964; GAMMA = 1.0
# Design effect: estimates are clustered within studies (857 estimates from 207 studies), so binomial counts use an
# effective number of independent observations n * 207/857 (about one per study).
DEFF = 207 / 857
BETA = np.ones(n) / n
res = {}


def cell(a, h):
    return [r for r in ROWS if r["prog_short"] == a and r["horizon"] == h][0]


def wilson(p, nn, z):
    d = 1 + z * z / nn
    c = (p + z * z / (2 * nn)) / d
    hw = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
    return max(c - hw, 0.0), min(c + hw, 1.0)


def se_cell(a, h, se_mult=1.0):
    """Standard error of a type-specific mean: horizon SE of Table 2 rescaled by sqrt(n_h / n_cell)."""
    r = cell(a, h); t = T2[h]
    return se_mult * float(t["se_mean_pp"]) * np.sqrt(int(t["n_effect"]) / int(r["n_effect"]))


def channel_grades(a, h, M=MBENCH, z=Z95, pse_flip=False, se_mult=1.0, p_override=None, m_override=None, n_scale=DEFF,
                   scale="sym"):
    """Normalised source grades (mu_1, nu_1, mu_2, nu_2) of one cell, and the intervals behind them."""
    r = cell(a, h)
    p = float(r["share_sig_positive_pct"]) / 100 if p_override is None else p_override
    L1, U1 = wilson(p, max(int(r["n_sign"]) * n_scale, 1.0), z)
    me = float(r["mean_effect_pp"]) if m_override is None else m_override
    if pse_flip and a == "PSE" and h == "ST" and m_override is None:
        me = -me
    se = se_cell(a, h, se_mult)
    lo, hi = me - z * se, me + z * se
    if scale == "sym":      # symmetric scale: effects in [-M, M] mapped to [0, 1], zero effect = 0.5
        g = lambda x: min(max((x + M) / (2 * M), 0.0), 1.0)
    else:                   # one-sided scale [0, M], clipped at zero (sensitivity case only; rewards imprecise effects, Section 3.1)
        g = lambda x: min(max(x / M, 0.0), 1.0)
    return (L1, 1 - U1, g(lo), 1 - g(hi)), dict(p=p, L1=L1, U1=U1, mean=me, se=se, lo=lo, hi=hi)


def encode(alts, kappa=KAPPA, **kw):
    k = np.asarray(kappa, float)
    T = np.zeros((len(alts), n, S)); F = np.zeros_like(T)
    for i, a in enumerate(alts):
        for j, h in enumerate(H):
            (mu1, nu1, mu2, nu2), _ = channel_grades(a, h, **kw)
            T[i, j] = k * np.array([mu1, mu2]); F[i, j] = k * np.array([nu1, nu2])
    return T, F


def go(A, p, beta=BETA, kappa=KAPPA, ref="absolute"):
    return run([A], "B" * n, kappa=kappa, gamma=GAMMA, p=p, w=np.array([1.0]), beta=beta, ref=ref)


LAB = [f"\\omega_{{{i+1}}}" for i in range(m)]


def ostr(C, labels=LAB):
    return " \\succ ".join(order(C, labels))


def discordant(C1, C2):
    return int(sum(np.sign(C1[i] - C1[j]) != np.sign(C2[i] - C2[j]) for i, j in itertools.combinations(range(len(C1)), 2)))


# ===================================================================== data checks
chk = {}
for h in H:
    rs = [r for r in ROWS if r["horizon"] == h]
    ne = sum(int(r["n_effect"]) for r in rs); ns = sum(int(r["n_sign"]) for r in rs)
    mean = sum(float(r["mean_effect_pp"]) * int(r["n_effect"]) for r in rs) / ne
    share = sum(float(r["share_sig_positive_pct"]) * int(r["n_sign"]) for r in rs) / ns
    chk[h] = dict(n_effect=ne, n_effect_t2=int(T2[h]["n_effect"]), n_sign=ns, n_sign_t2=int(T2[h]["n_sign"]),
                  mean=mean, mean_t2=float(T2[h]["mean_effect_pp"]), share=share, share_t2=float(T2[h]["pct_sig_positive"]))
    assert ne == int(T2[h]["n_effect"]) and ns == int(T2[h]["n_sign"])
res["checks"] = chk
rs = [r for r in ROWS if r["horizon"] == "ST"]
ne = sum(int(r["n_effect"]) for r in rs)
res["st_mean_flip"] = (sum(float(r["mean_effect_pp"]) * int(r["n_effect"]) for r in rs) - 2 * 3.6 * 14) / ne
res["n_estimates_total"] = sum(int(T2[h]["n_sign"]) for h in H)
res["n_effect_total"] = sum(int(T2[h]["n_effect"]) for h in H)
res["n_estimates_main"] = sum(int(cell(a, h)["n_sign"]) for a in ALTS for h in H)
res["n_effect_main"] = sum(int(cell(a, h)["n_effect"]) for a in ALTS for h in H)

# ===================================================================== base encoding and results
T, F = encode(ALTS)
A = (T, F)
grades = {}
for i, a in enumerate(ALTS + [EXTRA]):
    for j, h in enumerate(H):
        g, info = channel_grades(a, h)
        grades[f"{a}_{h}"] = dict(grades=list(g), **info)
res["grades"] = grades

B1 = go(A, 1); B2 = go(A, 2)
t, f = T.sum(-1), F.sum(-1)
score = (BETA[None] * (t - f)).sum(1)
res.update(C1=B1["C"].tolist(), C2=B2["C"].tolist(), Dp1=B1["Dp"].tolist(), Dn1=B1["Dn"].tolist(),
           Dp2=B2["Dp"].tolist(), Dn2=B2["Dn"].tolist(), score=score.tolist(),
           order1=order(B1["C"], ALTS), order2=order(B2["C"], ALTS))
# Theorem 6.3 (Hamming collapse): C^H = (1 + S_i)/2 with absolute references
res["collapse_err"] = float(np.max(np.abs(B1["C"] - 0.5 * (1 + score))))
# Theorem 6.5: D^2 = P/N + Lambda (equal kappa)
lam = ((T - T.mean(-1, keepdims=True)) ** 2).sum(-1) + ((F - F.mean(-1, keepdims=True)) ** 2).sum(-1)
P = (0.5 / S) * (BETA[None] * ((1 - t) ** 2 + f ** 2)).sum(1)
N = (0.5 / S) * (BETA[None] * (t ** 2 + (1 - f) ** 2)).sum(1)
Lam = 0.5 * (BETA[None] * lam).sum(1)
res["decomp_err"] = float(max(np.max(np.abs(B2["Dp"] ** 2 - P - Lam)), np.max(np.abs(B2["Dn"] ** 2 - N - Lam))))
CPhi = np.sqrt(N) / (np.sqrt(P) + np.sqrt(N))
res.update(lam=lam.tolist(), Lam=Lam.tolist(), P=P.tolist(), N=N.tolist(), CPhi=CPhi.tolist())
res["discordant_H_E"] = discordant(B1["C"], B2["C"])
res["discordant_Phi_E"] = discordant(CPhi, B2["C"])

# ===================================================================== Hamacher evidence profile across horizons
GAMMAS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20]
prof = {}
for g in GAMMAS:
    sc = []
    for i in range(m):
        Ti, Fi = hwa_sep([(T[i, j], F[i, j]) for j in range(n)], BETA, g, KAPPA)
        sc.append(float(Ti.sum() - Fi.sum()))
    prof[str(g)] = sc
res["profile"] = prof
res["profile_orders"] = {g: order(np.array(v), ALTS) for g, v in prof.items()}
res["profile_rank_invariant"] = len({tuple(v) for v in res["profile_orders"].values()}) == 1
prof_cells = {}
for i, a in enumerate(ALTS):
    Ti, Fi = hwa_sep([(T[i, j], F[i, j]) for j in range(n)], BETA, GAMMA, KAPPA)
    prof_cells[a] = [Ti.tolist(), Fi.tolist()]
res["profile_cells"] = prof_cells
# operator comparison on the horizon profile (Remark S3.7): separable RHWA, allocation operator, convex pool
opcmp = {}
for nm, fn in [("sep", lambda c: hwa_sep(c, BETA, GAMMA, KAPPA)), ("general", lambda c: hwa_general(c, BETA, GAMMA)),
               ("convex", lambda c: convex_pool(c, BETA))]:
    sc_, cells_ = [], {}
    for i, a in enumerate(ALTS):
        Ti, Fi = fn([(T[i, j], F[i, j]) for j in range(n)])
        sc_.append(float(np.sum(Ti) - np.sum(Fi))); cells_[a] = [list(map(float, Ti)), list(map(float, Fi))]
    opcmp[nm] = dict(score=sc_, order=order(np.array(sc_), ALTS), cells=cells_)
res["opcmp"] = opcmp
res["prs_profile_F2_sep"] = opcmp["sep"]["cells"]["PRS"][1][1]
res["prs_profile_F2_convex"] = opcmp["convex"]["cells"]["PRS"][1][1]

# ===================================================================== criterion weights: scenarios and SMAA
WSCEN = [("Equal (base)", BETA), ("Short-run emphasis $(0.5,0.3,0.2)$", np.array([.5, .3, .2])),
         ("Long-run emphasis $(0.2,0.3,0.5)$", np.array([.2, .3, .5])),
         ("Long term only $(0,0,1)$", np.array([0., 0., 1.])), ("Short term only $(1,0,0)$", np.array([1., 0., 0.]))]
md = maxdev_weights(A)
WSCEN.append((f"Maximising deviation $({md[0]:.2f},{md[1]:.2f},{md[2]:.2f})$", md))
res["maxdev_beta"] = md.tolist()
wtab = []
for name, b in WSCEN:
    x1, x2 = go(A, 1, beta=b), go(A, 2, beta=b)
    wtab.append(dict(name=name, beta=list(map(float, b)), o1=ostr(x1["C"]), o2=ostr(x2["C"]),
                     s1=discordant(x1["C"], B1["C"]), s2=discordant(x2["C"], B2["C"]),
                     top1=ALTS[int(np.argmax(x1["C"]))], top2=ALTS[int(np.argmax(x2["C"]))],
                     C1=x1["C"].tolist(), C2=x2["C"].tolist()))
res["wscen"] = wtab


def contrib(A_, refs, p):
    T_, F_ = A_; (Tp, Fp), (Tn, Fn) = refs
    cp = 0.5 * ((np.abs(T_ - Tp[None]) ** p).sum(-1) + (np.abs(F_ - Fp[None]) ** p).sum(-1))
    cn = 0.5 * ((np.abs(T_ - Tn[None]) ** p).sum(-1) + (np.abs(F_ - Fn[None]) ** p).sum(-1))
    return cp, cn


def smaa(p, alpha_vec, Nn=20000, seed=11):
    g_ = np.random.default_rng(seed)
    W = g_.dirichlet(alpha_vec, size=Nn)
    cp, cn = contrib(A, B1["refs"], p)
    C = (W @ cn.T) ** (1 / p) / ((W @ cp.T) ** (1 / p) + (W @ cn.T) ** (1 / p))
    rk = np.argsort(np.argsort(-C, axis=1), axis=1)
    acc = np.stack([(rk == q).mean(0) for q in range(m)], 1)
    pw = np.array([[(C[:, i] > C[:, l]).mean() for l in range(m)] for i in range(m)])
    base_rk = ranks(B1["C"] if p == 1 else B2["C"]) - 1
    return dict(acc=acc.tolist(), pw=pw.tolist(), same=float((rk == base_rk[None]).all(1).mean()))


res["smaa"] = {f"{nm}_p{p}": smaa(p, av) for p in (1, 2) for nm, av in [("global", np.ones(n)), ("local", 30 * BETA)]}
# how often do the Hamming and Euclidean rankings differ over the weight simplex?
_W = np.random.default_rng(11).dirichlet(np.ones(n), size=20000)
_ord = {}
for p in (1, 2):
    cp, cn = contrib(A, B1["refs"], p)
    _C = (_W @ cn.T) ** (1 / p) / ((_W @ cp.T) ** (1 / p) + (_W @ cn.T) ** (1 / p))
    _ord[p] = np.argsort(-_C, axis=1)
res["smaa_HE_differ"] = float((_ord[1] != _ord[2]).any(1).mean())

# ===================================================================== encoding and modelling robustness
k_prop = (res["n_estimates_total"] / (res["n_estimates_total"] + res["n_effect_total"]),
          res["n_effect_total"] / (res["n_estimates_total"] + res["n_effect_total"]))
res["kappa_prop"] = list(k_prop)
VARIANTS = [
    ("Base: $\\kappa=(0.5,0.5)$, $M=20$, symmetric effect scale, 95\\% intervals, design-effect $n$, equal $\\beta$", {}, {}),
    ("Benchmark effect $M=10$ points", dict(M=10.0), {}),
    ("Benchmark effect $M=15$ points", dict(M=15.0), {}),
    ("Benchmark effect $M=30$ points", dict(M=30.0), {}),
    ("90\\% intervals", dict(z=1.644854), {}),
    ("99\\% intervals", dict(z=2.575829), {}),
    ("Effect-size standard errors $\\times1.5$", dict(se_mult=1.5), {}),
    ("No design effect: vote-count intervals with $n$ = estimates", dict(n_scale=1.0), {}),
    ("One-sided effect scale $[0,M]$ clipped at zero (Section 3.1)", dict(scale="clip0"), {}),
    ("Public-sector short-term mean $-3.6$ (reconciles Table 2)", dict(pse_flip=True), {}),
    (f"$\\kappa=({k_prop[0]:.2f},{k_prop[1]:.2f})$, proportional to estimate counts", dict(kappa=k_prop), dict(kappa=k_prop)),
    (f"$\\kappa=({k_prop[1]:.2f},{k_prop[0]:.2f})$", dict(kappa=k_prop[::-1]), dict(kappa=k_prop[::-1])),
    ("Data-driven references $I^{\\pm}$", {}, dict(ref="data")),
    ("Distance order $p=3$ (in place of $p=2$)", {}, dict(p3=True)),
]
rob = []
for name, enc_kw, run_kw in VARIANTS:
    A_ = encode(ALTS, **enc_kw)
    kap = run_kw.get("kappa", KAPPA); ref = run_kw.get("ref", "absolute")
    x1 = go(A_, 1, kappa=kap, ref=ref)
    x2 = go(A_, 3 if run_kw.get("p3") else 2, kappa=kap, ref=ref)
    rob.append(dict(name=name, o1=ostr(x1["C"]), o2=ostr(x2["C"]), s1=discordant(x1["C"], B1["C"]), s2=discordant(x2["C"], B2["C"]),
                    C1=x1["C"].tolist(), C2=x2["C"].tolist()))
res["robust"] = rob
# benchmark M on a fine grid: where do TRN and JSA swap?
Ms_grid = np.round(np.arange(5, 40.01, 0.5), 2); swap = {1: [], 2: []}
for Mb in Ms_grid:
    A_ = encode(ALTS, M=float(Mb))
    for p in (1, 2):
        C_ = go(A_, p)["C"]; swap[p].append(float(C_[0] - C_[1]))
res["M_grid"] = Ms_grid.tolist(); res["M_trn_minus_jsa"] = {str(p): swap[p] for p in (1, 2)}
for p in (1, 2):
    d = np.array(swap[p]); idx = np.where(np.diff(np.sign(d)) != 0)[0]
    res[f"M_swap_p{p}"] = [float(Ms_grid[k] + (Ms_grid[k + 1] - Ms_grid[k]) * d[k] / (d[k] - d[k + 1])) for k in idx]

# ===================================================================== parametric bootstrap of the published statistics
def bootstrap(Bn=5000, seed=3):
    g_ = np.random.default_rng(seed)
    out = {1: [], 2: []}
    for _ in range(Bn):
        k = np.asarray(KAPPA)
        T_ = np.zeros((m, n, S)); F_ = np.zeros_like(T_)
        for i, a in enumerate(ALTS):
            for j, h in enumerate(H):
                r = cell(a, h); nn = int(r["n_sign"]); p0 = float(r["share_sig_positive_pct"]) / 100
                ne = max(1, int(round(nn * DEFF)))          # design-effect-adjusted number of draws
                ps = g_.binomial(ne, p0) / ne
                ms = g_.normal(float(r["mean_effect_pp"]), se_cell(a, h))
                (mu1, nu1, mu2, nu2), _ = channel_grades(a, h, p_override=ps, m_override=ms)
                T_[i, j] = k * [mu1, mu2]; F_[i, j] = k * [nu1, nu2]
        for p in (1, 2):
            out[p].append(go((T_, F_), p)["C"])
    stats = {}
    for p in (1, 2):
        Cs = np.array(out[p]); rk = np.argsort(np.argsort(-Cs, 1), 1)
        stats[p] = dict(mean=Cs.mean(0).tolist(), lo=np.percentile(Cs, 2.5, 0).tolist(), hi=np.percentile(Cs, 97.5, 0).tolist(),
                        first=(rk == 0).mean(0).tolist(), last=(rk == m - 1).mean(0).tolist(),
                        meanrank=(rk + 1).mean(0).tolist(), same=float((rk == (ranks(B1["C"] if p == 1 else B2["C"]) - 1)[None]).all(1).mean()),
                        trn_over_jsa=float((Cs[:, 0] > Cs[:, 1]).mean()))
    return stats


bs = bootstrap()
res["boot1"], res["boot2"] = bs[1], bs[2]

# ===================================================================== rank reversal
def rr_tests(ref, beta_mode):
    full = ALTS + [EXTRA]
    Afull = encode(full)
    def C_of(alts, Asub):
        b = BETA if beta_mode == "equal" else maxdev_weights(Asub)
        return run([Asub], "B" * n, kappa=KAPPA, gamma=GAMMA, p=2, w=np.array([1.0]), beta=b, ref=ref)["C"]
    base = dict(zip(full, C_of(full, Afull)))
    out = []
    def compare(alts, C):
        d = dict(zip(alts, C)); common = [a for a in alts if a in base and alts.count(a) == 1]
        return sum(np.sign(base[a] - base[b]) != np.sign(d[a] - d[b]) for a, b in itertools.combinations(common, 2))
    for q in full:                                    # remove one alternative
        alts = [a for a in full if a != q]; idx = [full.index(a) for a in alts]
        out.append(dict(test=f"remove {q}", bad=int(compare(alts, C_of(alts, (Afull[0][idx], Afull[1][idx]))))))
    for q in full:                                    # duplicate one alternative
        idx = list(range(len(full))) + [full.index(q)]
        alts = full + [q + "*"]
        C = C_of(alts, (Afull[0][idx], Afull[1][idx]))
        out.append(dict(test=f"duplicate {q}", bad=int(compare(alts, C))))
    return out


res["rr"] = {f"{ref}_{bm}": rr_tests(ref, bm) for ref in ("absolute", "data") for bm in ("equal", "maxdev")}
res["rr_max"] = {k: max(x["bad"] for x in v) for k, v in res["rr"].items()}
Afull = encode(ALTS + [EXTRA])
C5 = go(Afull, 2)["C"]
res["C2_with_other"] = C5.tolist()
res["order2_with_other"] = order(C5, ALTS + [EXTRA])

# ===================================================================== benchmarks
def ifwa_score(t_, f_, b):
    tt = 1 - np.prod((1 - t_) ** b[None], 1); ff = np.prod(np.where(f_ > 0, f_, 0) ** b[None], 1) * np.all(f_ > 0, 1)
    return tt - ff


def vikor(x, b, v=0.5):
    fs, fm = x.max(0), x.min(0); den = np.where(fs - fm > 0, fs - fm, 1)
    Sg = (b[None] * (fs - x) / den).sum(1); Rg = (b[None] * (fs - x) / den).max(1)
    Q = v * (Sg - Sg.min()) / max(Sg.max() - Sg.min(), 1e-12) + (1 - v) * (Rg - Rg.min()) / max(Rg.max() - Rg.min(), 1e-12)
    return -Q


def sk_topsis(t_, f_, b):
    tp, fp, tn, fn = t_.max(0), f_.min(0), t_.min(0), f_.max(0)
    d = lambda t1, f1, t2, f2: 0.5 * (b[None] * (np.abs(t1 - t2) + np.abs(f1 - f2) + np.abs((1 - t1 - f1) - (1 - t2 - f2)))).sum(1)
    Dp_, Dn_ = d(t_, f_, tp[None], fp[None]), d(t_, f_, tn[None], fn[None])
    return Dn_ / (Dp_ + Dn_)


def dhfs_topsis(A_, kap, b, lam_=2):
    T_, F_ = A_; k_ = np.asarray(kap)
    h = np.sort(T_ / k_, -1); gg = np.sort(F_ / k_, -1)
    hp, gp, hn, gn = h.max(0), gg.min(0), h.min(0), gg.max(0)
    d = lambda h1, g1, h2, g2: (0.5 * (b[None] * ((np.abs(h1 - h2) ** lam_).mean(-1) + (np.abs(g1 - g2) ** lam_).mean(-1))).sum(1)) ** (1 / lam_)
    Dp_, Dn_ = d(h, gg, hp[None], gp[None]), d(h, gg, hn[None], gn[None])
    return Dn_ / (Dp_ + Dn_)


def dempster_score(A_, kap, b):
    T_, F_ = A_; k_ = np.asarray(kap); sc = np.zeros(T_.shape[:2])
    for i in range(T_.shape[0]):
        for j in range(T_.shape[1]):
            ms = source_bpas(T_[i, j], F_[i, j], k_)
            mm = discount(ms[0], k_[0])
            for l in range(1, len(k_)):
                mm = dempster(mm, discount(ms[l], k_[l]))
            sc[i, j] = mm[0] - mm[1]
    return (b[None] * sc).sum(1)


share = np.array([[float(cell(a, h)["share_sig_positive_pct"]) for h in H] for a in ALTS])
meanef = np.array([[float(cell(a, h)["mean_effect_pp"]) for h in H] for a in ALTS])
BENCH = [("Average share of significantly positive estimates (vote count)", (BETA[None] * share).sum(1)),
         ("Average mean employment effect (effect size)", (BETA[None] * meanef).sum(1)),
         ("Intuitionistic reduction $\\Phi$, Euclidean TOPSIS", CPhi),
         ("IF-TOPSIS, Szmidt--Kacprzyk distance \\cite{boran2009multi,szmidt2000distances}", sk_topsis(t, f, BETA)),
         ("IFWA aggregation and score \\cite{xu2007intuitionistic}", ifwa_score(t, f, BETA)),
         ("VIKOR on intuitionistic scores \\cite{opricovic2004compromise}", vikor(t - f, BETA)),
         ("Dual hesitant TOPSIS, sorted channels \\cite{xu2011distance,zhu2012dual}", dhfs_topsis(A, KAPPA, BETA)),
         ("Dempster's rule on discounted channels \\cite{shafer1976mathematical}", dempster_score(A, KAPPA, BETA))]
res["bench"] = [dict(name=nm, o=ostr(sc), d1=discordant(sc, B1["C"]), d2=discordant(sc, B2["C"]), sc=list(map(float, sc))) for nm, sc in BENCH]

# ===================================================================== timing (complexity remark, supplementary Section S4)
tm = []
for (mm, nn, rr) in [(5, 4, 3), (20, 10, 5), (50, 20, 10), (100, 40, 20)]:
    g_ = np.random.default_rng(1); Ms_ = []
    for _ in range(rr):
        x = g_.dirichlet(np.ones(3), size=(mm, nn, 2)) * 0.5
        Ms_.append((x[..., 0], x[..., 1]))
    t0 = time.perf_counter()
    run(Ms_, ["B"] * nn, KAPPA, 1.0, p=2, w="consensus", beta=np.ones(nn) / nn)
    tm.append(dict(m=mm, n=nn, r=rr, sec=time.perf_counter() - t0))
res["timing"] = tm

# ===================================================================== CSV intermediates
with open(os.path.join(OUT, "csv", "encoded_grades.csv"), "w", newline="") as fh:
    w_ = csv.writer(fh)
    w_.writerow(["programme", "horizon", "share_sig_pos", "n_sign", "wilson_L", "wilson_U", "mean_pp", "se_pp", "ci_lo", "ci_hi",
                 "mu1", "nu1", "mu2", "nu2", "T1", "F1", "T2", "F2"])
    for a in ALTS + [EXTRA]:
        for h in H:
            g, info = channel_grades(a, h); k = KAPPA
            w_.writerow([a, h, info["p"], cell(a, h)["n_sign"], f"{info['L1']:.6f}", f"{info['U1']:.6f}", info["mean"], f"{info['se']:.6f}",
                         f"{info['lo']:.6f}", f"{info['hi']:.6f}"] + [f"{x:.6f}" for x in g] +
                        [f"{k[0]*g[0]:.6f}", f"{k[0]*g[1]:.6f}", f"{k[1]*g[2]:.6f}", f"{k[1]*g[3]:.6f}"])
with open(os.path.join(OUT, "csv", "closeness.csv"), "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["programme", "score", "C_hamming", "C_euclidean", "Lambda", "C_phi"])
    for i, a in enumerate(ALTS):
        w_.writerow([a, f"{score[i]:.6f}", f"{B1['C'][i]:.6f}", f"{B2['C'][i]:.6f}", f"{Lam[i]:.6f}", f"{CPhi[i]:.6f}"])

# ===================================================================== LaTeX tables
def W(name, s): open(os.path.join(OUT, "tables", name + ".tex"), "w").write(s)
f2 = lambda x: f"${x:.2f}$"; f3 = lambda x: f"${x:.3f}$"; f4 = lambda x: f"${x:.4f}$"


def pm(x):
    return f"${x:.1f}$" if x >= 0 else f"$-{abs(x):.1f}$"


# source data (Appendix B)
lines = []
for a in ALTS + [EXTRA]:
    tag = f"$\\omega_{{{ALTS.index(a)+1}}}$" if a in ALTS else "$\\omega_5$"
    r_ = [cell(a, h) for h in H]
    lines.append(f"{tag} & {a} & {NAMES[a]} & " + " & ".join(f"{r['share_sig_positive_pct']} ({r['n_sign']})" for r in r_) + " & " +
                 " & ".join(f"{pm(float(r['mean_effect_pp']))} ({r['n_effect']})" for r in r_) + " \\\\")
W("data", "\n".join(lines) + "\n")
# encoded grades
lines = []
for i, a in enumerate(ALTS):
    cells = []
    for h in H:
        g, _ = channel_grades(a, h)
        cells.append(f"$\\langle {g[0]:.2f},{g[1]:.2f}\\rangle$ & $\\langle {g[2]:.2f},{g[3]:.2f}\\rangle$")
    lines.append(f"$\\omega_{{{i+1}}}$ {a} & " + " & ".join(cells) + " \\\\")
W("grades", "\n".join(lines) + "\n")
# intervals behind the grades
lines = []
for i, a in enumerate(ALTS):
    cells = []
    for h in H:
        _, info = channel_grades(a, h)
        cells.append(f"[{info['L1']:.2f}, {info['U1']:.2f}] & [{pm(info['lo'])}, {pm(info['hi'])}]")
    lines.append(f"$\\omega_{{{i+1}}}$ {a} & " + " & ".join(cells) + " \\\\")
W("intervals", "\n".join(lines) + "\n")
# main results
rk1, rk2 = ranks(B1["C"]), ranks(B2["C"])
W("results", "\n".join(f"$\\omega_{{{i+1}}}$ & {ALTS[i]} & {f4(score[i])} & {f4(B1['C'][i])} & {rk1[i]} & {f4(B2['Dp'][i])} & "
                        f"{f4(B2['Dn'][i])} & {f4(B2['C'][i])} & {rk2[i]} & {f4(Lam[i])} & {f4(CPhi[i])} \\\\" for i in range(m)) + "\n")
# Hamacher profile vs gamma
lines = []
for i, a in enumerate(ALTS):
    lines.append(f"$\\omega_{{{i+1}}}$ {a} & " + " & ".join(f4(prof[str(g)][i]) for g in GAMMAS) + " \\\\")
lines.append("\\midrule\nRanking & " + " & ".join("$" + "\\succ ".join(f"\\omega_{{{ALTS.index(x)+1}}}" for x in res['profile_orders'][str(g)]) + "$" for g in GAMMAS) + " \\\\")
W("gamma", "\n".join(lines) + "\n")
# weight scenarios
W("wscen", "\n".join(f"{x['name']} & ${x['o1']}$ ({x['s1']}) & ${x['o2']}$ ({x['s2']}) \\\\" for x in wtab) + "\n")
# SMAA
lines = []
for i in range(m):
    row = [f"$\\omega_{{{i+1}}}$ {ALTS[i]}"]
    for key in ("global_p1", "global_p2", "local_p2"):
        acc = np.array(res["smaa"][key]["acc"])[i]
        row.append(" & ".join(f"{100*v:.1f}" for v in acc))
    lines.append(" & ".join(row) + " \\\\")
W("smaa", "\n".join(lines) + "\n")
# robustness
W("robust", "\n".join(f"{x['name']} & ${x['o1']}$ ({x['s1']}) & ${x['o2']}$ ({x['s2']}) \\\\" for x in rob) + "\n")
# bootstrap
lines = []
for i in range(m):
    b1, b2 = res["boot1"], res["boot2"]
    lines.append(f"$\\omega_{{{i+1}}}$ {ALTS[i]} & {f3(b1['mean'][i])} [{f3(b1['lo'][i])}, {f3(b1['hi'][i])}] & {100*b1['first'][i]:.1f} & {b1['meanrank'][i]:.2f} & "
                 f"{f3(b2['mean'][i])} [{f3(b2['lo'][i])}, {f3(b2['hi'][i])}] & {100*b2['first'][i]:.1f} & {b2['meanrank'][i]:.2f} \\\\")
W("bootstrap", "\n".join(lines) + "\n")
# rank reversal
lab_rr = {"absolute_equal": "Absolute references, equal $\\beta$ (base)", "absolute_maxdev": "Absolute references, maximising-deviation $\\beta$",
          "data_equal": "Data-driven references, equal $\\beta$", "data_maxdev": "Data-driven references, maximising-deviation $\\beta$"}
lines = []
for k_, v in res["rr"].items():
    rem = sum(x["bad"] for x in v if x["test"].startswith("remove")); dup = sum(x["bad"] for x in v if x["test"].startswith("duplicate"))
    lines.append(f"{lab_rr[k_]} & {rem} & {dup} \\\\")
W("rankrev", "\n".join(lines) + "\n")
# benchmarks
W("bench", "\n".join(f"{x['name']} & ${x['o']}$ & {x['d1']} & {x['d2']} \\\\" for x in res["bench"]) + "\n")

# ===================================================================== figures
from plotstyle import apply as _style, PCT
_style()
COLS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
mk = ["o", "s", "^", "D", "v"]
ALAB = [f"$\\omega_{i+1}$ ({ALTS[i]})" for i in range(m)]
CLAB = ["$q_1$\nshort term", "$q_2$\nmedium term", "$q_3$\nlong term"]

# (a) heat maps: score and channel disagreement
scell = t - f
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, Mx, ttl, cmap, fmt, vmin, vmax in [(axs[0], scell, r"cell score $t_{ij}-f_{ij}$", "RdYlGn", "${:.2f}$", -0.6, 0.6),
                                           (axs[1], lam, r"channel disagreement $\lambda_{ij}$", "Purples", "${:.3f}$", 0, lam.max())]:
    im = ax.imshow(Mx, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(CLAB, fontsize=7); ax.set_yticks(range(m)); ax.set_yticklabels(ALAB, fontsize=7.5)
    ax.set_title(ttl, fontsize=8)
    for i in range(m):
        for j in range(n):
            v = Mx[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (cmap == "Purples" and v > 0.6 * Mx.max()) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "heatmaps.png"), dpi=600); plt.close(fig)

# (b) the two channels as intervals (data view)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
off = np.linspace(-0.27, 0.27, m)
for i, a in enumerate(ALTS):
    for j, h in enumerate(H):
        _, info = channel_grades(a, h)
        y = j + off[i]
        axs[0].plot([info["L1"], info["U1"]], [y, y], color=COLS[i], lw=2)
        axs[0].plot([info["p"]], [y], marker=mk[i], color=COLS[i], ms=4, mec="k", mew=.4, label=ALAB[i] if j == 0 else None)
        axs[1].plot([info["lo"], info["hi"]], [y, y], color=COLS[i], lw=2)
        axs[1].plot([info["mean"]], [y], marker=mk[i], color=COLS[i], ms=4, mec="k", mew=.4)
axs[0].set_xlim(0, 1); axs[0].set_xlabel(f"share significantly positive\n(95{PCT} Wilson interval)")
axs[1].axvline(0, color="grey", lw=.6); axs[1].axvline(MBENCH, color="grey", lw=.6, ls="--")
axs[1].text(MBENCH, -0.62, f"$M={MBENCH:.0f}$", fontsize=7, ha="center", va="bottom")
axs[1].set_xlabel(f"mean employment effect, points\n(95{PCT} interval)")
axs[0].set_yticks(range(n)); axs[0].set_yticklabels(["short term", "medium term", "long term"]); axs[0].set_ylim(2.5, -0.75)
for ax in axs: ax.grid(alpha=.3, axis="x")
axs[0].set_title("channel 1: vote count", fontsize=8.5); axs[1].set_title("channel 2: effect size", fontsize=8.5)
h_, l_ = axs[0].get_legend_handles_labels()
axs[1].legend(h_, l_, fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "channels.png"), dpi=600); plt.close(fig)

# (c) decomposition of (D+)^2 (p = 2)
phi_part = (0.5 / S) * BETA[None] * ((1 - t) ** 2 + f ** 2)
lam_part = 0.5 * BETA[None] * lam
fig, ax = plt.subplots(figsize=(6.2, 2.7))
ccol = ["#4c72b0", "#dd8452", "#55a868"]; x = np.arange(m); bottom = np.zeros(m)
for j in range(n):
    ax.bar(x, phi_part[:, j], bottom=bottom, color=ccol[j], label=f"$q_{j+1}$ ({HLAB[j]})", edgecolor="white", lw=.4); bottom += phi_part[:, j]
for j in range(n):
    ax.bar(x, lam_part[:, j], bottom=bottom, color=ccol[j], hatch="////", alpha=.55, edgecolor="k", lw=.3); bottom += lam_part[:, j]
ax.set_xticks(x); ax.set_xticklabels(ALAB); ax.set_ylabel(r"contribution to $(D_i^{+})^2$")
h_, l_ = ax.get_legend_handles_labels(); h_.append(Patch(facecolor="white", edgecolor="k", hatch="////")); l_.append(r"channel disagreement $\Lambda$")
ax.legend(h_, l_, fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "decomposition.png"), dpi=600); plt.close(fig)

# (d) closeness as a function of disagreement (Theorem 6.5)
grid = np.linspace(0, 1.25 * Lam.max(), 300)
fig, ax = plt.subplots(figsize=(5.6, 2.7))
for i in range(m):
    Cg = np.sqrt(N[i] + grid) / (np.sqrt(P[i] + grid) + np.sqrt(N[i] + grid))
    ax.plot(grid, Cg, color=COLS[i], lw=1.2, label=ALAB[i])
    ax.plot([Lam[i]], [B2["C"][i]], marker=mk[i], color=COLS[i], ms=6, mec="k", mew=.5)
    ax.plot([0], [CPhi[i]], marker="|", color=COLS[i], ms=8)
ax.axhline(0.5, color="grey", lw=.6, ls=":")
ax.set_xlabel(r"channel disagreement $\Lambda$"); ax.set_ylabel(r"Euclidean closeness $C^E$")
ax.grid(alpha=.3); ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "disagreement_curves.png"), dpi=600); plt.close(fig)

# (e) Hamacher profile scores vs gamma
fig, ax = plt.subplots(figsize=(5.6, 2.6))
for i in range(m):
    ax.plot(GAMMAS, [prof[str(g)][i] for g in GAMMAS], marker=mk[i], ms=3.5, lw=1.1, color=COLS[i], label=ALAB[i])
ax.set_xscale("log"); ax.grid(alpha=.3); ax.set_xlabel(r"Hamacher parameter $\gamma$ (log scale)")
ax.set_ylabel(r"score of $\mathrm{RHWA}_{\gamma,\beta}$ profile")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "gamma_sensitivity.png"), dpi=600); plt.close(fig)

# (f) SMAA rank acceptabilities
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
rc = plt.cm.viridis(np.linspace(0.1, 0.9, m))
for ax, key_, ttl in [(axs[0], "global_p1", "Hamming, uniform simplex"), (axs[1], "global_p2", "Euclidean, uniform simplex")]:
    acc = np.array(res["smaa"][key_]["acc"]); bottom = np.zeros(m)
    for q in range(m):
        ax.bar(range(m), 100 * acc[:, q], bottom=bottom, color=rc[q], label=f"Rank {q+1}", edgecolor="white", lw=.4); bottom += 100 * acc[:, q]
    ax.set_xticks(range(m)); ax.set_xticklabels([f"$\\omega_{i+1}$\n{ALTS[i]}" for i in range(m)]); ax.set_title(ttl, fontsize=8.5)
axs[0].set_ylabel(f"rank acceptability ({PCT})")
axs[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "rank_acceptability.png"), dpi=600); plt.close(fig)

# (g) reliability sweep
k1s = np.round(np.arange(0.10, 0.901, 0.05), 2)
sweep = {1: [], 2: []}
for k1 in k1s:
    kap = (float(k1), float(1 - k1)); A_ = encode(ALTS, kappa=kap)
    for p in (1, 2):
        sweep[p].append(go(A_, p, kappa=kap)["C"])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
for ax, p in [(axs[0], 1), (axs[1], 2)]:
    Y = np.array(sweep[p])
    for i in range(m):
        ax.plot(k1s, Y[:, i], color=COLS[i], lw=1.1, marker=mk[i], ms=3, label=ALAB[i])
    ax.axvline(0.5, color="grey", lw=.6, ls=":"); ax.grid(alpha=.3)
    ax.set_xlabel(r"reliability of the vote count $\kappa_1$ ($\kappa_2=1-\kappa_1$)")
    ax.set_title("Hamming ($p=1$)" if p == 1 else "Euclidean ($p=2$)", fontsize=8.5)
axs[0].set_ylabel(r"closeness $C_i$")
axs[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "kappa_sweep.png"), dpi=600); plt.close(fig)
res["kappa_grid"] = k1s.tolist()
res["kappa_sweep_orders"] = {str(p): [order(c, ALTS) for c in sweep[p]] for p in (1, 2)}
res["kappa_sweep_top"] = {str(p): sorted({order(c, ALTS)[0] for c in sweep[p]}) for p in (1, 2)}
res["kappa_sweep_bottom"] = {str(p): sorted({order(c, ALTS)[-1] for c in sweep[p]}) for p in (1, 2)}
for p in (1, 2):
    d = np.array([c[0] - c[1] for c in sweep[p]]); idx = np.where(np.diff(np.sign(d)) != 0)[0]
    res[f"kappa_swap_p{p}"] = [float(k1s[k] + (k1s[k + 1] - k1s[k]) * d[k] / (d[k] - d[k + 1])) for k in idx]

# (h) evidence view: the two channels and their mixture, long term, PRS vs JSA
j = 2
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
bpa = {}
for ax, i in [(axs[0], ALTS.index("PRS")), (axs[1], ALTS.index("JSA"))]:
    ms = source_bpas(T[i, j], F[i, j], np.array(KAPPA)); mix = mixture(ms, KAPPA)
    bars = [ms[0], ms[1], mix]; bottom = np.zeros(3)
    for q, (nm, col) in enumerate([("$m(\\{Y\\})$ support", "#4daf4a"), ("$m(\\{N\\})$ against", "#e41a1c"), ("$m(\\Theta)$ uncommitted", "#bdbdbd")]):
        vals = np.array([b[q] for b in bars])
        ax.bar(range(3), vals, bottom=bottom, color=col, edgecolor="white", label=nm, width=.6)
        for xx, (v, bt) in enumerate(zip(vals, bottom)):
            if v > 0.06: ax.text(xx, bt + v / 2, f"${v:.2f}$", ha="center", va="center", fontsize=7)
        bottom += vals
    ax.set_xticks(range(3)); ax.set_xticklabels(["vote count\n$m_1$", "effect size\n$m_2$", "mixture\n$\\bar m=\\Phi$"], fontsize=7.5)
    ax.set_title(f"$\\omega_{i+1}$ ({ALTS[i]}), long term", fontsize=8.5)
    bpa[ALTS[i]] = [list(map(float, ms[0])), list(map(float, ms[1])), list(map(float, mix))]
axs[0].set_ylabel("mass"); axs[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "evidence_bpa.png"), dpi=600); plt.close(fig)
res["bpa_LT"] = bpa

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)

# ===================================================================== console summary
np.set_printoptions(precision=4, suppress=True)
print("checks", {h: (round(v["mean"], 2), v["mean_t2"], round(v["share"], 1), v["share_t2"]) for h, v in chk.items()})
print("C1", np.round(B1["C"], 4), res["order1"]); print("C2", np.round(B2["C"], 4), res["order2"])
print("score", np.round(score, 4), "Lam", np.round(Lam, 4), "CPhi", np.round(CPhi, 4))
print("collapse err", res["collapse_err"], "decomp err", res["decomp_err"], "disc H/E", res["discordant_H_E"], "Phi/E", res["discordant_Phi_E"])
print("profile orders", res["profile_orders"]); print("opcmp", {k: (v["order"], np.round(v["score"],4)) for k, v in opcmp.items()}, res["prs_profile_F2_sep"], res["prs_profile_F2_convex"])
for x in wtab: print("W", x["name"][:30], x["o1"], x["s1"], "|", x["o2"], x["s2"])
for x in rob: print("R", x["name"][:45], x["o1"], x["s1"], "|", x["o2"], x["s2"])
print("M swap", res["M_swap_p1"], res["M_swap_p2"])
for k_ in res["smaa"]: print("SMAA", k_, round(res["smaa"][k_]["same"], 3), np.round(np.array(res["smaa"][k_]["acc"])[:, 0] * 100, 1))
for p in (1, 2): b = res[f"boot{p}"]; print("boot", p, np.round(b["first"], 3), np.round(b["meanrank"], 2), round(b["same"], 3), round(b["trn_over_jsa"], 3))
print("HE differ", res["smaa_HE_differ"]); print("rr", res["rr_max"]); print("with OTH", res["order2_with_other"])
for x in res["bench"]: print("B", x["name"][:40], x["o"], x["d1"], x["d2"])
print("kappa top/bottom", res["kappa_sweep_top"], res["kappa_sweep_bottom"], res["kappa_swap_p1"], res["kappa_swap_p2"])
print("bpa", bpa)
