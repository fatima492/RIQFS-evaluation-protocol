"""
epp_compare.py -- comparative analysis (Sections 5.7 and 5.8).

1. Probability baselines. Every comparator is recomputed on the same 5,000 cluster-adjusted bootstrap draws as the
   protocol, so that methods are compared on probabilities rather than on a single rank order:
   protocol (both encodings), mean effect only, vote count only, a significance-count rule and a standardised average
   of the two point estimates. The covariate-adjusted contrasts of Card, Kluve and Weber are compared with independent
   normal draws of the pooled coefficients (their covariances are not published).
2. Established methods, like for like: IFWA, Szmidt-Kacprzyk TOPSIS, VIKOR, dual hesitant TOPSIS and Dempster's rule,
   applied to the same encoded data as the protocol under both encodings, all with fixed (not data-driven) references.
3. Simulation with a known ranking: synthetic meta-analytic summaries are generated from evaluations with known true
   effects in four scenarios, and each method's success in recovering the true ranking is recorded.

Writes outputs/epp_compare.json, outputs/tables/epp_c_*.tex, outputs/epp_compare_macros.tex.
"""
import os, sys, json, csv, itertools
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epp_encoding import cs, encode_alt, closeness_w, draw_cells, ALTS, H, m, n, BETA, KAPPA, DEFF

OUT, ROOT = cs.OUT, cs.ROOT
Z = cs.Z95
TRN, JSA, PRS, PSE = 0, 1, 2, 3
PROT_ORDER = (PRS, TRN, JSA, PSE)
res = {}
pc = lambda x: f"{100*x:.0f}"


def rank_stats(S, rng):
    """S: draws x alternatives scores. Ties are broken at random, so tied methods get fractional credit."""
    S = np.asarray(S, float) + 1e-9 * rng.standard_normal(np.shape(S))
    o = np.argsort(-S, 1)
    return dict(prs_first=float((o[:, 0] == PRS).mean()), pse_last=float((o[:, -1] == PSE).mean()),
                trn_jsa=float((S[:, TRN] > S[:, JSA]).mean()),
                order=float((o == np.array(PROT_ORDER)[None]).all(1).mean()))


def sig_rule(meanef, se):
    return (BETA[None] * np.where(meanef - Z * se > 0, 1, np.where(meanef + Z * se < 0, -1, 0))).sum(-1)


def std_avg(share, meanef):
    zs = (share - share.mean()) / share.std(); ze = (meanef - meanef.mean()) / meanef.std()
    return (BETA[None] * 0.5 * (zs + ze)).sum(-1)


# ===================================================================== 1. probability baselines on common draws
SE = np.array([[cs.se_cell(a, h) for h in H] for a in ALTS])
SH0 = np.array([[float(cs.cell(a, h)["share_sig_positive_pct"]) / 100 for h in H] for a in ALTS])
ME0 = np.array([[float(cs.cell(a, h)["mean_effect_pp"]) for h in H] for a in ALTS])
rng = np.random.default_rng(3)                       # same seed and draw order as the base-case bootstrap (Section 5.1)
sc = {k: [] for k in ("conv", "harm", "mean", "vote", "sig", "std")}
for _ in range(5000):
    d = draw_cells(rng)
    sh = np.array([[d[i][j][0] for j in range(n)] for i in range(m)]); me = np.array([[d[i][j][1] for j in range(n)] for i in range(m)])
    sc["conv"].append(closeness_w(encode_alt("base", "base", draws=d), BETA, 1)[0])
    sc["harm"].append(closeness_w(encode_alt("power", "signed", draws=d), BETA, 1)[0])
    sc["mean"].append((BETA[None] * me).sum(1)); sc["vote"].append((BETA[None] * sh).sum(1))
    sc["sig"].append(sig_rule(me, SE)); sc["std"].append(std_avg(sh, me))
r2 = np.random.default_rng(5)
probs = {k: rank_stats(np.array(v), r2) for k, v in sc.items()}
# adjusted pooled contrasts (CKW WP Table 5, col. 2): independent normal draws, training = 0
adj = {(r["term"], r["model"]): (float(r["coef"]), float(r["se"])) for r in csv.DictReader(open(os.path.join(ROOT, "data", "ckw2018_table5_adjusted.csv")))}
mu = np.array([0.0] + [adj[(t, "pooled_controls")][0] for t in ("JSA", "PRS", "PSE")])
sd = np.array([0.0] + [adj[(t, "pooled_controls")][1] for t in ("JSA", "PRS", "PSE")])
probs["adj"] = rank_stats(r2.normal(mu, sd, size=(20000, m)), r2)
res["probs"] = probs
LAB = [("conv", "Protocol, conventional encoding", "both channels"),
       ("harm", "Protocol, evidence-of-harm encoding", "both channels"),
       ("mean", "Mean effect only", "effect size"),
       ("vote", "Share of significantly positive estimates only", "vote count"),
       ("std", "Standardised average of the two point estimates", "both channels"),
       ("sig", "Significance-count rule (illustrative)", "effect size, significance only"),
       ("adj", "Covariate-adjusted contrasts, pooled model", "effect size, adjusted")]
lines = [f"{nm} & {src} & {pc(probs[k]['prs_first'])} & {pc(probs[k]['pse_last'])} & {pc(probs[k]['trn_jsa'])} & {pc(probs[k]['order'])} \\\\"
         for k, nm, src in LAB]
open(os.path.join(OUT, "tables", "epp_c_probs.tex"), "w").write("\n".join(lines) + "\n")
for k, nm, _ in LAB:
    print(f"{nm[:45]:45s}", {kk: round(v, 3) for kk, v in probs[k].items()})

# ===================================================================== 2. established methods, like for like
def methods(A, kappa=KAPPA):
    T, F = A; k = np.asarray(kappa); t, f = T.sum(-1), F.sum(-1); pi = 1 - t - f
    out = {}
    # IFWA (algebraic operations) and score
    tt = 1 - np.prod((1 - t) ** BETA[None], 1); ff = np.prod(np.where(f > 0, f, 1e-12) ** BETA[None], 1)
    out["IFWA aggregation and score"] = tt - ff
    # Szmidt-Kacprzyk distance TOPSIS with fixed references <1,0> and <0,1>
    dp = 0.5 * (BETA[None] * (np.abs(t - 1) + np.abs(f - 0) + np.abs(pi - 0))).sum(1)
    dn = 0.5 * (BETA[None] * (np.abs(t - 0) + np.abs(f - 1) + np.abs(pi - 0))).sum(1)
    out["IF-TOPSIS, Szmidt--Kacprzyk distance"] = dn / (dp + dn)
    # VIKOR on intuitionistic scores with fixed best (+1) and worst (-1)
    s = t - f; reg = (BETA[None] * (1 - s) / 2); Sg, Rg = reg.sum(1), reg.max(1)
    Q = 0.5 * (Sg - Sg.min()) / max(Sg.max() - Sg.min(), 1e-12) + 0.5 * (Rg - Rg.min()) / max(Rg.max() - Rg.min(), 1e-12)
    out["VIKOR on intuitionistic scores"] = -Q
    # dual hesitant TOPSIS on the sorted channel grades, fixed references h=1,g=0 and h=0,g=1
    hh = np.sort(T / k, -1); gg = np.sort(F / k, -1)
    dist = lambda h1, g1, h2, g2: np.sqrt(0.5 * (BETA[None] * (((h1 - h2) ** 2).mean(-1) + ((g1 - g2) ** 2).mean(-1))).sum(1))
    Dp, Dn = dist(hh, gg, 1.0, 0.0), dist(hh, gg, 0.0, 1.0)
    out["Dual hesitant TOPSIS, sorted channels"] = Dn / (Dp + Dn)
    # Dempster's rule on the reliability-discounted channels
    y1, n1 = T[..., 0], F[..., 0]; y2, n2 = T[..., 1], F[..., 1]; th1, th2 = 1 - y1 - n1, 1 - y2 - n2
    K = y1 * n2 + n1 * y2; Y = (y1 * y2 + y1 * th2 + th1 * y2) / (1 - K); N = (n1 * n2 + n1 * th2 + th1 * n2) / (1 - K)
    out["Dempster's rule on discounted channels"] = (BETA[None] * (Y - N)).sum(1)
    return out


def disc(a, b):
    return int(sum(np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]) for i, j in itertools.combinations(range(m), 2)))


ordstr = lambda sc_: "$" + " \\succ ".join("\\mathrm{" + ALTS[i] + "}" for i in np.argsort(-sc_)) + "$"
bench = []
for enc, kw in (("conv", dict(vote="base", effect="base")), ("harm", dict(vote="power", effect="signed"))):
    A = encode_alt(**kw); Cp = closeness_w(A, BETA, 1)[0]
    for nm, s_ in methods(A).items():
        bench.append(dict(enc=enc, name=nm, order=ordstr(s_), d=disc(s_, Cp)))
res["bench"] = bench
lines = []
names = list(dict.fromkeys(b["name"] for b in bench))
for nm in names:
    bc = [b for b in bench if b["name"] == nm and b["enc"] == "conv"][0]; bh = [b for b in bench if b["name"] == nm and b["enc"] == "harm"][0]
    lines.append(f"{nm} & {bc['order']} ({bc['d']}) & {bh['order']} ({bh['d']}) \\\\")
open(os.path.join(OUT, "tables", "epp_c_bench.tex"), "w").write("\n".join(lines) + "\n")
res["bench_all_agree"] = all(b["d"] == 0 for b in bench)
print("benchmarks:", [(b["enc"], b["name"][:20], b["d"]) for b in bench])

# ===================================================================== 3. simulation with a known ranking
NE_FRAC = 352 / 857          # share of classified estimates that also report an employment effect (as in CKW)
TAU = 4.0                    # between-evaluation heterogeneity of true effects (percentage points)
M = cs.MBENCH


def wilson(p, nn, z=Z):
    d = 1 + z * z / nn; c = (p + z * z / (2 * nn)) / d; hw = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
    return np.clip(c - hw, 0, 1), np.clip(c + hw, 0, 1)


def synth(theta, se_study, n_c, rng):
    """One synthetic meta-analysis: per cell, n_c evaluations with true effects theta + u, standard errors around se_study;
    returns share significantly positive, share significantly negative, mean effect (on the effect subset) and its se."""
    shp = np.zeros((m, n)); shn = np.zeros((m, n)); me = np.zeros((m, n)); se = np.zeros((m, n)); ne = np.zeros((m, n), int)
    for i in range(m):
        for j in range(n):
            k = int(n_c[i][j]); s_i = se_study[i] * np.exp(0.3 * rng.standard_normal(k))
            y = theta[i][j] + TAU * rng.standard_normal(k) + s_i * rng.standard_normal(k)
            zst = y / s_i; shp[i, j] = (zst > Z).mean(); shn[i, j] = (zst < -Z).mean()
            ke = max(2, int(round(NE_FRAC * k))); ye = y[:ke]
            me[i, j] = ye.mean(); se[i, j] = ye.std(ddof=1) / np.sqrt(ke); ne[i, j] = ke
    return shp, shn, me, se


def encode_sim(shp, shn, me, se, n_c, enc, kappa=KAPPA):
    k = np.asarray(kappa); T = np.zeros((m, n, 2)); F = np.zeros_like(T)
    # imputed negative shares as in the case study: horizon split of non-positive estimates, pooled over programmes
    for j in range(n):
        w = n_c[:, j]; pneg = (shn[:, j] * w).sum() / w.sum(); pins = ((1 - shp[:, j] - shn[:, j]) * w).sum() / w.sum()
        frac = pneg / max(pneg + pins, 1e-12)
        for i in range(m):
            L1, U1 = wilson(shp[i, j], n_c[i, j]); lo, hi = me[i, j] - Z * se[i, j], me[i, j] + Z * se[i, j]
            if enc == "conv":
                gs = lambda x: min(max((x + M) / (2 * M), 0.0), 1.0)
                mu1, nu1, mu2, nu2 = L1, 1 - U1, gs(lo), 1 - gs(hi)
            else:
                Lq, _ = wilson((1 - shp[i, j]) * frac, n_c[i, j])
                g = lambda x: min(max(x / M, 0.0), 1.0)
                mu1, nu1, mu2, nu2 = L1, Lq, g(lo), g(-hi)
            T[i, j] = k * [mu1, mu2]; F[i, j] = k * [nu1, nu2]
    return T, F


CK_N = np.array([[int(cs.cell(a, h)["n_sign"]) for h in H] for a in ALTS])
SCEN = {
    "equal": dict(label="Equal power", theta=[[3, 7, 9], [2, 4, 3], [5, 9, 12], [0, 1, 0]], se=[4, 4, 4, 4], n=[[60, 50, 25]] * 4,
                  pair=(0, 1)),
    "power": dict(label="Power differs (vote-count trap)", theta=[[3, 6, 7], [2, 3, 3], [5, 8, 11], [0, 0, 0]], se=[9, 1.5, 6, 6],
                  n=[[60, 50, 25]] * 4, pair=(0, 1)),
    "thin": dict(label="Few effect estimates (effect-size trap)", theta=[[3, 5, 6], [2, 4, 4], [6, 9, 12], [0, 1, 1]], se=[3, 3, 3, 3],
                 n=[[30, 25, 12], [30, 25, 12], [30, 25, 12], [12, 10, 5]], pair=(0, 1)),
    "ckw": dict(label="Calibrated to the case study", theta=ME0.tolist(), se=None, n=CK_N.tolist(), pair=(0, 1)),
}


def expected_share(theta, s):
    sd_ = np.sqrt(TAU ** 2 + s ** 2)
    return 1 - norm.cdf((Z * s - theta) / sd_)


# calibrate the study standard errors of the CKW-like scenario so that expected shares match the published ones
cal = []
for i in range(m):
    grid = np.linspace(0.5, 20, 400)
    err = [((expected_share(np.array(ME0[i]), s_) - SH0[i]) ** 2).sum() for s_ in grid]
    cal.append(float(grid[int(np.argmin(err))]))
SCEN["ckw"]["se"] = cal
res["sim_ckw_se"] = cal

KEFF = (0.25, 0.75)          # protocol variant that gives the effect-size channel three times the reliability of the vote count
METH = [("conv", "Protocol, conventional"), ("harm", "Protocol, evidence of harm"),
        ("harmeff", "Protocol, evidence of harm, $\\kappa=(0.25,0.75)$"), ("mean", "Mean effect only"),
        ("vote", "Vote count only"), ("std", "Standardised average"), ("sig", "Significance-count rule")]
sim = {}
R = 2000
for key, sp in SCEN.items():
    rng = np.random.default_rng(17)
    theta = np.array(sp["theta"], float); n_c = np.array(sp["n"]); tbar = (BETA[None] * theta).sum(1)
    true_order = np.argsort(-tbar); a_, b_ = sp["pair"]
    hits = {k: dict(top=0, bottom=0, pair=0, order=0) for k, _ in METH}
    flag = 0; flag_opp = 0                     # runs in which the channel verdicts for programme a_ differ / point in opposite directions
    for _ in range(R):
        shp, shn, me, se = synth(theta, sp["se"], n_c, rng)
        sco = {"conv": closeness_w(encode_sim(shp, shn, me, se, n_c, "conv"), BETA, 1)[0],
               "harm": closeness_w(encode_sim(shp, shn, me, se, n_c, "harm"), BETA, 1)[0],
               "harmeff": closeness_w(encode_sim(shp, shn, me, se, n_c, "harm", KEFF), BETA, 1, KEFF)[0],
               "mean": (BETA[None] * me).sum(1), "vote": (BETA[None] * shp).sum(1),
               "std": std_avg(shp, me), "sig": sig_rule(me, se)}
        for j in range(n):
            L1, U1 = wilson(shp[a_, j], n_c[a_, j]); v = "+" if L1 > 0.5 else ("-" if U1 < 0.5 else "0")
            e = "+" if me[a_, j] - Z * se[a_, j] > 0 else ("-" if me[a_, j] + Z * se[a_, j] < 0 else "0")
            if v != e:
                flag += 1; break
        for j in range(n):
            L1, U1 = wilson(shp[a_, j], n_c[a_, j]); v = "+" if L1 > 0.5 else ("-" if U1 < 0.5 else "0")
            e = "+" if me[a_, j] - Z * se[a_, j] > 0 else ("-" if me[a_, j] + Z * se[a_, j] < 0 else "0")
            if {v, e} == {"+", "-"}:
                flag_opp += 1; break
        for k, s_ in sco.items():
            s_ = s_ + 1e-9 * rng.standard_normal(m); o = np.argsort(-s_)
            hits[k]["top"] += o[0] == true_order[0]; hits[k]["bottom"] += o[-1] == true_order[-1]
            hits[k]["pair"] += (s_[a_] > s_[b_]) == (tbar[a_] > tbar[b_]); hits[k]["order"] += (o == true_order).all()
    sim[key] = dict(label=sp["label"], tbar=tbar.tolist(), se=list(map(float, sp["se"])),
                    acc={k: {kk: v / R for kk, v in h.items()} for k, h in hits.items()}, flag=flag / R, flag_opp=flag_opp / R)
    print(key, "flag", round(flag / R, 2), "opp", round(flag_opp / R, 2), {k: {kk: round(v, 2) for kk, v in a.items()} for k, a in sim[key]["acc"].items()})
res["sim"] = sim
lines = []
for key in SCEN:
    s_ = sim[key]
    for j, (k, nm) in enumerate(METH):
        a = s_["acc"][k]
        lines.append(f"{s_['label'] if j == 0 else ''} & {nm} & {pc(a['top'])} & {pc(a['bottom'])} & {pc(a['pair'])} & {pc(a['order'])} \\\\")
    if key != list(SCEN)[-1]:
        lines.append("\\midrule")
open(os.path.join(OUT, "tables", "epp_c_sim.tex"), "w").write("\n".join(lines) + "\n")
# scenario design table
lines = []
for key, sp in SCEN.items():
    th = np.array(sp["theta"]); tb = (BETA[None] * th).sum(1)
    lines.append(f"{sp['label']} & " + " & ".join(f"{tb[i]:.1f}" for i in range(m)) + " & " + " & ".join(f"{s_:.1f}" for s_ in sp["se"]) + " & " +
                 " & ".join(str(int(np.array(sp['n'])[i].sum())) for i in range(m)) + " \\\\")
open(os.path.join(OUT, "tables", "epp_c_simdesign.tex"), "w").write("\n".join(lines) + "\n")

json.dump(res, open(os.path.join(OUT, "epp_compare.json"), "w"), indent=1, default=float)

# ===================================================================== macros
mac = []
def Mm(k, v): mac.append(f"\\newcommand{{\\{k}}}{{{v}}}")
for key in SCEN:
    Mm(f"s{key}flag", pc(sim[key]["flag"])); Mm(f"s{key}flagopp", pc(sim[key]["flag_opp"]))
for k, tg in (("conv", "conv"), ("harm", "harm"), ("mean", "mean"), ("vote", "vote"), ("sig", "sig"), ("std", "std"), ("adj", "adj")):
    p_ = probs[k]
    Mm(f"cp{tg}prs", pc(p_["prs_first"])); Mm(f"cp{tg}pse", pc(p_["pse_last"])); Mm(f"cp{tg}trn", pc(p_["trn_jsa"])); Mm(f"cp{tg}ord", pc(p_["order"]))
for key in SCEN:
    for k, _ in METH:
        a = sim[key]["acc"][k]
        for kk in ("top", "bottom", "pair", "order"):
            Mm(f"s{key}{k}{kk}", pc(a[kk]))
open(os.path.join(OUT, "epp_compare_macros.tex"), "w").write("\n".join(mac) + "\n")
print("epp_compare done;", len(mac), "macros")
