"""
epp_review.py -- channel verdicts, practitioner baselines, adjusted contrasts and cost-effectiveness.

1. Plain-language channel verdicts by cell (conventional encoding): vote count '+' if the Wilson interval for the share of
   significantly positive estimates lies above 50%, '-' if below, '0' otherwise; effect size '+' if the interval for the
   mean effect lies above zero, '-' if below, '0' otherwise. A cell shows conflict when the verdicts differ.
2. Practitioner baselines: a significance-count rule, a standardised average of the point estimates, and the
   covariate-adjusted programme contrasts of Card, Kluve and Weber (NBER WP 21431, Table 5).
3. Cost-effectiveness: break-even cost ratios from the mean employment effects, and bootstrap probabilities that one
   programme is more cost-effective than another at given cost ratios.

Writes outputs/epp_review.json, outputs/tables/epp_r_*.tex and outputs/epp_review_macros.tex.
"""
import os, sys, json, csv, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epp_encoding import cs, encode_alt, closeness_w, ALTS, H, m, n, BETA, KAPPA, DEFF

OUT = cs.OUT; ROOT = cs.ROOT
res = {}
TRN, JSA, PRS, PSE = 0, 1, 2, 3
Z = cs.Z95
ACR = lambda idx: " $\\succ$ ".join(ALTS[i] for i in idx)

# ===================================================================== 1. channel verdicts (conventional encoding)
ver = {}
for a in ALTS:
    for h in H:
        r = cs.cell(a, h); p = float(r["share_sig_positive_pct"]) / 100
        L1, U1 = cs.wilson(p, max(int(r["n_sign"]) * DEFF, 1.0), Z)
        v = "+" if L1 > 0.5 else ("-" if U1 < 0.5 else "0")
        e = float(r["mean_effect_pp"]); se = cs.se_cell(a, h); lo, hi = e - Z * se, e + Z * se
        f = "+" if lo > 0 else ("-" if hi < 0 else "0")
        ver[f"{a}_{h}"] = dict(vote=v, effect=f, conflict=v != f, opposite={v, f} == {"+", "-"},
                               vote_int=[L1, U1], eff_int=[lo, hi])
res["verdicts"] = ver
n_conf = sum(x["conflict"] for x in ver.values()); n_opp = sum(x["opposite"] for x in ver.values())
res["n_conflict_cells"] = n_conf; res["n_opposite_cells"] = n_opp
SYM = {"+": "$+$", "-": "$-$", "0": "$0$"}
lines = []
for a in ALTS:
    cells = []
    for h in H:
        x = ver[f"{a}_{h}"]; s_ = f"{SYM[x['vote']]} / {SYM[x['effect']]}"
        cells.append(f"\\textbf{{{s_}}}" if x["conflict"] else s_)
    lines.append(f"{a} & " + " & ".join(cells) + " \\\\")
open(os.path.join(OUT, "tables", "epp_r_verdicts.tex"), "w").write("\n".join(lines) + "\n")

# ===================================================================== 2. practitioner baselines
share = np.array([[float(cs.cell(a, h)["share_sig_positive_pct"]) for h in H] for a in ALTS])
meanef = np.array([[float(cs.cell(a, h)["mean_effect_pp"]) for h in H] for a in ALTS])
se = np.array([[cs.se_cell(a, h) for h in H] for a in ALTS])
# (a) significance-count rule: +1 per horizon with an interval above zero, -1 below zero, weighted by beta
sig = np.where(meanef - Z * se > 0, 1, np.where(meanef + Z * se < 0, -1, 0))
score_sig = (BETA[None] * sig).sum(1)
# (b) standardised average of point estimates (each channel standardised over the 12 cells)
zs = (share - share.mean()) / share.std(); ze = (meanef - meanef.mean()) / meanef.std()
score_std = (BETA[None] * 0.5 * (zs + ze)).sum(1)
# (c) covariate-adjusted contrasts relative to training (CKW WP Table 5)
adj = {(r["term"], r["model"]): (float(r["coef"]), float(r["se"])) for r in csv.DictReader(open(os.path.join(ROOT, "data", "ckw2018_table5_adjusted.csv")))}
pooled = np.array([0.0] + [adj[(t, "pooled_controls")][0] for t in ("JSA", "PRS", "PSE")])
pooled_se = {t: adj[(t, "pooled_controls")][1] for t in ("JSA", "PRS", "PSE")}
hz = np.zeros((m, n))
for i, t in enumerate(("JSA", "PRS", "PSE"), start=1):
    b0 = adj[(t, "interactions_controls")][0]
    hz[i] = [b0, b0 + adj[(f"{t}_x_MT", "interactions_controls")][0], b0 + adj[(f"{t}_x_LT", "interactions_controls")][0]]
score_adj_h = (BETA[None] * hz).sum(1)
C_conv = closeness_w(encode_alt("base", "base"), BETA, 1)[0]
C_harm = closeness_w(encode_alt("power", "signed"), BETA, 1)[0]


def order_str(sc):
    o = np.argsort(-sc, kind="stable"); parts = [ALTS[o[0]]]
    for k in range(1, m):
        parts.append(("$=$" if np.isclose(sc[o[k]], sc[o[k - 1]]) else "$\\succ$")); parts.append(ALTS[o[k]])
    return " ".join(parts)


base_rows = [("Protocol, conventional encoding", C_conv),
             ("Protocol, evidence-of-harm encoding", C_harm),
             ("Significance-count rule (horizons with a significantly positive mean effect, minus negative)", score_sig),
             ("Standardised average of the two point estimates", score_std),
             ("Covariate-adjusted contrasts, pooled model (CKW Table 5, col.~2)", pooled),
             ("Covariate-adjusted contrasts by horizon, equal weights (CKW Table 5, col.~4)", score_adj_h)]
res["baselines"] = {nm: dict(order=order_str(sc), score=list(map(float, sc))) for nm, sc in base_rows}
open(os.path.join(OUT, "tables", "epp_r_baselines.tex"), "w").write(
    "\n".join(f"{nm} & {order_str(sc)} \\\\" for nm, sc in base_rows) + "\n")
res["adj_jsa_vs_trn"] = dict(coef=adj[("JSA", "pooled_controls")][0], se=adj[("JSA", "pooled_controls")][1])
res["sig_counts"] = {a: [int(x) for x in sig[i]] for i, a in enumerate(ALTS)}

# ===================================================================== 3. cost-effectiveness
Ebar = (BETA[None] * meanef).sum(1)                     # mean employment effect over horizons, percentage points
Ese = np.sqrt((BETA[None] ** 2 * se ** 2).sum(1))
res["Ebar"] = Ebar.tolist(); res["Ebar_se"] = Ese.tolist()
rng = np.random.default_rng(3)
D = rng.normal(meanef[None], se[None], size=(20000, m, n)); Eb = (BETA[None, None] * D).sum(2)
RATIOS = (1, 1.5, 2, 3, 5)
pairs = [(PRS, TRN), (PRS, JSA), (TRN, JSA), (PRS, PSE), (TRN, PSE), (JSA, PSE)]
ce = []
for i, l in pairs:
    be = Ebar[i] / Ebar[l] if Ebar[l] > 0 else None
    probs = [float((Eb[:, i] > r * Eb[:, l]).mean()) for r in RATIOS]
    ce.append(dict(i=ALTS[i], l=ALTS[l], ratio=be, probs=probs))
res["cost_effectiveness"] = ce
lines = []
for r_ in ce:
    lines.append(f"{r_['i']} over {r_['l']} & " + ("--" if r_["ratio"] is None else f"{r_['ratio']:.1f}") + " & " +
                 " & ".join(f"{100*p:.0f}" for p in r_["probs"]) + " \\\\")
open(os.path.join(OUT, "tables", "epp_r_costeff.tex"), "w").write("\n".join(lines) + "\n")
lines = [f"{a} & " + " & ".join(f"{meanef[i, j]:.1f}" for j in range(n)) + f" & {Ebar[i]:.2f} ({Ese[i]:.2f}) \\\\" for i, a in enumerate(ALTS)]
open(os.path.join(OUT, "tables", "epp_r_ebar.tex"), "w").write("\n".join(lines) + "\n")

json.dump(res, open(os.path.join(OUT, "epp_review.json"), "w"), indent=1, default=float)

# ===================================================================== macros
mac = []
def M(k, v): mac.append(f"\\newcommand{{\\{k}}}{{{v}}}")
M("nconflictcells", n_conf); M("noppositecells", n_opp)
for i, a in enumerate(ALTS):
    M(f"Ebar{a.lower()}", f"{Ebar[i]:.1f}")
for r_ in ce:
    tg = (r_["i"] + r_["l"]).lower()
    if r_["ratio"] is not None: M(f"ce{tg}", f"{r_['ratio']:.1f}")
    for r, p in zip(RATIOS, r_["probs"]):
        M(f"ce{tg}r{str(r).replace('.', 'p')}".replace("1p5", "onefive").replace("r1", "rone").replace("r2", "rtwo").replace("r3", "rthree").replace("r5", "rfive"), f"{100*p:.0f}")
M("adjjsacoef", f"{100*adj[('JSA','pooled_controls')][0]:.1f}"); M("adjjsase", f"{100*adj[('JSA','pooled_controls')][1]:.1f}")
M("adjprscoef", f"{100*adj[('PRS','pooled_controls')][0]:.1f}"); M("adjprsse", f"{100*adj[('PRS','pooled_controls')][1]:.1f}")
M("adjpsecoef", f"{100*adj[('PSE','pooled_controls')][0]:.1f}"); M("adjpsese", f"{100*adj[('PSE','pooled_controls')][1]:.1f}")
open(os.path.join(OUT, "epp_review_macros.tex"), "w").write("\n".join(mac) + "\n")
print("verdicts conflict cells", n_conf, "opposite", n_opp)
for nm, sc in base_rows: print(order_str(sc), "|", nm[:60])
print("Ebar", np.round(Ebar, 2), np.round(Ese, 2))
for r_ in ce: print(r_["i"], r_["l"], r_["ratio"], [round(p, 3) for p in r_["probs"]])
print("epp_review done;", len(mac), "macros")
