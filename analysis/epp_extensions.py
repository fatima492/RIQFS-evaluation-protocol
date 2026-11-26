"""
epp_extensions.py -- alternative encodings, decision table, pairwise probabilities and cost break-even.

1. Alternative, evaluation-oriented encodings of the two evidence channels
   (a) "power-aware" vote count: only significantly NEGATIVE estimates count against a programme;
       non-significant estimates become hesitancy. The share of significantly negative estimates by
       programme type is not published, so it is imputed from the horizon split of Table 2
       (sig. negative / (insignificant + sig. negative)) applied to the type's non-positive share.
   (b) "benefit/harm" effect-size encoding: membership = lower confidence bound of the gain,
       non-membership = lower confidence bound of the harm (both relative to M); a small, precisely
       estimated positive effect is then weak support, not evidence against.
   (c) both together ("evidence-of-harm" encoding).
2. A decision table for evaluators: reduction-based ranking, a separate conflict flag Lambda, and
   pairwise probabilities from the parametric bootstrap.
3. Costs: closed-form break-even cost gaps for the Hamming ranking when a cost criterion is added,
   with a numerical check for the Euclidean ranking.

Writes outputs/epp_results.json, outputs/tables/epp_*.tex, outputs/figures/pairwise.png,
outputs/figures/breakeven.png and outputs/epp_macros.tex.
"""
import os, sys, json, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "analysis"))
from epp_encoding import cs, neg_share, grades_alt, encode_alt, draw_cells   # imports case_study (runs the base case once)
from pipeline import run, ranks, order

OUT = cs.OUT; ALTS = cs.ALTS; H = cs.H; m, n = cs.m, cs.n
KAPPA = cs.KAPPA; BETA = cs.BETA; Z = cs.Z95; MB = cs.MBENCH
res = {}


def summary(A):
    T, F = A; t, f = T.sum(-1), F.sum(-1)
    x1, x2 = cs.go(A, 1), cs.go(A, 2)
    lam = ((T - T.mean(-1, keepdims=True)) ** 2).sum(-1) + ((F - F.mean(-1, keepdims=True)) ** 2).sum(-1)
    Lam = 0.5 * (BETA[None] * lam).sum(1)
    P = 0.25 * (BETA[None] * ((1 - t) ** 2 + f ** 2)).sum(1); N = 0.25 * (BETA[None] * (t ** 2 + (1 - f) ** 2)).sum(1)
    CPhi = np.sqrt(N) / (np.sqrt(P) + np.sqrt(N))
    hes = (BETA[None] * (1 - t - f)).sum(1)          # mean hesitancy (uncommitted evidence)
    return dict(C1=x1["C"], C2=x2["C"], Lam=Lam, lam=lam, CPhi=CPhi, hes=hes)


ENC = [("base", "base", "Conventional"),
       ("power", "base", "Only significant negatives count against (vote count)"),
       ("base", "signed", "Only evidence of harm counts against (effect size)"),
       ("power", "signed", "Evidence of harm (both changes)")]


def bootstrap(vote, effect, Bn=5000, seed=3):
    g_ = np.random.default_rng(seed); Cs = {1: [], 2: []}; Lams = []
    for _ in range(Bn):
        draws = draw_cells(g_)
        s = summary(encode_alt(vote, effect, draws))
        Cs[1].append(s["C1"]); Cs[2].append(s["C2"]); Lams.append(s["Lam"])
    out = {}
    for p in (1, 2):
        C = np.array(Cs[p]); rk = np.argsort(np.argsort(-C, 1), 1)
        out[p] = dict(first=(rk == 0).mean(0).tolist(), last=(rk == m - 1).mean(0).tolist(),
                      meanrank=(rk + 1).mean(0).tolist(),
                      pw=[[float((C[:, i] > C[:, l]).mean()) if i != l else float("nan") for l in range(m)] for i in range(m)])
    out["Lam_lo"] = np.percentile(np.array(Lams), 2.5, 0).tolist(); out["Lam_hi"] = np.percentile(np.array(Lams), 97.5, 0).tolist()
    return out


encres = []
for vote, effect, name in ENC:
    s = summary(encode_alt(vote, effect))
    b = bootstrap(vote, effect)
    encres.append(dict(vote=vote, effect=effect, name=name, C1=s["C1"].tolist(), C2=s["C2"].tolist(), Lam=s["Lam"].tolist(),
                       CPhi=s["CPhi"].tolist(), hes=s["hes"].tolist(), o1=order(s["C1"], ALTS), o2=order(s["C2"], ALTS),
                       s1=cs.discordant(s["C1"], cs.B1["C"]), s2=cs.discordant(s["C2"], cs.B2["C"]), boot=b))
    print(name, order(s["C1"], ALTS), order(s["C2"], ALTS), "PRS first %.3f / PSE last %.3f" % (b[1]["first"][2], b[1]["last"][3]),
          "TRN>JSA %.3f" % b[1]["pw"][0][1])
res["encodings"] = encres

# sanity: base encoding of this script reproduces the case-study base results
assert np.allclose(encres[0]["C2"], cs.B2["C"]) and np.allclose(encres[0]["C1"], cs.B1["C"])

# M sweep for the evidence-of-harm encoding: where does the middle pair swap?
grid = np.arange(5, 40.01, 0.5); dd = []
for Mb in grid:
    s = summary(encode_alt("power", "signed", M=float(Mb))); dd.append(float(s["C1"][0] - s["C1"][1]))
res["harm_M_grid"] = grid.tolist(); res["harm_M_trn_minus_jsa"] = dd
dd = np.array(dd); res["harm_M_trn_ahead_all"] = bool((dd > 0).all()); res["harm_M_trn_ahead_share"] = float((dd > 0).mean())

# ===================================================================== costs: break-even analysis
# Add a cost criterion with weight w and crisp grade <1-c, c> on each channel (c = cost index in [0,1],
# most expensive programme = 1). For the Hamming ranking with absolute references:
#   C_new = (1-w) C^H + w (1 - c)   (exactly), so i stays ahead of l iff  c_i - c_l < (1-w)/w (C^H_i - C^H_l).
C1 = cs.B1["C"]


def ham_new(c, w):
    return (1 - w) * C1 + w * (1 - np.asarray(c))


def euc_new(c, w):
    T, F = cs.A; k = np.asarray(KAPPA); c = np.asarray(c)
    Tn = np.concatenate([T, (k[None, :] * (1 - c)[:, None])[:, None, :]], 1)
    Fn = np.concatenate([F, (k[None, :] * c[:, None])[:, None, :]], 1)
    beta = np.r_[(1 - w) * BETA, w]
    return run([(Tn, Fn)], "B" * (n + 1), kappa=KAPPA, gamma=1.0, p=2, w=np.array([1.0]), beta=beta, ref="absolute")["C"]


W_LIST = [0.25, 0.5]
pairs = [(2, 1), (2, 0), (2, 3), (1, 0), (0, 3), (1, 3)]      # (i, l): max cost gap c_i - c_l that keeps i ahead
be = []
for i, l in pairs:
    row = dict(i=ALTS[i], l=ALTS[l], dC=float(C1[i] - C1[l]))
    for w in W_LIST:
        row[f"ham_{w}"] = float((1 - w) / w * (C1[i] - C1[l]))
        # Euclidean: other programme at cost 0, raise c_i until the order flips (bisection)
        lo_, hi_ = 0.0, 1.0
        f = lambda c_: euc_new([c_ if q == i else 0.0 for q in range(m)], w)
        base_sign = np.sign(f(0.0)[i] - f(0.0)[l])
        if base_sign != np.sign(C1[i] - C1[l]) or np.sign(f(1.0)[i] - f(1.0)[l]) == base_sign:
            row[f"euc_{w}"] = None
        else:
            for _ in range(60):
                mid = 0.5 * (lo_ + hi_)
                if np.sign(f(mid)[i] - f(mid)[l]) == base_sign: lo_ = mid
                else: hi_ = mid
            row[f"euc_{w}"] = 0.5 * (lo_ + hi_)
    be.append(row)
res["breakeven"] = be

# check of the closed form against a direct Hamming computation
_c = np.array([0.7, 0.2, 0.9, 1.0]); _w = 0.25
T, F = cs.A; k = np.asarray(KAPPA)
Tn = np.concatenate([T, (k[None, :] * (1 - _c)[:, None])[:, None, :]], 1); Fn = np.concatenate([F, (k[None, :] * _c[:, None])[:, None, :]], 1)
_direct = run([(Tn, Fn)], "B" * (n + 1), kappa=KAPPA, gamma=1.0, p=1, w=np.array([1.0]), beta=np.r_[(1 - _w) * BETA, _w], ref="absolute")["C"]
res["breakeven_closed_form_err"] = float(np.max(np.abs(_direct - ham_new(_c, _w))))
assert res["breakeven_closed_form_err"] < 1e-12

# which programme ranks first, over the cost index of PRS and JSA (TRN, PSE fixed at 1 = most expensive)
cg = np.linspace(0, 1, 201)
first_map = {}
for w in W_LIST:
    Mx = np.zeros((len(cg), len(cg)), int)
    for a_, cj in enumerate(cg):          # rows: JSA cost index
        for b_, cp in enumerate(cg):      # cols: PRS cost index
            Mx[a_, b_] = int(np.argmax(ham_new([1.0, cj, cp, 1.0], w)))
    first_map[w] = Mx
res["first_map_note"] = "Hamming; TRN and PSE at cost index 1; rows JSA cost, columns PRS cost"

# break-even gaps under the evidence-of-harm encoding (Hamming, closed form)
C1h = np.array(encres[3]["C1"])
res["breakeven_harm"] = [dict(i=ALTS[i], l=ALTS[l], dC=float(C1h[i] - C1h[l]),
                              **{f"ham_{w}": float((1 - w) / w * (C1h[i] - C1h[l])) for w in W_LIST}) for i, l in pairs]

# ===================================================================== decision table (base encoding)
base = encres[0]; harm = encres[3]
res["decision_table"] = dict(order_reduction=order(np.array(base["CPhi"]), ALTS))

# ===================================================================== tables
def Wt(name, s): open(os.path.join(OUT, "tables", name + ".tex"), "w").write(s)
LAB = {a: f"$\\omega_{{{i+1}}}$ {a}" for i, a in enumerate(ALTS)}
ostr = lambda o: "$" + " \\succ ".join(f"\\omega_{{{ALTS.index(a)+1}}}" for a in o) + "$"

# encodings table
lines = []
for e in encres:
    b = e["boot"]
    lines.append(f"{e['name']} & {ostr(e['o1'])} ({e['s1']}) & {ostr(e['o2'])} ({e['s2']}) & "
                 f"{100*b[1]['first'][2]:.0f} & {100*b[1]['last'][3]:.0f} & {100*b[1]['pw'][0][1]:.0f} \\\\")
Wt("epp_encodings", "\n".join(lines) + "\n")

# decision table: closeness, rank, bootstrap mean rank and first/last shares, uncommitted evidence, conflict with interval
# conflict is expressed as a percentage of its largest possible value (one channel full support, the other full opposition)
LAMMAX = 0.25 * float(np.sum(np.asarray(KAPPA) ** 2))
res["lam_max"] = LAMMAX
lines = []
for i, a in enumerate(ALTS):
    b = base["boot"][1]
    lines.append(f"{LAB[a]} & {base['C1'][i]:.3f} & {ranks(np.array(base['C1']))[i]} & {b['meanrank'][i]:.2f} & "
                 f"{100*b['first'][i]:.0f} & {100*b['last'][i]:.0f} & {base['hes'][i]:.2f} & {100*base['Lam'][i]/LAMMAX:.0f} & "
                 f"[{100*base['boot']['Lam_lo'][i]/LAMMAX:.0f}, {100*base['boot']['Lam_hi'][i]/LAMMAX:.0f}] \\\\")
Wt("epp_decision", "\n".join(lines) + "\n")

# pairwise probabilities (Hamming, base encoding)
pw = np.array(base["boot"][1]["pw"])
lines = []
for i, a in enumerate(ALTS):
    lines.append(f"{LAB[a]} & " + " & ".join("--" if i == l else f"{100*pw[i, l]:.0f}" for l in range(m)) + " \\\\")
Wt("epp_pairwise", "\n".join(lines) + "\n")

# break-even table
lines = []
fmt = lambda x: "rev." if x is None else f"{x:.2f}"
for r_ in be:
    i, l = ALTS.index(r_["i"]), ALTS.index(r_["l"])
    lines.append(f"$\\omega_{{{i+1}}}$ {r_['i']} over $\\omega_{{{l+1}}}$ {r_['l']} & {r_['dC']:.3f} & "
                 f"{r_['ham_0.25']:.2f} & {fmt(r_['euc_0.25'])} & {r_['ham_0.5']:.2f} & {fmt(r_['euc_0.5'])} \\\\")
Wt("epp_breakeven", "\n".join(lines) + "\n")

# encoded grades under the evidence-of-harm encoding
lines = []
for i, a in enumerate(ALTS):
    cells = []
    for h in H:
        g = grades_alt(a, h, "power", "signed")
        cells.append(f"$\\langle {g[0]:.2f},{g[1]:.2f}\\rangle$ & $\\langle {g[2]:.2f},{g[3]:.2f}\\rangle$")
    lines.append(f"$\\omega_{{{i+1}}}$ {a} & " + " & ".join(cells) + " \\\\")
Wt("epp_grades_harm", "\n".join(lines) + "\n")
res["neg_share_imputed"] = {f"{a}_{h}": neg_share(a, h, float(cs.cell(a, h)["share_sig_positive_pct"]) / 100) for a in ALTS for h in H}

# ===================================================================== figures
from plotstyle import apply as _style, PCT
_style()
COLS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
ALAB = [f"$\\omega_{i+1}$ ({a})" for i, a in enumerate(ALTS)]

# pairwise probability heat map, base and evidence-of-harm encodings
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, e, ttl in [(axs[0], base, "conventional encoding"), (axs[1], harm, "evidence-of-harm encoding")]:
    P_ = np.array(e["boot"][1]["pw"]); Pm = np.where(np.isnan(P_), 0.5, P_)
    im = ax.imshow(Pm, cmap="RdBu", vmin=0, vmax=1)
    for i in range(m):
        for l in range(m):
            ax.text(l, i, "--" if i == l else f"{100*P_[i, l]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if (i != l and abs(P_[i, l] - 0.5) > 0.35) else "black")
    ax.set_xticks(range(m)); ax.set_xticklabels(ALTS, fontsize=7.5); ax.set_yticks(range(m)); ax.set_yticklabels(ALAB, fontsize=7.5)
    ax.set_xlabel("ranked below"); ax.set_title(ttl, fontsize=8.5)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label(f"probability row ranked above column", fontsize=7.5)
fig.savefig(os.path.join(OUT, "figures", "pairwise.png"), dpi=600, bbox_inches="tight"); plt.close(fig)

# break-even map: first-ranked programme over the cost indices of JSA and PRS
from matplotlib.colors import ListedColormap
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
cmap = ListedColormap(COLS)
for ax, w in zip(axs, W_LIST):
    ax.imshow(first_map[w], origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", alpha=.8)
    ax.plot([0, 1], [0, 1], color="k", lw=.5, ls=":")
    ax.set_xlabel("cost index of PRS"); ax.set_title(f"cost weight $w={w}$", fontsize=8.5)
    present = sorted(set(first_map[w].ravel().tolist()))
    for q in present:
        ys, xs = np.where(first_map[w] == q)
        ax.text(cg[int(np.median(xs))], cg[int(np.median(ys))], f"{ALTS[q]} first", ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=.75, lw=0, pad=1.5))
axs[0].set_ylabel("cost index of JSA")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "breakeven.png"), dpi=600); plt.close(fig)

json.dump(res, open(os.path.join(OUT, "epp_results.json"), "w"), indent=1, default=float)

# ===================================================================== macros quoted in the text
mac = []
def M(k, v): mac.append(f"\\newcommand{{\\{k}}}{{{v}}}")
pc = lambda x: f"{100*x:.0f}"   # whole percentages: the probabilities are approximate
tags = ["base", "power", "signed", "harm"]
for tg, e in zip(tags, encres):
    b = e["boot"][1]
    M(f"enc{tg}prsfirst", pc(b["first"][2])); M(f"enc{tg}pselast", pc(b["last"][3])); M(f"enc{tg}trnjsa", pc(b["pw"][0][1]))
M("pwprsjsa", pc(pw[2, 1])); M("pwprstrn", pc(pw[2, 0])); M("pwtrnjsa", pc(pw[0, 1])); M("pwjsapse", pc(pw[1, 3])); M("pwtrnpse", pc(pw[0, 3]))
M("pwprspse", pc(pw[2, 3]))
for r_ in be:
    tg = (r_["i"] + r_["l"]).lower()
    M(f"be{tg}q", f"{r_['ham_0.25']:.2f}"); M(f"be{tg}h", f"{r_['ham_0.5']:.2f}")
M("harmMshare", pc(res["harm_M_trn_ahead_share"]))
for r_ in res["breakeven_harm"]:
    tg = (r_["i"] + r_["l"]).lower()
    M(f"beh{tg}q", f"{r_['ham_0.25']:.2f}"); M(f"beh{tg}h", f"{r_['ham_0.5']:.2f}")
for tg, e in zip(tags, encres):
    for i, a in enumerate(ALTS):
        M(f"C{tg}{a.lower()}", f"{e['C1'][i]:.3f}"); M(f"Lam{tg}{a.lower()}", f"{e['Lam'][i]:.4f}")
M("negshareST", f"{100*float(cs.T2['ST']['pct_sig_negative'])/(float(cs.T2['ST']['pct_insignificant'])+float(cs.T2['ST']['pct_sig_negative'])):.0f}")
M("negshareMT", f"{100*float(cs.T2['MT']['pct_sig_negative'])/(float(cs.T2['MT']['pct_insignificant'])+float(cs.T2['MT']['pct_sig_negative'])):.0f}")
M("negshareLT", f"{100*float(cs.T2['LT']['pct_sig_negative'])/(float(cs.T2['LT']['pct_insignificant'])+float(cs.T2['LT']['pct_sig_negative'])):.0f}")
open(os.path.join(OUT, "epp_macros.tex"), "w").write("\n".join(mac) + "\n")
print("break-even", json.dumps(be, indent=0))
print("extensions done;", len(mac), "macros")
