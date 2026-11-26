"""
epp_weights.py -- criterion-weight sensitivity (Section 5.3) and robustness of the
evidence-of-harm encoding (Section 5.4).

1. Exact weight space. The Hamming closeness is linear in the criterion weights, so the region of the weight
   simplex in which a programme ranks first is a polygon. We evaluate the ranking on a fine triangular grid
   (step 1/400), so shares of the weight space are area fractions, and find the smallest long-term weight that
   guarantees a result for every split of the remaining weight.
2. Ordinal weight information: long term >= medium term >= short term, and the reverse.
3. Concentrated weights around equal weights, Dirichlet(c * beta) for several concentrations c.
4. Joint uncertainty: parametric bootstrap of the published statistics combined with random weights.
5. Weights proportional to the numbers of estimates per horizon.
6. Robustness of the evidence-of-harm encoding: channel reliabilities, imputed negative shares, standard errors,
   benchmark M, the questionable public-sector short-term mean.

Writes outputs/epp_weights.json, outputs/tables/epp_w_*.tex, outputs/figures/weight_space.png and
outputs/epp_weights_macros.tex.
"""
import os, sys, json, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epp_encoding import cs, encode_alt, closeness_w, draw_cells, ALTS, H, m, n, KAPPA, BETA

OUT = cs.OUT
res = {}
TRN, JSA, PRS, PSE = 0, 1, 2, 3
ENC = {"conv": dict(vote="base", effect="base"), "harm": dict(vote="power", effect="signed")}

# ===================================================================== 1-2. exact weight space
STEP = 400
g = np.arange(STEP + 1) / STEP
WG = np.array([(a, b, 1 - a - b) for a in g for b in g if a + b <= 1 + 1e-12]); WG = np.clip(WG, 0, 1)
REG = {"all": np.ones(len(WG), bool),
       "lt_first": (WG[:, 2] >= WG[:, 1] - 1e-12) & (WG[:, 1] >= WG[:, 0] - 1e-12),
       "st_first": (WG[:, 0] >= WG[:, 1] - 1e-12) & (WG[:, 1] >= WG[:, 2] - 1e-12)}


def shares(C):
    o = np.argsort(-C, 1)
    return dict(prs_first=float((o[:, 0] == PRS).mean()), pse_last=float((o[:, -1] == PSE).mean()),
                trn_jsa=float((C[:, TRN] > C[:, JSA]).mean()),
                first={a: float((o[:, 0] == i).mean()) for i, a in enumerate(ALTS)},
                last={a: float((o[:, -1] == i).mean()) for i, a in enumerate(ALTS)})


def min_lt(C, cond):
    """Smallest long-term weight w such that cond holds for every weight vector with beta_3 >= w."""
    ok = cond(C)
    for w in np.arange(0, 1.0001, 0.01):
        if ok[WG[:, 2] >= w - 1e-12].all():
            return round(float(w), 2)
    return None


space = {}
for enc, kw in ENC.items():
    for flip in (False, True):
        A = encode_alt(**kw, pse_flip=flip)
        for p in (1, 2):
            C = closeness_w(A, WG, p)
            key = f"{enc}{'_flip' if flip else ''}_p{p}"
            space[key] = {r: shares(C[msk]) for r, msk in REG.items()}
            space[key]["min_lt_prs_first"] = min_lt(C, lambda C_: np.argmax(C_, 1) == PRS)
            space[key]["min_lt_pse_last"] = min_lt(C, lambda C_: np.argmin(C_, 1) == PSE)
            space[key]["n_orders"] = len({tuple(o) for o in np.argsort(-C, 1)})
res["space"] = space
for k, v in space.items():
    print(k, {r: (round(v[r]["prs_first"], 3), round(v[r]["pse_last"], 3), round(v[r]["trn_jsa"], 3)) for r in REG},
          "minLT PRS first", v["min_lt_prs_first"], "minLT PSE last", v["min_lt_pse_last"])

# ===================================================================== 3. concentrated weights
conc = {}
for enc, kw in ENC.items():
    A = encode_alt(**kw)
    for c in (3, 10, 30, 100):
        W = np.random.default_rng(11).dirichlet(c * BETA, size=20000)
        conc[f"{enc}_{c}"] = shares(closeness_w(A, W, 1))
res["concentration"] = conc

# ===================================================================== 4. joint: sampling x weights
def weights_sampler(kind, rng):
    if kind == "equal":
        return BETA.copy()
    w = rng.dirichlet(np.ones(n))
    if kind == "lt_first":
        return np.sort(w)                      # ascending: short <= medium <= long
    if kind == "st_first":
        return np.sort(w)[::-1]
    return w


joint = {}
for enc, kw in ENC.items():
    for kind in ("equal", "all", "lt_first", "st_first"):
        rng = np.random.default_rng(3); Cs = []   # seed 3: the equal-weight row reproduces the base-case bootstrap (Section 5.1)
        for _ in range(5000):
            A = encode_alt(**kw, draws=draw_cells(rng)); w = weights_sampler(kind, rng)
            Cs.append(closeness_w(A, w, 1)[0])
        joint[f"{enc}_{kind}"] = shares(np.array(Cs))
res["joint"] = joint
for k, v in joint.items():
    print("joint", k, round(v["prs_first"], 3), round(v["pse_last"], 3), round(v["trn_jsa"], 3))

# ===================================================================== 5. weights proportional to estimates
n_h = np.array([sum(int(cs.cell(a, h)["n_sign"]) for a in ALTS) for h in H], float)
w_est = n_h / n_h.sum()
res["w_est"] = w_est.tolist()
res["w_est_orders"] = {f"{enc}_p{p}": [ALTS[i] for i in np.argsort(-closeness_w(encode_alt(**kw), w_est, p)[0])]
                       for enc, kw in ENC.items() for p in (1, 2)}
print("weights prop. to estimates", np.round(w_est, 3), res["w_est_orders"])

# ===================================================================== 6. robustness of the evidence-of-harm encoding
A_h = encode_alt(**ENC["harm"])
C_h1, C_h2 = closeness_w(A_h, BETA, 1)[0], closeness_w(A_h, BETA, 2)[0]
# share of each programme's |balance| contributed by the vote-count channel
T_, F_ = A_h
bal = np.abs(T_ - F_).sum(1)
res["harm_vote_share"] = (bal[:, 0] / bal.sum(1)).tolist()


def disc(C1, C2):
    return int(sum(np.sign(C1[i] - C1[j]) != np.sign(C2[i] - C2[j]) for i, j in itertools.combinations(range(m), 2)))


def ostr(C):
    return "$" + " \\succ ".join(f"\\omega_{{{i+1}}}" for i in np.argsort(-C)) + "$"


def boot_harm(Bn=5000, seed=3, **kw):   # same draws as the base-case bootstrap (Section 5.1)
    rng = np.random.default_rng(seed); Cs = []
    for _ in range(Bn):
        A = encode_alt(**ENC["harm"], draws=draw_cells(rng), **kw)
        Cs.append(closeness_w(A, BETA, 1, kw.get("kappa", KAPPA))[0])
    return shares(np.array(Cs))


VAR = [("Evidence of harm (base: $\\kappa=(0.5,0.5)$, $M=20$, imputed negative shares)", {}),
       ("Imputed negative shares $\\times0.5$", dict(neg_mult=0.5)),
       ("Imputed negative shares $\\times2$", dict(neg_mult=2.0)),
       ("Effect-size standard errors $\\times1.5$", dict(se_mult=1.5)),
       ("Benchmark effect $M=10$ points", dict(M=10.0)),
       ("Benchmark effect $M=30$ points", dict(M=30.0)),
       ("Public-sector short-term mean $-3.6$", dict(pse_flip=True)),
       ("$\\kappa=(0.3,0.7)$: more weight on the effect-size channel", dict(kappa=(0.3, 0.7))),
       ("$\\kappa=(0.1,0.9)$: effect-size channel dominates", dict(kappa=(0.1, 0.9))),
       ("$\\kappa=(0.7,0.3)$: more weight on the vote-count channel", dict(kappa=(0.7, 0.3)))]
rob = []
for name, kw in VAR:
    kap = kw.get("kappa", KAPPA)
    A = encode_alt(**ENC["harm"], **kw)
    c1, c2 = closeness_w(A, BETA, 1, kap)[0], closeness_w(A, BETA, 2, kap)[0]
    b = boot_harm(**kw)
    rob.append(dict(name=name, o1=ostr(c1), o2=ostr(c2), s1=disc(c1, C_h1), s2=disc(c2, C_h2),
                    prs_first=b["prs_first"], pse_last=b["pse_last"], trn_jsa=b["trn_jsa"]))
    print("harm variant", name[:40], [ALTS[i] for i in np.argsort(-c1)], round(b["prs_first"], 3), round(b["trn_jsa"], 3))
res["harm_robust"] = rob
# full kappa sweep under the evidence-of-harm encoding
ks = np.round(np.arange(0.05, 0.951, 0.05), 2); sweep = []
for k1 in ks:
    kap = (float(k1), float(1 - k1)); A = encode_alt(**ENC["harm"], kappa=kap)
    sweep.append([ALTS[i] for i in np.argsort(-closeness_w(A, BETA, 1, kap)[0])])
res["harm_kappa_sweep"] = dict(k1=ks.tolist(), orders=sweep)
res["harm_kappa_prs_first_all"] = all(o[0] == "PRS" for o in sweep)
res["harm_kappa_trn_over_jsa_all"] = all(o.index("TRN") < o.index("JSA") for o in sweep)
print("harm kappa sweep", res["harm_kappa_prs_first_all"], res["harm_kappa_trn_over_jsa_all"], sweep[0], sweep[-1])
# benchmark M under the evidence-of-harm encoding: is PRS always first?
Ms = np.arange(5, 40.01, 0.5)
res["harm_M_prs_first_all"] = all(int(np.argmax(closeness_w(encode_alt(**ENC["harm"], M=float(M_)), BETA, 1)[0])) == PRS for M_ in Ms)

json.dump(res, open(os.path.join(OUT, "epp_weights.json"), "w"), indent=1)

# ===================================================================== tables
def Wt(name, s): open(os.path.join(OUT, "tables", name + ".tex"), "w").write(s)
pc = lambda x: f"{100*x:.0f}"   # whole percentages: the probabilities are approximate
ENCLAB = {"conv": "Conventional", "harm": "Evidence of harm"}
WLAB = {"equal": "Equal weights", "all": "No information (all weights)", "lt_first": "Long $\\ge$ medium $\\ge$ short",
        "st_first": "Short $\\ge$ medium $\\ge$ long"}
lines = []
for enc in ENC:
    for kind in ("equal", "all", "lt_first", "st_first"):
        j = joint[f"{enc}_{kind}"]
        if kind == "equal":
            C = closeness_w(encode_alt(**ENC[enc]), BETA, 1)[0]; o = np.argsort(-C)
            sp = [100.0 if o[0] == PRS else 0.0, 100.0 if o[-1] == PSE else 0.0, 100.0 if C[TRN] > C[JSA] else 0.0]
            sp = [f"{x:.0f}" for x in sp]
        else:
            s_ = space[f"{enc}_p1"][kind]; sp = [pc(s_["prs_first"]), pc(s_["pse_last"]), pc(s_["trn_jsa"])]
        lines.append(f"{ENCLAB[enc] if kind == 'equal' else ''} & {WLAB[kind]} & " + " & ".join(sp) + " & " +
                     " & ".join([pc(j["prs_first"]), pc(j["pse_last"]), pc(j["trn_jsa"])]) + " \\\\")
    if enc == "conv":
        lines.append("\\midrule")
Wt("epp_w_ordinal", "\n".join(lines) + "\n")

lines = []
for c in (3, 10, 30, 100):
    a, b = conc[f"conv_{c}"], conc[f"harm_{c}"]
    lines.append(f"{c} & " + " & ".join(pc(x) for x in (a["prs_first"], a["pse_last"], a["trn_jsa"], b["prs_first"], b["pse_last"], b["trn_jsa"])) + " \\\\")
Wt("epp_w_conc", "\n".join(lines) + "\n")

Wt("epp_w_harmrobust", "\n".join(f"{r['name']} & {r['o1']} ({r['s1']}) & {r['o2']} ({r['s2']}) & {pc(r['prs_first'])} & {pc(r['pse_last'])} & {pc(r['trn_jsa'])} \\\\"
                                  for r in rob) + "\n")

# ===================================================================== figure: exact rank regions
from plotstyle import apply as _style
_style()
COLS = {"TRN": "#1b9e77", "JSA": "#d95f02", "PRS": "#7570b3", "PSE": "#e7298a"}
gg = np.arange(301) / 300
W = np.clip(np.array([(a, b, 1 - a - b) for a in gg for b in gg if a + b <= 1 + 1e-12]), 0, 1)
X = W[:, 1] + 0.5 * W[:, 2]; Y = np.sqrt(3) / 2 * W[:, 2]
P = lambda w: (w[1] + 0.5 * w[2], np.sqrt(3) / 2 * w[2])
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.3))
for r, enc in enumerate(ENC):
    C = closeness_w(encode_alt(**ENC[enc]), W, 1); o = np.argsort(-C, 1)
    axs[r, 0].scatter(X, Y, c=[COLS[ALTS[i]] for i in o[:, 0]], s=1.6, marker="s", lw=0)
    axs[r, 1].scatter(X, Y, c=np.where(C[:, TRN] > C[:, JSA], COLS["TRN"], COLS["JSA"]), s=1.6, marker="s", lw=0)
    for c_, ttl in [(0, "ranked first"), (1, "TRN or JSA ahead")]:
        ax = axs[r, c_]
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], "k", lw=.8)
        tri = [P((1/3, 1/3, 1/3)), P((0, .5, .5)), P((0, 0, 1)), P((1/3, 1/3, 1/3))]      # long >= medium >= short
        ax.plot(*zip(*tri), color="white", lw=1.1, ls="-"); ax.plot(*zip(*tri), color="k", lw=.6, ls=":")
        ax.plot(*P((1/3, 1/3, 1/3)), "k*", ms=8)
        thr = space[f"{enc}_p1"]["min_lt_prs_first"]
        ax.plot([P((1 - thr, 0, thr))[0], P((0, 1 - thr, thr))[0]], [P((1 - thr, 0, thr))[1]] * 2, "k--", lw=.6)
        ax.set_title(f"{ENCLAB[enc].lower()}: {ttl}", fontsize=8.5); ax.axis("off"); ax.set_aspect("equal")
        ax.text(0, -0.03, "short term\nonly", fontsize=6.5, ha="center", va="top")
        ax.text(1, -0.03, "medium term\nonly", fontsize=6.5, ha="center", va="top")
        ax.text(0.5, np.sqrt(3) / 2 + 0.02, "long term only", fontsize=6.5, ha="center", va="bottom")
from matplotlib.patches import Patch
fig.legend([Patch(color=COLS[a]) for a in ("PRS", "TRN", "JSA")], ["PRS", "TRN", "JSA"], loc="lower center", ncol=3, fontsize=7.5, frameon=False)
fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(os.path.join(OUT, "figures", "weight_space.png"), dpi=600); plt.close(fig)

# ===================================================================== macros
mac = []
def M(k, v): mac.append(f"\\newcommand{{\\{k}}}{{{v}}}")
for enc in ENC:
    s1 = space[f"{enc}_p1"]
    E = enc
    M(f"w{E}allprs", pc(s1["all"]["prs_first"])); M(f"w{E}allpse", pc(s1["all"]["pse_last"])); M(f"w{E}alltrn", pc(s1["all"]["trn_jsa"]))
    M(f"w{E}allfirsttrn", pc(s1["all"]["first"]["TRN"])); M(f"w{E}allfirstjsa", pc(s1["all"]["first"]["JSA"]))
    M(f"w{E}ltprs", pc(s1["lt_first"]["prs_first"])); M(f"w{E}lttrn", pc(s1["lt_first"]["trn_jsa"])); M(f"w{E}ltpse", pc(s1["lt_first"]["pse_last"]))
    M(f"w{E}stprs", pc(s1["st_first"]["prs_first"])); M(f"w{E}sttrn", pc(s1["st_first"]["trn_jsa"])); M(f"w{E}stpse", pc(s1["st_first"]["pse_last"]))
    M(f"w{E}minltprs", f"{s1['min_lt_prs_first']:.2f}"); M(f"w{E}minltpse", f"{s1['min_lt_pse_last']:.2f}")
    for kind in ("equal", "all", "lt_first", "st_first"):
        j = joint[f"{enc}_{kind}"]; t = kind.replace("_first", "").replace("all", "all")
        M(f"j{E}{t}prs", pc(j["prs_first"])); M(f"j{E}{t}pse", pc(j["pse_last"])); M(f"j{E}{t}trn", pc(j["trn_jsa"]))
    for c in (3, 100):
        M(f"c{E}{'three' if c == 3 else 'hundred'}trn", pc(conc[f"{enc}_{c}"]["trn_jsa"]))
        M(f"c{E}{'three' if c == 3 else 'hundred'}prs", pc(conc[f"{enc}_{c}"]["prs_first"]))
M("wconvflipminltpse", f"{space['conv_flip_p1']['min_lt_pse_last']:.2f}")
M("wconvflipallpse", pc(space["conv_flip_p1"]["all"]["pse_last"]))
M("wconvminltpsetwo", f"{space['conv_p2']['min_lt_pse_last']:.2f}")
M("westst", f"{w_est[0]:.2f}"); M("westmt", f"{w_est[1]:.2f}"); M("westlt", f"{w_est[2]:.2f}")
vs = res["harm_vote_share"]
M("harmvotesharemin", f"{100*min(vs):.0f}"); M("harmvotesharemax", f"{100*max(vs):.0f}")
for i, a in enumerate(ALTS): M(f"harmvoteshare{a.lower()}", f"{100*vs[i]:.0f}")
open(os.path.join(OUT, "epp_weights_macros.tex"), "w").write("\n".join(mac) + "\n")
print("epp_weights done;", len(mac), "macros")
