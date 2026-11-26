"""Validated worked examples (Section S6 of the supplementary material)."""
import numpy as np, os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from riqfs_core import *  # noqa
OUT = os.path.join(ROOT, "outputs", "tables")
os.makedirs(OUT, exist_ok=True)

def c(t1, t2, f1, f2):
    x = (np.array([t1, t2]), np.array([f1, f2]))
    assert is_valid_cell(*x), (t1, t2, f1, f2)
    return x

def s(x, d=2):
    T, F = x
    return r"$\langle(" + ",".join(f"{v:.{d}f}" for v in T) + "),(" + ",".join(f"{v:.{d}f}" for v in F) + r")\rangle$"

def lab(k):
    return f"$(\\omega_{{{k[0]}}},q_{{{k[1]}}})$"

def table(cols, names, keys=None, d=2):
    if keys is None:
        keys = []
        for C in cols:
            for k in C:
                if k not in keys: keys.append(k)
        keys = sorted(keys)
    rows = []
    for k in keys:
        rows.append(lab(k) + " & " + " & ".join(s(C[k], d) if k in C else "--" for C in cols) + r" \\")
    head = r"$(\omega,q)$ & " + " & ".join(names) + r" \\"
    return head, "\n".join(rows)

snips = {}
def put(name, cols, names, keys=None, d=2):
    h, r = table(cols, names, keys, d)
    snips[name] = (h, r, len(cols))

# Example HR (full 3x3)
hr = {(1,1): c(.30,.20,.21,.07), (1,2): c(.20,.10,.32,.02), (1,3): c(.30,.25,.04,.32),
      (2,1): c(.20,.28,.09,.12), (2,2): c(.20,.30,.20,.10), (2,3): c(.11,.44,.14,.21),
      (3,1): c(.29,.25,.08,.30), (3,2): c(.21,.24,.07,.20), (3,3): c(.23,.10,.40,.20)}
rows = []
for i in (1, 2, 3):
    rows.append(f"$\\omega_{{{i}}}$ & " + " & ".join(s(hr[(i, j)]) for j in (1, 2, 3)) + r" \\")
open(f"{OUT}/ex_hr.tex", "w").write("\n".join(rows))
hes = {k: 1 - v[0].sum() - v[1].sum() for k, v in hr.items()}

D = [(1,1), (2,2), (3,3), (4,4)]
# inclusion example: F1 strongly in F2; G weakly but not strongly in F2
F1 = {D[0]: c(.10,.15,.30,.20), D[1]: c(.20,.10,.25,.15), D[2]: c(.05,.20,.40,.10), D[3]: c(.15,.15,.30,.25)}
F2 = {D[0]: c(.25,.20,.20,.10), D[1]: c(.30,.20,.20,.05), D[2]: c(.30,.25,.20,.05), D[3]: c(.20,.30,.10,.20)}
G  = {D[0]: c(.30,.00,.25,.05), D[1]: c(.05,.30,.20,.10), D[2]: c(.20,.15,.30,.10), D[3]: c(.30,.00,.20,.30)}
for k in D:
    assert strong_subset(F1[k], F2[k]); assert weak_subset(G[k], F2[k])
assert not all(strong_subset(G[k], F2[k]) for k in D)
put("ex_incl", [F1, G, F2], [r"$\mathcal{F}_1$", r"$\mathcal{G}$", r"$\mathcal{F}_2$"])

# complement
Fc = {D[0]: c(.33,.25,.20,.15), D[1]: c(.23,.34,.10,.19), D[2]: c(.40,.01,.30,.20), D[3]: c(.21,.11,.30,.25)}
put("ex_comp", [Fc, {k: comp(v) for k, v in Fc.items()}], [r"$\mathcal{F}$", r"$\mathcal{F}^{c}$"])

# union / intersection
G1 = {D[0]: c(.14,.26,.03,.05), D[1]: c(.50,.11,.02,.14), D[2]: c(.10,.20,.17,.41), D[3]: c(.24,.33,.11,.03)}
G2 = {D[0]: c(.30,.20,.13,.25), D[1]: c(.35,.19,.12,.24), D[2]: c(.29,.24,.13,.30), D[3]: c(.21,.07,.20,.10)}
put("ex_ui", [G1, G2, {k: union(G1[k], G2[k]) for k in D}, {k: inter(G1[k], G2[k]) for k in D}],
    [r"$\mathcal{A}$", r"$\mathcal{B}$", r"$\mathcal{A}\cup\mathcal{B}$", r"$\mathcal{A}\cap\mathcal{B}$"])

# tie-break illustration: equal aggregated membership
tA = c(.30,.20,.10,.20); tB = c(.20,.30,.25,.05)
tie = dict(A=s(tA), B=s(tB), U=s(union(tA, tB)), I=s(inter(tA, tB)))

# domain-based operations
H1 = {(1,1): c(.13,.19,.10,.20), (1,3): c(.25,.15,.10,.36), (2,1): c(.34,.20,.24,.10), (3,2): c(.01,.07,.60,.10)}
H2 = {(1,1): c(.20,.30,.11,.26), (1,2): c(.01,.30,.30,.30), (2,2): c(.10,.40,.21,.21), (3,2): c(.50,.30,.02,.10)}
put("ex_dom_in", [H1, H2], [r"$\mathcal{H}_1$", r"$\mathcal{H}_2$"])
put("ex_dom_out", [ext_inter(H1, H2), ext_union(H1, H2), res_union(H1, H2), res_inter(H1, H2), res_diff(H1, H2)],
    [r"$\mathcal{H}_1\cap_{E}\mathcal{H}_2$", r"$\mathcal{H}_1\cup_{E}\mathcal{H}_2$",
     r"$\mathcal{H}_1\cup_{R}\mathcal{H}_2$", r"$\mathcal{H}_1\cap_{R}\mathcal{H}_2$", r"$\mathcal{H}_1\setminus_{R}\mathcal{H}_2$"])

# excluded middle failure
a = c(.20,.10,.30,.10)
em = dict(A=s(a), U=s(union(a, comp(a))), I=s(inter(a, comp(a))))

# separable Hamacher example (kappa = (0.5,0.5))
KAP = (0.5, 0.5)
cells = [c(.30,.15,.10,.20), c(.20,.25,.25,.10), c(.40,.05,.05,.30)]
for x in cells: assert is_separable(*x, KAP)
w = np.array([.4, .35, .25])
ham = {}
for g in (0.5, 1, 2, 5):
    T, F = hwa_sep(cells, w, g, KAP); T2, F2 = hwa_sep_by_laws(cells, w, g, KAP)
    assert np.allclose(T, T2) and np.allclose(F, F2)
    ham[g] = (s((T, F), 4), T.sum(), F.sum())
# general (aggregate-then-allocate) operator: componentwise monotonicity failure
a1 = c(.2,.2,.2,.2); a2 = c(0,.2,.2,.2); a2b = c(.3,.2,.2,.2)
cwfail = (hwa_general([a1, a2], [.5,.5], 1)[0][1], hwa_general([a1, a2b], [.5,.5], 1)[0][1])
# discontinuity of the selection union
disc = dict(lo=s(union(c(.30,0,0,0), c(0,.299,0,0)),3), hi=s(union(c(.30,0,0,0), c(0,.301,0,0)),3))
# renormalised design counterexamples
ra, rb, rc = c(.1,.6,0,.2), c(.5,0,.3,.1), c(.6,.1,.2,0)
rx, ry = c(.3,.3,.2,0), c(.2,.3,0,.4)
ren = dict(lhs=s(runion(runion(ra,rb),rc),4), rhs=s(runion(ra,runion(rb,rc)),4),
           abs=s(runion(rx, rinter(rx, ry)),4))
# distance example between A and B (G1, G2)
TA = np.array([[G1[k][0] for k in D]]); FA = np.array([[G1[k][1] for k in D]])
TB = np.array([[G2[k][0] for k in D]]); FB = np.array([[G2[k][1] for k in D]])
W = np.array([[.4, .3, .2, .1]])
dex = dict(dH=dist((TA, FA), (TB, FB)), dE=dist((TA, FA), (TB, FB), p=2),
           dwH=dist((TA, FA), (TB, FB), W), dwE=dist((TA, FA), (TB, FB), W, 2),
           dagg=dist(reduce_phi((TA, FA)), reduce_phi((TB, FB))))
per_cell_H = [cell_dist(G1[k], G2[k]) for k in D]
per_cell_E2 = [((G1[k][0]-G2[k][0])**2).sum() + ((G1[k][1]-G2[k][1])**2).sum() for k in D]

# split table of domain operations into two parts
outE = [ext_inter(H1, H2), ext_union(H1, H2)]; outR = [res_union(H1, H2), res_inter(H1, H2), res_diff(H1, H2)]
hE, rE = table(outE, [r"$\mathcal{H}_1\cap_{E}\mathcal{H}_2$", r"$\mathcal{H}_1\cup_{E}\mathcal{H}_2$"])
hR, rR = table(outR, [r"$\mathcal{H}_1\cup_{R}\mathcal{H}_2$", r"$\mathcal{H}_1\cap_{R}\mathcal{H}_2$", r"$\mathcal{H}_1\setminus_{R}\mathcal{H}_2$"])
split = (r"\begin{table}[htbp]" + "\n" + r"\caption{Extended operations (Example~\ref{ex:dom}).}\label{tab:domout}" + "\n"
         + r"\centering\small" + "\n" + r"\begin{tabular}{@{}ccc@{}}" + "\n" + r"\toprule" + "\n" + hE + "\n" + r"\midrule" + "\n" + rE + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n\n"
         + r"\begin{table}[htbp]" + "\n" + r"\caption{Restricted operations (Example~\ref{ex:dom}).}\label{tab:domoutR}" + "\n"
         + r"\centering\small" + "\n" + r"\begin{tabular}{@{}cccc@{}}" + "\n" + r"\toprule" + "\n" + hR + "\n" + r"\midrule" + "\n" + rR + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
open(f"{OUT}/ex_dom_out_split.tex", "w").write(split)

# write snippets
for name, (h, r, nc) in snips.items():
    open(f"{OUT}/{name}.tex", "w").write(h + "\n\\midrule\n" + r)
json.dump(dict(cwfail=cwfail, disc=disc, ren=ren, tie=tie, em=em, ham={str(k): v for k, v in ham.items()}, dex=dex, per_cell_H=per_cell_H,
               per_cell_E2=per_cell_E2, hes={f"{k}": v for k, v in hes.items()}),
          open(os.path.join(ROOT, "outputs", "examples.json"), "w"), indent=1, default=float)
print(ham); print(cwfail); print(disc); print(ren); print(dex)
