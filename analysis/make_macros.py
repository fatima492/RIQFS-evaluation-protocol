"""make_macros.py -- writes outputs/macros.tex: every number quoted in the manuscript text."""
import os, sys, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from riqfs_core import dist
r = json.load(open(os.path.join(ROOT, "outputs", "results.json")))
e = json.load(open(os.path.join(ROOT, "outputs", "examples.json")))
N = "one two three four five six seven eight".split()
ALTS = ["trn", "jsa", "prs", "pse"]
out = []
def M(k, v): out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"; pc = lambda x: f"{100*x:.1f}"

# ---------------------------------------------------------------- case study (Sections 4-5)
M("nest", r["n_estimates_total"]); M("neff", r["n_effect_total"])
M("nestmain", r["n_estimates_main"]); M("neffmain", r["n_effect_main"])
M("stmeanprinted", f1(r["checks"]["ST"]["mean"])); M("stmeanflip", f1(r["st_mean_flip"]))
M("kpropone", f2(r["kappa_prop"][0])); M("kproptwo", f2(r["kappa_prop"][1]))
for i, a in enumerate(ALTS):
    M(f"CH{a}", f4(r["C1"][i])); M(f"CE{a}", f4(r["C2"][i])); M(f"Sc{a}", f4(r["score"][i]))
    M(f"Lam{a}", f4(r["Lam"][i])); M(f"Cphi{a}", f4(r["CPhi"][i]))
    M(f"bootHfirst{a}", pc(r["boot1"]["first"][i])); M(f"bootEfirst{a}", pc(r["boot2"]["first"][i]))
    M(f"bootHlast{a}", pc(r["boot1"]["last"][i])); M(f"bootElast{a}", pc(r["boot2"]["last"][i]))
    M(f"bootHrank{a}", f2(r["boot1"]["meanrank"][i])); M(f"bootErank{a}", f2(r["boot2"]["meanrank"][i]))
    for key, tag in (("global_p1", "GH"), ("global_p2", "GE"), ("local_p1", "LH"), ("local_p2", "LE")):
        acc = np.array(r["smaa"][key]["acc"])[i]
        M(f"smaa{tag}first{a}", pc(acc[0])); M(f"smaa{tag}last{a}", pc(acc[-1]))
for key, tag in (("global_p1", "GH"), ("global_p2", "GE"), ("local_p1", "LH"), ("local_p2", "LE")):
    M(f"smaa{tag}same", pc(r["smaa"][key]["same"]))
M("smaaHEdiffer", pc(r["smaa_HE_differ"]))
M("bootHsame", pc(r["boot1"]["same"])); M("bootEsame", pc(r["boot2"]["same"]))
M("bootHtrnjsa", pc(r["boot1"]["trn_over_jsa"])); M("bootEtrnjsa", pc(r["boot2"]["trn_over_jsa"]))
M("Mswapone", f1(r["M_swap_p1"][0]) if r["M_swap_p1"] else "--"); M("Mswaptwo", f1(r["M_swap_p2"][0]) if r["M_swap_p2"] else "--")
M("kswapone", f2(r["kappa_swap_p1"][0]) if r["kappa_swap_p1"] else "--")
for j in range(3): M(f"mdbeta{N[j]}", f2(r["maxdev_beta"][j]))
for k, v in r["rr_max"].items(): M("rr" + k.replace("_", ""), v)
M("prsFtwosep", f3(r["prs_profile_F2_sep"])); M("prsFtwoconvex", f3(r["prs_profile_F2_convex"]))
for i, a in enumerate(ALTS):
    M(f"prof{a}", f4(r["opcmp"]["sep"]["score"][i])); M(f"conv{a}", f4(r["opcmp"]["convex"]["score"][i])); M(f"gen{a}", f4(r["opcmp"]["general"]["score"][i]))
M("LamMaxOther", f4(max(r["Lam"][0], r["Lam"][2], r["Lam"][3])))
bp = r["bpa_LT"]
for a, tag in (("PRS", "prs"), ("JSA", "jsa")):
    for q, src in enumerate(("one", "two", "mix")):
        for c, nm in enumerate(("Y", "N", "T")):
            M(f"bpa{tag}{src}{nm}", f2(bp[a][q][c]))
M("collapseerr", f"{max(r['collapse_err'], 1e-16):.0e}".replace("e-", "\\times10^{-") + "}")
M("decomperr", f"{max(r['decomp_err'], 1e-17):.0e}".replace("e-", "\\times10^{-") + "}")
# grades quoted in the text
g = r["grades"]
for key, tag in (("TRN_ST", "trnst"), ("JSA_ST", "jsast"), ("PRS_LT", "prslt"), ("JSA_LT", "jsalt"), ("PSE_LT", "pselt"), ("TRN_LT", "trnlt")):
    x = g[key]
    M(f"g{tag}muone", f2(x["grades"][0])); M(f"g{tag}nuone", f2(x["grades"][1]))
    M(f"g{tag}mutwo", f2(x["grades"][2])); M(f"g{tag}nutwo", f2(x["grades"][3]))
    M(f"g{tag}lo", f1(x["lo"])); M(f"g{tag}hi", f1(x["hi"])); M(f"g{tag}se", f2(x["se"]))
    M(f"g{tag}L", f2(x["L1"])); M(f"g{tag}U", f2(x["U1"]))
for i, t in enumerate(r["timing"]): M(f"time{N[i]}", f"{1000*t['sec']:.1f}")

# ---------------------------------------------------------------- worked examples (supplementary Section S6)
for gm, nm in [("0.5", "half"), ("1", "one"), ("2", "two"), ("5", "five")]:
    M(f"ham{nm}", e["ham"][gm][0]); M(f"hamt{nm}", f4(e["ham"][gm][1])); M(f"hamf{nm}", f4(e["ham"][gm][2]))
M("cwfailA", f4(e["cwfail"][0])); M("cwfailB", f4(e["cwfail"][1]))
for k, v in e["disc"].items(): M(f"disc{k}", v)
for k, v in e["ren"].items(): M(f"ren{k}", v)
for k, v in e["tie"].items(): M(f"tie{k}", v)
for k, v in e["em"].items(): M(f"em{k}", v)
for k, v in e["dex"].items(): M(f"dex{k}", f4(v))
for i, v in enumerate(e["per_cell_H"]): M(f"pcH{N[i]}", f4(v))
for i, v in enumerate(e["per_cell_E2"]): M(f"pcE{N[i]}", f4(v))
M("pcEsum", f4(sum(e["per_cell_E2"])))
QA = (np.array([[[.40, .10], [.10, .10]]]), np.array([[[.10, .10], [.30, .20]]]))
QB = (np.array([[[.10, .10], [.40, .10]]]), np.array([[[.30, .20], [.10, .10]]]))
M("qdh", f4(dist(QA, QB))); M("qde", f4(dist(QA, QB, p=2)))

open(os.path.join(ROOT, "outputs", "macros.tex"), "w").write("\n".join(out) + "\n")
print(len(out), "macros")
