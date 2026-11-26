"""encoding_figure.py -- Figure 2 of the manuscript: how the two encodings turn published
intervals into support, opposition and uncommitted evidence.

(a) One cell (training, short term): the vote-count and effect-size intervals on their scales, and the
    resulting grades under the conventional and the evidence-of-harm encoding.
(b) Pooled grades (t, pi, f) of all twelve programme-horizon cells under both encodings (kappa = 0.5, 0.5).
Writes outputs/figures/encoding.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epp_encoding import cs, grades_alt, neg_share, ALTS, H, KAPPA, Z, MB, DEFF   # noqa: E402
from plotstyle import apply as _style, PCT                                                # noqa: E402

_style()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures")
CS, CO, CU = "#2C6E9B", "#C9731B", "#D9D9D9"          # support, opposition, uncommitted
HL = {"ST": "short", "MT": "medium", "LT": "long"}


def bar(ax, y, mu, nu, h=0.26, label=None):
    """Stacked bar on [0, 1]: support from the left, opposition from the right, uncommitted between."""
    ax.barh(y, mu, left=0, height=h, color=CS, edgecolor="white", lw=0.4)
    ax.barh(y, 1 - mu - nu, left=mu, height=h, color=CU, edgecolor="white", lw=0.4)
    ax.barh(y, nu, left=1 - nu, height=h, color=CO, edgecolor="white", lw=0.4)
    if label:
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=7)


fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42)

# ---------------------------------------------------------------- (a) one cell
a, h = "TRN", "ST"
r = cs.cell(a, h)
p = float(r["share_sig_positive_pct"]) / 100
nn = int(r["n_sign"]) * DEFF
L1, U1 = cs.wilson(p, nn, Z)
q = neg_share(a, h, p); Lq, _ = cs.wilson(q, nn, Z)
me = float(r["mean_effect_pp"]); se = cs.se_cell(a, h); lo, hi = me - Z * se, me + Z * se
c1 = grades_alt(a, h, "base", "base"); c2 = grades_alt(a, h, "harm", "harm")

axa = fig.add_subplot(gs[0, 0]); axa.set_xlim(-0.02, 1.02); axa.set_ylim(-0.2, 5.6); axa.axis("off")
axa.set_title(r"(a) Encoding one cell: training, short term", fontsize=8.5, loc="left")


def scale_line(y, lo_, hi_, pt, ticks, ticklab, title):
    axa.plot([0, 1], [y, y], color="black", lw=0.7)
    for t_, tl in zip(ticks, ticklab):
        axa.plot([t_, t_], [y - 0.06, y + 0.06], color="black", lw=0.6)
        axa.text(t_, y - 0.13, tl, ha="center", va="top", fontsize=6.5)
    axa.plot([lo_, hi_], [y + 0.2, y + 0.2], color="black", lw=2.2, solid_capstyle="butt")
    axa.plot([pt], [y + 0.2], "o", ms=3.5, color="white", mec="black", mew=0.8)
    axa.text(0, y + 0.42, title, ha="left", va="bottom", fontsize=7.5)


# vote-count channel on the share scale [0, 1]
yv = 4.55
scale_line(yv, L1, U1, p, [0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"],
           r"Vote count: %.0f%s significantly positive, interval $[%.2f,%.2f]$" % (100 * p, PCT, L1, U1))
bar(axa, yv - 0.62, c1[0], c1[1], label="conv.")
bar(axa, yv - 0.98, c2[0], c2[1], label="harm")
for x in (L1, U1):
    axa.plot([x, x], [yv + 0.2, yv], color="black", lw=0.4, ls=":")
    axa.plot([x, x], [yv - 0.34, yv - 0.48], color="black", lw=0.4, ls=":")

# effect-size channel on the symmetric scale [-M, M] -> [0, 1]
ye = 1.95
gsym = lambda x: (x + MB) / (2 * MB)
scale_line(ye, gsym(lo), gsym(hi), gsym(me), [0, .25, .5, .75, 1],
           [r"$-%d$" % MB, r"$-%d$" % (MB / 2), "0", r"$%d$" % (MB / 2), r"$%d$" % MB],
           r"Mean effect: $%.1f$ points, interval $[%.1f,%.1f]$, $M=%d$" % (me, lo, hi, MB))
bar(axa, ye - 0.62, c1[2], c1[3], label="conv.")
bar(axa, ye - 0.98, c2[2], c2[3], label="harm")
for x in (gsym(lo), gsym(hi)):
    axa.plot([x, x], [ye + 0.2, ye], color="black", lw=0.4, ls=":")
    axa.plot([x, x], [ye - 0.34, ye - 0.48], color="black", lw=0.4, ls=":")
axa.text(0.5, ye - 1.42, "evidence of harm: the interval reaches zero, so this channel is uncommitted",
         ha="center", va="top", fontsize=6.5, style="italic")

# ---------------------------------------------------------------- (b) pooled grades of all cells
axb = fig.add_subplot(gs[0, 1])
k = np.asarray(KAPPA)
ys, labs = [], []
y = 0.0
for a in ALTS:
    for h in H:
        g1 = grades_alt(a, h, "base", "base"); g2 = grades_alt(a, h, "harm", "harm")
        t1, f1 = k @ [g1[0], g1[2]], k @ [g1[1], g1[3]]
        t2, f2 = k @ [g2[0], g2[2]], k @ [g2[1], g2[3]]
        bar(axb, y - 0.15, t1, f1, h=0.27)
        bar(axb, y + 0.15, t2, f2, h=0.27)
        axb.plot([0.5 * (1 + (t1 - f1))] * 2, [y - 0.29, y - 0.01], color="black", lw=0.9)
        axb.plot([0.5 * (1 + (t2 - f2))] * 2, [y + 0.01, y + 0.29], color="black", lw=0.9)
        ys.append(y); labs.append(f"{a} {HL[h]}")
        y += 1.0
    y += 0.45
axb.set_yticks(ys); axb.set_yticklabels(labs, fontsize=6.8); axb.invert_yaxis()
axb.set_xlim(0, 1); axb.set_xlabel("share of the evidence", fontsize=7.5)
axb.tick_params(axis="x", labelsize=6.8)
axb.axvline(0.5, color="black", lw=0.4, ls=":")
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
axb.set_title(r"(b) Pooled grades, upper bar conventional," "\n" r"lower bar evidence of harm", fontsize=8.5, loc="left")

handles = [Patch(color=CS, label=r"support $\mu$ / $t$"), Patch(color=CU, label=r"uncommitted $\pi$"),
           Patch(color=CO, label=r"opposition $\nu$ / $f$"),
           plt.Line2D([], [], color="black", lw=0.9, label=r"score $\frac{1}{2}(1+t-f)$")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.17)
fig.savefig(os.path.join(OUT, "encoding.png"), dpi=600)
print("wrote encoding.png")
