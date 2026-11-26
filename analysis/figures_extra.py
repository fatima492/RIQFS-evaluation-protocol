"""
figures_extra.py -- Figure S1 of the supplementary material: discontinuity of the selection-based union (Design S) versus the
continuous renormalised union (Design C). Theory figure; it uses no case-study data.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from riqfs_core import *      # noqa

FIG = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG, exist_ok=True)
from plotstyle import apply as _style, PCT
_style()
COLS = plt.cm.tab10(np.arange(10))
out = {}

# ------------------------------------------------------------------ 1. discontinuity of Design S vs Design C
eta = np.linspace(-0.03, 0.03, 601)
a = (np.array([0.30, 0.0]), np.zeros(2))
S1 = []; S2 = []; C1 = []; C2 = []
for e in eta:
    b = (np.array([0.0, 0.30 + e]), np.zeros(2))
    s = union(a, b); c = runion(a, b)
    S1.append(s[0][0]); S2.append(s[0][1]); C1.append(c[0][0]); C2.append(c[0][1])
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
for ax, (y1, y2), ttl in [(axs[0], (S1, S2), "Design S (linear-order selection)"), (axs[1], (C1, C2), "Design C (component-wise, renormalised)")]:
    ax.plot(eta, y1, color=COLS[0], lw=1.6, label=r"$T^1$ of the union")
    ax.plot(eta, y2, color=COLS[3], lw=1.6, ls="--", label=r"$T^2$ of the union")
    ax.axvline(0, color="grey", lw=.6, ls=":")
    ax.set_title(ttl, fontsize=8.5); ax.set_xlabel(r"perturbation $\eta$ of $T^2_\beta=0.30+\eta$"); ax.grid(alpha=.3)
axs[0].set_ylabel("refined membership"); axs[1].legend(fontsize=7, loc="center right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "discontinuity.png"), dpi=600); plt.close(fig)

out["jump_T1_design_S"] = float(S1[300] - S1[301]) if len(S1) > 301 else None
json.dump(out, open(os.path.join(ROOT, "outputs", "figures_extra.json"), "w"), indent=1)
print("wrote discontinuity.png")
