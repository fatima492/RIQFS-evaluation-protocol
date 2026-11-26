"""plotstyle.py -- common figure style: all text typeset by LaTeX (Computer Modern), with a
mathtext fallback (Computer Modern glyphs) when no LaTeX installation is available."""
import os, shutil
import matplotlib.pyplot as plt

USETEX = (os.environ.get("RIQFS_NO_USETEX", "") != "1"
          and shutil.which("latex") is not None and shutil.which("dvipng") is not None)
PCT = r"\%" if USETEX else "%"


def apply():
    rc = {"font.size": 8.5, "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
          "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True}
    if USETEX:
        rc.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
                   "font.serif": ["Computer Modern Roman"]})
    else:
        rc.update({"font.serif": ["cmr10", "DejaVu Serif"]})
    plt.rcParams.update(rc)
