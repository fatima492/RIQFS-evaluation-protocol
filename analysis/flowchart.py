"""Render the TikZ figures to PNG (600 dpi): figures_src/framework.tex (research framework, not used in the article) and
figures_src/flowchart.tex (flowchart of Algorithm 1, Figure S2 of the supplementary material). Needs pdflatex (with TikZ) and pdftoppm (poppler);
if either is missing, the step is skipped and the shipped PNGs are kept.
"""
import os, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG, exist_ok=True)

if not (shutil.which("pdflatex") and shutil.which("pdftoppm")):
    print("pdflatex or pdftoppm not found; TikZ figures not regenerated")
else:
    for name in ("framework", "flowchart"):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(os.path.join(ROOT, "figures_src", name + ".tex"), tmp)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", name + ".tex"], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", name + ".pdf", name], cwd=tmp, check=True)
            shutil.copy(os.path.join(tmp, name + ".png"), os.path.join(FIG, name + ".png"))
        print("wrote", name + ".png")
