"""workflow_figure.py -- renders figures_src/workflow_epp.tex (Figure 1 of the manuscript) to PNG."""
import os, shutil, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src = os.path.join(ROOT, "figures_src")
if not (shutil.which("pdflatex") and shutil.which("pdftoppm")):
    print("pdflatex/pdftoppm not found: keeping the supplied workflow_epp.png"); sys.exit(0)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "workflow_epp.tex"], cwd=src, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", os.path.join(src, "workflow_epp.pdf"),
                os.path.join(ROOT, "outputs", "figures", "workflow_epp")], check=True)
for ext in ("aux", "log"):
    p = os.path.join(src, f"workflow_epp.{ext}")
    if os.path.exists(p): os.remove(p)
print("wrote workflow_epp.png")
