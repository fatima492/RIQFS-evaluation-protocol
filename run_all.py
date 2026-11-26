"""
run_all.py -- regenerate every number, table and figure of the manuscript and verify all theorems.

Usage:  python run_all.py            (full run, about 2-3 minutes)
        python run_all.py --no-tests (skip the theorem tests)

Requirements: pip install -r requirements.txt (numpy, scipy, matplotlib, pytest).
Figures are typeset with LaTeX when latex and dvipng are found (MiKTeX may ask to
install missing packages such as cm-super on first use); set RIQFS_NO_USETEX=1 to use
matplotlib's built-in Computer Modern fonts instead.
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in ("tables", "figures", "csv"):
    os.makedirs(os.path.join(ROOT, "outputs", _d), exist_ok=True)
STEPS = [
    ("Theorem verification (randomised property tests)", [sys.executable, "-m", "pytest", "-q", "tests"]),
    ("Worked examples (supplementary material)", [sys.executable, "analysis/examples.py"]),
    ("Case study on published data, robustness, benchmarks and figures (Sections 4-5, Appendices B-C)", [sys.executable, "analysis/case_study.py"]),
    ("Theory figure: discontinuity of Design S (supplementary material)", [sys.executable, "analysis/figures_extra.py"]),
    ("TikZ figures for the supplementary material (need pdflatex and pdftoppm)", [sys.executable, "analysis/flowchart.py"]),
    ("Numbers quoted in the text (outputs/macros.tex)", [sys.executable, "analysis/make_macros.py"]),
    ("Analyses: alternative encodings, decision table, pairwise probabilities, cost break-even (outputs/epp_*)", [sys.executable, "analysis/epp_extensions.py"]),
    ("Criterion-weight sensitivity and robustness of the evidence-of-harm encoding (outputs/epp_weights*)", [sys.executable, "analysis/epp_weights.py"]),
    ("Further analyses: channel verdicts, practitioner baselines, adjusted contrasts, cost-effectiveness (outputs/epp_review*)", [sys.executable, "analysis/epp_review.py"]),
    ("Comparative analysis: probability baselines, like-for-like benchmarks, simulation with known rankings (outputs/epp_compare*)", [sys.executable, "analysis/epp_compare.py"]),
    ("Encoding figure: Figure 2 (outputs/figures/encoding.png)", [sys.executable, "analysis/encoding_figure.py"]),
    ("Tables with programme acronyms (outputs/tables_plain)", [sys.executable, "analysis/plain_tables.py"]),
    ("Workflow figure (needs pdflatex and pdftoppm)", [sys.executable, "analysis/workflow_figure.py"]),
]
OPTIONAL = ("TikZ", "workflow figure")

if __name__ == "__main__":
    skip_tests = "--no-tests" in sys.argv
    import importlib.util
    missing = [m for m in ("numpy", "scipy", "matplotlib") if importlib.util.find_spec(m) is None]
    if missing:
        print("Missing packages: " + ", ".join(missing) + ". Run:  pip install -r requirements.txt"); sys.exit(1)
    if not skip_tests and importlib.util.find_spec("pytest") is None:
        print("WARNING: pytest is not installed, so the theorem tests are skipped. "
              "Install it with  pip install -r requirements.txt  to run them.")
        skip_tests = True
    t0 = time.time()
    for name, cmd in STEPS:
        if skip_tests and "pytest" in cmd:
            continue
        print(f"==> {name}")
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = "\n".join(r.stdout.strip().splitlines()[-3:])
        print("    " + tail.replace("\n", "\n    "))
        if r.returncode != 0:
            if any(k in name for k in OPTIONAL):
                print(f"    skipped (optional figure step failed; the supplied PNG is kept): {name}"); continue
            print(f"FAILED: {name}"); sys.exit(r.returncode)
    print(f"All steps completed in {time.time() - t0:.1f} s. Outputs are in {os.path.join(ROOT, 'outputs')}")
