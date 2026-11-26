# Ranking labour market programmes when vote counts and effect sizes disagree

Reproducibility package for the article **Ranking active labour market programmes for evaluation and planning when vote counts and effect sizes disagree**. The package also reproduces the supplementary material (formal development and worked examples).

All inputs are published statistics from D. Card, J. Kluve and A. Weber (2018), *Journal of the European Economic Association* 16(3), 894–931, https://doi.org/10.1093/jeea/jvx028 (see `data/SOURCE.md`).

The package regenerates **every number, table and figure** of the manuscript and the supplementary material from the published data. It also checks **every theorem, proposition and counterexample** of the supplementary material numerically.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py          # about 2-3 minutes (longer on the first run while LaTeX builds its font cache)
python run_all.py --no-tests   # skip the theorem tests
```

Outputs are written to `outputs/`. Numbering below follows the manuscript (Tables 1-8, Figures 1-5, Appendix Tables B.1-B.6 and C.1-C.8, Appendix Figures C.1-C.3) and the supplementary material (prefix S).

## What `run_all.py` does

`run_all.py` runs thirteen steps in order:

1. `pytest tests/` runs 14 randomised property tests that check the theory of the supplementary material (see the map below).
2. `analysis/examples.py` computes the worked examples of the supplementary material (Section S6).
3. `analysis/case_study.py` encodes the published data and runs the base case study, the robustness analyses and the benchmarks: Tables B.1, B.3, B.4, C.2, C.3 and C.5, Figures 3, C.1 and C.2, and Table S2 and Figure S3 of the supplementary material.
4. `analysis/figures_extra.py` draws Figure S1 (discontinuity of the selection-based union), a theory figure that uses no data.
5. `analysis/flowchart.py` renders the TikZ flowchart of Algorithm 1 (Figure S2, `figures_src/flowchart.tex`) and an overview figure that is not used in the article (`figures_src/framework.tex`). It needs `pdflatex` and `pdftoppm`; without them the step is skipped and the supplied PNGs are kept.
6. `analysis/make_macros.py` writes `outputs/macros.tex`, the base-case numbers quoted in the text.
7. `analysis/epp_extensions.py`: the evidence-of-harm encoding and its components (Table 3), the decision table (Table 4), pairwise bootstrap probabilities (Figure 4), the grades under the evidence-of-harm encoding (Table B.5) and the break-even cost gaps (Table C.6, Figure C.3).
8. `analysis/epp_weights.py`: criterion-weight sensitivity (Section 5.3: Figure 5, Table 5, Table C.1) and the robustness of the evidence-of-harm encoding (Table C.4). Both encodings are defined once, in `analysis/epp_encoding.py`.
9. `analysis/epp_review.py`: plain-language channel verdicts (Table 2), cost-effectiveness break-even ratios (Table 6), mean effects used for costs (Table B.6) and the covariate-adjusted contrasts of Card, Kluve and Weber (Table B.2, `data/ckw2018_table5_adjusted.csv`).
10. `analysis/epp_compare.py`: every comparator recomputed on the same bootstrap draws (Table 7), established methods applied like for like (Table C.7), and the simulation with known rankings (Table 8, design in Table C.8).
11. `analysis/encoding_figure.py` draws Figure 2 (how the encodings turn intervals into grades).
12. `analysis/plain_tables.py` writes `outputs/tables_plain/`, the table bodies with programme acronyms that the manuscript inputs.
13. `analysis/workflow_figure.py` renders Figure 1 (`figures_src/workflow_epp.tex`).

**Encoding conventions.** The conventional encoding places the mean effect on a symmetric scale from -M to +M (`scale="sym"` in `case_study.channel_grades`); a one-sided scale clipped at zero (`scale="clip0"`) is kept only as a sensitivity case (Section 3.1, Table C.3), because it favours imprecise effects. Vote-count intervals and bootstrap draws use a design-effect-adjusted number of observations, n x 207/857 (`DEFF` in `case_study.py`).

Plot text is typeset by LaTeX when `latex` and `dvipng` are found (`src/plotstyle.py`; the `cm-super` fonts are needed). Set `RIQFS_NO_USETEX=1` to use matplotlib's built-in Computer Modern fonts instead. All figures are PNG files at 600 dpi.

## Contents

| Path | Content |
|---|---|
| `data/ckw2018_table3a.csv` | Published statistics by programme type and horizon: share of significantly positive estimates and mean employment effect, with counts (Card, Kluve and Weber 2018, Table 3a) |
| `data/ckw2018_table2.csv` | Horizon totals and standard errors of the mean effect (same paper, Table 2) |
| `data/ckw2018_table5_adjusted.csv` | Covariate-adjusted programme contrasts (NBER working-paper version, Table 5) |
| `data/SOURCE.md` | Full citation, column definitions and the consistency checks of the transcription |
| `src/riqfs_core.py` | RIQFS mathematics: admissible order, both operation designs, Hamacher operational laws, the RHWA operator, distances, evidence-theory mapping |
| `src/pipeline.py` | The distance-based decision procedure (Algorithm 1 of the supplementary material) |
| `src/plotstyle.py` | Common figure style |
| `analysis/` | Scripts that produce the outputs (steps 2-13 above) |
| `figures_src/` | TikZ sources of Figure 1 and Figure S2; `figures_src/icons/` holds the icons of Figure 1 |
| `tests/test_theory.py` | Numerical checks of the theory (see the map below) |
| `outputs/tables_plain/*.tex` | Table bodies input verbatim by the manuscript |
| `outputs/tables/*.tex` | The same tables with symbolic programme labels |
| `outputs/figures/workflow_epp.png` | Figure 1 |
| `outputs/figures/encoding.png` | Figure 2 |
| `outputs/figures/channels.png` | Figure 3 |
| `outputs/figures/pairwise.png` | Figure 4 |
| `outputs/figures/weight_space.png` | Figure 5 |
| `outputs/figures/disagreement_curves.png` | Figure C.1 |
| `outputs/figures/kappa_sweep.png` | Figure C.2 |
| `outputs/figures/breakeven.png` | Figure C.3 |
| `outputs/figures/discontinuity.png`, `flowchart.png`, `gamma_sensitivity.png` | Figures S1, S2 and S3 of the supplementary material |
| `outputs/figures/heatmaps.png`, `decomposition.png`, `evidence_bpa.png`, `rank_acceptability.png`, `framework.png` | Additional diagnostics, not shown in the paper |
| `outputs/csv/encoded_grades.csv` | Encoded grades of every cell, with the intervals behind them |
| `outputs/csv/closeness.csv` | Scores and channel disagreement of each programme |
| `outputs/*.json` | All computed results |
| `outputs/*macros.tex` | Every number quoted in the text, as LaTeX macros |

## Map from results to tests

Numbers refer to the supplementary material; where a result is also stated in Appendix A of the manuscript, that number is given in brackets.

| Result | Test |
|---|---|
| Theorem S1.11 (closure, lattice, De Morgan laws) | `test_selection_lattice_demorgan` |
| Theorem S1.14 (no continuous union under a linear order) | `test_selection_union_discontinuous` |
| Proposition S1.16 (properties and counterexamples of Design C) | `test_renormalised_properties`, `test_renormalised_counterexamples` |
| Proposition S1.17 (sources as discounted evidence) [Prop. A.2] | `test_evidence_interpretation` |
| Theorem S2.2, Proposition S2.3 (metrics, equivalence) | `test_metric_and_equivalence` |
| Theorem S2.4, Proposition S2.5 (orders and metrics) | `test_order_metric_compatibility` |
| Lemma S3.1 (Hamacher generator) | `test_hamacher_identities` |
| Theorems S3.3 and S3.5 (operational laws, closed form of the operator) | `test_operational_laws_and_operator` |
| Theorem S3.5 (monotonicity), Remark S3.7 | `test_sep_operator_componentwise_monotone_and_general_not` |
| Theorem S3.6 (midpoint characterisation) | `test_kolmogorov_nagumo_midpoint` |
| Theorem S4.3 (Hamming distance gives the linear score) [Prop. A.3] | `test_hamming_collapse` |
| Theorem S4.5 (Euclidean distance and channel disagreement) [Prop. A.4] | `test_euclidean_disagreement_decomposition` |
| Theorem S4.6 (stability in the Hamacher parameter) | `test_gamma_bound` |

## Data statement

The case-study data are **published statistics**, transcribed from Tables 2 and 3a of Card, Kluve and Weber (2018) and from Table 5 of their NBER working paper (see `data/SOURCE.md`). `case_study.py` checks the transcription on every run: the counts add up to the published totals at every horizon, and the count-weighted means and shares reproduce the published overall values, except for one short-term mean (see `data/SOURCE.md`), which is treated as a sensitivity case. To analyse other evidence, replace the CSV files, keeping the same columns, and rerun `run_all.py`; every table, figure and macro file is then regenerated.

## Environment used for the reported results

Python 3.13.18, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1. The code uses only stable APIs and should also run with the minimum versions in `requirements.txt`. The random seeds are fixed (acceptability analysis seed 11, parametric bootstrap seed 3, tests seed 20261123), so reruns reproduce the reported numbers exactly; only the computing times (supplementary Proposition S4.7) depend on the machine.

## Running on Windows

Tested on Windows with Python 3.15 in VS Code and MiKTeX. Install the packages first (`pip install -r requirements.txt`); if pytest is missing, `run_all.py` skips the theorem tests with a warning. On the first run MiKTeX may open a dialog asking to install missing LaTeX packages (for example `cm-super`); accept it, or set `RIQFS_NO_USETEX=1` to draw the plots without LaTeX. Files written on Windows have CRLF line endings; their contents are otherwise identical.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). The data files in `data/` contain published statistics transcribed from Card, Kluve and Weber (2018) and their NBER working paper; please cite the original source when you reuse them. The icons in Figure 1 (`figures_src/icons/`) are from Tabler Icons (https://tabler.io/icons) and are distributed under their own MIT licence (`figures_src/icons/LICENSE_tabler_icons.txt`).
