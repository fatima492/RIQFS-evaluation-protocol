# Source of the case-study data

All numbers are transcribed from

> D. Card, J. Kluve, A. Weber (2018). What works? A meta analysis of recent active labor market program evaluations. *Journal of the European Economic Association* 16(3), 894–931. https://doi.org/10.1093/jeea/jvx028
> (NBER Working Paper 21431, revised April 2017, https://www.nber.org/papers/w21431)

* `ckw2018_table3a.csv`: Table 3a, rows "Training", "Job Search Assist." (includes sanction programmes), "Private Subsidy", "Public Sector Emp." and "Other"; columns 4-6 (mean programme effect on the probability of employment ×100, with the number of estimates) and columns 7-9 (percentage of estimates significantly positive, with the number of estimates classified by sign and significance).
* `ckw2018_table2.csv`: Table 2, subsample with an estimated effect on the employment probability (mean and its standard error, clustered by study) and the full-sample distribution of sign and significance.

Horizons: short term = up to one year after the programme ends; medium term = one to two years; long term = more than two years.

Consistency checks (see `analysis/case_study.py`): the counts in Table 3a add up to the totals in Table 2 for every horizon; the count-weighted type means reproduce the medium- and long-term means of Table 2 (5.4 and 8.7), and the count-weighted shares reproduce the overall shares (40, 52 and 61 per cent). At the short horizon the printed type means average to 2.3 rather than 1.6; reversing the sign of the public-sector short-term mean (3.6 → −3.6) would reconcile the two. The printed values are used, and the alternative is reported as a sensitivity case.

## Covariate-adjusted programme contrasts (`ckw2018_table5_adjusted.csv`)

Transcribed from Table 5 ("Estimated Program Effects on Probability of Employment") of NBER Working Paper 21431 (revised April 2017): linear regressions of the effect on the employment probability (352 estimates, standard errors clustered by study), omitted categories short term and training. `pooled_controls` = column 2 (horizon dummies, programme-type dummies and the additional controls: participant type, age, gender, country group, outcome variable, experimental design, programme duration, publication status, citation rank). `interactions_controls` = column 4 (programme type × horizon interactions with the same controls). Coefficients are in proportion units (0.042 = 4.2 percentage points). The published JEEA version may differ slightly; check before relying on the third decimal.
