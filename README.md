Sequential-test bisection:

It contains the simulation implementations to regenerate numerical summaries and current simulation figures.

The source files generate their own synthetic observations. No precomputed task archives, paper sources, compiled PDFs, screenshots or historical notebooks are needed to run this package. Those artifacts are in the separate full experiment release. This repository does not publish to GitHub or start computation automatically.

## Files

- `core/`: STB/ASTB, SA/ASA/PJ kernels, independent tuning/selection, checkpointed campaign, and raw-result audit/aggregation.
- `core/revision_design.json`: complete candidate grid, scoring rule, budgets, repetitions and stream domains.
- `core/setting_selection.json`: frozen independent screening decision, including all ten candidate scores. The formal run uses this decision and independently retunes comparison methods.
- `legacy/`: original kernels and design for the four quadratic confidence-sequence settings and the positive-slope cubic distribution experiment. These use their original random streams.
- `scripts/run.py`: run a fresh screening or formal campaign in a separate directory.
- `scripts/run_auxiliary.py`: run only the 3,000 auxiliary tasks needed by the current supplement.
- `scripts/summarize.py`: audit every expected raw task, aggregate results and export Tables 2–9.
- `scripts/plot.py`, `plot_helpers.py`: current simulation figures, including the three-panel Figure 3 and six-row distribution/QQ plots.
- `scripts/check.py`: small installation check; its outputs are not experiment results.
- `MANIFEST.json`: file hashes and hashes of the unchanged scientific source files.

## Install

Use **Python 3.11** on Linux or macOS:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check.py
```


## Reproduce the full numerical experiments

```sh
# Frozen common setting; repeats all independent formal tuning and evaluation.
python scripts/run.py formal --workers 4 --output results/formal

# Separate original quadratic coverage and positive-slope cubic QQ experiments.
python scripts/run_auxiliary.py --workers 4 --output results/auxiliary

# Audits all repetitions before producing numerical summaries.
python scripts/summarize.py

# Rebuilds figures directly from the audited generated CSVs.
python scripts/plot.py
```

The formal campaign creates **180 tuning and 6,800 evaluation tasks**. The auxiliary campaign creates **800 quadratic STB and 2,200 cubic ASTB tasks**.

The three formal families are linear, flat cubic and crossing jump. They share root $0.37$, signal amplitude $1$, independent Gaussian noise with standard deviation $0.05$, initial interval $[0,1]$, confidence error $0.05$ and master seed **42**. The grid is every integer power of ten from $10^2$ through $10^9$, including $10^8$. All method/family/budget comparisons use 200 repetitions. Linear ASTB distribution diagnostics use 400 repetitions at $10^3$–$10^6$ and 200 at $10^7$–$10^9$; the extra repetitions are excluded from rate-comparison summaries.

SA optimal denotes the variance-optimal coefficient for the linear model, or the minimum independent tuning MAE within the declared finite grid for other families. PJ is independently tuned over its coefficient/exponent grid. The full grid, scaling, tie rules, 60 formal tuning repetitions, screening criterion and exact root fractions are in the design JSON. Raw method IDs keep `SA selected`; plots display `SA optimal`.

The auxiliary quadratic experiment uses roots $\frac{1}{3}$ and $0.37$, each with noise $0.02$ and $0.2$. The auxiliary positive-slope cubic experiment uses $f(x) = x - \theta + 2(x - \theta)^3$ and noise $0.5$. It differs from the flat-cubic comparison. ASTB's all-observation fitted root is not projected onto the sampling interval. Its positive-slope normal limit is not asserted for flat cubic or jump signals.

## Current paper output mapping

All CSV names below are under `results/data/`. Table exports are under `results/tables/`; plots are under `results/figures/`.

| Paper item | Numerical source | Output |
|---|---|---|
| Tables 2–4: linear, flat cubic, jump errors | `error_summary.csv` | `table_2/3/4.csv` and `.tex` |
| Figure 3: (a) linear, (b) Flat cubic signal, (c) jump | `error_summary.csv` | `sim_error_combined.pdf/png` |
| Figure 4: transformed jump error | STB jump rows of `error_summary.csv` | `jump_transformed.pdf/png` |
| Tables 5, 7, 8; Figures 5, 7, 8: coverage and interval length | `coverage_length_summary.csv` | `table_5/7/8.csv`; `cs_gamma1/3/0.pdf/png` |
| Table 6; Figure 6: quadratic coverage | `legacy_gamma2_coverage.csv` | `table_6.csv`; `cs_gamma2.pdf/png` |
| Table 9: linear ASTB moments | `clt_summary.csv` | `table_9.csv` |
| Figures 9–10: linear ASTB distribution and QQ | `astb_all_replicates.csv`, linear rows | `sim_linear_hist_vertical.pdf/png`; `sim_linear_qq_vertical.pdf/png` |
| Figure 11: positive-slope cubic QQ | `legacy_cubic_qq_replicates.csv` | `sim_cubic_reference_qq_vertical.pdf/png` |

Figure 3's flat-cubic panel displays the original range $n \geq 10^3$; its title has no `detail` suffix. Distribution and QQ panels display $n = 10^4, 10^5, 10^6, 10^7, 10^8, 10^9$ in one vertical column. Tables export numeric values and $\log_{10}$ values rather than rounding tiny values to zero.
