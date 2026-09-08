# Running the scaling benchmarks

Both scripts are configured in code, so **no arguments are needed**: press
Run in VS Code with `code0821brian/.venv_opt` selected as the interpreter,
or run them from a terminal.

```
cd c:/Users/kai/liukh/RA/proximal/code0821brian
.venv_opt/Scripts/python.exe sksfolio-bnb-main/sksfolio-bnb-main/sksfolio/benchmarks/run_four_scaling.py
.venv_opt/Scripts/python.exe sksfolio-bnb-main/sksfolio-bnb-main/sksfolio/benchmarks/run_four_scaling_real.py
```

Roughly two hours each. Every run measures all rows fresh and archives any
previous CSV at the output path as `.superseded-<timestamp>.csv`. After an
interruption, add `--resume` to reuse what the dead run had finished.

## Built-in configuration

Edit these near the top of `run_four_scaling.py` rather than passing flags:

| constant | value | meaning |
|---|---|---|
| `DEFAULT_DIMENSIONS` | 10 ... 5000, 17 rungs | prox and relaxation ladder |
| `DEFAULT_SEED_COUNT` | 10 | instances per dimension |
| `DEFAULT_EXACT_DIMENSIONS` | 14 rungs, full up to 1000 plus 2000 and 5000 | exact sparse ladder |
| `DEFAULT_EXACT_SEED_COUNT` | 5 | instances per dimension, exact only |

`run_four_scaling_real.py` reuses all of these except
`DEFAULT_EXACT_DIMENSIONS_REAL`, whose top rung is 3000: the widest real
universe holds about 3000 tickers, so larger dimensions cannot be sampled.

The exact experiment gets its own, shorter ladder because every method
exhausts its time limit from about n=1500 upward. Those rows cost the full
budget (about 143 seconds per dimension and seed, across both scenarios and
all five methods) and report only the cap, so the extra rungs would add two
and a half hours to show what the 2000 and 5000 anchors already show.

## Outputs, written to `code0821brian/results`

| | synthetic | real |
|---|---|---|
| rows | `four_scaling_results.csv` | `four_scaling_results_real.csv` |
| summary | `four_scaling_results_summary.csv` | `four_scaling_results_real_summary.csv` |
| timing figure | `four_scaling_six_panel.svg` | `four_scaling_six_panel_real.svg` |
| precision figure | `four_scaling_precision.svg` | `four_scaling_precision_real.svg` |

Read the precision figure before the timing figure: it reports how closely
each method's objective matches Gurobi's, and what share of runs actually
solved rather than hitting a limit. A mean elapsed time only means
something for the runs that produced an answer.
