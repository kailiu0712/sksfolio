# Bertsimas--Cory-Wright benchmark

This directory separates two different numerical questions:

1. `paper_reference_results.csv` records representative published timings
   from Tables 7--9 of
   [*A Scalable Algorithm for Sparse Portfolio Selection*](https://arxiv.org/pdf/1811.00138).
2. `run_relaxations.py` compares `sksfolio` backends on the corresponding
   processed-panel dimensions and parameter shapes.

The numbers are not expected to match.  The paper timings solve the binary
cardinality problem with Algorithm 3 or CPLEX MISOCO on the authors' hardware.
`sksfolio` solves the continuous perspective relaxation.  This runner always
uses a deterministic synthetic factor generator; it has no original-returns
input path.
The authors'
[reference Julia repository](https://github.com/ryancorywright/SparsePortfolioSelection.jl)
contains the original table scripts and some source data.

The public manifest also enumerates the 30 OR-Library profiles from Tables 3
and 4.  They require the original `port1`--`port5` covariance data and are not
runnable through this synthetic historical-profile runner.

The default command runs all five PDHG variants, both PAVA implementations,
and native Python Gurobi on the processed S&P panel dimension of 499 assets,
rank 50, `k = 10`, and `gamma = 100 / sqrt(499)`:

```bash
python benchmarks/bertsimas_cory_wright_2022/run_relaxations.py \
  --output results/sp500_rank50_k10.csv
```

MOSEK and Julia are opt-in:

```bash
python benchmarks/bertsimas_cory_wright_2022/run_relaxations.py \
  --backend pdhg \
  --backend gurobi.julia \
  --backend mosek.python \
  --backend mosek.julia
```

No timing is used as a pass/fail assertion.  Tests independently validate the
two PAVA oracles, verify safe dual certificates on generated instances, and
check weak duality against an optimal Gurobi result when a license exists.
They do not treat an unconverged PDHG primal objective as an optimal value.

## Matched PDHG--FISTA comparison

`run_pdhg_fista.py` generates one synthetic instance and passes that same
in-memory problem to all five PDHG variants with both PAVA methods, FISTA with
both PAVA methods, and native Python Gurobi.  The matched runner permits only
the `unconstrained` profile, whose sole explicit linear constraint is
`1^T x = 1`.  FISTA has no variant label; its two rows differ only by the PAVA
implementation.

```bash
python benchmarks/bertsimas_cory_wright_2022/run_pdhg_fista.py \
  --max-iterations 5000 \
  --time-limit 600 \
  --output benchmarks/bertsimas_cory_wright_2022/sp500_rank50_k10_pdhg_fista_results.csv
```

This writes a separate file and does not replace the PDHG snapshot below.
Every row records a common synthetic instance identifier and the requested
PDHG and FISTA stopping and line-search settings.  The matched default uses a
`1e-8` first-order tolerance and a `1e-10` commercial-reference tolerance.
Rows include the independently evaluated primal-feasibility flag and, when a
safe certificate exists, its gap to the optimal Gurobi reference.  Compare
objective values only together with the recorded constraint violation and
residual.  A status of `iteration_limit` or `time_limit` is not convergence.
These are synthetic continuous-relaxation results, not a reproduction of the
paper's integer timings.

The checked-in `sp500_rank50_k10_pdhg_fista_results.csv` is the command above
with seed 7.  Both FISTA rows report `converged` after 46 iterations, maximum
primal violation below `5e-15`, objective difference from the optimal Gurobi
row below `4e-11`, and a verified safe-dual gap below `4e-10`.  All ten PDHG
rows report `iteration_limit` at 5,000 iterations; their maximum violations
range from about `2.3e-6` to `3.8e-3`.  Those PDHG rows are therefore retained
for matched timing and safe-bound context, not as converged primal solutions.
Runtime numbers describe this single run and are not pass/fail assertions.

## Four proximal strategies

`run_four_prox_variants.py` compares the four applicable implementations:

- PDHG with direct PAVA for `G_k`;
- outer FISTA with warm-started Brent/PAVA for the budget-only profile;
- outer FISTA with the warm-started dual-FISTA constrained prox;
- outer FISTA with the exact weak-majorization OSQP lift;
- native Python Gurobi as the reference.

```bash
python benchmarks/bertsimas_cory_wright_2022/run_four_prox_variants.py \
  --tolerance 1e-5 \
  --time-limit 60 \
  --output results/four_prox_budget.csv
```

Add `--regime constrained` for the budget-plus-minimum-return profile. The
budget-only Brent oracle is omitted in that profile because it cannot enforce
the extra row. The majorization formulation is exact, but has
`d k + 2 k - 1` variables and can be disabled by omitting
`--fista-prox-oracle majorization_qp` when OSQP is unavailable.

The checked smoke files
`output/benchmark/four_prox_sp500_budget_final.csv` and
`output/benchmark/four_prox_sp500_return_smoke.csv` were generated at 499
assets, rank 50, `k = 10`, partial-sort PAVA, one thread, and tolerance
`1e-5`. They are comparison artifacts rather than timing assertions.

The many-row comparison is:

```bash
python benchmarks/bertsimas_cory_wright_2022/run_many_constraints.py \
  --time-limit 60 \
  --output results/many_constraints.csv
```

Its default constrained profile stacks the budget and return rows with 20
sector bands, 40 style-factor bands, and 200 stress-loss rows, for 262 rows
in total. Every bound is centered on the saved feasible anchor.

For a converged three-row head-to-head, see
`sp500_rank50_k10_best_pdhg_fista_results.csv`.  On the same budget-only
instance and the same `1e-8` stopping tolerance, metric line-search/restart
PDHG with partial-sort PAVA converged in 6,050 iterations and 1.821 seconds,
while adaptive-restart FISTA with partial-sort PAVA converged in 46 iterations
and 0.0778 seconds.  Tight-tolerance Gurobi took 0.1030 seconds in total.  Both
first-order rows are primal feasible, and their safe-dual certificates verify
and obey weak duality.  These timings are one matched run, not a general
performance guarantee.

## Checked-in smoke result

`sp500_rank50_k10_results.csv` is a deterministic synthetic S&P-shaped run
with 499 assets, rank 50, `k = 10`, `gamma = 100 / sqrt(499)`, one thread,
and 5,000 PDHG iterations.  It contains every PDHG variant with both PAVA
methods plus native Gurobi.  The rows record the synthetic provenance, random
seed, stopping configuration, solver-only time, and total backend time.

The full-sort and partial-sort versions agree in their saved dual bounds to
machine precision.  Partial sorting reduced PDHG runtime in every pair in
this run.  All saved PDHG rows stopped at the iteration limit, so their primal
iterates are not a feasibility or convergence baseline.  The file is only a
snapshot of safe dual-bound and full-sort/partial-sort consistency.  Native
Gurobi was still much faster on this relaxation; the intended speed comparison
is at much larger dimensions, where full conic model construction and generic
solver memory become more important.
