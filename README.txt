Sparse portfolio solvers

This small package includes all 17 custom solver variants, two optional MOSEK configurations, and the exact synthetic testcase generator used in the experiments.
CASE.json selects 500 assets, a selection budget of 10, 200 additional tight exposure rows, and seed zero.
The generator also supports the larger tested cases without shipping large matrix files.

Quick start

Use Python 3.12 for the tested dependency versions.
From the extracted folder, run

    python -m pip install -r requirements.txt
    python -m sparsefolio --case CASE.json

The default method is alm_polished.
Results are written to results.json.
Use a fresh --output path when running again.
To install the package for import from other directories, run python -m pip install .

Run every custom variant

    python -m sparsefolio --case CASE.json --all --repeats 3 --limit 15 --output all-results.json

Slow variants can consume the entire time limit for each repetition.
Time limits are checked between numerical operations, so a single factorization may finish after the limit.

Compare with MOSEK

Install the optional Mosek Python package and configure a valid license.
No MOSEK installation or license is needed for the custom solvers.

    python -m pip install "Mosek==11.2.2"
    python -m sparsefolio --case CASE.json --all --with-mosek --repeats 3 --limit 15 --output comparison.json

The MOSEK settings use internal tolerances 1e-7 and 1e-9.
Every method must pass the same independent relative-gap target 1e-6 and feasibility target 1e-7.
The comparator is the fastest MOSEK setting that passes in every repetition.
A solver's own success status is not sufficient.

Use the Python API

    from sparsefolio import make_problem, solve, warmup
    warmup()
    problem = make_problem(n=500, k=10, exposures=200, band="tight", seed=0)
    result = solve(problem, method="alm_polished", time_limit=15)
    x = result["x"]
    print(result["solved"], result["metrics"])

To reproduce the larger case, set n=2000 and exposures=1000.
The CLI also accepts --n 2000 --exposures 1000 without --case.
When supplied, CASE.json overrides the corresponding CLI problem settings.

JIT compilation and timing

The scalar perspective prox, its coordinate kernel, and the perspective-value evaluator use Numba njit with disk caching.
Matrix factorizations use compiled SciPy LAPACK routines.
The outer Python control loops are not compiled by Numba.
The runner checks that the timed kernels have Numba nopython signatures.
It refuses a benchmark when NUMBA_DISABLE_JIT disables compilation.

Compilation or cache loading is reported separately from solve time.
Each method also receives an untimed small-instance warmup.
Measured solves start afresh from their own initialization inside the warmed process.
Reported times include solve setup and final independent accuracy assessment.
Data generation and imports are outside the timed solve.
The runner requests one CPU thread and executes methods sequentially in randomized order.
The packaged JIT-results.json and RESULTS.txt record fresh runs of this package.
They contain compact metrics and source hashes, without large histories or solution archives.

Available methods

    pdhg                  Fixed-step Condat-Vu primal-dual splitting
    adaptive_pdhg         Adaptive primal-dual splitting
    alm_apg               ALM with accelerated proximal-gradient inner solves
    newton_dense          Nested constrained prox with dense outer Newton solves
    newton_krylov         Nested constrained prox with outer GMRES
    newton_krylov_inexact  Nested Krylov method with inexact inner forcing
    alm_original          Original ALM with GMRES
    alm_fast              Faster scalar prox and structured ALM Newton solves
    alm_fast_tuned        Structured ALM with the fixed revised policy
    active_set            Independent warmup and joint primal-dual KKT polishing
    alm_polished          Recommended fixed ALM warmup plus KKT polishing
    alm_preconditioned    Same ALM polishing with clipped exposure-column scaling
    nested_fast_prox      Faster scalar prox inside the original nested method
    nested_fast_outer     Adds structured outer Newton solves
    nested_fast           Adds structured inner dual solves
    nested_fast_inexact   Structured nested method with inexact inner forcing
    ipm                   Experimental structured interior-point prototype
    mosek_fast            Optional conic reference with internal tolerance 1e-7
    mosek_reference       Optional conic reference with internal tolerance 1e-9

The IPM prototype uses the supplied feasible anchor when present.
It is included for completeness and can fail the common accuracy target.
The recommended hybrid starts from uniform holdings and zero multipliers.
All variants solve continuous root relaxations, not the integer selection problem.

The testcase

The objective is 0.5*||B*x||^2 + c.T*x + g(x).
The function g is the perspective relaxation of the diagonal quadratic with 0 <= z <= 1, 0 <= x <= u*z, and sum(z) <= k.
The constraints include budget, a mean-return lower bound, ten balanced sector rows, and additional correlated synthetic style rows.
All asset lower bounds are zero and upper bounds are one.
Tight halfwidths are 0.02 for sectors and 0.03 for styles.

The Problem object exposes the arrays directly.
K, lo, and hi contain all scaled constraints, including budget.
raw_K, raw_lo, and raw_hi contain the original units.
A = raw_K[1:], b_l = raw_lo[1:], and b_u = raw_hi[1:] exclude only the budget row.
B, c, gamma, u, and k define the objective and perspective domain.
Valid custom input requires finite B and c, positive finite gamma and u, compatible array dimensions, at least one constraint row, consistent constraint bounds, and integer 1 <= k <= n.
Only zero buy-in lower bounds are implemented.

Provenance and checks

The numerical solver and generator functions come from the benchmarked experiment.
Imports are adjusted for the package namespace.
The common perspective-value evaluator uses the corrected guarded scalar bracket for general positive caps and coefficients.
The previous evaluator's initial bracket was safe for the tested family but could fail on other inputs.
This package's fresh results use the corrected evaluator for every variant.
source_manifest.json records the original and packaged source hashes.

    python -m unittest discover -s tests -v

The tests cover Numba compilation, agreement of the fast and original prox, the corrected value bracket, 80 structured inner linear systems, optional imports, and the recommended solver's common certificate.
The experiments establish CPU results on this synthetic five-factor family.
They do not establish GPU or branch-and-bound performance.
The reported numerical gap and feasibility residual are separate checks, not an exact feasible primal upper bound.
