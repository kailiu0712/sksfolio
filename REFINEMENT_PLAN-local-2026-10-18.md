ALM-polished refinement plan, based on the local benchmark of 2026-10-18

This document proposes experiments. No refinement or solver tuning has been implemented. The baseline is the unchanged public `alm_polished` method. Preserve its two-phase ALM + primal-dual polishing architecture, perspective model, and independent accuracy test.

**Measured baseline and scope.** The complete original runner was executed for 18 methods, three repetitions each, with n=500, k=10, 200 additional tight exposure rows, seed=0, a 15-second limit, relative gap <=1e-6, and original-unit feasibility <=1e-7. The local ALM-polished median is 0.710782 s; MOSEK-fast is 1.356840 s. All five unit tests passed. See RESULTS-local-2026-10-18.txt and JIT-results-local-2026-10-18.json.

The original reference used an Apple M2 Ultra and different NumPy/SciPy/Numba versions. This machine is an Intel Core i7-10510U with approximately 16 GiB RAM. All package-source hashes match. Two of ten generated-array byte hashes match across environments; the original arrays are not supplied, so hash differences cannot establish the magnitude of elementwise differences. The ALM-polished objectives agree within 7.633e-17. For future local comparisons, construct one problem and feed identical arrays to both methods.

**What the diagnostic run says.** Separate profiling was performed after the official benchmark. Its timings are not substituted for the benchmark times. Both NumPy and SciPy OpenBLAS pools were verified at one thread. An initial diagnostic harness allowed a lazily loaded SciPy pool to use eight threads; that measurement was discarded and the saved PROFILE files contain the corrected run. The original CLI sets its thread environment before imports; those environment variables were initially unset on this machine.

The unprofiled diagnostic repetitions had a median of 1.016353 s, versus 0.710782 s in the earlier official run. The profiled run took 1.960050 s. These differences show substantial run noise and profiler overhead; use the following breakdown as directional evidence, not a calibrated prediction of savings.

| Work | Observation |
|---|---|
| ALM warm phase | 16 outer iterations; 167 Newton steps; 112 backtracks; 295 prox calls |
| ALM linear solves | 154 support solves; 13 Woodbury solves; largest direct system 201; no linear failures |
| Penalty adaptation | Five increases |
| Certification | 71 ALM certificate checks; 77 evaluate calls across the complete solve |
| Final polishing | Two KKT solves; no backtracks; support 18; active rows 18 |
| `_direct_direction` | 0.541 s cumulative, approximately 28% of profiled solve |
| ALM `point` | 0.501 s cumulative, approximately 26% |
| `evaluate` | 0.218 s cumulative, approximately 11% |
| Fast scalar prox | 0.055 s cumulative, approximately 3%; included in calling routines above |
| Polishing | 0.019 s cumulative, approximately 1% |

Cumulative times overlap; do not add these percentages. Almost all time is spent reaching the polishing start. Speeding up the final two KKT solves has little room to improve end-to-end time.

**Experiment 1: exploit the diagonal-minus-rank-one Jacobian during matrix assembly.**

Location: `sparsefolio/improvements/alm_fast.py::_direct_direction`, especially the support branch; later apply the same identity in `active_set.py::polish` if worthwhile.

The code constructs a dense J and multiplies it by another dense matrix even though J = diag(d) - h h^T / delta. For any matrix G,

    J G = d[:, None] * G - h[:, None] * (h @ G)[None, :] / delta.

In the support branch G = U_S^T U_S. Therefore the Newton matrix can be assembled as

    M = diag(1 + regularization - d)
        + tau * d[:, None] * G
        + h[:, None] * (h - tau * (h @ G))[None, :] / delta.

When delta is zero, omit the rank-one contribution. This eliminates a generic cubic matrix multiplication during assembly; the dense factorization still has its existing cost. Keep the current support/Woodbury choice and residual safeguards. This is a small algebraic change with no model or Newton-equation change, apart from floating-point operation order.

Validation: compare linear residuals against the current assembled matrix on actual recorded directions and representative active/capped pieces, then compare end-to-end solves. Do not judge success only by a matrix microbenchmark.

**Experiment 2: fuse the surrounding vector work with Numba and reuse buffers.**

Locations: `alm_fast.py::point`, `candidate_y`, and the arithmetic surrounding the existing prox; possibly a separate optimized certificate candidate routine.

The scalar prox, coordinate formulas, and perspective-value routine already use `njit(cache=True)`. The larger opportunity is the repeated Python/NumPy work around them: clipping, active masks, residuals, squared norms, perspective cost, and temporary arrays. Create small typed numeric kernels that fuse these passes and return arrays/scalars rather than dictionaries. Reuse per-solve work buffers, retaining independent current and trial states so rejected line searches cannot corrupt the current point. Hoist invariant masks, caps, and quantities such as tau*gamma and sqrt(tau*gamma) while tau remains fixed.

Keep large matrix products and factorizations in the existing BLAS/LAPACK path initially. A JIT wrapper around an already compiled BLAS call is not itself evidence of a speedup. Start with strict floating-point semantics. `fastmath=True` can change infinity/NaN behavior, which matters because the interval data contain infinite bounds. Extra Numba threads would change the one-thread benchmark and may cost more than they save for vectors of length 500. Disk caching reduces compilation/loading overhead, which the benchmark already excludes.

Numba's official guidance supports profiling real workloads, compiling numeric loops, and recognizing that linear algebra already delegates to BLAS/LAPACK: [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html).

**Experiment 3: reduce the ALM work before the two cheap polishing steps.**

Locations: `alm_polished.py::solve` and the configuration dictionary in `alm_fast.py::solve`.

The baseline warm result has relative gap about 1.05e-5 and feasibility about 9.01e-6, against warm targets 1e-4 and 1e-5. Feasibility is the tighter normalized gate at this endpoint. Merely loosening the gap target may therefore do nothing.

Try a small sequence of isolated changes:

- Warm feasibility target: 1e-5 (baseline), 3e-5, and 1e-4. Initially retain the 1e-4 warm gap target.
- Initial penalty factor: 3, 10 (baseline), and 30.
- If needed after those trials, inner forcing decay: 0.3, 0.4 (baseline), and 0.6.

Select on total warm-plus-polish time, success rate, and backtracking counts. An earlier handoff can make polishing much harder, so compare total work. Keep the final targets fixed at 1e-6 and 1e-7. Do not take the Cartesian product of every parameter initially. If a simple fixed handoff helps, a later small heuristic can require support/active-set stability before an early handoff, with a guarded return to ALM if polishing stalls.

**Experiment 4: cheaper certification and safe reuse of repeated calculations.**

Locations: `core.py::evaluate`, `alm_fast.py::certificate`, and wrapper/API handoffs.

The current solve performs 77 complete evaluations. Trial ideas are an ALM certificate stride of 10 instead of 5, removal of duplicate evaluations of exactly the same x,y pair, and sharing already computed quantities for that exact pair. A quantity evaluated at the Newton trial x cannot automatically be reused for its different proximal point px. Cache keys must account for the iterate and multipliers.

Keep the final original evaluator as the independent authority. Larger strides can delay stopping and lose the best observed candidate; measure whether reduced check cost outweighs extra Newton work. This is a small bookkeeping experiment, not permission to stop on a weaker metric.

**Experiment 5: modest variable scaling or linear-system equilibration.**

The generator already normalizes every constraint row to norm sqrt(500), approximately 22.36068. Repeating row normalization offers no new benefit. Also, the baseline uses direct linear solves, so a generic Krylov preconditioner is not a direct improvement to this path.

Two concrete candidates remain:

- Equilibrate a reduced Newton system before solving it: solve Dr M Dc w = Dr b and recover direction = Dc w. Use bounded positive row/column norm scalings; the KKT dual diagonal is zero, so naive diagonal inversion is inappropriate. Check the residual in the original system. This targets numerical conditioning, not a reduction in factorization complexity. With no recorded linear failures and only two polish solves, the expected benefit is uncertain.
- Scale asset variables with x = D w, choosing D_ii approximately 1/sqrt(gamma_i), with bounded scaling ratios. Transform B to BD, c to Dc, gamma_i to gamma_i D_ii^2, u_i to u_i/D_ii, and K to KD. Any additional row scaling must transform interval bounds and duals consistently. This preserves the perspective model but changes the geometry seen by the iterative method. Map weights and multipliers back and run the original evaluator. This has moderate implementation complexity and may alter cap activity and prox cost.

Try one or two inexpensive passes, and count preprocessing time inside each timed solve. Do not reuse preprocessing across benchmark repetitions unless every comparator is explicitly given the same different protocol. MOSEK discusses scaling as a numerical issue in its [modeling cookbook](https://docs.mosek.com/modeling-cookbook/practical.html).

**Experiment 6: a very small, provably equivalent presolve.**

Commercial-solver inspiration is useful for exact bound propagation and removing genuinely redundant constraints. On a nonnegative unit-budget portfolio, each row a satisfies min(a) <= a^T x <= max(a). These cheap bounds can identify redundant sides; exact duplicate rows can be merged by intersecting their intervals. Restore a valid full multiplier vector and evaluate all original constraints after solving.

This particular test contains many tight synthetic style bands, so easy reductions may be limited. The ten sector rows sum to the budget row, but that linear dependence alone does not permit dropping a sector inequality. Similarly, replacing a perspective cap u with a tighter inferred affine bound can strengthen the perspective relaxation; do not treat that as an equivalent presolve without proof.

MOSEK documents redundant-row and fixed-variable elimination, as well as the cost of presolve: [MOSEK presolve](https://docs.mosek.com/latest/pythonapi/presolver.html). Gurobi recommends comparing presolved numerical ranges because aggregation can worsen conditioning: [Gurobi numerical parameters](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/numeric_parameters.html). These are general design principles; the proposed changes are not reproductions of proprietary solver internals.

**Order and acceptance protocol.**

Start with experiments 1 and 2 as separate changes, then experiment 3. Follow with certificate bookkeeping. Investigate scaling and presolve after those measurements, because the profile gives them less direct support. Combine only changes that win independently.

For each candidate, hold n=500, k=10, exposures=200, tight bands, seed=0, the exact local input arrays, final tolerances, fresh state, and the 15-second limit fixed. Verify both BLAS pools and any Numba thread pool. Compile kernels and perform the same small-instance warmup outside timing. Count all per-solve setup/preprocessing and final evaluation inside timing.

The three original repetitions are sufficient to reproduce RESULTS.txt but are noisy for small refinements: baseline times ranged from 0.658 to 0.990 s, and later diagnostic runs varied further. Use at least 20 interleaved baseline/candidate pairs, randomizing order within pairs. Report median, interquartile range, all pass counts, and paired ratios. Confirm a candidate in a second batch before calling a small change a speedup. A practical initial target is a repeatable >=10% improvement with every run passing; that threshold is an experiment criterion, not a promised gain.

Use the existing five unit tests plus focused checks for any altered Jacobian algebra or variable/dual transformation. For unchanged arithmetic and parameter-only candidates, do not create tests that merely repeat configuration constants. After choosing a winner on the requested exact case, neighboring seeds or sizes can be a separate robustness check, clearly distinguished from this benchmark.

No candidate code, parameter sweep, alternative scaling, or presolve has been implemented in this task.
