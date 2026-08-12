# sksfolio 0.9 benchmark snapshot

These are reproducibility checks from 8 August 2026 on one Apple-silicon
macOS machine. They are not a claim of universal solver dominance.

## Large constrained instance

The generated instance has `d = 500`, factor rank `20`, `k = 25`, and 21
rows in `C`: one budget equality, one minimum-return bound, 10 sector bands,
4 style bands, and 5 one-sided stress limits. Every method receives the same
matrix and one thread.

### Continuous relaxation

| Method | Status | End-to-end seconds | Safe bound verified |
| --- | ---: | ---: | ---: |
| FISTA, dual L-BFGS-B prox | converged | 0.0433 | yes |
| FISTA, dual FISTA prox | converged | 0.1119 | yes |
| native Gurobi | optimal | 0.0870 | yes |
| native MOSEK | optimal | 0.1968 | yes |

The FISTA safe gaps were about `1.1e-9` and `2.0e-9`. The independently
reconstructed zero-row-multiplier certificates for the commercial primal
solutions are valid but much looser; their native solver bounds are reported
separately.

### Cardinality-constrained solve

All methods start from the same OSQP-polished feasible portfolio and use the
same absolute cutoff of `1e-4`.

| Method | End-to-end seconds | Absolute gap | Notes |
| --- | ---: | ---: | --- |
| safe-screened sksfolio BnB | 0.1256 | 1.25e-5 | 463 of 500 selectors screened at the root; no search nodes needed |
| native MOSEK perspective MISOCP | 0.3160 | 1.25e-5 | direct Python API |
| native Gurobi perspective MIQCP | 2.6611 | 3.86e-5 | direct Python API; 268 nodes |
| JuMP/MOSEK perspective MISOCP | 5.1447 | 1.25e-5 | includes Julia bridge/build overhead |
| JuMP/Gurobi perspective MISOCP | 8.1688 | 9.56e-5 | includes Julia bridge/build overhead; 283 nodes |

On this favorable screening instance, sksfolio is about `2.5x` faster than
native MOSEK and `21x` faster than native Gurobi at the matched cutoff. More
seeds, harder constraint geometry, and timeout counts are needed before
making a general performance claim.

The full record is
`outputs/student_benchmark/student_suite_v090_d500_k25_seed17.json`.

## Exact small-instance cross-check

For `d = 16`, rank `4`, `k = 4`, and 9 general rows, custom BnB, native
Gurobi, native MOSEK, JuMP/Gurobi, and JuMP/MOSEK returned the same checked
upper bound, `-0.006360341901474834`. Custom BnB closed the tree with zero
reported gap. The full record is
`outputs/student_benchmark/student_suite_v090_correctness_d16_k4_seed29.json`.

Julia's first use may spend substantial time resolving and precompiling its
project. The JSON retains those cold-start costs. For steady-state solver
comparisons, initialize Julia once, run multiple instances in the same Python
process, and report both end-to-end and solver-only times.
