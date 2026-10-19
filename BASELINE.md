# ALM-Newton baseline: 2026-10-18

The Git tag `baseline-almnewton-2026-10-18` identifies the baseline before any
proposed refinement is implemented. The solver source is unchanged from commit
`adeee752de52315a20c5dbf1728a894c1e446a41`.

## Saved evidence

- `RESULTS.txt` and `JIT-results.json`: original benchmark.
- `RESULTS-local-2026-10-18.txt` and `JIT-results-local-2026-10-18.json`:
  the local Windows benchmark, comparison, accuracy metrics, and source hashes.
- `PROFILE-local-2026-10-18.txt` and `.json`: separate diagnostic profiling;
  instrumented timings do not replace benchmark timings.
- `REFINEMENT_PLAN-local-2026-10-18.md`: proposed experiments only.
- `alm_polished_walkthrough.py` and `alm_polished_walkthrough_trace.json`:
  a three-asset numerical walkthrough; `ALM-Newton illustration.png` accompanies
  the method documentation.
- `analyze_alm_local.py`: report and profiling helper. Both helper scripts support
  this repository layout and the original workspace layout.

## Benchmark protocol

`CASE.json` fixes 500 assets, cardinality 10, 200 additional tight exposures,
and seed 0. There are 202 constraint rows including budget and mean return.
The CLI ran all 18 methods, three repetitions each, with a 15-second limit,
relative-gap tolerance `1e-6`, and raw feasibility tolerance `1e-7`.
The local `alm_polished` median was **0.710782 s**, passing 3/3 runs;
`mosek_fast` took **1.356840 s**, passing 3/3 runs.

The original and local runs use the same source and case configuration, but
their data fingerprints are not all identical across platforms. The local
report records that limitation and dependency versions. Future refinements
should be compared against this baseline on the same local generated arrays.

From the repository root, with dependencies installed and a valid MOSEK license
configured in your environment:

```powershell
python -m unittest discover -s tests -v
python alm_polished_walkthrough.py
python -m sparsefolio --case CASE.json --all --with-mosek --repeats 3 --limit 15 --tol 1e-6 --feas-tol 1e-7 --output JIT-results-local-new-run.json
```

The walkthrough regenerates its trace. `python analyze_alm_local.py report`
regenerates the dated report from saved JSON; `python analyze_alm_local.py profile`
reruns diagnostics and replaces the dated profile files. Preserve the tagged
artifacts when comparing subsequent runs. License files are not included.

## Return to this baseline

Create a separate checkout without disturbing ongoing refinement work:

```powershell
git fetch origin --tags
git worktree add ../sksfolio-baseline baseline-almnewton-2026-10-18
```

To compare a later commit with the baseline:

```powershell
git diff baseline-almnewton-2026-10-18 HEAD -- sparsefolio tests CASE.json
```
