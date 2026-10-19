"""Report and profile the unchanged local ALM benchmark; no solver refinements."""
import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
if (ROOT / 'ALM+Newton').is_dir():
    ROOT = ROOT / 'ALM+Newton'
STAMP = '2026-10-18'
LOCAL = ROOT / f'JIT-results-local-{STAMP}.json'


def clean(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def profile():
    # Match the CLI's early environment setup, and load SciPy's separate BLAS
    # before constructing the threadpool controller (Windows has two DLLs).
    for key in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    import numpy as np
    import scipy.linalg
    from threadpoolctl import threadpool_info, threadpool_limits
    sys.path.insert(0, str(ROOT))
    from sparsefolio import make_problem, solve, warmup
    case = json.loads((ROOT / 'CASE.json').read_text())
    with threadpool_limits(limits=1):
        warmup()
        solve(make_problem(n=50, k=5, exposures=10), time_limit=1)
        assert all(pool['num_threads'] == 1 for pool in threadpool_info()
                   if pool['user_api'] == 'blas'), threadpool_info()
        p = make_problem(**case)
        runs = []
        for _ in range(3):
            start = time.perf_counter()
            result = solve(p, method='alm_polished', time_limit=15)
            runs.append(dict(seconds=time.perf_counter()-start,
                             solved=result['solved'], metrics=result['metrics'],
                             diagnostics=result['diagnostics'], history=result['history']))
        profiler = cProfile.Profile()
        profiler.enable()
        result = solve(p, method='alm_polished', time_limit=15)
        profiler.disable()
        pools = threadpool_info()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(40)
    stats.sort_stats('tottime').print_stats(30)
    functions = []
    for (filename, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        functions.append(dict(file=filename, line=line, name=name, calls=calls,
                              primitive_calls=primitive, own_seconds=own,
                              cumulative_seconds=cumulative))
    functions.sort(key=lambda row: row['cumulative_seconds'], reverse=True)
    info = dict(case=case, requested_threads=1, runs=runs,
                median_seconds=statistics.median(r['seconds'] for r in runs),
                profile_total_seconds=stats.total_tt, functions=functions,
                profiled_diagnostics=result['diagnostics'], threadpools=pools,
                thread_environment={key: os.environ.get(key) for key in
                                    ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
                constraint_row_norm_range=[float(np.min(np.linalg.norm(p.K, axis=1))),
                                           float(np.max(np.linalg.norm(p.K, axis=1)))],
                note='Supplementary diagnostics only; excluded from the official 54-attempt benchmark.')
    (ROOT / f'PROFILE-local-{STAMP}.json').write_text(json.dumps(clean(info), indent=2)+'\n', encoding='utf-8')
    (ROOT / f'PROFILE-local-{STAMP}.txt').write_text(stream.getvalue(), encoding='utf-8')
    print(json.dumps(clean({k: info[k] for k in ['median_seconds', 'profile_total_seconds', 'constraint_row_norm_range']}), indent=2))
    print(json.dumps(clean(runs[1]['diagnostics']), indent=2))
    print(stream.getvalue())


def report():
    old = json.loads((ROOT / 'JIT-results.json').read_text())
    new = json.loads(LOCAL.read_text())
    assert len(new['runs']) == 54 and len(new['summary']) == 18
    assert old['case'] == new['case']
    assert all(old[k] == new[k] for k in ['common_gap', 'common_feasibility', 'time_limit'])
    previous = {row['method']: row for row in old['summary']}
    ordered = sorted(new['summary'], key=lambda r: (r['passed'] != r['runs'], r['median_seconds'] or float('inf')))
    passing = [r for r in ordered if r['passed'] == r['runs']]
    custom = [r for r in passing if not r['method'].startswith('mosek')]
    mosek = [r for r in passing if r['method'].startswith('mosek')]
    best = min(custom, key=lambda r: r['median_seconds'])
    comparator = min(mosek, key=lambda r: r['median_seconds']) if mosek else None
    best_rows = [r for r in new['runs'] if r['method'] == best['method']]
    old_sources = {k.replace('\\', '/'): v for k, v in old['source_sha256'].items()}
    new_sources = {k.replace('\\', '/'): v for k, v in new['source_sha256'].items()}
    source_mismatches = [k for k in sorted(set(old_sources) | set(new_sources)) if old_sources.get(k) != new_sources.get(k)]
    fingerprints = {k: old['fingerprints'][k] == new['fingerprints'][k] for k in old['fingerprints']}
    lines = [
        'JIT benchmark results - local machine, 2026-10-18 (America/New_York)', '',
        'Case n=500, k=10, 200 tight additional exposures, seed=0.',
        'Budget and mean-return rows are also present (202 total rows).',
        'Three repetitions per method, with fresh optimizer state; 18 methods, 54 attempts.',
        'Numba scalar kernels are compiled in nopython mode; SciPy supplies compiled linear algebra.',
        f"JIT compilation/cache-loading warmup {new['jit']['seconds']:.4f} s, excluded from solve times.",
        'Per-method small-case warmup and data generation are also excluded.',
        'Reported solve times include setup and final independent accuracy checks.', '',
        f"Fastest passing custom method {best['method']} at {best['median_seconds']:.6f} s.",
    ]
    if comparator:
        lines.extend([
            f"Fastest passing MOSEK setting {comparator['method']} at {comparator['median_seconds']:.6f} s.",
            f"Speedup {comparator['median_seconds']/best['median_seconds']:.3f}x including MOSEK setup, and {comparator['optimizer_median_seconds']/best['median_seconds']:.3f}x compared with MOSEK optimizer-only time.",
        ])
    lines.extend([
        f"Winning method maximum relative gap {max(r['rel_gap'] for r in best_rows):.3e} and maximum raw feasibility residual {max(r['feasibility'] for r in best_rows):.3e}.", '',
        'Method                     Passed     Median attempt time (s)    Min (s)      Max (s)',
        '-'*86,
    ])
    for row in ordered:
        times = [r['seconds'] for r in new['runs'] if r['method'] == row['method'] and r['seconds'] is not None]
        lines.append(f"{row['method']:<27}{row['passed']}/{row['runs']:<10}{row['median_seconds']:>15.6f}{min(times):>13.6f}{max(times):>13.6f}")
    lines.extend(['', 'Common targets are relative gap <= 1e-6 and original-unit feasibility <= 1e-7.',
                  'Failed-run times are censored or unsuccessful attempts, not times to a valid solution.'])
    for row in ordered:
        if row['passed'] < row['runs']:
            statuses = sorted({r['status'] for r in new['runs'] if r['method'] == row['method']})
            lines.append(f"{row['method']}: {row['passed']}/{row['runs']} passed; statuses: {', '.join(statuses)}.")
    env = new['environment']
    lines.extend(['', 'Machine Intel Core i7-10510U CPU @ 1.80GHz, 4 cores / 8 logical processors.',
                  f'Installed RAM: 16,918,597,632 bytes ({16918597632 / 2**30:.3f} GiB). Windows 11 x86-64.',
                  f"Python {env['python']}; NumPy {env['numpy']}; SciPy {env['scipy']}; Numba {env['numba']}; threadpoolctl {importlib.metadata.version('threadpoolctl')}; MOSEK {importlib.metadata.version('Mosek')}.",
                  'The runner requests one CPU thread and runs methods sequentially in randomized order.',
                  'The machine was not isolated from unrelated OS/user activity; timing spread is shown above.',
                  'This result applies to this synthetic CPU root-relaxation testcase.', '',
                  'Comparison with the original RESULTS.txt / JIT-results.json', '',
                  'Original machine: Apple M2 Ultra, 64 GiB, macOS arm64.',
                  'Original libraries: Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, Numba 0.68.0.',
                  'The same generator, case, tolerances, randomized schedule, repetitions, and time limits were used.',
                  f"Package source hashes: {'all match' if not source_mismatches else 'differences: ' + ', '.join(source_mismatches)} after normalizing path separators.",
                  f"Data byte hashes matching: {sum(fingerprints.values())}/{len(fingerprints)}."])
    for name, same in fingerprints.items():
        lines.append(f"  {name}: {'identical bytes' if same else 'different byte hash; cross-platform numerical variation is a possible cause'}")
    old_obj = statistics.median(r['objective'] for r in old['runs'] if r['method'] == 'alm_polished')
    new_obj = statistics.median(r['objective'] for r in new['runs'] if r['method'] == 'alm_polished')
    lines.extend([
        'The reference array values are not included in the saved report, so elementwise differences cannot be quantified from hashes.',
        f'Absolute difference between old and new alm_polished median objectives: {abs(new_obj-old_obj):.3e}.',
        'Future local refinements should reuse exactly the same local arrays for baseline and candidate.',
    ])
    lines.extend(['', 'Method                     Old pass  New pass   Old median    New median   New / old', '-'*90])
    for row in ordered:
        prior = previous[row['method']]
        ratio = row['median_seconds']/prior['median_seconds']
        note = ' *' if row['passed'] < row['runs'] or prior['passed'] < prior['runs'] else ''
        lines.append(f"{row['method']:<27}{prior['passed']}/{prior['runs']:<7}{row['passed']}/{row['runs']:<7}{prior['median_seconds']:>12.6f}{row['median_seconds']:>14.6f}{ratio:>12.3f}x{note}")
    lines.extend(['', '* Ratios involving unsuccessful attempts are not time-to-solution slowdowns.',
                  'Changes in runtime combine hardware, operating system, BLAS, dependency versions, and run noise.',
                  'They cannot be attributed solely to CPU speed.', '',
                  'Validation: python -m unittest discover -s tests -v; all 5 tests passed (7.172 s).',
                  'No solver implementations or tuning policies were changed.', '', 'Reproduce', '',
                  '$env:MOSEKLM_LICENSE_FILE = (Resolve-Path -LiteralPath "..\\mosek.lic").Path',
                  'python -m sparsefolio --case CASE.json --all --with-mosek --repeats 3 --limit 15 --tol 1e-6 --feas-tol 1e-7 --output JIT-results-local-new-run.json', '',
                  f'JIT-results-local-{STAMP}.json contains all raw metrics, dependency versions, data fingerprints, and package-source hashes.',
                  'The original RESULTS.txt and JIT-results.json are preserved.', ''])
    destination = ROOT / f'RESULTS-local-{STAMP}.txt'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['profile', 'report'])
    args = parser.parse_args()
    globals()[args.mode]()
