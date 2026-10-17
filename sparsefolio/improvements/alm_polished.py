"""Fixed ALM warmup followed by reduced-support primal-dual Newton polishing.

Every instance uses the same schedule.  A modest ALM accuracy identifies the
primal support and binding exposures without driving its penalty to a numerical
floor.  The joint KKT stage then enforces the requested independent certificate.
No reference solution or known feasible anchor is used.
"""
from __future__ import annotations

import time
from ..core import evaluate
from .alm_fast import solve as solve_alm
from .active_set import polish


def solve(problem, mode='alm_polished', time_limit=15.0, tol=1e-6,
          feas_tol=1e-7, max_iter=100, config=None):
    if mode != 'alm_polished':
        raise ValueError('mode must be alm_polished')
    settings = dict(warm_tol=1e-4, warm_feas_tol=1e-5,
                    warm_max_iter=300, warm_time_fraction=0.8)
    if config:
        settings.update(config)
    start = time.perf_counter()
    warm = solve_alm(problem, mode='alm_fast_tuned',
                     time_limit=time_limit*settings['warm_time_fraction'],
                     tol=max(tol, settings['warm_tol']),
                     feas_tol=max(feas_tol, settings['warm_feas_tol']),
                     max_iter=settings['warm_max_iter'])
    warm_seconds = time.perf_counter()-start
    warm_metrics = evaluate(problem, warm['x'], warm['y'])
    warm_merit = max(warm_metrics['rel_gap']/tol, warm_metrics['feasibility']/feas_tol)
    polished = None
    remaining = time_limit-(time.perf_counter()-start)
    if warm_merit > 1.0 and remaining > 0.0:
        polished = polish(problem, warm['x'], warm['y'], time_limit=remaining,
                          tol=tol, feas_tol=feas_tol, max_iter=max_iter)
        polished_metrics = evaluate(problem, polished['x'], polished['y'])
        polished_merit = max(polished_metrics['rel_gap']/tol,
                             polished_metrics['feasibility']/feas_tol)
    else:
        polished_merit = float('inf')
    if polished_merit < warm_merit:
        selected, metrics, merit = polished, polished_metrics, polished_merit
    else:
        selected, metrics, merit = warm, warm_metrics, warm_merit
    elapsed = time.perf_counter()-start
    status = 'converged' if merit <= 1.0 else (
        'time_limit' if elapsed >= time_limit else 'not_converged')
    return dict(x=selected['x'], y=selected['y'], status=status,
                iterations=warm['iterations']+(polished['iterations'] if polished else 0),
                metrics=metrics, seconds=elapsed,
                diagnostics=dict(mode=mode, seconds=elapsed, config=settings,
                                 warm_seconds=warm_seconds, warm_metrics=warm_metrics,
                                 warm_diagnostics=warm['diagnostics'],
                                 polish_diagnostics=polished['diagnostics'] if polished else None,
                                 selected_phase='polish' if selected is polished else 'warm'),
                history=polished['history'] if polished else warm.get('history', []))
