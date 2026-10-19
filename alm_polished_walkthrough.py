"""Trace the actual ALM-polished implementation on a three-asset example.

Run from the workspace: python alm_polished_walkthrough.py
The solver source is imported unchanged from ALM+Newton/sparsefolio or the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "ALM+Newton" if (ROOT / "ALM+Newton").is_dir() else ROOT
sys.path.insert(0, str(SOURCE_ROOT))

from sparsefolio import Problem, solve, warmup
from sparsefolio.core import evaluate, value_and_z
from sparsefolio.improvements.fast_core import prox
from sparsefolio.improvements.alm_fast import (
    _direct_direction, _norm_squared, solve as solve_alm,
)
from sparsefolio.improvements.active_set import polish


def make_toy():
    # Budget x1+x2+x3=1 and an upper exposure bound x1<=0.4.
    return Problem(
        B=np.eye(3), c=-np.array([0.5, 0.3, 0.2]),
        gamma=np.ones(3), u=np.ones(3), k=2,
        K=np.array([[1., 1., 1.], [1., 0., 0.]]),
        lo=np.array([1., -np.inf]), hi=np.array([1., 0.4]),
    )


def first_alm_step(p):
    """Expose the same intermediates computed by alm_fast.solve/point."""
    x, y = np.full(p.n, 1 / p.n), np.zeros(len(p.lo))
    lf, k2 = _norm_squared(p.B), _norm_squared(p.K)
    rho = 10 * max(1e-3, lf / max(k2, 1e-12))
    tau = 0.95 / (lf + rho * k2)

    def point(xx):
        shifted = p.K @ xx + y / rho
        clipped = np.clip(shifted, p.lo, p.hi)
        normal = rho * (shifted - clipped)
        gradient = p.B.T @ (p.B @ xx) + p.c + p.K.T @ normal
        v = xx - tau * gradient
        px, z, theta, d, h, den = prox(v, tau, p.gamma, p.u, p.k)
        residual = xx - px
        gv = 0.5 * np.sum(p.gamma * np.divide(px * px, z,
                                              out=np.zeros_like(px), where=z > 0))
        smooth = (0.5 * np.linalg.norm(p.B @ xx)**2 + p.c @ xx
                  + 0.5 * rho * np.linalg.norm(shifted - clipped)**2)
        envelope = smooth + gv - gradient @ residual + residual @ residual / (2 * tau)
        active = (p.lo == p.hi) | (shifted < p.lo) | (shifted > p.hi)
        jacobian = np.diag(d) - (np.outer(h, h) / den if den > 0 else 0)
        return dict(x=xx, shifted=shifted, clipped=clipped, normal=normal,
                    gradient=gradient, v=v, px=px, z=z, theta=theta,
                    d=d, h=h, denominator=den, jacobian=jacobian,
                    residual=residual, envelope=float(envelope), active=active)

    current = point(x)
    ka = p.K[current['active']]
    hessian = p.B.T @ p.B + rho * (ka.T @ ka)
    regularization = min(1e-2, max(1e-10, 0.1 * np.sqrt(np.linalg.norm(current['residual']))))
    U = np.vstack((p.B, np.sqrt(rho) * ka))
    direction, dimension, branch = _direct_direction(
        current['d'], current['h'], current['denominator'], U,
        tau, regularization, -current['residual'])
    jacobian = current['jacobian']
    matrix = ((1 + regularization) * np.eye(p.n) - jacobian
              + tau * jacobian @ hessian)
    linear_residual = np.linalg.norm(matrix @ direction + current['residual'])
    radius = max(0.1, 2 * np.linalg.norm(x))
    if np.linalg.norm(direction) > radius:
        direction *= radius / np.linalg.norm(direction)
    envelope_gradient = (current['residual'] - tau * hessian @ current['residual']) / tau
    accepted = None
    for backtracks in range(22):
        alpha = 0.5**backtracks
        trial_x = np.clip(x + alpha * direction, 0, p.u)
        trial = point(trial_x)
        predicted = float(envelope_gradient @ (trial_x - x))
        allowance = 5e-16 * max(1.0, abs(current['envelope']))
        residual_ok = np.linalg.norm(trial['residual']) <= (1 - 1e-3 * alpha) * np.linalg.norm(current['residual'])
        envelope_ok = predicted < 0 and trial['envelope'] <= current['envelope'] + 1e-4 * predicted + allowance
        if residual_ok or envelope_ok:
            accepted = dict(alpha=alpha, backtracks=backtracks, point=trial)
            break
    shifted_px = p.K @ current['px'] + y / rho
    # This is a certificate candidate, not an early outer-multiplier update.
    candidate_y = rho * (shifted_px - np.clip(shifted_px, p.lo, p.hi))
    return dict(lf=lf, k2=k2, rho=rho, tau=tau, initial=current,
                candidate_y_at_initial_px=candidate_y, hessian=hessian,
                newton_matrix=matrix, regularization=regularization,
                direction=direction, system_dimension=dimension, branch=branch,
                linear_residual=linear_residual, accepted=accepted)


def summary(result):
    return {key: result[key] for key in ('x', 'y', 'status', 'iterations', 'metrics', 'diagnostics', 'history')}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main():
    np.set_printoptions(precision=10, suppress=True)
    with threadpool_limits(limits=1):
        warmup()
        p = make_toy()
        first = first_alm_step(p)
        # This is the same phase schedule used by alm_polished.solve.
        warm = solve_alm(p, mode='alm_fast_tuned', time_limit=12,
                         tol=1e-4, feas_tol=1e-5, max_iter=300)
        polished = polish(p, warm['x'], warm['y'], time_limit=3,
                          tol=1e-6, feas_tol=1e-7, max_iter=5000)
        full = solve(p, method='alm_polished', time_limit=15)
        exact_x, exact_y = np.array([.4, .35, .25]), np.array([-.55, .15])
        _, exact_z = value_and_z(exact_x, p.gamma, p.u, p.k)
        exact_metrics = evaluate(p, exact_x, exact_y)
        np.testing.assert_allclose(full['x'], exact_x, atol=2e-6, rtol=0)
        np.testing.assert_allclose(polished['x'], full['x'], atol=2e-6, rtol=0)
        assert full['solved'], full['metrics']
        assert first['linear_residual'] < 1e-12
        report = dict(first_alm_step=first, warm=summary(warm),
                      polished=summary(polished), full=summary(full),
                      exact=dict(x=exact_x, y=exact_y, z=exact_z, metrics=exact_metrics))

    output = ROOT / 'alm_polished_walkthrough_trace.json'
    output.write_text(json.dumps(clean(report), indent=2) + '\n', encoding='utf-8')
    print('FIRST POINT:')
    for key in ('lf', 'k2', 'rho', 'tau', 'candidate_y_at_initial_px'):
        print(f'  {key}: {first[key]}')
    for key in ('x', 'gradient', 'v', 'px', 'z', 'theta', 'jacobian', 'residual'):
        print(f'  {key}: {first["initial"][key]}')
    print('FIRST NEWTON STEP:')
    print('  direction:', first['direction'])
    print('  accepted alpha:', first['accepted']['alpha'])
    print('  trial x:', first['accepted']['point']['x'])
    print('WARM PHASE, LAST CERTIFICATE PER OUTER ITERATION:')
    by_outer = {row['outer_iteration']: row for row in warm['history']}
    for outer, row in by_outer.items():
        print(f'  outer={outer:2d}, Newton attempts={row["iteration"]:2d}, '
              f'rho={row["rho"]:.6g}, feasibility={row["feasibility"]:.3e}, '
              f'rel_gap={row["rel_gap"]:.3e}')
    for name, result in (('WARM', warm), ('POLISHED', polished), ('FULL PUBLIC API', full)):
        print(f'{name}: x={result["x"]}, y={result["y"]}')
        print('  metrics:', result['metrics'])
        print('  iterations:', result['iterations'])
    print('POLISH HISTORY:')
    for row in polished['history']:
        print(f'  iteration={row["iteration"]}, feasibility={row["feasibility"]:.3e}, '
              f'rel_gap={row["rel_gap"]:.3e}, residual merit={row["residual"]:.3e}')
    print('EXACT:', exact_metrics, 'z=', exact_z)
    print('Trace saved to:', output)


if __name__ == '__main__':
    main()
