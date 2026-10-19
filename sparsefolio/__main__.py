"""Portable benchmark runner with compact output and explicit JIT warmup."""
import os

# This module is imported before NumPy because package exports are lazy.
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")


def main():
    import argparse
    import hashlib
    import json
    import platform
    import random
    import statistics
    import time
    from pathlib import Path
    import numpy as np
    import scipy
    import numba
    from threadpoolctl import threadpool_limits
    from .api import CUSTOM_METHODS, MOSEK_METHODS, solve, warmup
    from .datasets import make_problem

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, help="JSON with n, k, exposures, band, seed")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exposures", type=int, default=200)
    parser.add_argument("--band", choices=("tight", "wide", "inactive"), default="tight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=CUSTOM_METHODS+MOSEK_METHODS,
                        default=["alm_polished"])
    parser.add_argument("--all", action="store_true", help="Run all 17 custom variants")
    parser.add_argument("--with-mosek", action="store_true", help="Also run both licensed MOSEK settings")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=float, default=15.0)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--feas-tol", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.limit <= 0:
        parser.error("repeats and limit must be positive")
    if args.output.exists():
        parser.error("Use a new output path to avoid overwriting results")
    settings = {k: getattr(args, k) for k in ("n", "k", "exposures", "band", "seed")}
    if args.case:
        settings.update(json.loads(args.case.read_text()))
    selected = list(CUSTOM_METHODS) if args.all else list(args.methods)
    if args.with_mosek:
        selected.extend(MOSEK_METHODS)
    selected = list(dict.fromkeys(selected))

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        jit = warmup()
        print(json.dumps({"jit_warmup": jit}), flush=True)
        # Warm imports and library initialization equally for every method.
        small = make_problem(n=50, k=5, exposures=10, band="tight", seed=0)
        method_warmup, unavailable = {}, {}
        for method in selected:
            start = time.perf_counter()
            try:
                solve(small, method=method, time_limit=1.0, tol=args.tol, feas_tol=args.feas_tol)
            except Exception as error:
                unavailable[method] = repr(error)
            method_warmup[method] = time.perf_counter()-start
        p = make_problem(**settings)
        arrays = ("B", "c", "gamma", "u", "K", "lo", "hi", "raw_K", "raw_lo", "raw_hi")
        fingerprints = {name: hashlib.sha256(np.ascontiguousarray(getattr(p, name)).tobytes()).hexdigest()
                        for name in arrays}
        sources = {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(Path(__file__).parent.rglob("*.py"))}
        report = dict(case=settings, common_gap=args.tol, common_feasibility=args.feas_tol,
                      time_limit=args.limit, jit=jit, method_warmup_seconds=method_warmup,
                      unavailable=unavailable, fingerprints=fingerprints, source_sha256=sources,
                      environment=dict(python=platform.python_version(), platform=platform.platform(),
                                       numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                                       requested_threads=1), runs=[])
        for repeat in range(args.repeats):
            order = selected.copy()
            random.Random(717+settings["n"]+settings["exposures"]+2741*repeat).shuffle(order)
            for method in order:
                start = time.perf_counter()
                row = dict(method=method, repeat=repeat)
                if method in unavailable:
                    row.update(status="unavailable", solved=False, seconds=None, error=unavailable[method])
                else:
                    try:
                        result = solve(p, method=method, time_limit=args.limit,
                                       tol=args.tol, feas_tol=args.feas_tol)
                        row.update(seconds=time.perf_counter()-start, solved=result["solved"],
                                   status=result["status"], optimizer_seconds=result.get("solver_seconds"),
                                   **result["metrics"])
                    except Exception as error:
                        row.update(seconds=time.perf_counter()-start, solved=False,
                                   status="error", error=repr(error))
                report["runs"].append(clean(row))
                args.output.write_text(json.dumps(clean(report), indent=2)+"\n")
                print(json.dumps(clean(row)), flush=True)
        report["summary"] = []
        for method in selected:
            rows = [r for r in report["runs"] if r["method"] == method]
            durations = [r["seconds"] for r in rows if r["seconds"] is not None]
            optimizer = [r["optimizer_seconds"] for r in rows if r.get("optimizer_seconds") is not None]
            report["summary"].append(dict(method=method, passed=sum(r["solved"] for r in rows),
                                          runs=len(rows), median_seconds=statistics.median(durations) if durations else None,
                                          optimizer_median_seconds=statistics.median(optimizer) if optimizer else None))
        args.output.write_text(json.dumps(clean(report), indent=2)+"\n")
        print(json.dumps({"summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
