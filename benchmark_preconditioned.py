"""Interleaved baseline versus column-preconditioned ALM benchmark.

Usage: python benchmark_preconditioned.py --repeats 20
Compilation and problem creation are excluded; public-API solve time is included.
"""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from sparsefolio import make_problem, solve, warmup


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    attempts = [row for row in rows if row["type"] == "attempt"]
    for n, k in ((500, 10), (500, 200), (2000, 10), (2000, 200)):
        selected = [row for row in attempts if (row["n"], row["k"]) == (n, k)]
        baseline = {row["repeat"]: row for row in selected if row["method"] == "alm_polished"}
        scaled = {row["repeat"]: row for row in selected if row["method"] == "alm_preconditioned"}
        b = statistics.median(row["seconds"] for row in baseline.values())
        p = statistics.median(row["seconds"] for row in scaled.values())
        wins = sum(scaled[i]["seconds"] < baseline[i]["seconds"] for i in baseline.keys() & scaled.keys())
        print(f"({n}, {k}): baseline={b:.3f}s preconditioned={p:.3f}s "
              f"speedup={b/p:.3f}x wins={wins}/{len(baseline)} "
              f"solved={sum(row['solved'] for row in baseline.values())}/"
              f"{sum(row['solved'] for row in scaled.values())}")


def run(repeats, output):
    cases = ((500, 10), (500, 200), (2000, 10), (2000, 200))
    methods = ("alm_polished", "alm_preconditioned")
    with threadpool_limits(1):
        pools = [row for row in threadpool_info() if row["user_api"] == "blas"]
        if not pools or any(row["num_threads"] != 1 for row in pools):
            raise RuntimeError("Benchmark requires single-threaded BLAS")
        warmup()
        small = make_problem(n=50, k=5, exposures=10, seed=0)
        for method in methods:
            solve(small, method=method, time_limit=1)
        problems = {(n, k): make_problem(n=n, k=k, exposures=200,
                                          band="tight", seed=0)
                    for n, k in cases}
        with Path(output).open("w", encoding="utf-8") as handle:
            header = dict(type="protocol", cases=cases, repeats=repeats,
                          methods=methods, exposures=200, band="tight", seed=0,
                          limit=15, tol=1e-6, feas_tol=1e-7,
                          blas_threads=1, numpy=np.__version__)
            handle.write(json.dumps(header) + "\n")
            for n, k in cases:
                problem = problems[(n, k)]
                for repeat in range(repeats):
                    order = list(methods)
                    random.Random(94113 + 1000*n + 10*k + repeat).shuffle(order)
                    for method in order:
                        start = time.perf_counter()
                        cpu_start = time.process_time()
                        try:
                            result = solve(problem, method=method, time_limit=15,
                                           tol=1e-6, feas_tol=1e-7)
                            row = dict(type="attempt", n=n, k=k, repeat=repeat,
                                       method=method, seconds=time.perf_counter()-start,
                                       cpu_seconds=time.process_time()-cpu_start,
                                       solved=result["solved"], status=result["status"],
                                       objective=result["metrics"]["objective"],
                                       rel_gap=result["metrics"]["rel_gap"],
                                       feasibility=result["metrics"]["feasibility"],
                                       warm_steps=result["diagnostics"]["warm_diagnostics"]["newton_steps"],
                                       polish_steps=(result["diagnostics"]["polish_diagnostics"] or {}).get("kkt_solves", 0))
                        except Exception as exc:
                            row = dict(type="attempt", n=n, k=k, repeat=repeat,
                                       method=method, seconds=time.perf_counter()-start,
                                       cpu_seconds=time.process_time()-cpu_start,
                                       solved=False, status="exception", error=repr(exc))
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        print(f"n={n} k={k} r={repeat} {method}: "
                              f"{row['seconds']:.3f}s {row['status']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="PRECONDITIONED_RESULTS.jsonl")
    args = parser.parse_args()
    run(args.repeats, args.output)
    summarize(args.output)
