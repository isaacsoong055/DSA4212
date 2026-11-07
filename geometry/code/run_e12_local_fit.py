"""E12: does a LOCALLY learned tilt follow a bending valley?

The report separates local geometry (the valley's shape where the search is now;
CMA-ES re-adapts it from recent steps) from global geometry (one metric fitted to
all the data, as a stationary GP kernel learns). On Rosenbrock the valley bends,
so the two differ. E5 could not test this: at 200 evaluations no BO run got far
along the valley, and the trust-region BO of E10 fits its GP to ALL points since
its last restart, so its metric is not local either (only its search is).

Arms (2-D, 15 instances, 300 evaluations, noise fixed at 1e-6; trust regions use
local_mode="reflect", which only changes candidates that fall outside the box):
  bo-full         plain EI, full metric                global shape, global search
  tr-full         trust region, full metric            global shape, local search
  tr-full-local   trust region, GP fitted only to the  local shape, local search
                  points inside the current region (at least the 20 nearest)
  tr-ard, tr-ard-local   the same two with an axis-aligned box (cannot tilt)
Problems: Rosenbrock f8 (aligned) and f9 (rotated) as in E5. Control: the rotated
kappa = 10,000 quadratic, whose shape is the same everywhere, with tr-full and
tr-full-local only.

Design note: a 3-instance pilot on f9 (150 evaluations) with the minimum at
2d + 4 = 8 points collapsed (log10 f -0.18 to +0.41 against -4.25 to -5.77 for
the global fit): 8 points cannot pin down a tilt. A minimum of 20 recovered
part of the gap (-2.18 to -3.48), so the local arms use 20.

Predictions, written before running (the pilot already argues against the first): on Rosenbrock tr-full-local beats tr-full
(a local fit follows the bend); on the quadratic control the two are level (the
local and global shapes coincide); the axis-aligned arms trail the tilting ones.

Writes results/e12_runs.csv and results/e12_summary.csv; checkpoint
results/e12_partial.jsonl.
Usage: python code/run_e12_local_fit.py [--workers N] [--report-only]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import itertools
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analysis
import bo
import problems
from run_e5_other_functions import rosenbrock

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
PARTIAL = RESULTS / "e12_partial.jsonl"
DIMENSION, BOX, CAP, N_INSTANCES = 2, 5.0, 300, 15
CHECKPOINTS = (100, 200, 300)
ROSEN_METHODS = ("bo-full", "tr-full", "tr-full-local", "tr-ard", "tr-ard-local")
CONTROL_METHODS = ("tr-full", "tr-full-local")
PROBLEMS = ("rosenbrock:none", "rosenbrock:random", "quadratic:random")


def objective(problem, instance):
    family, geo = problem.split(":")
    if family == "rosenbrock":
        return rosenbrock(instance, geo)
    f, _ = problems.make_ellipsoid(DIMENSION, 10000, instance=instance, rotation=geo)
    return f


def run_one(task):
    problem, instance, method = task
    f = objective(problem, instance)
    started = time.time()
    if method == "bo-full":
        out = bo.bayes_opt(f, DIMENSION, -BOX, BOX, "full", seed=instance, budget=CAP,
                           noise_variance=1e-6)
    else:
        metric = method.split("-")[1]
        out = bo.trust_region_bo(f, DIMENSION, -BOX, BOX, metric, seed=instance, budget=CAP,
                                 local_mode="reflect", local_fit=method.endswith("-local"),
                                 local_min_points=20)
    return {"problem": problem, "instance": instance, "method": method,
            "seconds": time.time() - started,
            "best_so_far": np.minimum.accumulate(out["y"]).tolist()}


def all_tasks():
    t = [(p, i, m) for p in PROBLEMS[:2] for i in range(N_INSTANCES) for m in ROSEN_METHODS]
    t += [(PROBLEMS[2], i, m) for i in range(N_INSTANCES) for m in CONTROL_METHODS]
    return sorted(t, key=lambda x: x[2] != "bo-full")          # slowest first


def sweep(workers):
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["problem"], r["instance"], r["method"]))
    todo = [t for t in all_tasks() if t not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"[{i}/{len(todo)}] {rec['method']} {rec['problem']} inst={rec['instance']} "
                  f"{rec['seconds']:.0f}s elapsed {time.time() - started:.0f}s", flush=True)


def report():
    pd.set_option("display.width", 220)
    rows = []
    for line in PARTIAL.read_text().splitlines():
        r = json.loads(line)
        b = np.asarray(r["best_so_far"])
        rows.append({k: r[k] for k in ("problem", "instance", "method", "seconds")}
                    | {f"log10_f_{n}": float(np.log10(b[n - 1] + 1e-300)) for n in CHECKPOINTS})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e12_runs.csv", index=False)
    print("\n== median log10 best f at 100 / 200 / 300 evaluations")
    print(df.pivot_table(index=["problem", "method"], values=[f"log10_f_{n}" for n in CHECKPOINTS],
                         aggfunc="median")[[f"log10_f_{n}" for n in CHECKPOINTS]].round(2).to_string())
    sel = lambda p, m: df[(df.problem == p) & (df.method == m)].sort_values("instance")["log10_f_300"].values
    out = []
    print("\n== paired at 300 evaluations (negative = first arm better)")
    pairs = [("tr-full-local", "tr-full"), ("tr-ard-local", "tr-ard"), ("tr-full", "bo-full"),
             ("tr-full-local", "bo-full"), ("tr-ard", "tr-full"), ("tr-ard-local", "tr-full-local")]
    for p in PROBLEMS:
        for a, b in pairs:
            if len(sel(p, a)) == 0 or len(sel(p, b)) == 0:
                continue
            res = analysis.bootstrap_paired_difference(sel(p, a), sel(p, b), n_boot=5000)
            out.append({"problem": p, "comparison": f"{a} - {b}", **res})
            flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
            print(f"  {p:18s} {a:14s} - {b:14s} {res['difference']:+6.2f} "
                  f"[{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]{flag}")
    print("\n== median wall-clock seconds per run")
    print(df.groupby("method").seconds.median().round(0).to_string())
    pd.DataFrame(out).to_csv(RESULTS / "e12_summary.csv", index=False)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    report()
