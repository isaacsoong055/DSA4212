"""E8: is the geometry result specific to the squared-exponential kernel?

E2 (with the BO fix) used a squared-exponential (SE) kernel throughout. This
re-runs its key cells with the Matern-5/2 kernel, BoTorch's default, with
everything else identical: 2-D ellipsoids, kappa in {100, 10000}, aligned and
Haar-rotated, 15 instances (seed = instance), box [-5, 5]^2, cap 200, noise
fixed at 1e-6, MAP fit with the same prior. ARD and full metrics are the same
choice in both kernels (the metric M sits inside r^2 = d^T M d).

Prediction, written before running: the geometry result does not depend on
the kernel's smoothness, so with Matern-5/2 (a) BO-ARD pays a rotation penalty
at kappa = 10,000 and BO-full does not, and (b) ARD minus full on the rotated
kappa = 10,000 problem is positive, as with SE (+1.40 [+0.68, +1.69]).

The SE runs are read from results/e2_bofix_runs_partial.jsonl (same machine,
same settings), so SE vs Matern is paired by instance.

Writes results/e8_matern_runs.csv and results/e8_matern_summary.csv; raw
checkpoint results/e8_matern_partial.jsonl.
Usage: python code/run_e8_matern.py [--workers N] [--report-only]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # repeatable BO runs (E2e)
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

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
PARTIAL = RESULTS / "e8_matern_partial.jsonl"
SE_PARTIAL = RESULTS / "e2_bofix_runs_partial.jsonl"
DIMENSION, BOX, CAP, N_INSTANCES = 2, 5.0, 200, 15
CONDITIONS = (100, 10000)
GEOMETRIES = ("none", "random")
METRICS = ("ard", "full")
CHECKPOINTS = (50, 200)


def run_one(task):
    condition, geometry, instance, metric = task
    f, _ = problems.make_ellipsoid(DIMENSION, condition, instance=instance, rotation=geometry)
    values = []

    def counted(x):
        v = float(f(x))
        values.append(v)
        return v

    started = time.time()
    bo.bayes_opt(counted, DIMENSION, -BOX, BOX, metric, seed=instance, budget=CAP,
                 noise_variance=1e-6, kernel="matern52")
    return {"condition": condition, "geometry": geometry, "instance": instance,
            "method": f"bo-{metric}", "kernel": "matern52", "seconds": time.time() - started,
            "best_so_far": np.minimum.accumulate(values).tolist()}


def sweep(workers):
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["geometry"], r["instance"], r["method"][3:]))
    todo = [t for t in itertools.product(CONDITIONS, GEOMETRIES, range(N_INSTANCES), METRICS)
            if t not in done]
    todo.sort(key=lambda t: t[3] != "full")
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"[{i}/{len(todo)}] {rec['method']} k={rec['condition']} {rec['geometry']} "
                  f"inst={rec['instance']} {rec['seconds']:.0f}s elapsed "
                  f"{time.time() - started:.0f}s", flush=True)


def table():
    rows = []
    sources = [(PARTIAL, "matern52"), (SE_PARTIAL, "se")]
    for path, kernel in sources:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r["method"].startswith("bo-") or r["condition"] not in CONDITIONS:
                continue
            best = np.asarray(r["best_so_far"])
            rows.append({"kernel": kernel, "condition": r["condition"], "geometry": r["geometry"],
                         "instance": r["instance"], "method": r["method"],
                         "seconds": r["seconds"],
                         **{f"log10_f_{n}": float(np.log10(best[n - 1] + 1e-300))
                            for n in CHECKPOINTS}})
    return pd.DataFrame(rows)


def report(df):
    pd.set_option("display.width", 200)
    print("\n== median log10 f at 200 evaluations")
    print(df.pivot_table(index=["condition", "geometry"], columns=["kernel", "method"],
                         values="log10_f_200", aggfunc="median").round(2).to_string())
    sel = lambda k, c, g, m: df[(df.kernel == k) & (df.condition == c) & (df.geometry == g)
                                & (df.method == m)].sort_values("instance")["log10_f_200"].values
    out = []

    def show(label, res, extra):
        out.append({**extra, **res})
        flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
        print(f"  {label:48s} {res['difference']:+6.2f} [{res['ci_low']:+6.2f}, "
              f"{res['ci_high']:+6.2f}]{flag}")

    print("\n== rotation penalty at 200, rotated - aligned (unpaired; + = rotation hurts)")
    rng = np.random.default_rng(0)
    for k, c, m in itertools.product(("matern52", "se"), CONDITIONS, ("bo-ard", "bo-full")):
        a, b = sel(k, c, "random", m), sel(k, c, "none", m)
        draws = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
                 for _ in range(5000)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        show(f"{k:9s} k={c:<6} {m}", {"difference": float(np.median(a) - np.median(b)),
                                       "ci_low": float(lo), "ci_high": float(hi)},
             {"comparison": f"{m}: rotated - aligned", "kernel": k, "condition": c})

    print("\n== ARD - full, paired by instance (+ = ARD worse)")
    for k, c, g in itertools.product(("matern52", "se"), CONDITIONS, GEOMETRIES):
        show(f"{k:9s} k={c:<6} {g}",
             analysis.bootstrap_paired_difference(sel(k, c, g, "bo-ard"), sel(k, c, g, "bo-full"),
                                                  n_boot=5000),
             {"comparison": "bo-ard - bo-full", "kernel": k, "condition": c, "geometry": g})

    print("\n== Matern - SE, paired by instance (negative = Matern better)")
    for c, g, m in itertools.product(CONDITIONS, GEOMETRIES, ("bo-ard", "bo-full")):
        show(f"k={c:<6} {g:7s} {m}",
             analysis.bootstrap_paired_difference(sel("matern52", c, g, m), sel("se", c, g, m),
                                                  n_boot=5000),
             {"comparison": f"{m}: matern52 - se", "condition": c, "geometry": g})
    print("\n== median wall-clock seconds per run")
    print(df.groupby(["kernel", "method"]).seconds.median().round(0).to_string())
    return pd.DataFrame(out)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    df = table()
    df.to_csv(RESULTS / "e8_matern_runs.csv", index=False)
    report(df).to_csv(RESULTS / "e8_matern_summary.csv", index=False)
