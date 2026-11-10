"""E10: does a TuRBO-style trust region inherit ARD's axis restriction?

The report argues that TuRBO (Eriksson et al. 2019) inherits the restriction
under test, because its trust region is an axis-aligned box whose sides are
set per coordinate from the ARD lengthscales. This tests the argument with
bo.trust_region_bo: one trust region, TuRBO-1's expand/shrink/restart rules,
EI over 1,000 candidates inside the region.
  tr-ard   axis-aligned box, sides from the ARD lengthscales (published TuRBO)
  tr-full  box rotated to the full metric's eigenvectors (can tilt)

Problems as E2: 2-D ellipsoids, kappa in {100, 10000}, aligned and
Haar-rotated, 15 instances (seed = instance), box [-5, 5]^2, cap 200, noise
fixed at 1e-6.

Prediction, written before running: tr-ard pays a rotation penalty at
kappa = 10,000 and tr-full does not; on rotated problems tr-full beats tr-ard,
on aligned ones they are level or tr-ard is ahead.

Writes results/e10_trust_region_runs.csv and results/e10_trust_region_summary.csv;
raw checkpoint results/e10_trust_region_partial.jsonl.
Usage: python code/run_e10_trust_region.py [--workers N] [--report-only]
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
PARTIAL = RESULTS / "e10_trust_region_partial.jsonl"
DIMENSION, BOX, CAP, N_INSTANCES = 2, 5.0, 200, 15
CONDITIONS = (100, 10000)
GEOMETRIES = ("none", "random")
METRICS = ("ard", "full")
CHECKPOINTS = (50, 200)


def run_one(task):
    condition, geometry, instance, metric = task
    f, _ = problems.make_ellipsoid(DIMENSION, condition, instance=instance, rotation=geometry)
    started = time.time()
    out = bo.trust_region_bo(f, DIMENSION, -BOX, BOX, metric, seed=instance, budget=CAP)
    return {"condition": condition, "geometry": geometry, "instance": instance,
            "method": f"tr-{metric}", "seconds": time.time() - started,
            "best_so_far": np.minimum.accumulate(out["y"]).tolist()}


def sweep(workers):
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["geometry"], r["instance"], r["method"][3:]))
    todo = [t for t in itertools.product(CONDITIONS, GEOMETRIES, range(N_INSTANCES), METRICS)
            if t not in done]
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
    for line in PARTIAL.read_text().splitlines():
        r = json.loads(line)
        best = np.asarray(r["best_so_far"])
        rows.append({k: r[k] for k in ("condition", "geometry", "instance", "method", "seconds")}
                    | {f"log10_f_{n}": float(np.log10(best[n - 1] + 1e-300)) for n in CHECKPOINTS})
    return pd.DataFrame(rows)


def report(df):
    pd.set_option("display.width", 200)
    print("\n== median log10 f at 50 and 200 evaluations")
    print(df.pivot_table(index=["condition", "geometry"], columns="method",
                         values=["log10_f_50", "log10_f_200"], aggfunc="median").round(2).to_string())
    sel = lambda c, g, m: df[(df.condition == c) & (df.geometry == g) & (df.method == m)] \
        .sort_values("instance")["log10_f_200"].values
    out = []

    def show(label, res, extra):
        out.append({**extra, **res})
        flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
        print(f"  {label:36s} {res['difference']:+6.2f} [{res['ci_low']:+6.2f}, "
              f"{res['ci_high']:+6.2f}]{flag}")

    print("\n== rotation penalty at 200, rotated - aligned (unpaired; + = rotation hurts)")
    rng = np.random.default_rng(0)
    for c, m in itertools.product(CONDITIONS, ("tr-ard", "tr-full")):
        a, b = sel(c, "random", m), sel(c, "none", m)
        draws = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
                 for _ in range(5000)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        show(f"k={c:<6} {m}", {"difference": float(np.median(a) - np.median(b)),
                                "ci_low": float(lo), "ci_high": float(hi)},
             {"comparison": f"{m}: rotated - aligned", "condition": c})
    print("\n== tr-ard - tr-full, paired by instance (+ = ARD box worse)")
    for c, g in itertools.product(CONDITIONS, GEOMETRIES):
        show(f"k={c:<6} {g}", analysis.bootstrap_paired_difference(
            sel(c, g, "tr-ard"), sel(c, g, "tr-full"), n_boot=5000),
            {"comparison": "tr-ard - tr-full", "condition": c, "geometry": g})
    print("\n== median wall-clock seconds per run")
    print(df.groupby("method").seconds.median().round(0).to_string())
    return pd.DataFrame(out)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    df = table()
    df.to_csv(RESULTS / "e10_trust_region_runs.csv", index=False)
    report(df).to_csv(RESULTS / "e10_trust_region_summary.csv", index=False)
