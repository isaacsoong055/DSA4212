"""E3b: why does the 'easier' change of units hurt BO-full in E3?

In E3, units s = (10, 1) (g(u) = f(S u), box still [-5, 5]^2 in the new units)
turned the aligned kappa = 100 bowl into a round one, which helped CMA-ES but
cost BO-full +0.91 [+0.62, +1.47] decades at 200 evaluations (ARD +0.45, n.s.).

Explanation to test. In the original units the bowl is unchanged, but BO's box
is ten times wider along x1 ([-50, 50]). BO with plain EI stalls when its 200
uniform candidates stop landing in the region where f is below the incumbent
(Section 3.4 / E2 local candidates), and a ten times larger box makes that
happen at a roughly ten times higher f.

Test: remove that cause and nothing else. Re-run E3's units-default:10,1 cell
with 200 extra candidates around the incumbent (bo.bayes_opt
local_candidates=200, as in E2 --local-cand), noise fixed at 1e-6.
Baseline: the same BO on the original bowl, read from E2 --local-cand
(results/e2_lc_runs_partial.jsonl, condition 100, aligned), which uses the same
problem, seed, box and budget, so the comparison is paired by instance.

Prediction, written before running: with local candidates the units penalty
for BO-full is gone (difference <= 0, interval covering 0 or below it). If it
stays near +0.9, the candidate search is not the cause; the box also sets the
initial design and the lengthscale prior's location, which this does not change.

Writes results/e3b_units_check_runs.csv and results/e3b_units_check_summary.csv;
raw checkpoint results/e3b_units_check_partial.jsonl.
Usage: python code/run_e3b_units_check.py [--workers N] [--report-only]
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
PARTIAL = RESULTS / "e3b_units_check_partial.jsonl"
BASELINE = RESULTS / "e2_lc_runs_partial.jsonl"
DIMENSION, CONDITION, BOX, CAP, N_INSTANCES = 2, 100, 5.0, 200, 15
SCALES = np.array([10.0, 1.0])
BO_OPTIONS = {"noise_variance": 1e-6, "local_candidates": 200}
METHODS = ("bo-ard", "bo-full")


def run_one(task):
    instance, method = task
    base, _ = problems.make_ellipsoid(DIMENSION, CONDITION, instance=instance, rotation="none")
    seen = problems.rescale_coordinates(base, SCALES)
    values = []

    def counted(u):
        v = float(seen(u))
        values.append(v)
        return v

    started = time.time()
    bo.bayes_opt(counted, DIMENSION, -BOX, BOX, method[3:], seed=instance, budget=CAP,
                 **BO_OPTIONS)
    return {"instance": instance, "method": method, "seconds": time.time() - started,
            "best_so_far": np.minimum.accumulate(values).tolist()}


def sweep(workers):
    RESULTS.mkdir(exist_ok=True)
    done = set()
    if PARTIAL.exists():
        done = {(r["instance"], r["method"]) for r in map(json.loads,
                                                         PARTIAL.read_text().splitlines())}
    todo = [t for t in itertools.product(range(N_INSTANCES), METHODS) if t not in done]
    todo.sort(key=lambda t: t[1] != "bo-full")          # slower runs first
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"[{i}/{len(todo)}] {rec['method']} inst={rec['instance']} "
                  f"{rec['seconds']:.0f}s, final f={rec['best_so_far'][-1]:.2e}, "
                  f"elapsed {time.time() - started:.0f}s", flush=True)


def final_log10(records):
    return {(r["instance"], r["method"]): float(np.log10(r["best_so_far"][CAP - 1] + 1e-300))
            for r in records}


def report():
    units = final_log10(map(json.loads, PARTIAL.read_text().splitlines()))
    base = final_log10(r for r in map(json.loads, BASELINE.read_text().splitlines())
                       if r["condition"] == CONDITION and r["geometry"] == "none"
                       and r["method"] in METHODS)
    rows, summary = [], []
    for (inst, m), v in sorted(units.items()):
        rows.append({"instance": inst, "method": m, "log10_f_200_units": v,
                     "log10_f_200_baseline": base[(inst, m)]})
    df = pd.DataFrame(rows)
    print("median log10 f at 200 evaluations, BO with local candidates")
    print(df.groupby("method")[["log10_f_200_baseline", "log10_f_200_units"]]
            .median().round(2).to_string())
    print("\nunits (10,1) minus original units, paired by instance (+ = units hurt)")
    for m in METHODS:
        sub = df[df.method == m].sort_values("instance")
        res = analysis.bootstrap_paired_difference(sub.log10_f_200_units.values,
                                                   sub.log10_f_200_baseline.values, n_boot=5000)
        summary.append({"method": m, **res})
        print(f"  {m:8s} {res['difference']:+6.2f} [{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]"
              f"   (E3, plain EI: bo-full +0.91 [+0.62, +1.47], bo-ard +0.45 [-0.44, +1.07])")
    df.to_csv(RESULTS / "e3b_units_check_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS / "e3b_units_check_summary.csv", index=False)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    report()
