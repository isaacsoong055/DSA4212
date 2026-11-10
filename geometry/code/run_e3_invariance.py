"""E3: which theoretical invariances survive the practical implementation?

Hansen lists the invariances CMA-ES is designed to have: to order-preserving
transformations of the objective value (it reads only rankings), and to any
invertible linear map of the search space *provided the initial mean and
covariance are mapped with it*. BO has neither by design: EI reads f values,
and the kernel and box are stated in the problem's units. This experiment
tests two kinds of transformation at the E2 budget rule.

1. Change of units: g(u) = f(S u), S = diag(s), so u_i is x_i measured in units
   s_i times larger. Two scale vectors on an aligned kappa = 100 base whose
   flat axis is x1 and steep axis is x2:
       s = (1, 10)   effective conditioning becomes 10,000
       s = (10, 1)   effective conditioning becomes 1
   Two variants:
     units-default  every method keeps its default set-up (the user did not
                    adapt anything to the units). All four methods. This tests
                    whether a method's adaptation absorbs the change. A
                    diagonal rescaling is axis-aligned, so ARD, the diagonal
                    and the full models can all represent it.
     units-matched  CMA-ES with the initial covariance mapped as the theory
                    requires, C0 = S^-2 (mean: the box centre, still 0). The
                    theory predicts invariance. BO is not run here: its
                    equivalent, normalising each input to its own box, is
                    invariant by construction, and bo.py takes one box width
                    for all coordinates.

2. Strictly increasing maps of the objective value, T(f): log1p, cube, sqrt and
   plus1 (f + 1, COCO's non-zero f_opt). CMA-ES should be exactly invariant,
   with identical trajectories. BO log-transforms and standardises its targets,
   which cancels powers (log f^p = p log f) but not the transform's floor,
   which is relative to the spread of the observed values, nor an additive
   shift.

Every evaluation is scored on the untransformed problem (f at the corresponding
point), so the yardstick is the same in every condition. Budget rule as in E2:
cap 200, box [-5, 5]^2, CMA-ES starts at its centre with sigma = 3, BO with the
noise fix adopted in E2e (noise_variance = 1e-6). 15 instances, seed = instance,
so every condition is paired with the baseline by instance.

Writes results/e3_runs.csv (one row per run), results/e3_summary.csv; the raw
checkpoint is results/e3_runs_partial.jsonl.
Usage: python code/run_e3_invariance.py [--workers N] [--report-only]
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
import cmaes
import problems

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
PARTIAL = RESULTS / "e3_runs_partial.jsonl"
DIMENSION = 2
CONDITION = 100
N_INSTANCES = 15
CAP = 200
BOX = 5.0
SIGMA = 0.3 * 2 * BOX
BO_OPTIONS = {"noise_variance": 1e-6}
CHECKPOINTS = (50, 100, 200)
TARGETS = (1e-3, 1e-5)
SCALES = {"1,10": (1.0, 10.0), "10,1": (10.0, 1.0)}
TRANSFORMS = ("log1p", "cube", "sqrt", "plus1")
METHODS = ("cmaes-full", "cmaes-diagonal", "bo-ard", "bo-full")

CONDITIONS = (["baseline"]
              + [f"units-default:{k}" for k in SCALES]
              + [f"units-matched:{k}" for k in SCALES]
              + [f"monotone:{t}" for t in TRANSFORMS])


def applies(condition, method):
    return not (condition.startswith("units-matched") and method.startswith("bo-"))


class Counting:
    """Logs the untransformed value of every evaluation, in call order."""

    def __init__(self, f):
        self.f, self.values = f, []

    def __call__(self, x):
        v = float(self.f(x))
        self.values.append(v)
        return v


def run_one(task):
    condition, instance, method = task
    base, _ = problems.make_ellipsoid(DIMENSION, CONDITION, instance=instance,
                                      rotation="none")
    initial_covariance = None
    if condition.startswith("units"):
        scales = np.array(SCALES[condition.split(":")[1]])
        counted = Counting(problems.rescale_coordinates(base, scales))
        seen = counted
        if condition.startswith("units-matched"):
            initial_covariance = np.diag(1.0 / scales ** 2)
    else:
        counted = Counting(base)
        seen = counted
        if condition.startswith("monotone"):
            transform = problems.MONOTONE_TRANSFORMS[condition.split(":")[1]]

            def seen(x):
                return float(transform(counted(x)))

    started = time.time()
    if method.startswith("bo-"):
        bo.bayes_opt(seen, DIMENSION, -BOX, BOX, method[3:], seed=instance, budget=CAP,
                     **BO_OPTIONS)
    else:
        cmaes.cma_es(seen, DIMENSION, initial_mean=np.zeros(DIMENSION), seed=instance,
                     variant=method.split("-")[1], max_evaluations=CAP, initial_sigma=SIGMA,
                     target=0.0, record_history=False, initial_covariance=initial_covariance)
    return {"condition": condition, "instance": instance, "method": method,
            "seconds": time.time() - started,
            "best_so_far": np.minimum.accumulate(counted.values).tolist()}


def all_tasks():
    tasks = [t for t in itertools.product(CONDITIONS, range(N_INSTANCES), METHODS)
             if applies(t[0], t[2])]
    return sorted(tasks, key=lambda t: not t[2].startswith("bo-"))   # slow ones first


def sweep(workers):
    RESULTS.mkdir(exist_ok=True)
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["instance"], r["method"]))
    todo = [t for t in all_tasks() if t not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if rec["method"].startswith("bo-") or i % 50 == 0:
                print(f"[{i}/{len(todo)}] {rec['method']} {rec['condition']} "
                      f"inst={rec['instance']} {rec['seconds']:.0f}s "
                      f"elapsed {time.time() - started:.0f}s", flush=True)


def table():
    rows = []
    for line in PARTIAL.read_text().splitlines():
        r = json.loads(line)
        best = np.asarray(r["best_so_far"])
        padded = np.concatenate([best, np.full(CAP - len(best), best[-1])])
        row = {k: r[k] for k in ("condition", "instance", "method", "seconds")}
        row["evaluations"] = len(best)
        for n in CHECKPOINTS:
            row[f"log10_f_{n}"] = float(np.log10(padded[n - 1] + 1e-300))
        for t in TARGETS:
            hit = np.flatnonzero(best <= t)
            row[f"runtime_{t:g}"] = float(hit[0] + 1) if hit.size else np.nan
        row["curve"] = " ".join(f"{v:.17g}" for v in padded)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "method", "instance"])


def report(df):
    pd.set_option("display.width", 220)
    print("\n== median log10 f (untransformed) at fixed budgets")
    print(df.pivot_table(index=["condition"], columns="method",
                         values=[f"log10_f_{n}" for n in (50, 200)], aggfunc="median")
            .round(2).reindex(CONDITIONS).to_string())

    print("\n== % of runs reaching each target within", CAP, "evaluations")
    for t in TARGETS:
        print(f"-- target {t:g}")
        print((100 * df.assign(hit=df[f"runtime_{t:g}"].notna())
                 .pivot_table(index="condition", columns="method", values="hit",
                              aggfunc="mean")).round(0).reindex(CONDITIONS).to_string())

    base = df[df.condition == "baseline"].set_index(["method", "instance"])
    out = []
    print("\n== condition minus baseline, paired by instance: median log10 f "
          "(+ = worse than baseline); identical = runs whose 200-step curves match the "
          "baseline's exactly")
    for condition, method in itertools.product(CONDITIONS[1:], METHODS):
        if not applies(condition, method):
            continue
        sub = df[(df.condition == condition) & (df.method == method)].set_index("instance")
        b = base.loc[method].loc[sub.index]
        identical = int((sub.curve == b.curve).sum())
        for n in (50, 200):
            res = analysis.bootstrap_paired_difference(sub[f"log10_f_{n}"].values,
                                                       b[f"log10_f_{n}"].values, n_boot=5000)
            out.append({"condition": condition, "method": method, "budget": n,
                        "identical_runs": identical, "n_runs": len(sub), **res})
            if n == 200:
                flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
                print(f"  {condition:20s} {method:15s} n=200 {res['difference']:+6.2f} "
                      f"[{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]  "
                      f"identical {identical}/{len(sub)}{flag}")
    return pd.DataFrame(out)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    df = table()
    df.drop(columns="curve").to_csv(RESULTS / "e3_runs.csv", index=False)
    report(df).to_csv(RESULTS / "e3_summary.csv", index=False)
