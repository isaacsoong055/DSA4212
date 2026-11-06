"""E2: BO versus CMA-ES head to head, under one budget rule for both families.

Question. E1 shows that a search distribution restricted to the coordinate
axes fails on rotated ill-conditioned problems. The standard BO surrogate
carries the same restriction (an ARD kernel is a diagonal metric). This
experiment runs the 2x2 of {surrogate, population} x {axis-aligned,
rotation-aware} on identical problems, so that (a) the geometry explanation is
tested in both families and (b) the two families are compared with each other.

Budget rule (chosen so the two families can be compared at all budgets, not
only at one that favours either):

* Unit: objective evaluations, counted identically for every method by
  wrapping the objective. The initial design BO evaluates counts too.
* Every evaluation's value is logged, so each run yields a best-so-far curve
  f_best(n). One dataset is then read two ways:
    - runtime to target (COCO): the first n with f_best(n) - f* <= t, for
      t in TARGETS; runs that never reach t are censored, not dropped, and
      success rates are always reported beside runtimes;
    - fixed budget: f_best(n) at n in CHECKPOINTS, compared with paired
      bootstrap intervals.
* Cap: CAP evaluations for every method. Separately, CMA-ES is continued to
  LONG_CAP = 2000 x d evaluations, the E1 budget, so its behaviour outside
  BO's regime is still visible; those runs are labelled and never mixed into
  the capped comparison.
* Same starting information: both methods are given the box [-BOX, BOX]^d.
  BO searches it; CMA-ES starts at its centre with sigma = SIGMA_FRACTION x
  box width. Neither method is tuned beyond its defaults.
* Wall-clock seconds per run are logged, because BO spends far more
  computation choosing each point and the evaluation count hides that.

Problems: 2-D ellipsoids with optimum on a sphere of radius 2 (f* = 0),
condition numbers CONDITIONS, axis-aligned or Haar-randomly rotated, N_INSTANCES
COCO-style instances. Run seed = instance index for every method, so
cross-method comparisons are paired on identical problems.

Note on rotation comparisons: problems.make_ellipsoid draws the rotation from
the same generator as the optimum's location, so an aligned and a rotated
problem with the same instance index have different optima. Aligned-versus-
rotated differences are therefore bootstrapped unpaired.

Runs are checkpointed to results/e2_runs_partial.jsonl so an interrupted sweep
resumes where it stopped. Writes results/e2_runs.csv (one row per run and
target), results/e2_trajectories.csv (best-so-far per evaluation, capped runs)
and results/e2_summary.csv.

--bofix re-runs only the BO methods with the fix from E2e (bo.py options in
BOFIX_OPTIONS) and writes results/e2_bofix_* instead; the CMA-ES runs, which the
fix does not touch, are copied from the original checkpoint. The original
E2 files are left as they are, for the before/after comparison.

--refit-local re-runs the BO methods with the ORIGINAL settings (noise fitted)
on the current machine, with BLAS pinned to one thread, into results/e2_local_*.
Together with --bofix this makes the before/after comparison paired on one
machine (the original e2_* runs came from another machine, before pinning).

--local-cand adds Stage C on top of the fix: 200 extra acquisition candidates
around the incumbent (bo.bayes_opt local_candidates=200), into results/e2_lc_*.
Stage C was added after the fixed-noise logs showed that the remaining stall
near 1e-4 is an acquisition-search failure: in 98-99% of late iterations a
dense search near the incumbent finds >10x the EI of the point chosen.
Prediction, written before running: BO now reaches 1e-5 on kappa = 1, and the
rotation result (ARD pays at kappa = 10,000, full does not) persists.

Usage: python code/run_e2_bo_metric.py [--workers N] [--report-only]
                                       [--bofix | --refit-local | --local-cand]
"""

import os
# One BLAS thread per worker: multithreaded BLAS makes BO runs non-repeatable
# (E2e, results/e2e_determinism.csv).
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
import cmaes
import problems

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
PARTIAL = RESULTS / "e2_runs_partial.jsonl"
BOFIX = "--bofix" in sys.argv
LOCAL = "--refit-local" in sys.argv
LOCAL_CAND = "--local-cand" in sys.argv
assert BOFIX + LOCAL + LOCAL_CAND <= 1, "choose one of --bofix, --refit-local, --local-cand"
PREFIX = ("e2_bofix" if BOFIX else "e2_local" if LOCAL else "e2_lc" if LOCAL_CAND else "e2")
# The fix adopted after E2e Stage B (code/run_e2e_stage_b.py, notebook 8.6).
# B2, noise fixed at 1e-6: the cause found in Stage A, -1.17 decades at 200
# evaluations. B5's extra fixed floor (-0.28 [-0.68, +0.15] vs B2) was not
# significant and is untested at high kappa, so it is not adopted.
BOFIX_OPTIONS = {"noise_variance": 1e-6}
BO_OPTIONS = (BOFIX_OPTIONS if BOFIX else
              {**BOFIX_OPTIONS, "local_candidates": 200} if LOCAL_CAND else {})
if BOFIX or LOCAL or LOCAL_CAND:
    ORIGINAL_PARTIAL, PARTIAL = PARTIAL, RESULTS / f"{PREFIX}_runs_partial.jsonl"
DIMENSION = 2
BOX = 5.0
SIGMA_FRACTION = 0.3
CAP = 200
LONG_CAP = 2000 * DIMENSION
TARGETS = (1e-1, 1e-3, 1e-5)
CHECKPOINTS = (20, 50, 100, 200)
CONDITIONS = (1, 100, 10000)
GEOMETRIES = ("none", "random")
N_INSTANCES = 15
METHODS = ("bo-full", "bo-ard", "cmaes-full", "cmaes-diagonal", "cmaes-full-long",
           "cmaes-diagonal-long")


class CountingObjective:
    """Wraps an objective and logs every value it returns, in call order."""

    def __init__(self, objective):
        self.objective = objective
        self.values = []

    def __call__(self, x):
        value = float(self.objective(x))
        self.values.append(value)
        return value


def run_one(task):
    condition, geometry, instance, method = task
    objective, meta = problems.make_ellipsoid(DIMENSION, condition, instance=instance,
                                              rotation=geometry)
    counted = CountingObjective(objective)
    started = time.time()
    if method.startswith("bo-"):
        bo.bayes_opt(counted, DIMENSION, -BOX, BOX, method[3:], seed=instance, budget=CAP,
                     **BO_OPTIONS)
        cap = CAP
    else:
        variant = method.split("-")[1]
        cap = LONG_CAP if method.endswith("-long") else CAP
        cmaes.cma_es(counted, DIMENSION, initial_mean=np.zeros(DIMENSION), seed=instance,
                     variant=variant, max_evaluations=cap,
                     initial_sigma=SIGMA_FRACTION * 2 * BOX, target=min(TARGETS) / 10,
                     record_history=False)
    seconds = time.time() - started
    best = np.minimum.accumulate(np.asarray(counted.values))   # f* = 0
    return {"condition": condition, "geometry": geometry, "instance": instance,
            "method": method, "cap": cap, "evaluations": len(best),
            "seconds": seconds, "best_so_far": best.tolist()}


def all_tasks():
    tasks = list(itertools.product(CONDITIONS, GEOMETRIES, range(N_INSTANCES), METHODS))
    # Longest first, so the workers finish together.
    cost = {"bo-full": 0, "bo-ard": 1}
    return sorted(tasks, key=lambda t: cost.get(t[3], 2))


def sweep(workers):
    RESULTS.mkdir(exist_ok=True)
    if (BOFIX or LOCAL or LOCAL_CAND) and not PARTIAL.exists():
        cma = [line for line in ORIGINAL_PARTIAL.read_text().splitlines()
               if not json.loads(line)["method"].startswith("bo-")]
        PARTIAL.write_text("".join(line + "\n" for line in cma))
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            rec = json.loads(line)
            done.add((rec["condition"], rec["geometry"], rec["instance"], rec["method"]))
    todo = [t for t in all_tasks() if t not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"[{i}/{len(todo)}] {rec['method']} k={rec['condition']} "
                  f"{rec['geometry']} inst={rec['instance']} {rec['seconds']:.0f}s "
                  f"elapsed {time.time() - started:.0f}s", flush=True)


def runtime_to(best, target):
    hit = np.flatnonzero(np.asarray(best) <= target)
    return float(hit[0] + 1) if hit.size else np.nan


def tables():
    records = [json.loads(line) for line in PARTIAL.read_text().splitlines()]
    rows, traj = [], []
    for rec in records:
        best = rec["best_so_far"]
        base = {k: rec[k] for k in ("condition", "geometry", "instance", "method",
                                    "cap", "evaluations", "seconds")}
        for t in TARGETS:
            rt = runtime_to(best, t)
            rows.append({**base, "target": t, "runtime": rt, "reached": bool(np.isfinite(rt))})
        if rec["cap"] == CAP:
            padded = best + [best[-1]] * (CAP - len(best))   # stopped early = stays solved
            for n, v in enumerate(padded, 1):
                traj.append({**{k: base[k] for k in ("condition", "geometry", "instance",
                                                     "method")},
                             "evaluation": n, "best_so_far": v})
    return pd.DataFrame(rows), pd.DataFrame(traj)


def bootstrap_unpaired_difference(a, b, n_boot=10000, seed=0, statistic=np.median):
    """Bootstrap CI for statistic(a) - statistic(b), resampling each group separately."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    draws = np.array([statistic(rng.choice(a, len(a))) - statistic(rng.choice(b, len(b)))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(statistic(a) - statistic(b)),
            "ci_low": float(lo), "ci_high": float(hi)}


def log10_at(traj, n):
    sub = traj[traj.evaluation == n]
    return sub.assign(log10_f=np.log10(sub.best_so_far + 1e-300))


def report(runs, traj):
    capped = runs[runs.cap == CAP]
    pd.set_option("display.width", 200)

    print("\n== success rate (%) within the shared cap of", CAP, "evaluations")
    print((100 * capped.pivot_table(index=["condition", "geometry", "target"],
                                    columns="method", values="reached",
                                    aggfunc="mean")).round(0).to_string())

    print("\n== median runtime to target among successful runs (evaluations)")
    print(capped.pivot_table(index=["condition", "geometry", "target"], columns="method",
                             values="runtime", aggfunc="median").round(0).to_string())

    long = runs[runs.cap == LONG_CAP]
    print(f"\n== CMA-ES continued to {LONG_CAP} evaluations: success % and median runtime")
    print(long.groupby(["condition", "geometry", "target", "method"])
              .agg(success=("reached", lambda s: round(100 * s.mean())),
                   median_runtime=("runtime", "median")).unstack("method").to_string())

    print("\n== median log10 best-so-far at fixed budgets")
    at = pd.concat([log10_at(traj, n).assign(budget=n) for n in CHECKPOINTS])
    print(at.pivot_table(index=["condition", "geometry", "budget"], columns="method",
                         values="log10_f", aggfunc="median").round(2).to_string())

    summary = []
    print("\n== paired bootstrap, cross-family at fixed budget: "
          "median log10 f, BO minus CMA-ES (negative = BO better)")
    pairs = (("bo-full", "cmaes-full"), ("bo-ard", "cmaes-diagonal"),
             ("bo-ard", "cmaes-full"))
    for (cond, geo), n, (a, b) in itertools.product(
            itertools.product(CONDITIONS, GEOMETRIES), CHECKPOINTS, pairs):
        sub = at[(at.condition == cond) & (at.geometry == geo) & (at.budget == n)]
        xa = sub[sub.method == a].sort_values("instance").log10_f.values
        xb = sub[sub.method == b].sort_values("instance").log10_f.values
        out = analysis.bootstrap_paired_difference(xa, xb, n_boot=5000)
        summary.append({"comparison": f"{a} - {b}", "condition": cond, "geometry": geo,
                        "budget": n, **out})
        print(f"  k={cond:<6} {geo:7s} n={n:<4} {a:>8} - {b:<15} "
              f"{out['difference']:+6.2f} [{out['ci_low']:+6.2f}, {out['ci_high']:+6.2f}]")

    print("\n== rotation penalty at n =", CAP, ": median log10 f, rotated minus aligned "
          "(unpaired bootstrap; positive = rotation hurts)")
    final = at[at.budget == CAP]
    for cond, method in itertools.product(CONDITIONS, [m for m in METHODS
                                                       if not m.endswith("-long")]):
        sub = final[(final.condition == cond) & (final.method == method)]
        out = bootstrap_unpaired_difference(sub[sub.geometry == "random"].log10_f,
                                            sub[sub.geometry == "none"].log10_f, n_boot=5000)
        summary.append({"comparison": f"{method}: rotated - aligned", "condition": cond,
                        "geometry": "random-none", "budget": CAP, **out})
        print(f"  k={cond:<6} {method:16s} {out['difference']:+6.2f} "
              f"[{out['ci_low']:+6.2f}, {out['ci_high']:+6.2f}]")

    print("\n== median wall-clock seconds per run (capped runs)")
    print(capped.drop_duplicates(["condition", "geometry", "instance", "method"])
                .pivot_table(index="condition", columns="method", values="seconds",
                             aggfunc="median").round(2).to_string())
    return pd.DataFrame(summary)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    runs, traj = tables()
    runs.to_csv(RESULTS / f"{PREFIX}_runs.csv", index=False)
    traj.to_csv(RESULTS / f"{PREFIX}_trajectories.csv", index=False)
    report(runs, traj).to_csv(RESULTS / f"{PREFIX}_summary.csv", index=False)
