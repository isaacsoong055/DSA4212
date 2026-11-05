"""E5: beyond the quadratic. A curved valley (Rosenbrock) and a noisy objective.

Same budget rule as E2: 2-D, box [-5, 5]^2, cap 200, 15 instances (seed =
instance), CMA-ES from the box centre with sigma = 3, BO with the noise fix
(noise_variance = 1e-6) unless stated.

1. Rosenbrock, COCO f8 (aligned) and f9 (rotated):
       f(x) = sum_i 100 (z_{i+1} - z_i^2)^2 + (1 - z_i)^2,   z = Q (x - x*) + 1,
   with x* on the radius-2 sphere as in E2, Q = I or Haar-random, f* = 0.
   Its valley bends, so the best search direction changes along the way.
   Prediction, written before running: a full GP metric is ONE global tilt,
   so on the rotated version its advantage over ARD should be smaller than on
   the rotated quadratic (E2 with the fix, 8.7: ARD minus full = +1.40
   [+0.68, +1.69] decades at kappa = 10,000). CMA-ES re-learns its covariance
   as it moves, so full should still beat diagonal on the rotated version.

2. Noisy quadratic: the aligned kappa = 100 ellipsoid of E2/E3 plus additive
   Gaussian noise, sigma in {0.01, 1}. Additive rather than COCO's
   multiplicative noise because it does not vanish at the optimum: this is
   the case where assuming no noise is wrong everywhere.
   Methods: BO-ARD with the fix (noise fixed at 1e-6), BO-ARD with the noise
   fitted (the original bo.py default), full and diagonal CMA-ES.
   Scoring: every method's recommendation is the evaluated point with the
   lowest OBSERVED (noisy) value so far, scored on the noise-free f; a method
   fooled by noise pays for it. The best noise-free value ever evaluated is
   logged too ("oracle"), but it is not something a method could report.
   Prediction: fixed-noise BO degrades badly at sigma = 1, fitted noise
   copes better, CMA-ES (rank-based) degrades gracefully.
   The noise stream is seeded by instance, so every method sees the same
   sequence of noise draws in call order.

Writes results/e5_runs.csv, results/e5_trajectories.csv, results/e5_summary.csv;
raw checkpoint results/e5_runs_partial.jsonl.
Usage: python code/run_e5_other_functions.py [--workers N] [--report-only]
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
PARTIAL = RESULTS / "e5_runs_partial.jsonl"
DIMENSION = 2
BOX = 5.0
CAP = 200
SIGMA = 0.3 * 2 * BOX
N_INSTANCES = 15
CHECKPOINTS = (20, 50, 100, 200)
PROBLEMS = ("rosenbrock:none", "rosenbrock:random", "noisy:0.01", "noisy:1")
METHODS = {
    "rosenbrock": ("bo-ard", "bo-full", "cmaes-full", "cmaes-diagonal"),
    "noisy": ("bo-ard", "bo-ard-fitnoise", "cmaes-full", "cmaes-diagonal"),
}
BO_FIX = {"noise_variance": 1e-6}


def rosenbrock(instance, rotation):
    """COCO-style Rosenbrock with the optimum at x* (f* = 0)."""
    _, meta = problems.make_ellipsoid(DIMENSION, 1, instance=instance, rotation=rotation)
    optimum = meta["optimum"]
    q = np.eye(DIMENSION)
    if rotation == "random":
        q = problems.haar_rotation(DIMENSION, np.random.default_rng(3000 + instance))

    def f(x):
        z = q @ (np.asarray(x, float) - optimum) + 1.0
        return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))
    return f


class Logged:
    """Returns (possibly noisy) values; logs the true and the observed value of every call."""

    def __init__(self, f, noise_sd=0.0, seed=None):
        self.f, self.noise_sd = f, noise_sd
        self.rng = np.random.default_rng(seed)
        self.true, self.observed = [], []

    def __call__(self, x):
        t = float(self.f(x))
        o = t + (self.noise_sd * float(self.rng.normal()) if self.noise_sd else 0.0)
        self.true.append(t)
        self.observed.append(o)
        return o


def run_one(task):
    problem, instance, method = task
    family, setting = problem.split(":")
    if family == "rosenbrock":
        logged = Logged(rosenbrock(instance, setting))
    else:
        base, _ = problems.make_ellipsoid(DIMENSION, 100, instance=instance, rotation="none")
        logged = Logged(base, noise_sd=float(setting), seed=20_000 + instance)
    started = time.time()
    if method.startswith("bo-"):
        metric = "full" if method == "bo-full" else "ard"
        options = {} if method == "bo-ard-fitnoise" else BO_FIX
        bo.bayes_opt(logged, DIMENSION, -BOX, BOX, metric, seed=instance, budget=CAP, **options)
    else:
        cmaes.cma_es(logged, DIMENSION, initial_mean=np.zeros(DIMENSION), seed=instance,
                     variant=method.split("-")[1], max_evaluations=CAP, initial_sigma=SIGMA,
                     target=-np.inf, record_history=False)
    true, observed = np.asarray(logged.true), np.asarray(logged.observed)
    # Recommendation after n evaluations: the point with the lowest observed value so far.
    incumbent = np.array([int(np.argmin(observed[:n + 1])) for n in range(len(observed))])
    return {"problem": problem, "instance": instance, "method": method,
            "seconds": time.time() - started,
            "recommended_true": true[incumbent].tolist(),
            "oracle_true": np.minimum.accumulate(true).tolist()}


def all_tasks():
    tasks = [(p, i, m) for p in PROBLEMS for i in range(N_INSTANCES)
             for m in METHODS[p.split(":")[0]]]
    return sorted(tasks, key=lambda t: not t[2].startswith("bo-"))


def sweep(workers):
    RESULTS.mkdir(exist_ok=True)
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
            if rec["method"].startswith("bo-") or i % 40 == 0:
                print(f"[{i}/{len(todo)}] {rec['method']} {rec['problem']} inst={rec['instance']} "
                      f"{rec['seconds']:.0f}s elapsed {time.time() - started:.0f}s", flush=True)


def tables():
    runs, traj = [], []
    for line in PARTIAL.read_text().splitlines():
        r = json.loads(line)
        rec = np.asarray(r["recommended_true"]); orc = np.asarray(r["oracle_true"])
        pad = lambda v: np.concatenate([v, np.full(CAP - len(v), v[-1])])
        rec, orc = pad(rec), pad(orc)
        row = {k: r[k] for k in ("problem", "instance", "method", "seconds")}
        for n in CHECKPOINTS:
            row[f"log10_f_{n}"] = float(np.log10(rec[n - 1] + 1e-300))
            row[f"log10_oracle_{n}"] = float(np.log10(orc[n - 1] + 1e-300))
        runs.append(row)
        for n in range(CAP):
            traj.append({"problem": r["problem"], "instance": r["instance"], "method": r["method"],
                         "evaluation": n + 1, "recommended_true": rec[n], "oracle_true": orc[n]})
    return pd.DataFrame(runs), pd.DataFrame(traj)


def paired(runs, problem, a, b, n, col="log10_f"):
    x = runs[(runs.problem == problem) & (runs.method == a)].sort_values("instance")[f"{col}_{n}"]
    y = runs[(runs.problem == problem) & (runs.method == b)].sort_values("instance")[f"{col}_{n}"]
    return analysis.bootstrap_paired_difference(x.values, y.values, n_boot=5000)


def unpaired(runs, method, n):
    a = runs[(runs.problem == "rosenbrock:random") & (runs.method == method)][f"log10_f_{n}"]
    b = runs[(runs.problem == "rosenbrock:none") & (runs.method == method)][f"log10_f_{n}"]
    rng = np.random.default_rng(0)
    draws = [np.median(rng.choice(a.values, len(a))) - np.median(rng.choice(b.values, len(b)))
             for _ in range(5000)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(a.median() - b.median()), "ci_low": float(lo), "ci_high": float(hi)}


def report(runs):
    pd.set_option("display.width", 220)
    print("\n== median log10 f of the recommended point (true f), at fixed budgets")
    print(runs.pivot_table(index=["problem", "method"],
                           values=[f"log10_f_{n}" for n in CHECKPOINTS], aggfunc="median")
              .round(2).to_string())
    print("\n== noisy problems: median log10 of the best TRUE value ever evaluated (oracle)")
    print(runs[runs.problem.str.startswith("noisy")]
          .pivot_table(index=["problem", "method"], values=[f"log10_oracle_{n}" for n in (50, 200)],
                       aggfunc="median").round(2).to_string())

    out = []
    def show(label, res, extra):
        out.append({**extra, **res})
        flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
        print(f"  {label:58s} {res['difference']:+6.2f} [{res['ci_low']:+6.2f}, "
              f"{res['ci_high']:+6.2f}]{flag}")

    print("\n== Rosenbrock, paired within geometry, at 200 (+ = first method worse)")
    for geo, (a, b) in itertools.product(("none", "random"),
                                         (("bo-ard", "bo-full"), ("cmaes-diagonal", "cmaes-full"),
                                          ("bo-full", "cmaes-full"))):
        show(f"rosenbrock:{geo}  {a} - {b}", paired(runs, f"rosenbrock:{geo}", a, b, 200),
             {"comparison": f"{a} - {b}", "problem": f"rosenbrock:{geo}", "budget": 200})
    print("\n== Rosenbrock rotation penalty at 200, rotated - aligned (unpaired; + = hurts)")
    for m in METHODS["rosenbrock"]:
        show(f"{m}", unpaired(runs, m, 200),
             {"comparison": f"{m}: rotated - aligned", "problem": "rosenbrock", "budget": 200})

    print("\n== noisy quadratic, paired, at 50 and 200 (+ = first method worse)")
    for prob, (a, b), n in itertools.product(("noisy:0.01", "noisy:1"),
                                             (("bo-ard", "bo-ard-fitnoise"),
                                              ("bo-ard", "cmaes-full"),
                                              ("bo-ard-fitnoise", "cmaes-full")), (50, 200)):
        show(f"{prob}  {a} - {b}  n={n}", paired(runs, prob, a, b, n),
             {"comparison": f"{a} - {b}", "problem": prob, "budget": n})
    return pd.DataFrame(out)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    runs, traj = tables()
    runs.to_csv(RESULTS / "e5_runs.csv", index=False)
    traj.to_csv(RESULTS / "e5_trajectories.csv", index=False)
    report(runs).to_csv(RESULTS / "e5_summary.csv", index=False)
