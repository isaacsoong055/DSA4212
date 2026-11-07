"""E6: noisy evaluations. How to split the budget, and which point to recommend.

Problem: the aligned kappa = 100 ellipsoid of E2-E5 (2-D, box [-5, 5]^2) with
additive Gaussian noise, sigma in {0.01, 0.1, 1}. 15 instances, 200 noisy
evaluations per run (every evaluation counts, repeats included). The noise
stream is seeded by instance, so every variant sees the same draws in call
order. Methods: full CMA-ES (from the box centre, sigma0 = 3) and BO-ARD with
the noise fix (noise_variance = 1e-6), as in E5.

Budget splits (Part B):
  cmaes-base      lambda = 6, every evaluation a new point
  cmaes-resample  each candidate evaluated 4 times, ranked by the average
  cmaes-pop24     lambda = 24 (averaging through a larger population)
  cmaes-reserve   search with 160, then re-evaluate the 5 best-observed
                  points 8 times each (40)
  bo-base         every evaluation a new point
  bo-replicate    each chosen point (and the initial design) evaluated 4 times;
                  the GP is given the averages
  bo-reserve      search with 160, then re-evaluate the 5 best-observed
                  points 8 times each (40)

Recommendation rules (Part A), all scored on the noise-free f:
  R1 lowest observed   the point with the lowest observed value (the average,
                       for variants that average)
  R2 model estimate    CMA-ES: the final mean of the search distribution.
                       BO: the evaluated point with the lowest posterior mean
                       of a GP refitted at the end WITH the noise estimated (the
                       search GP fixes the noise near zero, so its own mean
                       would just repeat the noisy readings and R2 = R1)
  R3 re-evaluated      reserve variants only: the lowest average over the
                       original reading plus 8 re-evaluations

Predictions, written before running:
  P1 R2 beats R1 for CMA-ES, increasingly with sigma (the mean averages over
     the population; the lowest reading is a lucky draw).
  P2 R2 beats R1 for BO at sigma >= 0.1.
  P3 For CMA-ES a larger population beats resampling at the same budget
     (the usual finding in the ES literature).
  P4 The reserve helps both families at sigma = 1 and costs at sigma = 0.01.
  P5 BO replicates hurt at sigma = 0.01 (a quarter of the points) and help at
     sigma = 1.

Writes results/e6_runs.csv (one row per run and rule) and results/e6_summary.csv;
raw checkpoint results/e6_runs_partial.jsonl.
Usage: python code/run_e6_noise_budget.py [--workers N] [--report-only]
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
PARTIAL = RESULTS / "e6_runs_partial.jsonl"
DIMENSION = 2
BOX = 5.0
CAP = 200
SIGMA0 = 0.3 * 2 * BOX
N_INSTANCES = 15
NOISE = (0.01, 0.1, 1.0)
REPEATS = 4                    # resampling / replicates
RESERVE, TOP_K, RE_EVALS = 40, 5, 8
VARIANTS = ("cmaes-base", "cmaes-resample", "cmaes-pop24", "cmaes-reserve",
            "bo-base", "bo-replicate", "bo-reserve")
BO_FIX = {"noise_variance": 1e-6}


class Noisy:
    """Noisy objective that logs the true and observed value of every evaluation."""

    def __init__(self, f, sd, seed):
        self.f, self.sd, self.rng = f, sd, np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return float(self.f(x)) + self.sd * float(self.rng.normal())


class Averaged:
    """Evaluates each point `repeats` times; logs every distinct point and its average."""

    def __init__(self, noisy, repeats=1):
        self.noisy, self.repeats = noisy, repeats
        self.points, self.means = [], []

    def __call__(self, x):
        v = float(np.mean([self.noisy(x) for _ in range(self.repeats)]))
        self.points.append(np.asarray(x, float).copy())
        self.means.append(v)
        return v


def reevaluate(noisy, points, means):
    """Re-evaluate the TOP_K best-observed distinct points RE_EVALS times; return the best."""
    order = np.argsort(means)[:TOP_K]
    best, best_avg = None, np.inf
    for i in order:
        avg = (means[i] + sum(noisy(points[i]) for _ in range(RE_EVALS))) / (1 + RE_EVALS)
        if avg < best_avg:
            best, best_avg = points[i], avg
    return best


def bo_posterior_minimiser(points, values, seed):
    """R2 for BO: evaluated point with the lowest posterior mean, noise estimated."""
    x, y = np.asarray(points), np.asarray(values)
    z = bo._transform_targets(y)
    model = bo.fit_gp(x, z, "ard", np.random.default_rng(seed), input_scale=2 * BOX)
    mean, _ = bo.predict(model, x)
    return x[int(np.argmin(mean))]


def run_one(task):
    assert RESERVE == TOP_K * RE_EVALS, "the reserve must pay exactly for the re-evaluations"
    sigma, instance, variant = task
    f, _ = problems.make_ellipsoid(DIMENSION, 100, instance=instance, rotation="none")
    noisy = Noisy(f, sigma, seed=30_000 + instance)
    reserve = variant.endswith("reserve")
    search_budget = CAP - RESERVE if reserve else CAP
    repeats = REPEATS if variant in ("cmaes-resample", "bo-replicate") else 1
    wrapped = Averaged(noisy, repeats)
    started = time.time()
    rec = {}
    if variant.startswith("cmaes"):
        out = cmaes.cma_es(wrapped, DIMENSION, initial_mean=np.zeros(DIMENSION), seed=instance,
                           variant="full", max_evaluations=search_budget // repeats,
                           initial_sigma=SIGMA0, target=-np.inf, record_history=False,
                           population_size=24 if variant == "cmaes-pop24" else None)
        rec["R2"] = np.asarray(out["mean"])
    else:
        bo.bayes_opt(wrapped, DIMENSION, -BOX, BOX, "ard", seed=instance,
                     budget=search_budget // repeats, **BO_FIX)
        rec["R2"] = bo_posterior_minimiser(wrapped.points, wrapped.means, seed=instance)
    rec["R1"] = wrapped.points[int(np.argmin(wrapped.means))]
    if reserve:
        rec["R3"] = reevaluate(noisy, wrapped.points, wrapped.means)
    true_all = [f(p) for p in wrapped.points]
    return {"sigma": sigma, "instance": instance, "variant": variant,
            "seconds": time.time() - started, "evaluations": noisy.calls,
            "distinct_points": len(wrapped.points),
            "oracle": float(np.min(true_all)),
            "true": {rule: float(f(p)) for rule, p in rec.items()}}


def all_tasks():
    tasks = list(itertools.product(NOISE, range(N_INSTANCES), VARIANTS))
    return sorted(tasks, key=lambda t: not t[2].startswith("bo"))


def sweep(workers):
    RESULTS.mkdir(exist_ok=True)
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sigma"], r["instance"], r["variant"]))
    todo = [t for t in all_tasks() if t not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if rec["variant"].startswith("bo") or i % 50 == 0:
                print(f"[{i}/{len(todo)}] {rec['variant']} sigma={rec['sigma']} "
                      f"inst={rec['instance']} {rec['seconds']:.0f}s "
                      f"elapsed {time.time() - started:.0f}s", flush=True)


def table():
    rows = []
    for line in PARTIAL.read_text().splitlines():
        r = json.loads(line)
        for rule, v in r["true"].items():
            rows.append({"sigma": r["sigma"], "instance": r["instance"], "variant": r["variant"],
                         "family": r["variant"].split("-")[0], "rule": rule,
                         "log10_f": float(np.log10(v + 1e-300)),
                         "log10_oracle": float(np.log10(r["oracle"] + 1e-300)),
                         "evaluations": r["evaluations"], "distinct_points": r["distinct_points"],
                         "seconds": r["seconds"]})
    return pd.DataFrame(rows)


def values(df, sigma, variant, rule):
    return df[(df.sigma == sigma) & (df.variant == variant) & (df.rule == rule)] \
        .sort_values("instance").log10_f.values


def report(df):
    pd.set_option("display.width", 220)
    print("\n== median log10 f (noise-free) of the recommended point, by variant and rule")
    print(df.pivot_table(index=["variant", "rule"], columns="sigma", values="log10_f",
                         aggfunc="median").round(2).to_string())
    print("\n== evaluations used and distinct points (median)")
    print(df[df.rule == "R1"].groupby("variant")[["evaluations", "distinct_points"]]
            .median().to_string())

    out = []
    def show(label, a, b, extra):
        res = analysis.bootstrap_paired_difference(a, b, n_boot=5000)
        out.append({**extra, **res})
        flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
        print(f"  {label:52s} {res['difference']:+6.2f} [{res['ci_low']:+6.2f}, "
              f"{res['ci_high']:+6.2f}]{flag}")

    print("\n== Part A: rules on the same runs, R2 - R1 (negative = model estimate better)")
    for sigma, variant in itertools.product(NOISE, VARIANTS):
        show(f"sigma={sigma:<5} {variant:15s} R2 - R1",
             values(df, sigma, variant, "R2"), values(df, sigma, variant, "R1"),
             {"part": "A", "sigma": sigma, "comparison": f"{variant}: R2 - R1"})
    for sigma, variant in itertools.product(NOISE, ("cmaes-reserve", "bo-reserve")):
        show(f"sigma={sigma:<5} {variant:15s} R3 - R2",
             values(df, sigma, variant, "R3"), values(df, sigma, variant, "R2"),
             {"part": "A", "sigma": sigma, "comparison": f"{variant}: R3 - R2"})

    print("\n== Part B: split minus base, both under R2 (negative = split better)")
    for sigma, (variant, base) in itertools.product(
            NOISE, (("cmaes-resample", "cmaes-base"), ("cmaes-pop24", "cmaes-base"),
                    ("cmaes-reserve", "cmaes-base"), ("bo-replicate", "bo-base"),
                    ("bo-reserve", "bo-base"))):
        show(f"sigma={sigma:<5} {variant} - {base} (R2)",
             values(df, sigma, variant, "R2"), values(df, sigma, base, "R2"),
             {"part": "B", "sigma": sigma, "comparison": f"{variant} - {base} (R2)"})
    print("\n== P3: CMA-ES population vs resampling, both R2")
    for sigma in NOISE:
        show(f"sigma={sigma:<5} cmaes-pop24 - cmaes-resample (R2)",
             values(df, sigma, "cmaes-pop24", "R2"), values(df, sigma, "cmaes-resample", "R2"),
             {"part": "B", "sigma": sigma, "comparison": "cmaes-pop24 - cmaes-resample (R2)"})
    print("\n== Reserve with R3 vs base with R2 (the full cost of re-evaluation)")
    for sigma, fam in itertools.product(NOISE, ("cmaes", "bo")):
        show(f"sigma={sigma:<5} {fam}-reserve R3 - {fam}-base R2",
             values(df, sigma, f"{fam}-reserve", "R3"), values(df, sigma, f"{fam}-base", "R2"),
             {"part": "B", "sigma": sigma, "comparison": f"{fam}-reserve R3 - {fam}-base R2"})
    return pd.DataFrame(out)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    df = table()
    df.to_csv(RESULTS / "e6_runs.csv", index=False)
    report(df).to_csv(RESULTS / "e6_summary.csv", index=False)
