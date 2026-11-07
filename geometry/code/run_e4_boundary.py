"""E4: boundary constraints. What happens when the optimum is near the box edge?

The two families treat the box differently. BO searches only inside it (the
acquisition optimiser is bounded), and GP-EI is known to over-sample the box
boundary, where posterior variance is largest (Siivola et al., "Correcting
boundary over-exploration deficiencies in Bayesian optimization with virtual
derivative sign observations", 2018). The CMA-ES implementation has no box
handling at all, so it is tested twice: as in E2 (unconstrained; the share of
its evaluations outside the box is logged), and with a reflection boundary
transform, which evaluates f at x mirrored back into the box. The reflection is
a simplified, non-smoothed version of pycma's default BoundTransform.

Confound and control. Every method starts at the box centre, so an optimum near
an edge is also far from the start. An interior control at the same distance
separates the two:

    centre        radius 2, random direction (as in E2)       distance 2
    interior-far  (+-3.5, +-3.5), 1.5 from both faces          distance 4.95
    near-edge     4.9 along a random axis, 0.1 from the face   distance 4.9
    on-edge       5.0 along the same axis, on the face         distance 5.0
    corner        (+-5, +-5), on two faces                     distance 7.07

Primary comparison: near-edge and on-edge against interior-far, paired by
instance. Signs and the face axis are drawn per instance and shared across
conditions. The corner has no same-distance control.

Problems: 2-D aligned ellipsoids, kappa in {1, 100}, 15 instances. Budget rule
as in E2: cap 200, box [-5, 5]^2, CMA-ES from the centre with sigma = 3, BO-ARD
with the noise fix (noise_variance = 1e-6). BO-full is not run: on aligned
problems the kernel's geometry is not the question here.

Writes results/e4_runs.csv and results/e4_summary.csv; raw checkpoint
results/e4_runs_partial.jsonl.
Usage: python code/run_e4_boundary.py [--workers N] [--report-only]
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
PARTIAL = RESULTS / "e4_runs_partial.jsonl"
DIMENSION = 2
BOX = 5.0
CAP = 200
SIGMA = 0.3 * 2 * BOX
N_INSTANCES = 15
BO_OPTIONS = {"noise_variance": 1e-6}
CONDITIONS = (1, 100)
POSITIONS = ("centre", "interior-far", "near-edge", "on-edge", "corner")
METHODS = ("bo-ard", "cmaes-full", "cmaes-full-reflect")
CHECKPOINTS = (50, 100, 200)
TARGETS = (1e-3, 1e-5)
ON_BOUND = 1e-9     # a coordinate within this of +-BOX counts as on the boundary


def optimum_for(position, instance, default):
    """The optimum's location; draws are shared across positions of one instance."""
    rng = np.random.default_rng(700 + instance)
    signs = rng.choice([-1.0, 1.0], DIMENSION)
    axis = int(rng.integers(DIMENSION))
    if position == "centre":
        return np.asarray(default, float)
    if position == "interior-far":
        return 3.5 * signs
    if position in ("near-edge", "on-edge"):
        x = np.zeros(DIMENSION)
        x[axis] = signs[axis] * (4.9 if position == "near-edge" else BOX)
        return x
    if position == "corner":
        return BOX * signs
    raise ValueError(position)


def reflect(x):
    """Mirror x back into [-BOX, BOX] (periodic reflection; identity inside the box)."""
    width = 2 * BOX
    t = np.mod(np.asarray(x, float) + BOX, 2 * width)
    return np.where(t > width, 2 * width - t, t) - BOX


class Counting:
    """Logs every evaluated point and its value."""

    def __init__(self, f):
        self.f, self.values, self.points = f, [], []

    def __call__(self, x):
        x = np.asarray(x, float)
        v = float(self.f(x))
        self.values.append(v)
        self.points.append(x.copy())
        return v


def run_one(task):
    condition, position, instance, method = task
    _, meta = problems.make_ellipsoid(DIMENSION, condition, instance=instance, rotation="none")
    hessian = meta["hessian"]
    optimum = optimum_for(position, instance, meta["optimum"])

    def f(x):
        d = np.asarray(x, float) - optimum
        return 0.5 * float(d @ hessian @ d)

    counted = Counting(f)
    started = time.time()
    if method == "bo-ard":
        bo.bayes_opt(counted, DIMENSION, -BOX, BOX, "ard", seed=instance, budget=CAP,
                     **BO_OPTIONS)
        proposed = np.asarray(counted.points)
    else:
        proposed = []
        if method == "cmaes-full-reflect":
            def seen(x):
                proposed.append(np.asarray(x, float).copy())   # CMA-ES's own sample
                return counted(reflect(x))
        else:
            seen = counted
        cmaes.cma_es(seen, DIMENSION, initial_mean=np.zeros(DIMENSION), seed=instance,
                     variant="full", max_evaluations=CAP, initial_sigma=SIGMA, target=0.0,
                     record_history=False)
        proposed = np.asarray(proposed) if proposed else np.asarray(counted.points)
    points = np.asarray(counted.points)          # where f was actually evaluated
    return {"condition": condition, "position": position, "instance": instance,
            "method": method, "seconds": time.time() - started,
            "optimum": optimum.tolist(),
            "best_so_far": np.minimum.accumulate(counted.values).tolist(),
            "pct_on_boundary": float(100 * np.mean(np.any(np.abs(points) >= BOX - ON_BOUND,
                                                         axis=1))),
            "pct_outside_box": float(100 * np.mean(np.any(np.abs(points) > BOX + ON_BOUND,
                                                         axis=1))),
            "pct_proposed_outside": float(100 * np.mean(np.any(np.abs(proposed) > BOX,
                                                               axis=1)))}


def all_tasks():
    tasks = list(itertools.product(CONDITIONS, POSITIONS, range(N_INSTANCES), METHODS))
    return sorted(tasks, key=lambda t: t[3] != "bo-ard")        # slow ones first


def sweep(workers):
    RESULTS.mkdir(exist_ok=True)
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["position"], r["instance"], r["method"]))
    todo = [t for t in all_tasks() if t not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if rec["method"] == "bo-ard" or i % 50 == 0:
                print(f"[{i}/{len(todo)}] {rec['method']} k={rec['condition']} "
                      f"{rec['position']} inst={rec['instance']} {rec['seconds']:.0f}s "
                      f"elapsed {time.time() - started:.0f}s", flush=True)


def table():
    rows = []
    for line in PARTIAL.read_text().splitlines():
        r = json.loads(line)
        best = np.asarray(r["best_so_far"])
        padded = np.concatenate([best, np.full(CAP - len(best), best[-1])])
        row = {k: r[k] for k in ("condition", "position", "instance", "method", "seconds",
                                 "pct_on_boundary", "pct_outside_box", "pct_proposed_outside")}
        for n in CHECKPOINTS:
            row[f"log10_f_{n}"] = float(np.log10(padded[n - 1] + 1e-300))
        for t in TARGETS:
            row[f"reached_{t:g}"] = bool((best <= t).any())
        rows.append(row)
    return pd.DataFrame(rows)


def report(df):
    pd.set_option("display.width", 220)
    print("\n== median log10 f at 50 and 200 evaluations")
    print(df.pivot_table(index=["condition", "position"], columns="method",
                         values=["log10_f_50", "log10_f_200"], aggfunc="median")
            .round(2).to_string())
    print("\n== % of runs reaching 1e-3 within", CAP)
    print((100 * df.pivot_table(index=["condition", "position"], columns="method",
                                values="reached_0.001", aggfunc="mean")).round(0).to_string())
    print("\n== % of evaluations on the box boundary / outside the box (median over runs)")
    print(df.pivot_table(index=["condition", "position"], columns="method",
                         values=["pct_on_boundary", "pct_outside_box"], aggfunc="median")
            .round(1).to_string())

    out = []
    print("\n== position minus interior-far, paired by instance: median log10 f "
          "(+ = worse than the same-distance interior control)")
    for cond, pos, method, n in itertools.product(CONDITIONS, POSITIONS, METHODS, (50, 200)):
        if pos == "interior-far":
            continue
        a = df[(df.condition == cond) & (df.position == pos) & (df.method == method)] \
            .sort_values("instance")
        b = df[(df.condition == cond) & (df.position == "interior-far") & (df.method == method)] \
            .sort_values("instance")
        res = analysis.bootstrap_paired_difference(a[f"log10_f_{n}"].values,
                                                   b[f"log10_f_{n}"].values, n_boot=5000)
        out.append({"condition": cond, "position": pos, "method": method, "budget": n, **res})
        if n == 200:
            flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
            print(f"  k={cond:<4} {pos:10s} {method:19s} {res['difference']:+6.2f} "
                  f"[{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]{flag}")
    return pd.DataFrame(out)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    df = table()
    df.to_csv(RESULTS / "e4_runs.csv", index=False)
    report(df).to_csv(RESULTS / "e4_summary.csv", index=False)
