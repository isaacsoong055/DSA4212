"""E4b: is BO drawn to the box edge because the GP is most uncertain there?

E4 found BO-ARD placing 8-14% of its evaluations exactly on the box edge.
Landing exactly on an edge (to 1e-9) can only come from the bounded L-BFGS-B
polish of EI (or the final clip): candidates are uniform inside the box. So at
those points EI's slope points out of the box. The usual explanation is that
the GP's posterior uncertainty is largest at the edge, where it has no data
beyond, and EI rewards uncertainty. E4 did not measure this.

This re-runs E4's BO-ARD cells with a trace hook (it does not touch the random
stream, so the runs are E4's runs) and logs, at every iteration:
  chosen_on_edge       the chosen point lies on the box edge
  start_on_edge        the best uniform candidate (the polish's start) does
  sd_rank_chosen       share of the 200 uniform candidates whose posterior SD is
                       below the chosen point's (1 = most uncertain of all)
  mean_rank_chosen     share of candidates whose posterior mean is above the
                       chosen point's (1 = best predicted value of all)
  sd_edge_band / sd_interior
                       median posterior SD of candidates within 0.5 of an edge,
                       and of the others
  dist_polish          distance the polish moved from its start
Problems: 2-D aligned ellipsoids, kappa in {1, 100}, optimum near the edge and
the interior control at the same distance (E4's near-edge and interior-far),
15 instances, 200 evaluations, noise fixed at 1e-6.

Prediction, written before running: if uncertainty pulls EI to the edge, then
(a) candidates near the edge have higher posterior SD than interior ones, and
(b) at edge picks the chosen point is among the most uncertain candidates
(median sd_rank_chosen well above 0.5, e.g. > 0.8) rather than among the best
predicted (mean_rank_chosen). If edge picks instead have low uncertainty and a
good predicted mean, the explanation fails: BO is exploiting a good region that
touches the edge.

Writes results/e4b_trace.csv (one row per run and iteration) and
results/e4b_summary.csv; checkpoint results/e4b_partial.jsonl.
Usage: python code/run_e4b_edge_uncertainty.py [--workers N] [--report-only]
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
import bo
import problems
from run_e4_boundary import BOX, CAP, DIMENSION, BO_OPTIONS, ON_BOUND, optimum_for

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
PARTIAL = RESULTS / "e4b_partial.jsonl"
N_INSTANCES = 15
CONDITIONS = (1, 100)
POSITIONS = ("near-edge", "interior-far")
EDGE_BAND = 0.5


def on_edge(points):
    return np.any(np.abs(np.atleast_2d(points)) >= BOX - ON_BOUND, axis=1)


def run_one(task):
    condition, position, instance = task
    _, meta = problems.make_ellipsoid(DIMENSION, condition, instance=instance, rotation="none")
    hessian = meta["hessian"]
    optimum = optimum_for(position, instance, meta["optimum"])

    def f(x):
        d = np.asarray(x, float) - optimum
        return 0.5 * float(d @ hessian @ d)

    rows = []

    def trace(state):
        model, cand = state["model"], state["candidates"]
        chosen, start = state["chosen"].reshape(1, -1), state["start"].reshape(1, -1)
        mean_c, sd_c = bo.predict(model, cand)
        mean_x, sd_x = bo.predict(model, chosen)
        band = np.any(np.abs(cand) >= BOX - EDGE_BAND, axis=1)
        rows.append({"n": len(state["y"]),
                     "chosen_on_edge": bool(on_edge(chosen)[0]),
                     "start_on_edge": bool(on_edge(start)[0]),
                     "sd_rank_chosen": float(np.mean(sd_c < sd_x[0])),
                     "mean_rank_chosen": float(np.mean(mean_c > mean_x[0])),
                     "sd_chosen": float(sd_x[0]),
                     "sd_edge_band": float(np.median(sd_c[band])) if band.any() else np.nan,
                     "sd_interior": float(np.median(sd_c[~band])) if (~band).any() else np.nan,
                     "dist_polish": float(np.linalg.norm(chosen - start))})

    started = time.time()
    out = bo.bayes_opt(f, DIMENSION, -BOX, BOX, "ard", seed=instance, budget=CAP,
                       trace=trace, **BO_OPTIONS)
    return {"condition": condition, "position": position, "instance": instance,
            "seconds": time.time() - started,
            "pct_on_boundary": float(100 * np.mean(on_edge(out["x"]))),
            "trace": rows}


def sweep(workers):
    RESULTS.mkdir(exist_ok=True)
    done = set()
    if PARTIAL.exists():
        done = {(r["condition"], r["position"], r["instance"])
                for r in map(json.loads, PARTIAL.read_text().splitlines())}
    todo = [t for t in itertools.product(CONDITIONS, POSITIONS, range(N_INSTANCES))
            if t not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"[{i}/{len(todo)}] k={rec['condition']} {rec['position']} "
                  f"inst={rec['instance']} {rec['seconds']:.0f}s, "
                  f"{rec['pct_on_boundary']:.0f}% on edge, elapsed {time.time() - started:.0f}s",
                  flush=True)


def report():
    recs = [json.loads(line) for line in PARTIAL.read_text().splitlines()]
    trace = pd.DataFrame([{"condition": r["condition"], "position": r["position"],
                           "instance": r["instance"], **row}
                          for r in recs for row in r["trace"]])
    trace.to_csv(RESULTS / "e4b_trace.csv", index=False)
    pd.set_option("display.width", 200)
    print("% of runs' evaluations on the edge (median), as E4:")
    print(pd.DataFrame(recs).groupby(["condition", "position"]).pct_on_boundary
            .median().round(1).to_string())
    print("\n(a) is the GP more uncertain near the edge? median posterior SD of candidates")
    a = trace.groupby(["condition", "position"])[["sd_edge_band", "sd_interior"]].median()
    print(a.round(3).to_string())
    print("\n(b) where do edge picks come from? medians over iterations, by whether the pick is on the edge")
    b = (trace.groupby(["condition", "position", "chosen_on_edge"])
              .agg(iterations=("n", "size"), sd_rank=("sd_rank_chosen", "median"),
                   mean_rank=("mean_rank_chosen", "median"),
                   start_on_edge=("start_on_edge", "mean"),
                   dist_polish=("dist_polish", "median")))
    print(b.round(3).to_string())
    b.reset_index().to_csv(RESULTS / "e4b_summary.csv", index=False)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    report()
