"""E2e, Stage A: why does BO plateau near 1e-2.6 on the round bowl?

Instrument only; the optimiser's behaviour is unchanged. Re-runs BO-ARD and
BO-full on the E2 setup (2-D, kappa = 1, aligned, box [-5, 5]^2, 200
evaluations, seed = instance) and logs, at every iteration, what the GP and
the acquisition step saw. Each hypothesis predicts a different signature:

  H1  floor in the log transform: f_best approaches the floor
      max(spread * 1e-6, 1e-12)                       -> column f_best_over_floor
  H2  log transform makes the optimum a cusp the GP explains as noise: the
      fitted noise rises as f_best falls, and the GP's errors at the best
      points exceed the gap between the best two     -> noise_sd, worst_resid_top5, gap
  H3  acquisition search misses the small improving region: EI found by a
      dense search near the incumbent is far above the EI of the point
      chosen, and chosen points land far from the incumbent
                                                      -> local_max_ei vs chosen_ei,
                                                         dist_chosen_incumbent
  H4  EI underflows: max EI over candidates is ~0 or exactly 0
                                                      -> cand_max_ei, cand_frac_zero

Diagnostics use their own random generator, so the BO run is identical to
E2's; the script checks this against results/e2_runs_partial.jsonl.

Writes results/e2e_plateau_trace.csv (one row per run and iteration).
--determinism-check instead repeats two runs with and without the hook and
writes results/e2e_determinism.csv.
Usage: python code/run_e2e_bo_plateau.py [--workers N] [--determinism-check]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker
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
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bo
import problems

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
DIMENSION, BOX, CAP = 2, 5.0, 200
CONDITION, GEOMETRY = 1, "none"
N_INSTANCES = 15
METHODS = ("bo-ard", "bo-full")
LOCAL_RADII = (1e-3, 1e-2, 1e-1, 1.0)
LOCAL_POINTS = 500


def transformed_value(f, y, target_floor=None):
    """Where the value f lands on the GP's scale, using the transform of data y."""
    y = np.asarray(y, float)
    floor = bo.transform_floor(y, target_floor)
    shift = 0.0 if y.min() > 0 else (floor - y.min())
    logged = np.log10(y + shift + floor)
    return (np.log10(f + shift + floor) - logged.mean()) / (logged.std() + 1e-12), floor


def run_one(task):
    """task = (instance, method) or (instance, method, bayes_opt options) for Stage B."""
    instance, method = task[:2]
    options = task[2] if len(task) > 2 else {}
    metric = method[3:]
    objective, meta = problems.make_ellipsoid(DIMENSION, CONDITION, instance=instance,
                                              rotation=GEOMETRY)
    optimum = meta["optimum"]
    diag_rng = np.random.default_rng(10_000 + instance)    # never the BO generator
    rows = []

    def trace(s):
        x, y, z, model = s["x"], s["y"], s["standardised"], s["model"]
        inc = s["incumbent"]
        best = int(np.argmin(y))
        x_inc = x[best]
        _, cut = bo._precision_matrix(model["theta"], DIMENSION, metric)
        ell = np.sort(np.exp(bo.eigen_log_lengthscales(model["precision"], metric)))
        z_opt, floor = transformed_value(0.0, y, options.get("target_floor"))

        mean_inc, sd_inc = bo.predict(model, x_inc[None])
        top = np.argsort(z)[:5]
        mean_top, _ = bo.predict(model, x[top])
        mean_opt, sd_opt = bo.predict(model, optimum[None])
        ei_opt = bo.expected_improvement(model, optimum[None], inc)[0]

        # Dense search for EI around the incumbent, at several radii.
        local = {}
        best_local, best_local_ei = x_inc, -np.inf
        for r in LOCAL_RADII:
            u = diag_rng.normal(size=(LOCAL_POINTS, DIMENSION))
            u *= (r * diag_rng.uniform(size=(LOCAL_POINTS, 1)) ** (1 / DIMENSION)
                  / np.linalg.norm(u, axis=1, keepdims=True))
            pts = np.clip(x_inc + u, -BOX, BOX)
            ei = bo.expected_improvement(model, pts, inc)
            k = int(np.argmax(ei))
            local[r] = float(ei[k])
            if ei[k] > best_local_ei:
                best_local, best_local_ei = pts[k], float(ei[k])
        polished = minimize(lambda p: -float(bo.expected_improvement(model, p[None], inc)[0]),
                            best_local, method="L-BFGS-B", bounds=[(-BOX, BOX)] * DIMENSION)

        # Plain EI, recomputed, so the columns mean the same under every acquisition.
        cand_ei = bo.expected_improvement(model, s["candidates"], inc)
        chosen = s["chosen"]
        f_chosen = float(objective(chosen))
        rows.append({
            "instance": instance, "method": method, "n": len(y),
            "f_best": float(y[best]), "floor": floor, "f_best_over_floor": float(y[best]) / floor,
            "dist_inc_opt": float(np.linalg.norm(x_inc - optimum)),
            "noise_var": float(np.exp(2 * model["theta"][cut])),
            "noise_sd": float(np.exp(model["theta"][cut])),
            "ell_short": float(ell[0]), "ell_long": float(ell[-1]),
            "z_best": float(inc), "gap": float(np.sort(z)[1] - np.sort(z)[0]),
            "resid_inc": float(mean_inc[0] - z[best]), "sd_inc": float(sd_inc[0]),
            "worst_resid_top5": float(np.max(np.abs(mean_top - z[top]))),
            "z_opt_true": float(z_opt), "mean_opt": float(mean_opt[0]),
            "sd_opt": float(sd_opt[0]), "ei_opt": float(ei_opt),
            "cand_max_ei": float(cand_ei.max()),
            "cand_frac_zero": float(np.mean(cand_ei <= 0)),
            "cand_min_dist_opt": float(np.min(np.linalg.norm(s["candidates"] - optimum, axis=1))),
            "chosen_ei": float(bo.expected_improvement(model, chosen[None], inc)[0]),
            **{f"local_ei_r{r:g}": v for r, v in local.items()},
            "local_max_ei": max(best_local_ei, -float(polished.fun)),
            "dist_chosen_incumbent": float(np.linalg.norm(chosen - x_inc)),
            "dist_start_incumbent": float(np.linalg.norm(s["start"] - x_inc)),
            "f_chosen": f_chosen, "improved": f_chosen < float(y[best]),
            # with acquisition="logei" these two are log EI, not EI
            "acq_cand_max": float(np.max(s["candidate_ei"])), "acq_chosen": s["chosen_ei"],
        })

    started = time.time()
    out = bo.bayes_opt(objective, DIMENSION, -BOX, BOX, metric, seed=instance,
                       budget=CAP, trace=trace, **options)
    best_so_far = np.minimum.accumulate(out["y"]).tolist()
    return {"instance": instance, "method": method, "seconds": time.time() - started,
            "rows": rows, "best_so_far": best_so_far}


def check_against_e2(records):
    """Do the instrumented runs reproduce E2's best-so-far curves?"""
    path = RESULTS / "e2_runs_partial.jsonl"
    if not path.exists():
        print("no E2 checkpoint to compare against")
        return
    ref = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["condition"] == CONDITION and r["geometry"] == GEOMETRY and r["method"] in METHODS:
            ref[(r["instance"], r["method"])] = np.asarray(r["best_so_far"])
    worst = 0.0
    for rec in records:
        key = (rec["instance"], rec["method"])
        if key in ref:
            a = np.log10(np.asarray(rec["best_so_far"]) + 1e-300)
            b = np.log10(ref[key] + 1e-300)
            worst = max(worst, float(np.max(np.abs(a - b))))
    print(f"reproduction check vs E2: {len(ref)} reference runs, "
          f"max |log10 best-so-far difference| = {worst:.3g}")


def plain_run(task):
    """The same BO run with no trace hook, for the determinism check."""
    instance, method = task
    objective, _ = problems.make_ellipsoid(DIMENSION, CONDITION, instance=instance,
                                           rotation=GEOMETRY)
    out = bo.bayes_opt(objective, DIMENSION, -BOX, BOX, method[3:], seed=instance, budget=CAP)
    return np.minimum.accumulate(out["y"])


def _determinism_job(arg):
    kind, task = arg
    return plain_run(task) if kind == "plain" else np.asarray(run_one(task)["best_so_far"])


def determinism_check(workers, tasks=((2, "bo-ard"), (9, "bo-full"))):
    """Run each task twice without the hook and once with it; compare best-so-far curves.

    Separates two questions: does the hook change the run (plain vs traced), and
    is a BO run repeatable at all on one machine (plain vs plain)?
    """
    jobs = [(k, t) for t in tasks for k in ("plain", "plain", "traced")]
    with mp.Pool(workers) as pool:
        out = pool.map(_determinism_job, jobs)
    rows = []
    for j, task in enumerate(tasks):
        a, b, c = out[3 * j:3 * j + 3]
        for name, u, v in (("plain vs plain", a, b), ("plain vs traced", a, c)):
            rows.append({"instance": task[0], "method": task[1], "comparison": name,
                         "identical": bool(np.array_equal(u, v)),
                         "max_rel_diff": float(np.max(np.abs(u / v - 1))),
                         "final_log10_a": float(np.log10(u[-1])),
                         "final_log10_b": float(np.log10(v[-1]))})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "e2e_determinism.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--determinism-check" in sys.argv:
        determinism_check(workers)
        sys.exit()
    tasks = list(itertools.product(range(N_INSTANCES), METHODS))
    tasks.sort(key=lambda t: t[1] != "bo-full")             # slower metric first
    started, records = time.time(), []
    with mp.Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_one, tasks), 1):
            records.append(rec)
            print(f"[{i}/{len(tasks)}] {rec['method']} inst={rec['instance']} "
                  f"{rec['seconds']:.0f}s, final f={rec['best_so_far'][-1]:.2e}, "
                  f"elapsed {time.time() - started:.0f}s", flush=True)
    RESULTS.mkdir(exist_ok=True)
    trace = pd.DataFrame([row for rec in records for row in rec["rows"]])
    trace = trace.sort_values(["method", "instance", "n"])
    trace.to_csv(RESULTS / "e2e_plateau_trace.csv", index=False)
    check_against_e2(records)
