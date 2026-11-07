"""E11: the boundary trap. Why does BO stick at the box edge, and does reflecting
local candidates instead of clipping them remove it?

Background. With local candidates (e2_lc) or a trust region (E10), 4-8 of 15
BO runs per aligned ill-conditioned cell end stuck at f ~ 5, the incumbent on
the valley floor at the box edge; plain EI with the same noise fix almost never
does (1 run). Both trapped methods CLIP out-of-box candidates onto the box,
which piles about half of them onto the edge when the incumbent sits on it.

Part A, diagnosis (clip, as e2_lc; aligned kappa in {100, 10000}, ARD and full,
15 instances): per iteration, log whether the incumbent and the chosen point
are on the edge, the EI of the chosen point, and, among the 200 GLOBAL
candidates, how many lie in the true improving set {f < f_best} and the best
EI among them. The question: when better points ARE among the candidates, why
does EI prefer the edge? The run must reproduce e2_lc exactly (same code path;
the trace hook does not touch the random stream).

Part B, remedy for local candidates: e2_lc's cells (kappa in {100, 10000},
aligned and rotated, ARD and full) with local_mode="reflect". Paired against
e2_lc (clip).

Part C, remedy for the trust region: E10's cells with local_mode="reflect",
paired against E10 (clip).

Predictions, written before running: reflection cuts stuck runs (f > 1 at 200)
on aligned cells to about zero for both methods, and the rotation result
persists (ARD behind full on rotated kappa = 10,000). Part A: in trapped runs
the global candidates DO contain improving points, but their EI is far below
the chosen edge point's.

Writes results/e11_trace.csv (Part A, per iteration), results/e11_runs.csv,
results/e11_summary.csv; checkpoint results/e11_partial.jsonl.
Usage: python code/run_e11_boundary_trap.py [--workers N] [--report-only]
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
import analysis
import bo
import problems

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
PARTIAL = RESULTS / "e11_partial.jsonl"
DIMENSION, BOX, CAP, N_INSTANCES, N_GLOBAL = 2, 5.0, 200, 15, 200
EDGE = 1e-9
CONDITIONS = (100, 10000)
GEOMETRIES = ("none", "random")


def on_edge(p):
    return bool(np.any(np.abs(np.asarray(p)) >= BOX - EDGE))


def run_one(task):
    part, condition, geometry, instance, method = task
    f, _ = problems.make_ellipsoid(DIMENSION, condition, instance=instance, rotation=geometry)
    rows = []

    def trace(s):
        y = s["y"]
        f_best = float(y.min())
        cand, ei = s["candidates"], s["candidate_ei"]
        glob_true = np.array([f(c) for c in cand[:N_GLOBAL]])
        improving = glob_true < f_best
        rows.append({"n": len(y), "f_best": f_best,
                     "incumbent_on_edge": on_edge(s["x"][int(np.argmin(y))]),
                     "chosen_on_edge": on_edge(s["chosen"]),
                     "chosen_ei": s["chosen_ei"],
                     "n_global_improving": int(improving.sum()),
                     "best_ei_global_improving": float(ei[:N_GLOBAL][improving].max())
                     if improving.any() else np.nan,
                     "best_ei_local": float(ei[N_GLOBAL:].max()) if len(ei) > N_GLOBAL else np.nan})

    started = time.time()
    if method.startswith("tr-"):
        out = bo.trust_region_bo(f, DIMENSION, -BOX, BOX, method.split("-")[1], seed=instance,
                                 budget=CAP, local_mode="reflect")
    else:
        mode = "clip" if part == "A" else "reflect"
        out = bo.bayes_opt(f, DIMENSION, -BOX, BOX, method[3:], seed=instance, budget=CAP,
                           noise_variance=1e-6, local_candidates=200, local_mode=mode,
                           trace=trace if part == "A" else None)
    points = np.asarray(out["x"])
    return {"part": part, "condition": condition, "geometry": geometry, "instance": instance,
            "method": method, "seconds": time.time() - started,
            "best_so_far": np.minimum.accumulate(out["y"]).tolist(),
            "pct_on_edge": float(100 * np.mean([on_edge(p) for p in points])),
            "trace": rows}


def all_tasks():
    a = [("A", c, "none", i, m) for c, i, m in
         itertools.product(CONDITIONS, range(N_INSTANCES), ("bo-ard", "bo-full"))]
    b = [("B", c, g, i, m) for c, g, i, m in
         itertools.product(CONDITIONS, GEOMETRIES, range(N_INSTANCES), ("bo-ard", "bo-full"))]
    c = [("C", cc, g, i, m) for cc, g, i, m in
         itertools.product(CONDITIONS, GEOMETRIES, range(N_INSTANCES), ("tr-ard", "tr-full"))]
    return a + b + c


def sweep(workers):
    done = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["part"], r["condition"], r["geometry"], r["instance"], r["method"]))
    todo = [t for t in all_tasks() if t not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go", flush=True)
    started = time.time()
    with mp.Pool(workers) as pool, PARTIAL.open("a") as sink:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            print(f"[{i}/{len(todo)}] part {rec['part']} {rec['method']} k={rec['condition']} "
                  f"{rec['geometry']} inst={rec['instance']} {rec['seconds']:.0f}s "
                  f"elapsed {time.time() - started:.0f}s", flush=True)


def load_reference(path, method_filter):
    ref = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if method_filter(r["method"]):
            ref[(r["condition"], r["geometry"], r["instance"], r["method"])] = \
                float(np.log10(r["best_so_far"][-1] + 1e-300))
    return ref


def report():
    pd.set_option("display.width", 220)
    recs = [json.loads(l) for l in PARTIAL.read_text().splitlines()]
    runs = pd.DataFrame([{k: r[k] for k in ("part", "condition", "geometry", "instance", "method",
                                             "seconds", "pct_on_edge")}
                         | {"log10_f_200": float(np.log10(r["best_so_far"][-1] + 1e-300))}
                         for r in recs])
    trace = pd.DataFrame([{**t, "condition": r["condition"], "instance": r["instance"],
                           "method": r["method"], "final_log10": float(np.log10(r["best_so_far"][-1]))}
                          for r in recs if r["part"] == "A" for t in r["trace"]])
    runs.to_csv(RESULTS / "e11_runs.csv", index=False)
    trace.to_csv(RESULTS / "e11_trace.csv", index=False)
    lc = load_reference(RESULTS / "e2_lc_runs_partial.jsonl", lambda m: m.startswith("bo-"))
    tr = load_reference(RESULTS / "e10_trust_region_partial.jsonl", lambda m: m.startswith("tr-"))
    out = []

    a = runs[runs.part == "A"]
    diffs = [abs(r.log10_f_200 - lc[(r.condition, r.geometry, r.instance, r.method)])
             for r in a.itertuples()]
    print(f"\n== Part A reproduces e2_lc: max |log10 f difference| = {max(diffs):.2e}")
    a = a.assign(stuck=a.log10_f_200 > 0)
    print("Part A: % of evaluations on the box edge, stuck vs not stuck runs (median)")
    print(a.groupby(["condition", "method", "stuck"]).pct_on_edge.median().round(0).unstack().to_string())
    late = trace[trace.n > 100].assign(stuck=trace.final_log10 > 0)
    print("\nPart A, iterations after 100: medians (stuck runs vs others)")
    agg = late.groupby(["stuck"]).agg(
        incumbent_on_edge=("incumbent_on_edge", "mean"), chosen_on_edge=("chosen_on_edge", "mean"),
        n_global_improving=("n_global_improving", "median"),
        log10_chosen_ei=("chosen_ei", lambda v: np.log10(np.median(v) + 1e-300)),
        log10_best_ei_global_improving=("best_ei_global_improving",
                                        lambda v: np.log10(np.nanmedian(v) + 1e-300)),
        log10_best_ei_local=("best_ei_local", lambda v: np.log10(np.nanmedian(v) + 1e-300)))
    print(agg.round(3).to_string())

    def compare(part, ref, label):
        sub = runs[runs.part == part]
        print(f"\n== Part {part}: {label} (reflect) vs clip, at 200")
        for (c, g, m), grp in sub.groupby(["condition", "geometry", "method"]):
            grp = grp.sort_values("instance")
            old = np.array([ref[(c, g, i, m)] for i in grp.instance])
            new = grp.log10_f_200.values
            res = analysis.bootstrap_paired_difference(new, old, n_boot=5000)
            out.append({"part": part, "condition": c, "geometry": g, "method": m,
                        "comparison": "reflect - clip", "stuck_clip": int((old > 0).sum()),
                        "stuck_reflect": int((new > 0).sum()), **res})
            print(f"  k={c:<6} {g:7s} {m:8s} stuck {int((old > 0).sum()):2d} -> {int((new > 0).sum()):2d}"
                  f"   reflect - clip {res['difference']:+6.2f} [{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]")
        for c in CONDITIONS:
            ms = sorted(sub.method.unique())
            v = lambda g, m: sub[(sub.condition == c) & (sub.geometry == g) & (sub.method == m)] \
                .sort_values("instance").log10_f_200.values
            for g in GEOMETRIES:
                res = analysis.bootstrap_paired_difference(v(g, ms[0]), v(g, ms[1]), n_boot=5000)
                out.append({"part": part, "condition": c, "geometry": g,
                            "comparison": f"{ms[0]} - {ms[1]}", **res})
                print(f"  k={c:<6} {g:7s} {ms[0]} - {ms[1]} (+ = axis-restricted worse) "
                      f"{res['difference']:+6.2f} [{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]")

    compare("B", lc, "local candidates")
    compare("C", tr, "trust region")
    pd.DataFrame(out).to_csv(RESULTS / "e11_summary.csv", index=False)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--report-only" not in sys.argv:
        sweep(workers)
    report()
