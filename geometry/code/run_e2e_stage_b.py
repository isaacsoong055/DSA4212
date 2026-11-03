"""E2e, Stage B: does one change lift BO off its plateau on the round bowl?

Stage A (run_e2e_bo_plateau.py) found that the GP treats the best observations
as noise (H2), so EI against the observed best underflows (H4) and BO samples
almost at random. This tests one change at a time on BO-ARD, kappa = 1,
aligned, the same 15 instances and 200 evaluations as Stage A:

  B2  noise fixed at 1e-6 on the standardised scale (the objective is noise-free)
  B4  LogEI in place of EI (Ament et al., NeurIPS 2023)
  B5  B2 plus a fixed floor of 1e-12 in the log transform (added after B2
      stalled about 5 floors above the default relative floor)

B1 (changing the log transform) is held: the transform is there for numerical
stability, since raw values span many orders of magnitude. B3 (local candidates)
is dropped: Stage A found no extra EI near the incumbent.

The baseline is Stage A's BO-ARD runs (results/e2e_plateau_trace.csv), paired
by instance. Support: a change lifts BO to 1e-5. If nothing helps, the plateau
is BO's behaviour at this budget.

Writes results/e2e_stage_b_trace.csv (same columns as Stage A plus variant) and
results/e2e_stage_b_summary.csv.
--variants "B5 ..." runs only the named variants and replaces their rows in the
saved trace; the others are kept.
Usage: python code/run_e2e_stage_b.py [--workers N] [--report-only] [--variants NAME ...]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import itertools
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analysis
import run_e2e_bo_plateau as stage_a

RESULTS = stage_a.RESULTS
VARIANTS = {"B2 fixed noise": {"noise_variance": 1e-6},
            "B4 LogEI": {"acquisition": "logei"},
            "B5 fixed noise + floor 1e-12": {"noise_variance": 1e-6, "target_floor": 1e-12}}
METHOD = "bo-ard"
CHECKPOINTS = (20, 50, 100, 200)
TARGETS = (1e-3, 1e-5)


def run_variant(task):
    variant, instance = task
    rec = stage_a.run_one((instance, METHOD, VARIANTS[variant]))
    for row in rec["rows"]:
        row["variant"] = variant
    rec["variant"] = variant
    return rec


def best_so_far(trace):
    """Best value after each evaluation n = 9..200, rebuilt from the trace rows."""
    t = trace.sort_values("n")
    after = np.minimum(t.f_best.values, t.f_chosen.values)      # after evaluation n + 1
    return pd.Series(np.minimum.accumulate(after), index=t.n.values + 1)


def summarise(trace):
    rows = []
    for (variant, instance), g in trace.groupby(["variant", "instance"]):
        curve = best_so_far(g)
        row = {"variant": variant, "instance": instance}
        for n in CHECKPOINTS:
            row[f"log10_f_{n}"] = float(np.log10(curve.loc[:n].iloc[-1]))
        for t in TARGETS:
            row[f"reached_{t:g}"] = bool((curve <= t).any())
        late = g[g.n > 50]
        row["median_noise_sd"] = float(late.noise_sd.median())
        row["pct_ei_below_1e-12"] = float(100 * (late.cand_max_ei < 1e-12).mean())
        row["median_dist_chosen_incumbent"] = float(late.dist_chosen_incumbent.median())
        row["pct_improving"] = float(100 * late.improved.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def report(per_run):
    pd.set_option("display.width", 200)
    cols = [c for c in per_run.columns if c not in ("variant", "instance")]
    print("Medians over 15 instances (reached_*: % of runs); BO-ARD, kappa = 1, aligned")
    table = per_run.groupby("variant")[cols].median(numeric_only=True)
    for t in TARGETS:
        table[f"reached_{t:g}"] = 100 * per_run.groupby("variant")[f"reached_{t:g}"].mean()
    print(table.T.round(2).to_string())

    base = per_run[per_run.variant == "baseline"].sort_values("instance")
    out = []
    print("\nPaired bootstrap, variant minus baseline, median log10 f (negative = variant better)")
    for variant, n in itertools.product(VARIANTS, CHECKPOINTS):
        v = per_run[per_run.variant == variant].sort_values("instance")
        res = analysis.bootstrap_paired_difference(v[f"log10_f_{n}"].values,
                                                   base[f"log10_f_{n}"].values, n_boot=5000)
        out.append({"variant": variant, "budget": n, **res})
        print(f"  {variant:15s} n={n:<4} {res['difference']:+6.2f} "
              f"[{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]")
    return pd.DataFrame(out)


def load_baseline():
    a = pd.read_csv(RESULTS / "e2e_plateau_trace.csv")
    return a[a.method == METHOD].assign(variant="baseline")


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    path = RESULTS / "e2e_stage_b_trace.csv"
    if "--report-only" not in sys.argv:
        chosen = list(VARIANTS)
        if "--variants" in sys.argv:
            chosen = [a for a in sys.argv[sys.argv.index("--variants") + 1:]
                      if not a.startswith("--")]
            assert all(v in VARIANTS for v in chosen), chosen
        tasks = list(itertools.product(chosen, range(stage_a.N_INSTANCES)))
        started, records = time.time(), []
        with mp.Pool(workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(run_variant, tasks), 1):
                records.append(rec)
                print(f"[{i}/{len(tasks)}] {rec['variant']} inst={rec['instance']} "
                      f"{rec['seconds']:.0f}s, final f={rec['best_so_far'][-1]:.2e}, "
                      f"elapsed {time.time() - started:.0f}s", flush=True)
        trace = pd.DataFrame([row for rec in records for row in rec["rows"]])
        if path.exists():
            kept = pd.read_csv(path)
            trace = pd.concat([kept[~kept.variant.isin(chosen)], trace], ignore_index=True)
        trace.sort_values(["variant", "instance", "n"]).to_csv(path, index=False)
    trace = pd.concat([load_baseline(), pd.read_csv(path)], ignore_index=True)
    per_run = summarise(trace)
    per_run.to_csv(RESULTS / "e2e_stage_b_runs.csv", index=False)
    report(per_run).to_csv(RESULTS / "e2e_stage_b_summary.csv", index=False)
