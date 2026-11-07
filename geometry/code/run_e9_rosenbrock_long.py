"""E9: Rosenbrock with a budget long enough for CMA-ES to converge (CMA-ES only).

E5 ran Rosenbrock (COCO f8 aligned, f9 rotated) for 200 evaluations; nobody
converged, so its prediction "full CMA-ES beats diagonal on the rotated
version" could not be tested. Here CMA-ES runs to 2,000 x d = 4,000
evaluations, as E1 and E2's long runs, same instances and start (box centre
of [-5, 5]^2, sigma = 3).

Prediction: full and diagonal CMA-ES both converge on f8 and f9 (the valley
bends, so no single orientation is right for long), but full needs fewer
evaluations near the optimum, where the valley is a fixed tilted narrow
quadratic (local kappa ~ 2,500 at 63.5 degrees, even for "aligned" f8).

Writes results/e9_rosenbrock_long.csv.
Usage: python code/run_e9_rosenbrock_long.py
"""

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analysis
import cmaes
from run_e5_other_functions import rosenbrock

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
BUDGET = 4000
TARGETS = (1e-1, 1e-3, 1e-5, 1e-8)


def main():
    rows = []
    for geo, inst, variant in itertools.product(("none", "random"), range(15),
                                                ("full", "diagonal")):
        f = rosenbrock(inst, geo)
        values = []

        def counted(x):
            v = f(x)
            values.append(v)
            return v

        cmaes.cma_es(counted, 2, np.zeros(2), seed=inst, variant=variant,
                     max_evaluations=BUDGET, initial_sigma=3.0, target=1e-9,
                     record_history=False)
        best = np.minimum.accumulate(values)
        row = {"problem": f"rosenbrock:{geo}", "instance": inst, "variant": variant,
               "final_log10_f": float(np.log10(best[-1] + 1e-300))}
        for t in TARGETS:
            hit = np.flatnonzero(best <= t)
            row[f"runtime_{t:g}"] = float(hit[0] + 1) if hit.size else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e9_rosenbrock_long.csv", index=False)
    pd.set_option("display.width", 200)
    for t in TARGETS:
        col = f"runtime_{t:g}"
        print(f"target {t:g}: % reaching / median evaluations among successes")
        print(df.groupby(["problem", "variant"])[col]
                .agg(success=lambda v: round(100 * v.notna().mean()), median="median")
                .to_string())
    print("\nfull - diagonal, paired by instance: evaluations to 1e-5 (failures count as 4,000)")
    for prob in ("rosenbrock:none", "rosenbrock:random"):
        sub = df[df.problem == prob]
        a = sub[sub.variant == "full"].sort_values("instance")["runtime_1e-05"].fillna(BUDGET)
        b = sub[sub.variant == "diagonal"].sort_values("instance")["runtime_1e-05"].fillna(BUDGET)
        r = analysis.bootstrap_paired_difference(a.values, b.values, n_boot=5000)
        print(f"  {prob:18s} {r['difference']:+7.0f} [{r['ci_low']:+7.0f}, {r['ci_high']:+7.0f}]"
              " evaluations (negative = full faster)")


if __name__ == "__main__":
    main()
