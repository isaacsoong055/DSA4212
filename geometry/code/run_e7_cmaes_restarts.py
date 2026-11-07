"""E7: would restarts change the CMA-ES results? (CMA-ES only; seconds.)

The CMA-ES implementation has no restarts. Restarts help an ES escape a poor
local state (a local minimum, or a collapsed step size). On our problems the
only CMA-ES failure is diagonal CMA-ES on rotated, ill-conditioned quadratics
(E1, E2). Claim to test: that failure is structural, because a restarted
diagonal run still cannot tilt, so restarts should not repair it.

Problems: the E2 2-D ellipsoids (kappa in {100, 10000}, aligned and rotated,
15 instances; optimum on the radius-2 sphere), start at the centre of
[-5, 5]^2 with sigma = 3. Budget 4,000 evaluations, as E2's long runs.
  single   one run of 4,000 evaluations (lambda = 6)
  ipop     four runs of 1,000 evaluations each, population doubling
           (lambda = 6, 12, 24, 48), each from the box centre with its own
           seed; the best value over all four is kept (IPOP-style; Auger and
           Hansen 2005 restart on a stopping criterion instead, which our
           implementation does not have, so the budget is split evenly)

Writes results/e7_restarts.csv.
Usage: python code/run_e7_cmaes_restarts.py
"""

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cmaes
import problems

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
BUDGET = 4000
POPULATIONS = (6, 12, 24, 48)
TARGETS = (1e-3, 1e-5)


def run(f, variant, seed, scheme):
    if scheme == "single":
        return cmaes.cma_es(f, 2, np.zeros(2), seed=seed, variant=variant,
                            max_evaluations=BUDGET, initial_sigma=3.0, target=1e-6,
                            record_history=False)["best_value"]
    share = BUDGET // len(POPULATIONS)
    return min(cmaes.cma_es(f, 2, np.zeros(2), seed=1000 * k + seed, variant=variant,
                            max_evaluations=share, initial_sigma=3.0, target=1e-6,
                            population_size=lam, record_history=False)["best_value"]
               for k, lam in enumerate(POPULATIONS))


def main():
    rows = []
    for kappa, geo, inst, variant, scheme in itertools.product(
            (100, 10000), ("none", "random"), range(15), ("full", "diagonal"),
            ("single", "ipop")):
        f, _ = problems.make_ellipsoid(2, kappa, instance=inst, rotation=geo)
        best = run(f, variant, inst, scheme)
        rows.append({"condition": kappa, "geometry": geo, "instance": inst,
                     "variant": variant, "scheme": scheme, "best_value": best,
                     **{f"reached_{t:g}": best <= t for t in TARGETS}})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "e7_restarts.csv", index=False)
    pd.set_option("display.width", 200)
    print("% of 15 instances reaching 1e-5 within 4,000 evaluations")
    print((100 * df.pivot_table(index=["condition", "geometry"], columns=["variant", "scheme"],
                                values="reached_1e-05", aggfunc="mean")).round(0).to_string())
    print("\nmedian log10 best value")
    print(df.assign(l=np.log10(df.best_value + 1e-300))
            .pivot_table(index=["condition", "geometry"], columns=["variant", "scheme"],
                         values="l", aggfunc="median").round(2).to_string())


if __name__ == "__main__":
    main()
