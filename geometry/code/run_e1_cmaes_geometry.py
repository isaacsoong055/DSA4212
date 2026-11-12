"""E1: does learning a rotation matter?

Rewrite of the original notebook experiment with its two weaknesses fixed:

1. The optimum is no longer at the origin.  Each instance places it on a
   sphere of radius 2, so the search cannot start at the solution.
2. Rotation is swept over many angles and, separately, drawn Haar-random per
   instance, instead of testing only 0 and 45 degrees.  45 degrees in 2-D is a
   special, maximally adversarial angle and a single angle cannot distinguish
   "not invariant" from "unlucky".

The ablation itself is unchanged: full-covariance CMA-ES against a
diagonal-covariance CMA-ES that can stretch axes but not rotate.

Writes results/e1_angle_sweep_*.csv and results/e1_random_rotation_*.csv.
"""

import itertools
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analysis
import cmaes
import problems

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
START = 2.0          # search starts at (START, ..., START)
SIGMA0 = 2.0
TARGET = 1e-8
N_INSTANCES = 15     # COCO-style: instances, not just seeds


def one_run(objective, dimension, seed, variant, budget):
    return cmaes.cma_es(objective, dimension,
                        initial_mean=np.full(dimension, START),
                        seed=seed, variant=variant,
                        max_evaluations=budget, initial_sigma=SIGMA0,
                        target=TARGET)


def angle_sweep(conditions=(1, 10, 100, 1000),
                angles=(0, 5, 15, 30, 45, 60, 75, 90),
                budget=4000):
    """2-D: how performance varies continuously with the rotation angle."""
    records = []
    for cond, angle, inst in itertools.product(conditions, angles, range(N_INSTANCES)):
        objective, _ = problems.make_ellipsoid(2, cond, instance=inst, rotation=angle)
        for variant in cmaes.VARIANTS:
            r = one_run(objective, 2, inst, variant, budget)
            records.append({"condition": cond, "angle": angle, "instance": inst,
                            "variant": variant, "best_value": r["best_value"],
                            "evaluations": r["evaluations"],
                            "population_size": r["population_size"],
                            "history": r["history"]})
    return records


def random_rotation(conditions=(1, 10, 100, 1000), dimension=2, budget=4000):
    """Invariance test proper: Haar-random rotations against no rotation.

    CMA-ES's rotation invariance is distributional, not pathwise: with C = I the
    first generation samples the same points either way, but the rotated
    objective ranks them differently, so individual trajectories diverge.  The
    claim to test is therefore that the two runtime DISTRIBUTIONS agree, which
    needs many instances, not a paired comparison of two runs.
    """
    records = []
    for cond, inst in itertools.product(conditions, range(2 * N_INSTANCES)):
        for geometry in ("none", "random"):
            objective, _ = problems.make_ellipsoid(dimension, cond, instance=inst,
                                                   rotation=geometry)
            for variant in cmaes.VARIANTS:
                r = one_run(objective, dimension, inst, variant, budget)
                records.append({"condition": cond, "geometry": geometry,
                                "dimension": dimension, "instance": inst,
                                "variant": variant, "best_value": r["best_value"],
                                "evaluations": r["evaluations"],
                                "population_size": r["population_size"],
                                "history": r["history"]})
    return records


def save(records, name, group_columns):
    RESULTS.mkdir(exist_ok=True)
    runs = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in records])
    runs.to_csv(RESULTS / f"{name}_runs.csv", index=False)
    analysis.runtime_table(records).to_csv(RESULTS / f"{name}_runtimes.csv", index=False)
    print(f"\n--- {name} ---")
    print(analysis.summarise(runs, group_columns).to_string(index=False))
    return runs


if __name__ == "__main__":
    t0 = time.time()
    save(angle_sweep(), "e1_angle_sweep", ["condition", "angle", "variant"])
    save(random_rotation(), "e1_random_rotation", ["condition", "geometry", "variant"])
    print(f"\ntotal {time.time() - t0:.1f}s -> {RESULTS}")
