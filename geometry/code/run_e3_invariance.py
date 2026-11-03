"""E3: which theoretical invariances survive the practical implementation?

Hansen lists the invariances CMA-ES is designed to have: invariance to
order-preserving transformations of the objective value (the algorithm reads
only the ranking), to rigid transformations of the search space (rotation,
reflection, translation) provided the initial point is transformed with it, and
to any invertible linear transformation provided C and the initial point are
set accordingly.  Each of those conditions is a thing that can fail in a real
implementation, and the conditional clauses are where the interesting results
live.

Four transformations, applied one at a time to a fixed base problem, against
four methods:

  rotation   -- Haar-random rotation of the search space.
                Expect: full CMA-ES invariant; diagonal CMA-ES and ARD-BO not.
  rescaling  -- change of units, x_i measured in different units.
                Expect: a diagonal model is invariant in PRINCIPLE, but only
                once it has estimated the scales, so with a small budget the
                invariance may not materialise.  For BO the box is rescaled
                too, which is itself extra information.
  translation -- move the optimum within the domain.
                Expect: free for CMA-ES; BO with a fixed box and a constant
                prior mean can suffer when the optimum nears a boundary.
  monotone    -- strictly increasing map of the objective VALUE only.
                Expect: CMA-ES exactly invariant (rank-based); EI is not,
                because it reads f values and not their ranks.  This is the
                sharpest and cheapest discriminator in the project, and it is
                a different mechanism from geometry.

Writes results/e3_invariance.csv.
"""

import itertools
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
DIMENSION = 2
CONDITION = 100
N_INSTANCES = 10
BUDGET = 80
BOX = 5.0
START = 2.0

METHODS = (("cmaes", "full"), ("cmaes", "diagonal"), ("bo", "full"), ("bo", "ard"))


def build(condition_name, instance):
    """Return (objective, box, note) for one transformation condition.

    Every condition keeps the same underlying ellipsoid; only the stated
    transformation is applied, so any performance change is attributable to it.
    """
    base, meta = problems.make_ellipsoid(DIMENSION, CONDITION, instance=instance,
                                         rotation="none")
    rng = np.random.default_rng(500 + instance)

    if condition_name == "baseline":
        return base, BOX, ""
    if condition_name == "rotation":
        rotated, _ = problems.make_ellipsoid(DIMENSION, CONDITION, instance=instance,
                                             rotation="random")
        return rotated, BOX, "Haar-random rotation"
    if condition_name == "rescaling":
        scales = np.array([1.0, 10.0])
        # The box is rescaled with the coordinates, so BO is given the same
        # relative region, not a harder one.
        return problems.rescale_coordinates(base, scales), BOX, "x2 in units of 10"
    if condition_name == "translation":
        shift = rng.uniform(1.5, 3.0, DIMENSION) * rng.choice([-1, 1], DIMENSION)
        return problems.translate(base, shift), BOX, f"optimum shifted by {np.round(shift,2)}"
    if condition_name.startswith("monotone:"):
        name = condition_name.split(":", 1)[1]
        return problems.transform_value(base, name), BOX, f"f -> {name}(f)"
    raise ValueError(condition_name)


def evaluate_recommendation(objective_raw, point):
    """Score the recommended point on the UNTRANSFORMED objective.

    A monotone transform changes the numbers a method reports, so final values
    are not comparable across conditions unless the recommendation is re-scored
    on a common objective.  This is the same discipline the project brief asks
    for with noisy objectives: assess the recommendation, not the best
    observation.
    """
    return objective_raw(point)


CONDITIONS = ("baseline", "rotation", "rescaling", "translation",
              "monotone:log1p", "monotone:cube", "monotone:sqrt")


def run():
    records = []
    for condition_name, inst in itertools.product(CONDITIONS, range(N_INSTANCES)):
        objective, box, note = build(condition_name, inst)
        raw, _ = problems.make_ellipsoid(DIMENSION, CONDITION, instance=inst,
                                         rotation="none")
        # For conditions that move the search space, the common yardstick is the
        # transformed problem itself; for value-only transforms it is the raw f.
        scorer = raw if condition_name.startswith("monotone:") else objective

        for family, model in METHODS:
            t0 = time.time()
            if family == "bo":
                r = bo.bayes_opt(objective, DIMENSION, -box, box, model, inst,
                                 budget=BUDGET)
            else:
                r = cmaes.cma_es(objective, DIMENSION,
                                 initial_mean=np.full(DIMENSION, START),
                                 seed=inst, variant=model, max_evaluations=BUDGET,
                                 target=1e-8)
            seconds = time.time() - t0
            records.append({"condition": condition_name, "note": note,
                            "instance": inst, "family": family, "model": model,
                            "method": f"{family}-{model}",
                            "scored_value": evaluate_recommendation(scorer, r["best_point"]),
                            "reported_value": r["best_value"],
                            "evaluations": r["evaluations"], "seconds": seconds})
    return pd.DataFrame(records)


def report(df):
    print("\nmedian objective of the RECOMMENDED point, scored on a common objective")
    print(df.pivot_table(index="condition", columns="method", values="scored_value",
                         aggfunc="median").to_string(float_format=lambda x: f"{x:.3e}"))

    print("\ninvariance check: ratio to the baseline condition (1.0 = invariant)")
    piv = df.pivot_table(index="condition", columns="method", values="scored_value",
                         aggfunc="median")
    print((piv / piv.loc["baseline"]).to_string(float_format=lambda x: f"{x:.3g}"))

    print("\npaired bootstrap vs baseline, log10 scored value")
    for condition_name in CONDITIONS:
        if condition_name == "baseline":
            continue
        for method in sorted(df.method.unique()):
            a = np.log10(df[(df.condition == condition_name) & (df.method == method)]
                         .sort_values("instance").scored_value.values + 1e-300)
            b = np.log10(df[(df.condition == "baseline") & (df.method == method)]
                         .sort_values("instance").scored_value.values + 1e-300)
            out = analysis.bootstrap_paired_difference(a, b, n_boot=5000)
            flag = "" if out["ci_low"] <= 0 <= out["ci_high"] else "  <-- not invariant"
            print(f"  {condition_name:16s} {method:16s} dlog10 = {out['difference']:+.2f} "
                  f"[{out['ci_low']:+.2f}, {out['ci_high']:+.2f}]{flag}")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    df = run()
    df.to_csv(RESULTS / "e3_invariance.csv", index=False)
    report(df)
    print(f"\ntotal {time.time() - t0:.1f}s -> {RESULTS / 'e3_invariance.csv'}")
