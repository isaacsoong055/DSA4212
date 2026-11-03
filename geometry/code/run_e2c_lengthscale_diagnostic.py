"""E2c: read the kernel's internal state instead of inferring it from performance.

Every other experiment here measures an OUTCOME (did the optimiser reach the
target) and infers a cause. This one measures the cause directly: after fitting
a GP to a design of points on an ellipsoid, we read back the lengthscales it
learned and ask whether they describe the problem's actual geometry.

Three parts, each answering a different objection:

Part 1 -- what the kernel learns.
    Fit ARD and full-metric kernels on the same problems and report the
    anisotropy each one recovers, plus the direction its long axis points.
    There is a quantitative prediction to check, not just a direction of
    effect: for f = 0.5 x^T H x, a step of size s along an eigenvector with
    eigenvalue a changes f by about a*s^2/2, so the step giving a fixed change
    scales as 1/sqrt(a). The natural lengthscale ratio between the flattest
    and steepest directions is therefore sqrt(condition number), NOT the
    condition number. On an axis-aligned problem ARD should recover roughly
    that; on a rotated one it should collapse towards 1.

Part 2 -- what the data says, with no model at all.
    The typical size of f(x + h*d) - f(x) for random base points x, measured
    along each coordinate axis and along the problem's true flat direction.
    This is a property of the objective, computed without fitting anything, so
    it cannot be blamed on our GP implementation. It is also, in essence, the
    quantity the marginal likelihood is matching when it chooses lengthscales:
    a large typical gap at a small separation means low correlation at that
    separation, which means a short lengthscale.

Part 3 was removed. It summed the fitted kernel's similarity from points on
    the valley floor to a query further along the same line. The quantity was
    non-standard and its scale arbitrary: the 0-to-3 range was an artefact of
    choosing three probe points, and the offsets and the choice to sum rather
    than average were equally free. It was also redundant -- Part 1 shows the
    mechanism and Part 4 shows the consequence on a measure that needs no
    explanation. Prefer an established metric over an invented one.

Part 3 -- held-out predictive accuracy (THE STANDARD MEASURE).
    Fit on the design, then predict the objective at points the GP has never
    seen, and score those predictions. This is the ordinary way a GP is
    evaluated, so the numbers mean something to a reader who has never seen
    this project. Two held-out sets are scored separately: points along the
    valley floor, where the geometry is the whole question, and points drawn
    uniformly from the box, which checks whether any gap is specific to the
    valley or is just a worse model everywhere.

    Scores are computed on the GP's own transformed target scale, using the
    transform fitted to the TRAINING targets alone, so no test information
    leaks into the standardisation. Root mean squared error measures the point
    prediction; mean log predictive density also scores the uncertainty, which
    matters here because a model that is wrong but admits it is less damaging
    to Bayesian optimisation than one that is wrong and confident.

    An earlier version of this experiment also scored a held-out set placed
    along the valley floor. It was dropped, and the reason is worth recording
    because it is a result in its own right: a uniform design over the box
    barely samples a narrow valley at all. At condition 10000 the trench is a
    hundred times longer than it is wide, so forty uniform points land almost
    entirely on the walls, and every objective value on the valley floor lies
    far below anything in the training set. Scoring there measures
    extrapolation into an unsampled region, which every model fails and which
    therefore separates none of them -- in the smoke test both kernels scored
    an RMSE near 2.0 on an axis-aligned problem where ARD is known to be
    competent. The uniform held-out set below samples the same distribution
    the model was fitted on, which is the standard arrangement and which does
    discriminate.

Resolved issue (E2d, run_e2d_fit_diagnostic.py): with plain maximum likelihood
the fitted lengthscale ratio overshot sqrt(condition) on aligned problems. The
cause was the hard bound ell <= e^4 in bo.fit_gp: along the flat direction the
likelihood optimum is at ell -> infinity, so the bound, not the data, set the
ratio at condition >= 1000. bo.fit_gp now fits by MAP with a rotation-invariant
LogNormal lengthscale prior (see bo.py), and this script uses it. Absolute
ratios at high condition numbers are still partly set by that prior; the
prior-sensitivity check in E2d says how much.

The headline claim this supports: ARD's failure is a FAMILY restriction, not a
fitting failure. Part 1 shows the parameters are estimated competently when the
structure is axis-aligned; Part 2 shows the flat direction is plainly present
in the data either way; Part 3 shows ARD cannot act on it. No quantity of data
revises a restriction imposed before any data arrived.

Writes results/e2c_lengthscales.csv, results/e2c_structure.csv and
results/e2c_heldout.csv.
"""

import itertools
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bo
import problems

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
DIMENSION = 2
CONDITIONS = (1, 10, 100, 1000, 10000)
GEOMETRIES = ("none", 45, "random")
N_INSTANCES = 10
N_DESIGN = 40          # points the GP is fitted to
BOX = 5.0
N_RESTARTS = bo.DEFAULT_RESTARTS   # MAP restarts; same as the BO default (10)


def geometry_label(geometry):
    return {"none": "aligned", "random": "random rotation"}.get(geometry, f"{geometry} deg")


def principal_directions(hessian):
    """Eigenvalues ascending, with their eigenvectors. Column 0 is flattest."""
    values, vectors = np.linalg.eigh(hessian)
    order = np.argsort(values)
    return values[order], vectors[:, order]


def axis_angle(vector):
    """Orientation of an undirected axis, in degrees on [0, 180)."""
    return float(np.rad2deg(np.arctan2(vector[1], vector[0])) % 180.0)


def angle_difference(a, b):
    """Smallest angle between two undirected axes, in degrees on [0, 90]."""
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def kernel_lengthscales(precision):
    """Lengthscales along the kernel metric's own eigen-directions, longest first.

    The kernel uses r^2 = (x - x')^T M (x - x'). Along an eigenvector of M with
    eigenvalue lam, r^2 = lam * s^2 for a step s, so that direction behaves like
    a lengthscale of 1/sqrt(lam). A long lengthscale is therefore a SMALL
    eigenvalue of M.
    """
    values, vectors = np.linalg.eigh(precision)
    lengths = 1.0 / np.sqrt(np.maximum(values, 1e-12))
    order = np.argsort(-lengths)
    return lengths[order], vectors[:, order]


def design(objective, instance):
    rng = np.random.default_rng(90000 + instance)
    x = rng.uniform(-BOX, BOX, size=(N_DESIGN, DIMENSION))
    y = np.array([objective(p) for p in x])
    return x, y, rng


def part1_lengthscales():
    rows = []
    for condition, geometry, instance in itertools.product(CONDITIONS, GEOMETRIES,
                                                           range(N_INSTANCES)):
        objective, meta = problems.make_ellipsoid(DIMENSION, condition,
                                                  instance=instance, rotation=geometry)
        _, true_vectors = principal_directions(meta["hessian"])
        true_angle = axis_angle(true_vectors[:, 0])     # flattest direction
        x, y, rng = design(objective, instance)
        targets = bo._transform_targets(y)

        for metric in bo.METRICS:
            model = bo.fit_gp(x, targets, metric, rng, n_restarts=N_RESTARTS,
                              input_scale=2 * BOX)
            lengths, vectors = kernel_lengthscales(model["precision"])
            learned_angle = axis_angle(vectors[:, 0])   # long-lengthscale direction
            rows.append({
                "condition": condition,
                "geometry": geometry_label(geometry),
                "instance": instance,
                "metric": metric,
                "ell_long": lengths[0],
                "ell_short": lengths[-1],
                "ratio": lengths[0] / lengths[-1],
                "predicted_ratio": np.sqrt(condition),
                "learned_long_axis_deg": learned_angle,
                "true_flat_axis_deg": true_angle,
                # On an isotropic problem (condition 1) every direction is
                # equivalent, so "the flat direction" is arbitrary and an axis
                # error is meaningless rather than zero. Record it as missing.
                "axis_error_deg": (np.nan if condition == 1
                                   else angle_difference(learned_angle, true_angle)),
                "nlml": model["nlml"],
            })
    return pd.DataFrame(rows)


def part2_structure(step=0.1, n_samples=60000):
    """Typical |f(x+h*d) - f(x)|, computed from the objective with no model."""
    rows = []
    for condition, geometry, instance in itertools.product(CONDITIONS, GEOMETRIES,
                                                           range(N_INSTANCES)):
        objective, meta = problems.make_ellipsoid(DIMENSION, condition,
                                                  instance=instance, rotation=geometry)
        hessian = meta["hessian"]
        _, true_vectors = principal_directions(hessian)
        rng = np.random.default_rng(70000 + instance)
        base = rng.uniform(-3.0, 3.0, size=(n_samples, DIMENSION))

        def rms_gap(direction):
            d = np.asarray(direction, dtype=float)
            d = d / np.linalg.norm(d)
            here = 0.5 * np.einsum("ij,jk,ik->i", base, hessian, base)
            moved = base + step * d
            there = 0.5 * np.einsum("ij,jk,ik->i", moved, hessian, moved)
            return float(np.sqrt(np.mean((there - here) ** 2)))

        rows.append({
            "condition": condition,
            "geometry": geometry_label(geometry),
            "instance": instance,
            "step": step,
            "gap_axis_1": rms_gap([1.0, 0.0]),
            "gap_axis_2": rms_gap([0.0, 1.0]),
            "gap_flat_direction": rms_gap(true_vectors[:, 0]),
        })
    return pd.DataFrame(rows)


def training_transform(y_train):
    """Mirror bo._transform_targets, but return the parameters so the same
    transform can be applied to held-out targets.

    Standardising test targets with their own mean and standard deviation
    would leak information about the test set into the score, so the training
    statistics are the only ones used.
    """
    y_train = np.asarray(y_train, dtype=float)
    spread = float(y_train.max() - y_train.min())
    floor = max(spread * 1e-6, 1e-12)
    shift = 0.0 if y_train.min() > 0 else (floor - y_train.min())
    logged = np.log10(y_train + shift + floor)
    centre, scale = float(logged.mean()), float(logged.std()) + 1e-12

    def apply(y):
        return (np.log10(np.asarray(y, dtype=float) + shift + floor) - centre) / scale

    return apply


def part3_heldout(n_uniform=200):
    """Score each fitted GP on points it has never seen."""
    rows = []
    for condition, geometry, instance in itertools.product(CONDITIONS, GEOMETRIES,
                                                           range(N_INSTANCES)):
        objective, meta = problems.make_ellipsoid(DIMENSION, condition,
                                                  instance=instance, rotation=geometry)
        x_train, y_train, rng = design(objective, instance)
        to_target = training_transform(y_train)
        targets = to_target(y_train)

        test_rng = np.random.default_rng(40000 + instance)
        held_out = {"uniform": test_rng.uniform(-BOX, BOX, size=(n_uniform, DIMENSION))}

        for metric in bo.METRICS:
            model = bo.fit_gp(x_train, targets, metric, rng, n_restarts=N_RESTARTS,
                              input_scale=2 * BOX)
            noise = float(np.exp(2 * model["theta"][bo._n_parameters(DIMENSION, metric) - 1]))

            for name, points in held_out.items():
                if len(points) == 0:
                    continue
                truth = to_target([objective(p) for p in points])
                mean, sd = bo.predict(model, points)
                variance = sd ** 2 + noise
                residual = truth - mean
                rows.append({
                    "condition": condition,
                    "geometry": geometry_label(geometry),
                    "instance": instance,
                    "metric": metric,
                    "held_out_set": name,
                    "n_points": len(points),
                    "rmse": float(np.sqrt(np.mean(residual ** 2))),
                    "log_predictive_density": float(np.mean(
                        -0.5 * np.log(2 * np.pi * variance) - residual ** 2 / (2 * variance))),
                })
    return pd.DataFrame(rows)


def _field(points, hessian, optimum, kind, shift, floor):
    """Evaluate either the objective or the transformed field the GP is fitted to."""
    delta = points - optimum
    raw = 0.5 * np.einsum("ij,jk,ik->i", delta, hessian, delta)
    if kind == "f":
        return raw
    return np.log10(raw + shift + floor)


def _rms_change(base, direction, step, hessian, optimum, kind, shift, floor):
    """Typical size of the field's change over a step of the given size."""
    here = _field(base, hessian, optimum, kind, shift, floor)
    there = _field(base + step * direction, hessian, optimum, kind, shift, floor)
    return float(np.sqrt(np.mean((there - here) ** 2)))


def _step_for_change(target, base, direction, hessian, optimum, kind, shift, floor,
                     low=1e-6, high=100.0, iterations=40):
    """Bisect for the step size whose typical change equals ``target``.

    Returns NaN when the target is unreachable inside the bracket, which keeps
    an unmatched direction visible as missing rather than silently clipped.
    """
    args = (hessian, optimum, kind, shift, floor)
    if _rms_change(base, direction, high, *args) < target:
        return np.nan
    if _rms_change(base, direction, low, *args) > target:
        return np.nan
    for _ in range(iterations):
        mid = np.sqrt(low * high)
        if _rms_change(base, direction, mid, *args) < target:
            low = mid
        else:
            high = mid
    return float(np.sqrt(low * high))


def transform_constants(y_train, floor_scale=1.0):
    """The shift and floor bo._transform_targets would apply to these targets."""
    y_train = np.asarray(y_train, dtype=float)
    spread = float(y_train.max() - y_train.min())
    floor = max(spread * 1e-6, 1e-12) * floor_scale
    shift = 0.0 if y_train.min() > 0 else (floor - y_train.min())
    return shift, floor


def part4_reference_check(n_base=20000, reference_steps=(0.01, 0.05),
                          floor_scales=(1.0, 100.0)):
    """Measure the lengthscale ratio of the field the GP is actually fitted to.

    Part 1 scores the fitted lengthscale ratio against sqrt(condition). That
    prediction is derived for the objective f, but bo.py fits the GP to
    log10(f) standardised. If the transformed field has a different natural
    anisotropy, the reference line in the Part 1 chart is in the wrong place
    and "fitted divided by sqrt(kappa)" is not measuring calibration error.

    Method, model-free throughout. Pick a step along the steep principal
    direction, measure the typical change in the field over that step, then
    bisect for the step along the flat direction producing the SAME typical
    change. The ratio of those two steps is the field's lengthscale ratio.
    Standardisation is ignored on purpose: it rescales both directions
    equally and cancels in a ratio. The log and the shift do not cancel, so
    the constants bo.py would use are reproduced exactly.

    The raw objective is measured alongside as a CONTROL. The method must
    return sqrt(condition) on f, because that is what sqrt(condition) means.
    If the control fails, the measurement is broken and the log-field number
    carries no weight.

    Two reference step sizes and two floor values are recorded per problem, so
    that any dependence on those arbitrary choices is visible in the output
    rather than hidden behind an average.
    """
    rows = []
    for condition, geometry, instance in itertools.product(CONDITIONS, GEOMETRIES,
                                                           range(N_INSTANCES)):
        objective, meta = problems.make_ellipsoid(DIMENSION, condition,
                                                  instance=instance, rotation=geometry)
        hessian, optimum = meta["hessian"], meta["optimum"]
        values, vectors = principal_directions(hessian)
        flat, steep = vectors[:, 0], vectors[:, -1]

        _, y_train, _ = design(objective, instance)
        base = np.random.default_rng(60000 + instance).uniform(
            -BOX, BOX, size=(n_base, DIMENSION))

        for kind in ("f", "log10_f"):
            scales = (1.0,) if kind == "f" else floor_scales
            for floor_scale in scales:
                shift, floor = transform_constants(y_train, floor_scale)
                for step in reference_steps:
                    args = (hessian, optimum, kind, shift, floor)
                    target = _rms_change(base, steep, step, *args)
                    flat_step = _step_for_change(target, base, flat, *args)
                    rows.append({
                        "condition": condition,
                        "geometry": geometry_label(geometry),
                        "instance": instance,
                        "field": kind,
                        "floor_scale": floor_scale,
                        "reference_step": step,
                        "steep_step": step,
                        "flat_step": flat_step,
                        "measured_ratio": flat_step / step,
                        "sqrt_condition": float(np.sqrt(condition)),
                    })
    return pd.DataFrame(rows)


def report(lengths, structure, heldout, reference=None):
    print("\n" + "=" * 78)
    print("PART 1  median anisotropy the kernel recovers  (predicted = sqrt(condition))")
    print("=" * 78)
    table = (lengths.groupby(["condition", "geometry", "metric"])
                    .agg(ratio=("ratio", "median"),
                         predicted=("predicted_ratio", "first"),
                         axis_error=("axis_error_deg", "median"))
                    .reset_index())
    print(table.pivot_table(index=["condition", "predicted", "geometry"],
                            columns="metric", values="ratio")
               .to_string(float_format=lambda v: f"{v:.1f}"))

    print("\nmedian error in the long axis direction, degrees (0 = found the valley)")
    print(table.pivot_table(index=["condition", "geometry"], columns="metric",
                            values="axis_error")
               .to_string(float_format=lambda v: f"{v:.1f}"))

    print("\n" + "=" * 78)
    print("PART 2  what the data says, no model fitted (median, step 0.1)")
    print("=" * 78)
    print(structure.groupby(["condition", "geometry"])
                   .agg(axis_1=("gap_axis_1", "median"),
                        axis_2=("gap_axis_2", "median"),
                        flat_direction=("gap_flat_direction", "median"))
                   .to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print("PART 3  held-out predictive accuracy (median over instances)")
    print("=" * 78)
    for name in ("uniform",):
        subset = heldout[heldout.held_out_set == name]
        print(f"\nRMSE on held-out {name} points, GP target scale (lower is better)")
        print(subset.pivot_table(index=["condition", "geometry"], columns="metric",
                                 values="rmse", aggfunc="median")
                    .to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\nMean log predictive density, held-out {name} points (higher is better)")
        print(subset.pivot_table(index=["condition", "geometry"], columns="metric",
                                 values="log_predictive_density", aggfunc="median")
                    .to_string(float_format=lambda v: f"{v:.3f}"))

    if reference is None:
        return
    print("\n" + "=" * 78)
    print("PART 4  is sqrt(condition) the right reference for the fitted field?")
    print("=" * 78)
    control = reference[reference.field == "f"]
    print("\nCONTROL -- measured ratio on the raw objective (must equal sqrt(condition))")
    print(control.pivot_table(index=["condition", "sqrt_condition"],
                              values="measured_ratio", aggfunc="median")
                 .to_string(float_format=lambda v: f"{v:.3f}"))
    logged = reference[reference.field == "log10_f"]
    print("\nMeasured ratio on log10(f), the field the GP is fitted to")
    print(logged.pivot_table(index=["condition", "sqrt_condition"],
                             columns=["floor_scale", "reference_step"],
                             values="measured_ratio", aggfunc="median")
                .to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    start = time.time()

    lengths = part1_lengthscales()
    lengths.to_csv(RESULTS / "e2c_lengthscales.csv", index=False)

    structure = part2_structure()
    structure.to_csv(RESULTS / "e2c_structure.csv", index=False)

    heldout = part3_heldout()
    heldout.to_csv(RESULTS / "e2c_heldout.csv", index=False)

    reference = part4_reference_check()
    reference.to_csv(RESULTS / "e2c_reference.csv", index=False)

    report(lengths, structure, heldout, reference)
    print(f"\ntotal {time.time() - start:.1f}s -> {RESULTS}")
