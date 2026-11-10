"""E2d: is bo.fit_gp finding the marginal-likelihood optimum, or hitting its bounds?

E2c found that fitted lengthscale ratios overshoot the sqrt(condition)
prediction, and three explanations fit: (1) an implementation fault, with
parameters pinned at the bounds declared in bo.fit_gp; (2) ordinary
small-sample overfitting; (3) a wrong reference value. This script tests
explanation (1) only. It changes nothing in bo.py: it re-runs the exact
restart procedure of bo.fit_gp (same bounds, same start distribution, same
L-BFGS-B settings) on the exact E2c designs, but keeps EVERY restart instead
of only the best, so we can see

  * whether the best fit has any parameter at a bound, and which kind
    (log-lengthscale +-4, off-diagonal +-50, log-noise -6 or 0);
  * how much the best fit improves when restarts go from 3 (BO default)
    and 6 (E2c) to 20;
  * whether restarts that are nearly as good (within 1 nat of the best)
    disagree about the lengthscale ratio, i.e. whether the likelihood is a
    flat ridge in the ratio direction.

Part 2 refits with only the log-lengthscale bound widened (results/e2d_widened.csv).
Part 3 checks the MAP fit that replaced the original one (results/e2d_map.csv).

Writes results/e2d_restarts.csv (one row per restart).
"""

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bo
import problems
from run_e2c_lengthscale_diagnostic import (CONDITIONS, GEOMETRIES, N_INSTANCES,
                                            design, geometry_label,
                                            kernel_lengthscales, axis_angle,
                                            angle_difference, principal_directions)

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
N_RESTARTS = 20
BOUND_TOL = 1e-3


def bounds_for(dimension, metric):
    """Identical to the bounds built inside bo.fit_gp."""
    n_par = bo._n_parameters(dimension, metric)
    n_scale = bo._n_scales(dimension, metric)
    return ([(-4.0, 4.0)] * n_scale
            + [(-50.0, 50.0)] * (n_par - 1 - n_scale)
            + [(-6.0, 0.0)]), n_par, n_scale


def one_restart(x, y, metric, rng):
    """One restart of bo.fit_gp, unchanged: same start draw, same optimiser."""
    dimension = x.shape[1]
    bounds, n_par, n_scale = bounds_for(dimension, metric)
    start = np.zeros(n_par)
    start[:n_scale] = rng.normal(0.0, 0.5, n_scale)
    if metric == "full":
        start[dimension:n_par - 1] = rng.normal(0.0, 0.2, n_par - 1 - dimension)
    start[-1] = -3.0
    found = minimize(bo._negative_log_marginal_likelihood, start,
                     args=(x, y, dimension, metric), method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": 200})
    return found, bounds, n_scale


def at_bound_flags(theta, bounds, n_scale):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    hit = (np.abs(theta - lo) < BOUND_TOL) | (np.abs(theta - hi) < BOUND_TOL)
    n_off = len(theta) - 1 - n_scale
    return {"scale_at_bound": bool(hit[:n_scale].any()),
            "offdiag_at_bound": bool(hit[n_scale:n_scale + n_off].any()) if n_off else False,
            "noise_at_lower": bool(abs(theta[-1] - lo[-1]) < BOUND_TOL),
            "noise_at_upper": bool(abs(theta[-1] - hi[-1]) < BOUND_TOL)}


def main():
    rows = []
    for condition, geometry, instance in itertools.product(CONDITIONS, GEOMETRIES,
                                                           range(N_INSTANCES)):
        objective, meta = problems.make_ellipsoid(2, condition, instance=instance,
                                                  rotation=geometry)
        _, true_vectors = principal_directions(meta["hessian"])
        true_angle = axis_angle(true_vectors[:, 0])
        x, y, _ = design(objective, instance)
        targets = bo._transform_targets(y)
        for metric in bo.METRICS:
            # Deterministic seed per cell (Python's hash() is salted per process).
            rng = np.random.default_rng([condition, GEOMETRIES.index(geometry),
                                         instance, bo.METRICS.index(metric)])
            for restart in range(N_RESTARTS):
                found, bounds, n_scale = one_restart(x, targets, metric, rng)
                precision, _ = bo._precision_matrix(found.x, 2, metric)
                lengths, vectors = kernel_lengthscales(precision)
                rows.append({
                    "condition": condition, "geometry": geometry_label(geometry),
                    "instance": instance, "metric": metric, "restart": restart,
                    "nlml": float(found.fun), "converged": bool(found.success),
                    "ratio": lengths[0] / lengths[-1],
                    "predicted_ratio": np.sqrt(condition),
                    "axis_error_deg": (np.nan if condition == 1 else
                                       angle_difference(axis_angle(vectors[:, 0]), true_angle)),
                    "log_noise": float(found.x[-1]),
                    "theta": " ".join(f"{t:.4g}" for t in found.x),
                    **at_bound_flags(found.x, bounds, n_scale),
                })
        print(f"done condition={condition} geometry={geometry} instance={instance}", flush=True)
    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "e2d_restarts.csv", index=False)
    print(f"wrote {len(out)} rows")
    return out


def widened_refit(restarts, wide=10.0):
    """Part 2: does the likelihood want to go past the log-lengthscale bound?

    Start from the best of the 20 restarts and continue L-BFGS-B with only the
    log-lengthscale (log-diagonal) bound widened from +-4 to +-wide. A positive
    nlml_gain means the original bound was cutting off a better fit.
    """
    keys = ["condition", "geometry", "instance", "metric"]
    best = restarts.loc[restarts.groupby(keys).nlml.idxmin()].set_index(keys)
    rows = []
    for condition, geometry, instance in itertools.product((100, 1000, 10000),
                                                           GEOMETRIES, range(N_INSTANCES)):
        objective, _ = problems.make_ellipsoid(2, condition, instance=instance,
                                               rotation=geometry)
        x, y, _ = design(objective, instance)
        targets = bo._transform_targets(y)
        for metric in bo.METRICS:
            bounds, n_par, n_scale = bounds_for(2, metric)
            bounds = [(-wide, wide)] * n_scale + bounds[n_scale:]
            old = best.loc[(condition, geometry_label(geometry), instance, metric)]
            theta0 = np.array([float(t) for t in old.theta.split()])
            found = minimize(bo._negative_log_marginal_likelihood, theta0,
                             args=(x, targets, 2, metric), method="L-BFGS-B",
                             bounds=bounds, options={"maxiter": 500})
            lengths, _ = kernel_lengthscales(bo._precision_matrix(found.x, 2, metric)[0])
            rows.append({"condition": condition, "geometry": geometry_label(geometry),
                         "instance": instance, "metric": metric,
                         "nlml_gain": float(old.nlml - found.fun),
                         "ratio_old": old.ratio, "ratio_new": lengths[0] / lengths[-1],
                         "ell_long_new": lengths[0], "ell_short_new": lengths[-1],
                         "at_wide_bound": bool((np.abs(np.abs(found.x[:n_scale]) - wide)
                                                < BOUND_TOL).any())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "e2d_widened.csv", index=False)
    print(f"wrote {len(out)} rows")
    return out


def map_check():
    """Part 3: the MAP fit now in bo.fit_gp -- restart agreement and prior sensitivity.

    Same designs as above. At the default prior width (sigma = sqrt 3) each fit
    uses 20 restarts, so best-of-3 and best-of-10 can be compared with best-of-20
    on the MAP objective. The prior width is then halved and doubled (10
    restarts each) to show which fitted quantities the prior, rather than the
    data, determines. Writes results/e2d_map.csv.
    """
    widths = {"half": bo.PRIOR_SCALE / 2, "default": bo.PRIOR_SCALE,
              "double": bo.PRIOR_SCALE * 2}
    rows = []
    for condition, geometry, instance in itertools.product(CONDITIONS, GEOMETRIES,
                                                           range(N_INSTANCES)):
        objective, meta = problems.make_ellipsoid(2, condition, instance=instance,
                                                  rotation=geometry)
        _, true_vectors = principal_directions(meta["hessian"])
        true_angle = axis_angle(true_vectors[:, 0])
        x, y, _ = design(objective, instance)
        targets = bo._transform_targets(y)
        for metric in bo.METRICS:
            for label, width in widths.items():
                n_restarts = 20 if label == "default" else 10
                rng = np.random.default_rng([condition, GEOMETRIES.index(geometry), instance,
                                             bo.METRICS.index(metric), n_restarts])
                model = bo.fit_gp(x, targets, metric, rng, n_restarts=n_restarts,
                                  input_scale=2 * 5.0, prior_scale=width)
                lengths, vectors = kernel_lengthscales(model["precision"])
                values = np.array([v for v, _ in model["restarts"]])
                best_k = {k: values[:k].min() for k in (3, 10)}
                rows.append({
                    "condition": condition, "geometry": geometry_label(geometry),
                    "instance": instance, "metric": metric,
                    "prior_width": label, "n_restarts": n_restarts,
                    "objective": model["objective"], "nlml": model["nlml"],
                    "gap_best3": best_k[3] - values.min(),
                    "gap_best10": best_k[10] - values.min(),
                    "ell_long": lengths[0], "ell_short": lengths[-1],
                    "ratio": lengths[0] / lengths[-1],
                    "predicted_ratio": np.sqrt(condition),
                    "axis_error_deg": (np.nan if condition == 1 else
                                       angle_difference(axis_angle(vectors[:, 0]), true_angle)),
                    "scale_at_bound": bool((np.abs(np.abs(model["theta"][:2])
                                                   - bo.MAP_LOG_SCALE_BOUND) < BOUND_TOL).any()),
                })
        print(f"map done condition={condition} geometry={geometry} instance={instance}",
              flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "e2d_map.csv", index=False)
    print(f"wrote {len(out)} rows")
    return out


if __name__ == "__main__":
    # Parts 1-2 diagnose the ORIGINAL maximum-likelihood fit (its bounds are
    # copied above, so they do not depend on the current bo.fit_gp); part 3
    # checks the MAP fit that replaced it.
    if "--part2-only" in sys.argv:
        widened_refit(pd.read_csv(RESULTS / "e2d_restarts.csv"))
    elif "--part3-only" in sys.argv:
        map_check()
    else:
        widened_refit(main())
        map_check()
