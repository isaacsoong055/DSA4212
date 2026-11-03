"""Shared performance measurement and uncertainty quantification.

Following COCO, the primary performance measure is runtime in objective
evaluations, recorded against several target precisions rather than one, and
aggregated as an empirical cumulative distribution over (problem, target)
pairs.  Runtime in evaluations is independent of platform, language and
coding style, which is what makes numbers comparable across methods and
machines.

Two conventions matter for this project:

* A run that never reaches a target contributes a censored observation, not a
  missing one.  Reporting only the median of the runs that happened to succeed
  flatters a method that fails often, so success rate is always reported
  alongside any runtime summary.
* Comparisons use PAIRED differences over a shared instance/seed list, and the
  uncertainty reported is the uncertainty in the difference, not in each
  method's mean separately.
"""

import numpy as np
import pandas as pd

DEFAULT_TARGETS = np.logspace(2, -8, 21)  # 1e2 down to 1e-8


def evaluations_to_target(history, population_size, target):
    """First evaluation count at which best-so-far fell to or below ``target``.

    ``history`` is the best-so-far value after each generation, so generation g
    corresponds to (g + 1) * population_size evaluations.  Returns NaN when the
    target was never reached (a censored observation).
    """
    history = np.asarray(history, dtype=float)
    hit = np.flatnonzero(history <= target)
    if hit.size == 0:
        return np.nan
    return float((hit[0] + 1) * population_size)


def runtime_table(records, targets=None):
    """Long-format table of one row per (run, target).

    ``records`` must contain 'history', 'population_size', 'evaluations' and
    whatever grouping columns the experiment varies.
    """
    targets = DEFAULT_TARGETS if targets is None else np.asarray(targets)
    rows = []
    for rec in records:
        for t in targets:
            rt = evaluations_to_target(rec["history"], rec["population_size"], t)
            row = {k: v for k, v in rec.items() if k != "history"}
            row["target"] = float(t)
            row["runtime"] = rt
            row["reached"] = bool(np.isfinite(rt))
            rows.append(row)
    return pd.DataFrame(rows)


def ecdf(runtimes, budget, grid=None):
    """Fraction of (problem, target) pairs solved within x evaluations.

    Censored runs count as unsolved at every x, which is the COCO convention and
    the reason the curve plateaus below 1 for a method that does not always
    converge.
    """
    runtimes = np.asarray(runtimes, dtype=float)
    grid = np.logspace(0, np.log10(budget), 200) if grid is None else np.asarray(grid)
    solved = np.array([np.mean(np.nan_to_num(runtimes, nan=np.inf) <= x) for x in grid])
    return grid, solved


def bootstrap_paired_difference(a, b, n_boot=10000, seed=0, statistic=np.median):
    """Bootstrap CI for statistic(a) - statistic(b) over paired observations.

    ``a`` and ``b`` must be aligned: entry i of each is the same instance and
    seed, run with the two methods being compared.  Non-finite entries are
    replaced by the shared penalty value so censored runs still contribute,
    which is why the caller passes already-penalised arrays for runtime
    comparisons.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    rng = np.random.default_rng(seed)
    n = len(a)
    observed = statistic(a) - statistic(b)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = statistic(a[idx]) - statistic(b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))}


def penalised_runtime(runtime, budget, penalty_factor=10.0):
    """Replace a censored runtime with budget * penalty_factor.

    This is a stated convention, not a measurement.  Report the penalty factor
    and check that conclusions do not depend on it.
    """
    runtime = np.asarray(runtime, dtype=float)
    return np.where(np.isfinite(runtime), runtime, budget * penalty_factor)


def summarise(df, by, value="best_value", target=1e-8):
    """Success rate plus median outcome per group, in one table."""
    out = (df.assign(success=df[value] <= target)
             .groupby(by)
             .agg(runs=(value, "size"),
                  success_rate=("success", "mean"),
                  median_value=(value, "median"),
                  median_evaluations=("evaluations", "median"))
             .reset_index())
    out["success_rate"] = (100 * out["success_rate"]).round(1)
    return out
