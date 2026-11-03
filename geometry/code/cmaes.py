"""CMA-ES with a switchable covariance model.

This is the implementation written for the geometry notebook, moved into a
module so experiment scripts and the notebook share one copy.  The update
equations are unchanged from that notebook (they follow Hansen's tutorial:
rank-one plus rank-mu covariance updates, cumulative step-size adaptation,
and the h_sigma stall guard).

Two changes relative to the notebook version:

1. ``initial_mean`` is now required.  In the notebook it defaulted to the
   origin, which was also the location of every optimum, so the search began
   at the solution and the run reduced to shrinking sigma.  Callers must now
   state where the search starts.
2. ``record_history`` lets callers skip storing per-generation traces when
   only the final outcome is needed, which keeps large sweeps cheap.

The ``variant`` argument is the ablation: "full" keeps the whole covariance
matrix, "diagonal" zeroes the off-diagonal entries after every update.  The
diagonal variant can still stretch each axis independently but cannot
represent a rotation, which makes it a stand-in for any search distribution
whose geometry is constrained to the coordinate axes.
"""

import numpy as np

VARIANTS = ("full", "diagonal")


def default_population_size(dimension):
    return 4 + int(3 * np.log(dimension))


def cma_es(objective, dimension, initial_mean, seed=0, variant="full",
           max_evaluations=1500, initial_sigma=2.0, target=1e-8,
           population_size=None, record_history=True):
    """Minimise ``objective`` with full- or diagonal-covariance CMA-ES.

    Returns a dict with the best value found, the number of objective
    evaluations consumed, the final state of the search distribution, and
    (optionally) the best-so-far value after each generation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")

    rng = np.random.default_rng(seed)
    n = int(dimension)
    lam = default_population_size(n) if population_size is None else int(population_size)
    mu = lam // 2

    # Better ranks receive larger positive weights.
    raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = raw_weights / raw_weights.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)

    # Standard CMA-ES learning rates and damping for the evolution paths.
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    expected_norm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    mean = np.array(initial_mean, dtype=float)
    if mean.shape != (n,):
        raise ValueError(f"initial_mean must have shape ({n},)")
    sigma = float(initial_sigma)
    covariance = np.eye(n)
    path_sigma = np.zeros(n)
    path_c = np.zeros(n)
    best_value = np.inf
    best_point = mean.copy()
    history = []
    evaluations = 0
    generation = 0

    while evaluations + lam <= max_evaluations and best_value > target:
        generation += 1

        # Factor the covariance to draw candidates with the current shape and
        # orientation.  The diagonal variant never has off-diagonal entries to
        # factor, so its cloud stays aligned with the axes.
        if variant == "diagonal":
            variances = np.maximum(np.diag(covariance), 1e-20)
            transform = np.diag(np.sqrt(variances))
            inverse_sqrt = np.diag(1 / np.sqrt(variances))
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            inverse_sqrt = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T

        standard_samples = rng.normal(size=(lam, n))
        steps = standard_samples @ transform.T
        candidates = mean + sigma * steps
        values = np.array([objective(point) for point in candidates])
        evaluations += lam
        order = np.argsort(values)
        if values[order[0]] < best_value:
            best_value = float(values[order[0]])
            best_point = candidates[order[0]].copy()
        if record_history:
            history.append(best_value)

        selected_steps = steps[order[:mu]]
        old_mean = mean.copy()
        weighted_step = np.sum(weights[:, None] * selected_steps, axis=0)
        mean = old_mean + sigma * weighted_step

        # p_sigma measures whether recent steps line up; it drives global sigma.
        whitened_step = inverse_sqrt @ weighted_step
        path_sigma = ((1 - c_sigma) * path_sigma
                      + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * whitened_step)
        correction = np.sqrt(1 - (1 - c_sigma) ** (2 * generation))
        h_sigma = (np.linalg.norm(path_sigma) / max(correction, 1e-20) / expected_norm
                   < 1.4 + 2 / (n + 1))

        # p_c accumulates movement in the original coordinates for rank-one learning.
        path_c = ((1 - c_c) * path_c
                  + float(h_sigma) * np.sqrt(c_c * (2 - c_c) * mu_eff) * weighted_step)

        # Rank-mu learning adds spread along the successful selected steps.
        rank_mu = sum(w * np.outer(step, step)
                      for w, step in zip(weights, selected_steps))
        covariance = ((1 - c1 - c_mu) * covariance
                      + c1 * (np.outer(path_c, path_c)
                              + (1 - float(h_sigma)) * c_c * (2 - c_c) * covariance)
                      + c_mu * rank_mu)
        if variant == "diagonal":
            covariance = np.diag(np.maximum(np.diag(covariance), 1e-20))

        # CSA changes the overall cloud size; the covariance sets its relative shape.
        sigma *= np.exp((c_sigma / d_sigma)
                        * (np.linalg.norm(path_sigma) / expected_norm - 1))

    return {"best_value": best_value,
            "best_point": best_point,
            "evaluations": evaluations,
            "generations": generation,
            "mean": mean,
            "sigma": sigma,
            "covariance": covariance,
            "population_size": lam,
            "history": np.asarray(history) if record_history else None}
