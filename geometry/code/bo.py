"""Gaussian-process Bayesian optimisation with a switchable kernel metric.

The point of this module is the ``metric`` argument, which is the BO-side
counterpart of the diagonal/full ablation in ``cmaes.py``.

The squared distance inside the kernel is  r^2 = (x - x')^T M (x - x').

    metric="iso"   M = I / lengthscale^2: one lengthscale shared by every
                   direction.  The simplest baseline; it can neither stretch
                   nor rotate.  Not in METRICS (see below), so the E2 and E2c
                   scripts, which loop over METRICS, are unchanged.

    metric="ard"   M is diagonal.  This is the standard kernel in the BO
                   literature: Frazier's tutorial writes the squared
                   exponential as exp(-sum_i alpha_i (x_i - x'_i)^2), which
                   is a diagonal metric, and TuRBO's trust region inherits
                   the same restriction because it is an axis-aligned
                   hyperrectangle whose side lengths are set per coordinate
                   from the ARD lengthscales.  An ARD kernel can stretch the
                   axes but cannot represent a rotation.

    metric="full"  M = L L^T with L lower triangular, so the kernel can learn
                   a correlated (rotated) metric.  This is the surrogate-side
                   analogue of a full covariance matrix in CMA-ES.

Everything else is held fixed between the two so the kernel metric is the only
thing that differs: same acquisition function, same acquisition optimiser,
same initial design for a given seed, same number of marginal-likelihood
restarts, same noise floor.

Hyperparameters are fitted by maximum a posteriori (MAP): the log marginal
likelihood plus a LogNormal prior on the lengthscales, maximised with L-BFGS-B
from several starts.  The prior is the dimension-scaled default used by
BoTorch (Hvarfner, Hellsten and Nardi, "Vanilla Bayesian Optimization Performs
Great in High Dimensions", ICML 2024): log ell ~ N(sqrt(2) + log(d)/2, sqrt(3))
for inputs scaled to the unit cube.  Our inputs live in a box of width W, so a
lengthscale ell in our units is W * ell_unit and the prior's location shifts
by log(W).

Why a prior rather than plain maximum likelihood (diagnosed in E2d,
run_e2d_fit_diagnostic.py): along a flat direction the data carry no
information beyond the box width, so the likelihood optimum is ell -> infinity.
The original fit stopped at a hidden hard bound (ell <= e^4 ~ 55), which then
set every fitted lengthscale ratio at high condition numbers.  A prior replaces
that hidden wall with a stated, cited, smooth default which the data override
wherever they are informative.

The prior is placed on the lengthscales along the eigen-directions of the
kernel metric M, ell_j = lambda_j(M)^(-1/2), not on the raw parameters.  For
ARD these are exactly the per-axis lengthscales, so this is BoTorch's prior.
For the full metric this keeps the prior rotation-invariant: a prior on the
diagonal of the Cholesky factor would make axis-aligned metrics cheaper and
reintroduce the very restriction this project tests.  As in GPyTorch, the
prior is evaluated on the lengthscales directly, with no Jacobian term.

``prior=None`` restores the original maximum-likelihood fit with its original
bounds, so earlier results remain reproducible.  Targets are log-transformed and
standardised before the GP sees them, because an ill-conditioned quadratic
spans many orders of magnitude and a stationary GP on raw values is dominated
by the few largest observations.
"""

from math import log, sqrt

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import erfcx
from scipy.stats import norm

METRICS = ("ard", "full")
# Every metric fit_gp accepts.  METRICS is kept as the original pair because
# the E2/E2c scripts loop over it and share one random generator across the
# loop; adding "iso" there would change their fits and their saved CSVs.
ALL_METRICS = ("iso",) + METRICS


def _n_scales(dimension, metric):
    """Number of log-lengthscale parameters: one for iso, one per axis otherwise."""
    return 1 if metric == "iso" else dimension


def _n_parameters(dimension, metric):
    """Length scales (+ off-diagonals for the full metric) plus one noise term."""
    if metric == "iso":
        return 2
    extra = 0 if metric == "ard" else dimension * (dimension - 1) // 2
    return dimension + extra + 1


def _precision_matrix(theta, dimension, metric):
    """Map unconstrained parameters to the kernel's precision matrix M."""
    if metric == "iso":
        return np.exp(2 * theta[0]) * np.eye(dimension), 1
    if metric == "ard":
        return np.diag(np.exp(2 * theta[:dimension])), dimension
    lower = np.zeros((dimension, dimension))
    lower[np.diag_indices(dimension)] = np.exp(theta[:dimension])
    off = np.tril_indices(dimension, -1)
    lower[off] = theta[dimension:dimension + len(off[0])]
    return lower @ lower.T, dimension + len(off[0])


def _squared_distance(a, b, precision):
    delta = a[:, None, :] - b[None, :, :]
    return np.einsum("ijk,kl,ijl->ij", delta, precision, delta)


KERNELS = ("se", "matern52")
_SQRT5 = sqrt(5.0)


def _kernel(r2, kernel="se"):
    """Stationary kernel as a function of the squared metric distance r^2 = d^T M d.

    "se" is the squared exponential used throughout; "matern52" is the Matern-5/2
    kernel, BoTorch's default, which is rougher (twice differentiable rather than
    infinitely). Both see the same metric M, so ARD/full is the same choice in each.
    """
    if kernel == "se":
        return np.exp(-0.5 * r2)
    r = np.sqrt(np.maximum(r2, 0.0))
    return (1.0 + _SQRT5 * r + 5.0 * r2 / 3.0) * np.exp(-_SQRT5 * r)


def _kernel_dr2(r2, kernel="se"):
    """d k / d(r^2), used by the exact likelihood gradient."""
    if kernel == "se":
        return -0.5 * np.exp(-0.5 * r2)
    r = np.sqrt(np.maximum(r2, 0.0))
    return -(5.0 / 6.0) * (1.0 + _SQRT5 * r) * np.exp(-_SQRT5 * r)


def _negative_log_marginal_likelihood(theta, x, y, dimension, metric, kernel="se"):
    precision, cut = _precision_matrix(theta, dimension, metric)
    noise = np.exp(2 * theta[cut])
    gram = _kernel(_squared_distance(x, x, precision), kernel)
    gram += (noise + 1e-8) * np.eye(len(x))
    try:
        factor = cho_factor(gram, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = cho_solve(factor, y)
    return 0.5 * y @ alpha + np.sum(np.log(np.diag(factor[0])))


# Lengthscale prior (see module docstring).
PRIOR_LOC_UNIT = sqrt(2.0)       # location for unit-cube inputs, before the log(d)/2 term
PRIOR_SCALE = sqrt(3.0)          # standard deviation of log(ell)
LEGACY_LOG_SCALE_BOUND = 4.0     # original MLE bound on log-diagonal parameters (ell <= e^4)
MAP_LOG_SCALE_BOUND = 10.0       # numerical guard only under MAP; the prior does the work
DEFAULT_RESTARTS = 10            # was 3; E2d found best-of-3 misses the optimum in 7% of fits


def lengthscale_prior_loc(dimension, input_scale):
    """Prior location for log(ell) in the problem's own units (box width input_scale)."""
    return PRIOR_LOC_UNIT + 0.5 * log(dimension) + log(input_scale)


def eigen_log_lengthscales(precision, metric):
    """log(ell_j) along the metric's eigen-directions: ell_j = lambda_j^(-1/2)."""
    if metric == "iso":
        return np.array([-0.5 * np.log(max(precision[0, 0], 1e-300))])
    values = np.linalg.eigvalsh(precision)
    return -0.5 * np.log(np.maximum(values, 1e-300))


def _negative_log_posterior(theta, x, y, dimension, metric, prior_loc, prior_scale, kernel="se"):
    """Negative log marginal likelihood plus the LogNormal lengthscale penalty."""
    nlml = _negative_log_marginal_likelihood(theta, x, y, dimension, metric, kernel)
    if nlml >= 1e10:
        return nlml
    precision, _ = _precision_matrix(theta, dimension, metric)
    log_ell = eigen_log_lengthscales(precision, metric)
    return nlml + float(np.sum((log_ell - prior_loc) ** 2)) / (2.0 * prior_scale ** 2)


def _metric_gradient_to_theta(grad_m, theta, dimension, metric):
    """Chain rule from dObjective/dM (symmetric d x d) to the metric's parameters."""
    if metric == "iso":
        return np.array([np.trace(grad_m) * 2 * np.exp(2 * theta[0])])
    if metric == "ard":
        return np.diag(grad_m) * 2 * np.exp(2 * theta[:dimension])
    lower = np.zeros((dimension, dimension))
    lower[np.diag_indices(dimension)] = np.exp(theta[:dimension])
    off = np.tril_indices(dimension, -1)
    lower[off] = theta[dimension:dimension + len(off[0])]
    d_lower = 2 * grad_m @ lower                     # d tr(G L L^T) / dL
    return np.concatenate([np.diag(d_lower) * np.exp(theta[:dimension]), d_lower[off]])


def _nlml_and_gradient(theta, x, y, dimension, metric, kernel="se"):
    """Same value as _negative_log_marginal_likelihood, plus its exact gradient.

    dNLML/dtheta_k = 1/2 tr((K^-1 - alpha alpha^T) dK/dtheta_k)  (Rasmussen and
    Williams 2006, eq. 5.9).  The kernel depends on theta only through the
    metric M, so the gradient is first formed with respect to M and then
    chained to the parameters.  Replaces finite differences, which cost
    (number of parameters + 1) likelihood evaluations per gradient.
    """
    precision, cut = _precision_matrix(theta, dimension, metric)
    noise = np.exp(2 * theta[cut])
    delta = x[:, None, :] - x[None, :, :]
    r2 = np.einsum("ijk,kl,ijl->ij", delta, precision, delta)
    gram = _kernel(r2, kernel) + (noise + 1e-8) * np.eye(len(x))
    try:
        factor = cho_factor(gram, lower=True)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(theta)
    alpha = cho_solve(factor, y)
    value = 0.5 * y @ alpha + np.sum(np.log(np.diag(factor[0])))
    w = cho_solve(factor, np.eye(len(x))) - np.outer(alpha, alpha)
    # dK_ij/dM = k'(r_ij^2) d_ij d_ij^T, so dNLML/dM = 1/2 sum_ij W_ij k'(r_ij^2) d_ij d_ij^T.
    weighted = w * _kernel_dr2(r2, kernel)
    grad_m = 0.5 * np.einsum("ij,ijk,ijl->kl", weighted, delta, delta)
    grad = np.zeros_like(theta)
    grad[:cut] = _metric_gradient_to_theta(grad_m, theta, dimension, metric)
    grad[cut] = np.trace(w) * noise
    return float(value), grad


def _neg_log_posterior_and_gradient(theta, x, y, dimension, metric, prior_loc, prior_scale,
                                    kernel="se"):
    """MAP objective of _negative_log_posterior with its exact gradient.

    Prior term P = sum_j (log ell_j - mu)^2 / (2 s^2) with log ell_j = -1/2 log lambda_j.
    For distinct eigenvalues d lambda_j = v_j^T dM v_j, so dP/dM = sum_j
    (dP/d lambda_j) v_j v_j^T, chained to theta like the likelihood term.
    """
    value, grad = _nlml_and_gradient(theta, x, y, dimension, metric, kernel)
    if value >= 1e10:
        return value, grad
    precision, cut = _precision_matrix(theta, dimension, metric)
    if metric == "iso":
        log_ell = -theta[0]
        return (value + (log_ell - prior_loc) ** 2 / (2 * prior_scale ** 2),
                grad + np.r_[-(log_ell - prior_loc) / prior_scale ** 2,
                             np.zeros(len(theta) - 1)])
    lam, vec = np.linalg.eigh(precision)
    lam = np.maximum(lam, 1e-300)
    log_ell = -0.5 * np.log(lam)
    resid = log_ell - prior_loc
    d_lam = (resid / prior_scale ** 2) * (-0.5 / lam)
    grad_m = (vec * d_lam) @ vec.T
    grad[:cut] += _metric_gradient_to_theta(grad_m, theta, dimension, metric)
    return value + float(np.sum(resid ** 2)) / (2 * prior_scale ** 2), grad


def fit_gp(x, y, metric, rng, n_restarts=DEFAULT_RESTARTS, prior="lognormal",
           input_scale=None, prior_scale=PRIOR_SCALE, noise_variance=None, kernel="se"):
    """Fit kernel hyperparameters by MAP (default) or maximum likelihood (prior=None).

    input_scale is the width of the search box in the problem's units; it sets
    the prior's location.  If omitted it is taken as the widest coordinate range
    of the design, which is close to the box width for a space-filling design;
    callers that know the box should pass it.

    noise_variance=None fits the noise (the default).  A number fixes it, on the
    standardised scale, instead (Stage B variant B2: the objective is noise-free,
    and Stage A found the fitted noise absorbing the best observations).
    """
    dimension = x.shape[1]
    n_par = _n_parameters(dimension, metric)
    n_scale = _n_scales(dimension, metric)
    if prior is None:
        scale_bound = LEGACY_LOG_SCALE_BOUND
        objective, extra = _nlml_and_gradient, (x, y, dimension, metric, kernel)
    elif prior == "lognormal":
        scale_bound = MAP_LOG_SCALE_BOUND
        if input_scale is None:
            input_scale = float(np.max(x.max(axis=0) - x.min(axis=0)))
        prior_loc = lengthscale_prior_loc(dimension, input_scale)
        objective = _neg_log_posterior_and_gradient
        extra = (x, y, dimension, metric, prior_loc, prior_scale, kernel)
    else:
        raise ValueError("prior must be None or 'lognormal'")
    bounds = ([(-scale_bound, scale_bound)] * n_scale
              + [(-50.0, 50.0)] * (n_par - 1 - n_scale)
              + [(-6.0, 0.0)])
    if noise_variance is not None:
        fixed = 0.5 * log(noise_variance)     # theta_noise = log(noise SD)
        bounds[-1] = (fixed, fixed)

    best, restarts = None, []
    for restart in range(n_restarts):
        start = np.zeros(n_par)
        if prior is not None and restart == 0:
            # BoTorch initialises at the prior mode; theta = -log(ell) on the diagonal.
            start[:n_scale] = -(prior_loc - prior_scale ** 2)
        else:
            start[:n_scale] = rng.normal(0.0, 0.5, n_scale)
            if metric == "full":
                start[dimension:n_par - 1] = rng.normal(0.0, 0.2, n_par - 1 - dimension)
        start[-1] = -3.0 if noise_variance is None else bounds[-1][0]
        found = minimize(objective, start, args=extra, method="L-BFGS-B", jac=True,
                         bounds=bounds, options={"maxiter": 200})
        restarts.append((float(found.fun), found.x.copy()))
        if best is None or found.fun < best.fun:
            best = found

    precision, cut = _precision_matrix(best.x, dimension, metric)
    noise = np.exp(2 * best.x[cut])
    gram = _kernel(_squared_distance(x, x, precision), kernel)
    gram += (noise + 1e-8) * np.eye(len(x))
    factor = cho_factor(gram, lower=True)
    nlml = _negative_log_marginal_likelihood(best.x, x, y, dimension, metric, kernel)
    return {"precision": precision, "factor": factor,
            "alpha": cho_solve(factor, y), "x": x,
            "theta": best.x, "nlml": float(nlml), "objective": float(best.fun),
            "restarts": restarts, "kernel": kernel}


def predict(model, query):
    """Posterior mean and standard deviation at ``query`` (unit prior variance)."""
    cross = _kernel(_squared_distance(query, model["x"], model["precision"]),
                    model.get("kernel", "se"))
    mean = cross @ model["alpha"]
    reduction = np.einsum("ij,ij->i", cross, cho_solve(model["factor"], cross.T).T)
    return mean, np.sqrt(np.maximum(1.0 - reduction, 1e-12))


def expected_improvement(model, query, incumbent):
    """EI for minimisation, on the GP's (transformed) output scale."""
    mean, sd = predict(model, query)
    z = (incumbent - mean) / sd
    return (incumbent - mean) * norm.cdf(z) + sd * norm.pdf(z)


_C1 = 0.5 * log(2 * np.pi)
_C2 = 0.5 * log(np.pi / 2)


def _log1mexp(a):
    """log(1 - exp(a)) for a < 0, accurate at both ends (Maechler 2012)."""
    a = np.asarray(a, float)
    return np.where(a > -log(2), np.log(-np.expm1(a)), np.log1p(-np.exp(a)))


def _log_h(z):
    """log(phi(z) + z Phi(z)), the EI kernel, without underflow.

    Ament et al., "Unexpected Improvements to Expected Improvement for Bayesian
    Optimization", NeurIPS 2023, eq. 9.  Plain EI underflows to 0 once z is
    below about -38, which leaves the acquisition step with no signal; this
    stays finite and smooth for any z.
    """
    z = np.asarray(z, float)
    out = np.empty_like(z)
    upper = z > -1
    zu = z[upper]
    out[upper] = np.log(norm.pdf(zu) + zu * norm.cdf(zu))
    zl = z[~upper]
    tiny = zl < -1 / np.sqrt(np.finfo(float).eps)
    mid = -zl[~tiny]
    lower = np.empty_like(zl)
    lower[~tiny] = (-0.5 * mid ** 2 - _C1
                    + _log1mexp(np.log(erfcx(mid / np.sqrt(2)) * mid) + _C2))
    lower[tiny] = -0.5 * zl[tiny] ** 2 - _C1 - 2 * np.log(-zl[tiny])
    out[~upper] = lower
    return out


def log_expected_improvement(model, query, incumbent):
    """log EI for minimisation (Stage B variant B4); same maximiser as EI."""
    mean, sd = predict(model, query)
    return _log_h((incumbent - mean) / sd) + np.log(sd)


def transform_floor(y, target_floor=None):
    """The floor added inside the log: relative by default, or a fixed value."""
    y = np.asarray(y, dtype=float)
    if target_floor is not None:
        return float(target_floor)
    return max(float(y.max() - y.min()) * 1e-6, 1e-12)


def _transform_targets(y, target_floor=None):
    """Log-scale then standardise the observations the GP is fitted to.

    An ill-conditioned quadratic spans many orders of magnitude, and a
    stationary GP fitted to raw values is dominated by the handful of largest
    observations.  Taking logs first makes the residual structure visible.
    The shift keeps the logarithm finite without turning the incumbent into an
    outlier: it is tiny relative to the observed spread rather than a fixed
    floor like 1e-12.

    target_floor fixes the floor instead (Stage B variant B5). Stage B found BO
    with fixed noise stalling a few floors above the relative floor, where the
    log flattens and further improvement is nearly invisible to the GP.
    """
    y = np.asarray(y, dtype=float)
    floor = transform_floor(y, target_floor)
    shift = 0.0 if y.min() > 0 else (floor - y.min())
    logged = np.log10(y + shift + floor)
    return (logged - logged.mean()) / (logged.std() + 1e-12)


def reflect_into_box(points, lower, upper):
    """Mirror points back into [lower, upper] (periodic reflection; identity inside the box).

    Clipping instead puts every out-of-box point exactly ON the boundary, which piles
    local candidates onto an edge whenever the incumbent sits on one (the boundary trap).
    """
    width = upper - lower
    t = np.mod(np.asarray(points, float) - lower, 2 * width)
    return np.where(t > width, 2 * width - t, t) + lower


def bayes_opt(objective, dimension, lower, upper, metric, seed,
              n_initial=None, budget=80, n_candidates=200,
              n_restarts=DEFAULT_RESTARTS, prior="lognormal", prior_scale=PRIOR_SCALE,
              trace=None, noise_variance=None, acquisition="ei", target_floor=None,
              kernel="se", local_candidates=0, local_mode="clip"):
    """Minimise ``objective`` with GP-EI Bayesian optimisation.

    ``lower``/``upper`` define the box the acquisition function searches, which
    BO requires and CMA-ES does not.  Record it: a box is extra information
    about where the optimum lies, so a fair comparison has to say what each
    method was told.

    ``trace``, if given, is called once per iteration with the state used to
    choose the next point (model, data, candidates and their EI, chosen point).
    It is for diagnostics only: it must not touch ``rng``, and with trace=None
    the run is identical to before.

    Stage B options (defaults reproduce the earlier runs exactly):
    ``noise_variance`` fixes the GP noise instead of fitting it (B2), and
    ``acquisition="logei"`` maximises log EI instead of EI (B4), and
    ``target_floor`` fixes the floor in the log transform (B5).
    ``local_candidates`` adds that many acquisition candidates drawn around the
    incumbent, at scales log-uniform in [1e-3, 1] x box width / 10 (Stage C; the
    same idea as BoTorch's sampling around the best points). With fixed noise,
    the improving region shrinks to about 0.1 wide, and 200 uniform candidates
    in the whole box almost never land in it.
    ``local_mode`` says how local candidates outside the box are brought back:
    "clip" (the default, as in the original runs) or "reflect".  With logei,
    the trace's candidate_ei and chosen_ei are log EI values.
    """
    if metric not in ALL_METRICS:
        raise ValueError(f"metric must be one of {ALL_METRICS}")
    if acquisition not in ("ei", "logei"):
        raise ValueError("acquisition must be 'ei' or 'logei'")
    acquire = expected_improvement if acquisition == "ei" else log_expected_improvement
    rng = np.random.default_rng(seed)
    n_initial = 2 * dimension + 4 if n_initial is None else n_initial

    x = rng.uniform(lower, upper, size=(n_initial, dimension))
    y = np.array([objective(point) for point in x])

    for _ in range(budget - n_initial):
        standardised = _transform_targets(y, target_floor)
        model = fit_gp(x, standardised, metric, rng, n_restarts=n_restarts,
                       prior=prior, input_scale=float(upper - lower),
                       prior_scale=prior_scale, noise_variance=noise_variance,
                       kernel=kernel)
        incumbent = standardised.min()

        def negative_ei(point):
            return -float(acquire(model, point.reshape(1, -1), incumbent)[0])

        candidates = rng.uniform(lower, upper, size=(n_candidates, dimension))
        if local_candidates:
            centre = x[int(np.argmin(y))]
            scale = (upper - lower) / 10 * 10 ** rng.uniform(-3, 0, size=(local_candidates, 1))
            near = centre + scale * rng.normal(size=(local_candidates, dimension))
            near = (np.clip(near, lower, upper) if local_mode == "clip"
                    else reflect_into_box(near, lower, upper))
            candidates = np.vstack([candidates, near])
        scores = [negative_ei(c) for c in candidates]
        start = candidates[int(np.argmin(scores))]
        polished = minimize(negative_ei, start, method="L-BFGS-B",
                            bounds=[(lower, upper)] * dimension)
        nxt = np.clip(polished.x, lower, upper)

        if trace is not None:
            trace({"x": x, "y": y, "standardised": standardised, "model": model,
                   "incumbent": incumbent, "candidates": candidates,
                   "candidate_ei": -np.asarray(scores), "start": start,
                   "chosen": nxt, "chosen_ei": -float(polished.fun)})

        x = np.vstack([x, nxt])
        y = np.append(y, objective(nxt))

    order = int(np.argmin(y))
    return {"best_value": float(y[order]), "best_point": x[order],
            "evaluations": len(y), "history": np.minimum.accumulate(y),
            "x": x, "y": y}


def trust_region_bo(objective, dimension, lower, upper, metric, seed, budget=200,
                    n_initial=None, n_candidates=1000, noise_variance=1e-6,
                    n_restarts=DEFAULT_RESTARTS, local_mode="clip", local_fit=False,
                    local_min_points=None):
    """TuRBO-style trust-region BO (Eriksson et al. 2019), one trust region, EI inside it.

    Follows TuRBO-1's trust-region rules in the unit-cube scale of the box:
    side length L starts at 0.8, doubles after 3 consecutive successes (up to
    1.6), halves after max(4, d) consecutive failures, and the search restarts
    from a fresh design when L < 2^-7. A success is an improvement of at least
    1e-3 x |f_best|. Differences from the paper: EI over 1,000 uniform
    candidates in the region instead of Thompson sampling, and no batch.

    The region's shape is the point of the comparison:
      metric="ard"   an axis-aligned box whose sides are scaled by the ARD
                     lengthscales (the published TuRBO; it cannot tilt);
      metric="full"  a box rotated to the eigenvectors of the full metric M,
                     sides scaled by its eigen-lengthscales (it can tilt).
    In both, the side scales are normalised to geometric mean 1, as in TuRBO.

    ``local_mode`` = "clip" (default) or "reflect", as in bayes_opt.
    ``local_fit=True`` fits the GP only to the points inside the current region
    (the previous iteration's region; at least 2d + 4 points, the nearest ones in
    that region's own scaled coordinates if fewer are inside; ``local_min_points``
    raises that minimum), so the learned
    metric reflects the local shape of the function. TuRBO itself fits all the
    data of the current run (since the last restart), as the default here does.
    """
    rng = np.random.default_rng(seed)
    width = float(upper - lower)
    n_initial = 2 * dimension + 4 if n_initial is None else n_initial
    l_init, l_max, l_min = 0.8, 1.6, 2.0 ** -7
    fail_tol, succ_tol = max(4, dimension), 3

    xs, ys = [], []
    while len(ys) < budget:
        # (Re)start: a fresh design, a fresh trust region; earlier data are not reused.
        x = rng.uniform(lower, upper, size=(min(n_initial, budget - len(ys)), dimension))
        y = np.array([objective(p) for p in x])
        xs.extend(x); ys.extend(y)
        length, n_succ, n_fail = l_init, 0, 0
        region = None                      # (centre, axes, half-sides) of the last region
        while len(ys) < budget and length >= l_min:
            xf, yf = x, y
            if local_fit and region is not None:
                c0, ax0, half0 = region
                u = np.abs((x - c0) @ ax0) / half0           # scaled distance per region axis
                inside = np.all(u <= 1.0, axis=1)
                k_min = n_initial if local_min_points is None else local_min_points
                if inside.sum() < k_min:
                    inside = np.argsort(np.max(u, axis=1))[:k_min]
                xf, yf = x[inside], y[inside]
            z = _transform_targets(yf)
            model = fit_gp(xf, z, metric, rng, n_restarts=n_restarts, input_scale=width,
                           noise_variance=noise_variance)
            if metric == "ard":
                axes = np.eye(dimension)
                scales = 1.0 / np.sqrt(np.diag(model["precision"]))
            else:
                vals, axes = np.linalg.eigh(model["precision"])
                scales = 1.0 / np.sqrt(np.maximum(vals, 1e-300))
            scales = scales / np.exp(np.mean(np.log(scales)))
            centre = x[int(np.argmin(y))]
            u = rng.uniform(-0.5, 0.5, size=(n_candidates, dimension)) * scales * length * width
            raw = centre + u @ axes.T
            candidates = (np.clip(raw, lower, upper) if local_mode == "clip"
                          else reflect_into_box(raw, lower, upper))
            region = (centre, axes, 0.5 * scales * length * width)
            ei = expected_improvement(model, candidates, z.min())
            nxt = candidates[int(np.argmax(ei))]
            value = objective(nxt)
            if value < y.min() - 1e-3 * abs(y.min()):
                n_succ, n_fail = n_succ + 1, 0
            else:
                n_succ, n_fail = 0, n_fail + 1
            if n_succ == succ_tol:
                length, n_succ = min(2.0 * length, l_max), 0
            elif n_fail == fail_tol:
                length, n_fail = length / 2.0, 0
            x = np.vstack([x, nxt]); y = np.append(y, value)
            xs.append(nxt); ys.append(value)
    ys = np.asarray(ys[:budget])
    return {"best_value": float(ys.min()), "y": ys, "x": np.asarray(xs[:budget])}
