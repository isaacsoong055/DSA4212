"""Optimizers behind one interface: run(landscape, N, seed, config) -> RunLog.

Every objective call goes through RunLog.eval, which enforces the budget N
(restarts included) and records one row per evaluation.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np

from benchmarks import HI, LO, Landscape

# Independent random streams per seed, so variants share what they should:
# all CMA-ES variants on seed k share x0; all BO variants share the initial design.
STREAM_X0, STREAM_CMA, STREAM_SOBOL, STREAM_RANDOM, STREAM_RESTART = 1, 2, 3, 4, 5


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


class RunLog:
    """Budgeted objective wrapper plus per-evaluation log."""

    def __init__(self, landscape: Landscape, N: int, method: str, config_id: str, seed: int):
        self.ls, self.N = landscape, N
        self.base = dict(method=method, config_id=config_id, landscape=landscape.name, seed=seed)
        self.rows: list[dict] = []
        self.best = math.inf
        self.meta: dict = {}

    @property
    def remaining(self) -> int:
        return self.N - len(self.rows)

    def eval(self, x, **diag) -> float:
        if self.remaining <= 0:
            raise RuntimeError("budget exhausted")
        x = np.asarray(x, dtype=float)
        if np.any(x < LO - 1e-9) or np.any(x > HI + 1e-9):
            raise ValueError(f"point outside domain: {x}")
        x = np.clip(x, LO, HI)
        f = self.ls(x)
        improved = f < self.best
        self.best = min(self.best, f)
        row = dict(self.base, eval_idx=len(self.rows), f=f, best_so_far=self.best,
                   _improved=improved, _x=x, **diag)
        row.update({f"x_{i}": float(v) for i, v in enumerate(x)})
        self.rows.append(row)
        return f

    def finalize(self) -> "RunLog":
        """Basin check on each new best point (other rows left empty)."""
        assert len(self.rows) == self.N, (len(self.rows), self.N)
        for r in self.rows:
            if r.pop("_improved"):
                r["in_basin"] = self.ls.in_global_basin(r["_x"])
            r.pop("_x")
        return self


# ----------------------------------------------------------------- random

def run_random(ls: Landscape, N: int, seed: int, cfg: dict, config_id: str) -> RunLog:
    log = RunLog(ls, N, "random", config_id, seed)
    X = _rng(seed, STREAM_RANDOM).uniform(LO, HI, size=(N, ls.d))
    for x in X:
        log.eval(x, opt_time_s=0.0)
    return log.finalize()


# ----------------------------------------------------------------- CMA-ES

def default_popsize(d: int) -> int:
    return 4 + int(3 * math.log(d))


def run_cmaes(ls: Landscape, N: int, seed: int, cfg: dict, config_id: str) -> RunLog:
    """CMA-ES with optional IPOP/BIPOP restarts under a hard total budget N.

    cfg: popsize (None = default), restarts ('none'|'ipop'|'bipop'), sigma0,
    x0_halfwidth (first start ~ U[-h, h]^d), tolx, tolfun. Optional x0_from_well = r starts at
    distance r from the well centre in a random direction (oracle start, Exp 3 mechanism arm).
    Optional sigma0_in_w = k sets sigma0 = k * w, and x0_from_well_w = m starts m * w from the well
    (Exp 3 ratio test, so step size and start distance scale with the well width).
    Restarts redraw x0 uniformly in the full box [-5, 5]^d.
    With restarts='none' stop conditions are ignored and the run uses all N
    evaluations; the first generation at which they would have fired is
    recorded in log.meta['stop_fired'] for the restart-trigger pilot.
    """
    import cma

    d = ls.d
    log = RunLog(ls, N, "cmaes", config_id, seed)
    lam_def = cfg.get("popsize") or default_popsize(d)
    sigma0, mode = cfg["sigma0"], cfg["restarts"]
    if cfg.get("sigma0_in_w") is not None:  # step size as a multiple of the well width (Exp 3 ratio test)
        sigma0 = cfg["sigma0_in_w"] * ls.meta["w"]
    if cfg.get("x0_from_well_w") is not None:
        # Oracle start at a multiple of w from the well centre, random direction, redrawn until inside the box.
        rng_x0, r = _rng(seed, STREAM_X0), cfg["x0_from_well_w"] * ls.meta["w"]
        while True:
            u = rng_x0.normal(size=d)
            x0 = ls.x_star + r * u / np.linalg.norm(u)
            if np.all(np.abs(x0) <= 4.5):
                break
    elif cfg.get("x0_from_well") is not None:
        # Oracle start (Exp 3 mechanism arm only): at distance r from the well centre, random direction.
        u = _rng(seed, STREAM_X0).normal(size=d)
        x0 = ls.x_star + cfg["x0_from_well"] * u / np.linalg.norm(u)
    else:
        x0 = _rng(seed, STREAM_X0).uniform(-cfg["x0_halfwidth"], cfg["x0_halfwidth"], d)
    rng_restart = _rng(seed, STREAM_RESTART)
    cma_seeds = _rng(seed, STREAM_CMA).integers(1, 2**31 - 1, size=10_000)

    lam, sigma, restart, gen_total = lam_def, sigma0, 0, 0
    lam_large, budget_large, budget_small = lam_def, 0, 0
    regime = "large"
    stop_fired = None
    dist: list[dict] = []
    while log.remaining > 0:
        opts = dict(popsize=lam, bounds=[[LO] * d, [HI] * d], seed=int(cma_seeds[restart]),
                    tolx=cfg["tolx"], tolfun=cfg["tolfun"], maxiter=math.inf,
                    maxfevals=math.inf, verbose=-9, verb_log=0, verb_disp=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = cma.CMAEvolutionStrategy(x0, sigma, opts)
        used_before = len(log.rows)
        while log.remaining > 0:
            t0 = time.perf_counter()
            X = es.ask()
            t_ask = time.perf_counter() - t0
            diag = dict(sigma=float(es.sigma), cov_trace=float(es.sigma**2 * np.trace(es.C)),
                        gen=gen_total, restart=restart, popsize=lam)
            dist.append(dict(gen=gen_total, restart=restart, sigma=float(es.sigma),
                             mean=np.array(es.boundary_handler.transform(es.mean), float),
                             C=np.array(es.C, float)))  # in-memory only, for plotting
            k = min(len(X), log.remaining)
            fs = [log.eval(X[i], **diag) for i in range(k)]
            if k < len(X):  # budget ends mid-generation: no update
                for r in log.rows[-k:]:
                    r["opt_time_s"] = t_ask / k
                break
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                es.tell(X, fs)
                stop = es.stop()
            t_tell = time.perf_counter() - t0
            for r in log.rows[-k:]:
                r["opt_time_s"] = (t_ask + t_tell) / k
            gen_total += 1
            if stop and stop_fired is None and mode == "none":
                stop_fired = dict(gen=gen_total, eval=len(log.rows), why=",".join(stop))
            if stop and mode != "none":
                break
        if mode == "none":
            break
        # Book-keep budget per BIPOP regime, then choose the next restart.
        used = len(log.rows) - used_before
        if regime == "large":
            budget_large += used
        else:
            budget_small += used
        restart += 1
        x0 = rng_restart.uniform(LO, HI, d)
        if mode == "ipop":
            lam, sigma = 2 * lam, sigma0
        elif mode == "bipop":
            if restart > 1 and budget_small < budget_large:
                regime = "small"
                u1, u2 = rng_restart.uniform(size=2)
                lam = int(lam_def * (0.5 * lam_large / lam_def) ** (u1**2))
                sigma = sigma0 * 10 ** (-2 * u2)
            else:
                regime = "large"
                lam_large *= 2
                lam, sigma = lam_large, sigma0
    log.meta.update(stop_fired=stop_fired, n_restarts=restart if mode != "none" else 0, dist=dist)
    return log.finalize()


# ----------------------------------------------------------------- BO

KERNEL_NU = {"matern52": 2.5, "matern32": 1.5, "matern12": 0.5}
LS_BOUNDS_UNIT = (0.005, 2.0)  # lengthscale bounds on the unit cube = [0.05, 20] domain units
OUTPUTSCALE_BOUNDS = (1e-3, 1e3)
FIXED_NOISE = 1e-6  # noise variance in standardised output units
LS_INIT_UNIT = 0.2


def _kernel(model):
    k = model.covar_module
    return k.base_kernel if hasattr(k, "base_kernel") else k


def _build_gp(X, Y, gp: str, bounds, kernel: str = "matern52", cfg: dict | None = None):
    """Matern-5/2 ARD GP. 'mle_fixed': no priors, fixed tiny noise, bounded
    lengthscales. 'map_default': BoTorch's dim-scaled LogNormal prior and
    learned noise with its default prior (same Matern-5/2 kernel). `kernel`
    (matern52 | matern32 | matern12 | rbf) applies to the 'mle_*' settings only.
    One-factor variants of 'mle_fixed' (Exp 6), set in cfg: ls_cap (upper lengthscale bound in domain units, default 20),
    noise (fixed noise variance in standardised units, default 1e-6); gp 'mle_learned_noise' (noise fitted, no prior,
    in [1e-6, 1]) and 'mle_ls_prior' (BoTorch's dim-scaled lengthscale prior added, noise fixed)."""
    cfg = cfg or {}
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

    d = X.shape[-1]
    common = dict(input_transform=Normalize(d, bounds=bounds), outcome_transform=Standardize(1))
    if gp in ("mle_fixed", "mle_learned_noise", "mle_ls_prior"):
        hi = LS_BOUNDS_UNIT[1] if cfg.get("ls_cap") is None else cfg["ls_cap"] / (bounds[1] - bounds[0]).mean().item()
        ls_con = Interval(LS_BOUNDS_UNIT[0], hi)
        prior = None
        if gp == "mle_ls_prior":
            from gpytorch.priors import LogNormalPrior
            prior = LogNormalPrior(loc=math.sqrt(2) + math.log(d) * 0.5, scale=math.sqrt(3))  # BoTorch's dim-scaled prior
        kw = dict(ard_num_dims=d, lengthscale_constraint=ls_con, lengthscale_prior=prior)
        base = RBFKernel(**kw) if kernel == "rbf" else MaternKernel(nu=KERNEL_NU[kernel], **kw)
        covar = ScaleKernel(base, outputscale_constraint=Interval(*OUTPUTSCALE_BOUNDS))
        base.lengthscale = min(LS_INIT_UNIT, hi / 2)  # starts inside the interval when the cap is below the usual start
        covar.outputscale = 1.0
        if gp == "mle_learned_noise":
            from gpytorch.likelihoods import GaussianLikelihood
            lik = GaussianLikelihood(noise_constraint=Interval(1e-6, 1.0))
            lik.noise = 1e-3
            return SingleTaskGP(X, Y, likelihood=lik, covar_module=covar, **common)
        yvar = torch.full_like(Y, cfg.get("noise", FIXED_NOISE)) * Y.var()
        return SingleTaskGP(X, Y, train_Yvar=yvar, covar_module=covar, **common)
    if gp == "map_default":
        from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
        covar = get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel=False)
        return SingleTaskGP(X, Y, covar_module=covar, **common)
    raise ValueError(gp)


def _hyper_state(model, gp: str) -> dict:
    """Hyperparameters that can be carried to a model with more data."""
    keep = ("covar_module.", "mean_module.") + (("likelihood.",) if gp in ("map_default", "mle_learned_noise") else ())
    return {k: v.clone() for k, v in model.state_dict().items() if k.startswith(keep)}


def _count_jitter(caught) -> int:
    return sum(1 for w in caught if w.category.__name__ == "NumericalWarning"
               and "jitter" in str(w.message).lower())


def run_bo(ls: Landscape, N: int, seed: int, cfg: dict, config_id: str) -> RunLog:
    """GP-BO. cfg: acq ('ei'|'lcb'|'ts'), xi, beta, n_init, refit_every, gp, kernel,
    num_restarts, raw_samples, ts_candidates. Optional x_init_from_well = r puts the first initial point r from the well."""
    import torch
    from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
    from botorch.exceptions import ModelFittingError
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine

    torch.manual_seed(seed)
    dt = torch.float64
    d = ls.d
    log = RunLog(ls, N, "bo", config_id, seed)
    bounds = torch.tensor(ls.bounds, dtype=dt)
    sobol = SobolEngine(d, scramble=True, seed=int(_rng(seed, STREAM_SOBOL).integers(2**31 - 1)))

    def to_box(u):
        return bounds[0] + (bounds[1] - bounds[0]) * u.to(dt)

    init = to_box(sobol.draw(cfg["n_init"]))
    if cfg.get("x_init_from_well") is not None:
        # Diagnostic arm (Exp 4 oracle): the first initial point is replaced by one at this distance from the well
        # centre, random direction. The other n_init - 1 points and the budget are unchanged.
        u = _rng(seed, STREAM_X0).normal(size=d)
        init[0] = torch.tensor(ls.x_star + cfg["x_init_from_well"] * u / np.linalg.norm(u), dtype=dt)
    for x in init:
        log.eval(x.numpy(), opt_time_s=0.0)

    state, it, n_fit_fail, n_jitter_total = None, 0, 0, 0
    while log.remaining > 0:
        t0 = time.perf_counter()
        X = torch.tensor(np.array([[r[f"x_{i}"] for i in range(d)] for r in log.rows]), dtype=dt)
        Y = torch.tensor([[r["f"]] for r in log.rows], dtype=dt)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = _build_gp(X, Y, cfg["gp"], bounds, cfg.get("kernel", "matern52"), cfg)
            if state is not None:
                model.load_state_dict(state, strict=False)
            if state is None or it % cfg["refit_every"] == 0:
                try:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
                except ModelFittingError:
                    n_fit_fail += 1
                    if state is not None:
                        model.load_state_dict(state, strict=False)
            state = _hyper_state(model, cfg["gp"])
            model.eval()
            if cfg["acq"] == "ts":
                cand = to_box(sobol.draw(cfg["ts_candidates"]))
                with torch.no_grad():
                    sample = model.posterior(cand).rsample(torch.Size([1])).reshape(-1)
                x_next = cand[sample.argmin()]
            else:
                if cfg["acq"] == "ei":
                    acq = LogExpectedImprovement(model, best_f=Y.min() - cfg["xi"], maximize=False)
                else:
                    acq = UpperConfidenceBound(model, beta=cfg["beta"], maximize=False)
                x_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=cfg["num_restarts"],
                                          raw_samples=cfg["raw_samples"])
                x_next = x_next.reshape(-1)
        n_jit = _count_jitter(caught)
        n_jitter_total += n_jit
        lengthscales = (_kernel(model).lengthscale.detach().reshape(-1) * (HI - LO)).tolist()
        log.eval(x_next.detach().numpy(), lengthscales=lengthscales, n_jitter=n_jit,
                 opt_time_s=time.perf_counter() - t0)
        it += 1
    log.meta.update(n_fit_fail=n_fit_fail, n_jitter=n_jitter_total)
    return log.finalize()


# ----------------------------------------------------------------- dispatch

METHODS = {"random": run_random, "cmaes": run_cmaes, "bo": run_bo}


def run(landscape: Landscape, N: int, seed: int, config: dict, config_id: str = "") -> RunLog:
    """Run one (method, config, landscape, seed). config['method'] picks the optimizer."""
    return METHODS[config["method"]](landscape, N, seed, config, config_id or config["method"])
