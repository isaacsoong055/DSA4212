"""Frozen method configurations, seed lists and experiment definitions.

Do not tune these after seeing results (see CLAUDE.md).
"""
from __future__ import annotations

# Seed lists are fixed in advance and only ever extended by appending.
SEEDS = list(range(1000, 1200))        # experiment seeds; full runs use a prefix
PILOT_SEEDS = list(range(9000, 9200))  # epsilon pilot (random sampling only)
FRESH_SEEDS = list(range(2000, 2050))  # Exp 6 best-variant line only (selection rule in pilot_log.md)

N_FULL, N_FAST = 300, 100
N_LONG = 1500   # Exp 1 side check only (CMA-ES, never plotted on BO axes)
N_SEEDS_FULL, N_SEEDS_FAST = 50, 5

CMAES_BASE = dict(method="cmaes", popsize=None, restarts="none", sigma0=2.0,
                  x0_halfwidth=4.0, tolx=3e-3, tolfun=1e-5)  # frozen, see pilot_log.md
BO_BASE = dict(method="bo", gp="mle_fixed", acq="ei", xi=0.0, beta=None, n_init=10,
               refit_every=5, num_restarts=10, raw_samples=4096, ts_candidates=4096)

CONFIGS: dict[str, dict] = {
    "random": dict(method="random"),
    # Exp 0 baselines
    "cma_default": CMAES_BASE,
    "bo_default": BO_BASE,
    # Exp 1: lambda = 1x, 2x, 4x default (d=2: 6, 12, 24)
    "cma_lam12": dict(CMAES_BASE, popsize=12),
    "cma_lam24": dict(CMAES_BASE, popsize=24),
    # Exp 2: restarts
    "cma_ipop": dict(CMAES_BASE, restarts="ipop"),
    "cma_bipop": dict(CMAES_BASE, restarts="bipop"),
    # Exp 3: BO exploration vs exploitation (bo_default is EI with xi=0). xi is in units of f; the well is ~2.2
    # deep, so the grid reaches near-pure exploration (grid fixed 2026-10-25 from the landscape, not from results)
    "bo_ei_xi0.01": dict(BO_BASE, xi=0.01),
    "bo_ei_xi0.1": dict(BO_BASE, xi=0.1),
    "bo_ei_xi0.5": dict(BO_BASE, xi=0.5),
    "bo_ei_xi1": dict(BO_BASE, xi=1.0),
    "bo_ei_xi2": dict(BO_BASE, xi=2.0),
    "bo_lcb_b0.5": dict(BO_BASE, acq="lcb", beta=0.5),
    "bo_lcb_b2": dict(BO_BASE, acq="lcb", beta=2.0),
    "bo_lcb_b8": dict(BO_BASE, acq="lcb", beta=8.0),
    "bo_lcb_b32": dict(BO_BASE, acq="lcb", beta=32.0),
    "bo_ts": dict(BO_BASE, acq="ts", ts_candidates=2048),  # 4096 dominated runtime (pilot_log.md)
    "bo_map": dict(BO_BASE, gp="map_default"),  # lengthscale prior AND learned noise: not a one-factor change
    # Exp 4: BO kernel type (baseline is bo_default = Matern-5/2)
    "bo_k_m32": dict(BO_BASE, kernel="matern32"),
    "bo_k_m12": dict(BO_BASE, kernel="matern12"),
    "bo_k_rbf": dict(BO_BASE, kernel="rbf"),
}

PRIMARY = "primary_far_w0.3"

# Experiments with fully specified configs. 5, 6-best and 7 depend on the Exp 1-4 results
# and are added once those are in (configs chosen by the rule in direction.md).
EXPERIMENTS = {
    "exp0": dict(configs=["cma_default", "bo_default", "random"], landscapes=[PRIMARY]),
    "exp1": dict(configs=["cma_default", "cma_lam12", "cma_lam24"], landscapes=[PRIMARY]),
    "exp2": dict(configs=["cma_default", "cma_ipop", "cma_bipop"], landscapes=[PRIMARY]),
    "exp3": dict(configs=["bo_default", "bo_ei_xi0.01", "bo_ei_xi0.1", "bo_ei_xi0.5", "bo_ei_xi1", "bo_ei_xi2",
                          "bo_lcb_b0.5", "bo_lcb_b2", "bo_lcb_b8", "bo_lcb_b32", "bo_ts", "bo_map"],
                 landscapes=[PRIMARY]),
    "exp4": dict(configs=["bo_default", "bo_k_m32", "bo_k_m12", "bo_k_rbf"], landscapes=[PRIMARY]),
    "exp6_baseline": dict(configs=["cma_default", "bo_default", "random"],
                          # near well only at w <= 0.3: at w >= 0.6 its basin merges with the decoy
                          landscapes=[f"primary_far_w{w}" for w in (0.1, 0.3, 0.6, 1.0)]
                                     + [f"primary_near_w{w}" for w in (0.1, 0.3)]),
}

# Side check for Exp 1 (decided 2026-10-25, see pilot_log.md): the same three lambda settings at a 5x budget, so a
# lambda that is cut off by the N = 300 budget can be told apart from one that cannot succeed. CMA-ES only.
LONG_EXPERIMENTS = {
    "exp1_long": dict(configs=["cma_default", "cma_lam12", "cma_lam24"], landscapes=[PRIMARY]),
}
