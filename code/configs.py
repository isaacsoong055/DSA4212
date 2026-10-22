"""Frozen method configurations, seed lists and experiment definitions.

Do not tune these after seeing results (see CLAUDE.md).
"""
from __future__ import annotations

# Seed lists are fixed in advance and only ever extended by appending.
SEEDS = list(range(1000, 1200))        # experiment seeds; full runs use a prefix
PILOT_SEEDS = list(range(9000, 9200))  # epsilon pilot (random sampling only)
FRESH_SEEDS = list(range(2000, 2050))  # Exp E best-variant line only (selection rule in pilot_log.md)

N_FULL, N_FAST = 300, 100
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
    # Exp A: lambda = 1x, 2x, 4x default (d=2: 6, 12, 24)
    "cma_lam12": dict(CMAES_BASE, popsize=12),
    "cma_lam24": dict(CMAES_BASE, popsize=24),
    # Exp B: restarts
    "cma_ipop": dict(CMAES_BASE, restarts="ipop"),
    "cma_bipop": dict(CMAES_BASE, restarts="bipop"),
    # Exp C: BO exploration (bo_default is EI with xi=0)
    "bo_ei_xi0.01": dict(BO_BASE, xi=0.01),
    "bo_ei_xi0.1": dict(BO_BASE, xi=0.1),
    "bo_lcb_b0.5": dict(BO_BASE, acq="lcb", beta=0.5),
    "bo_lcb_b2": dict(BO_BASE, acq="lcb", beta=2.0),
    "bo_lcb_b8": dict(BO_BASE, acq="lcb", beta=8.0),
    "bo_ts": dict(BO_BASE, acq="ts", ts_candidates=2048),  # 4096 dominated runtime (pilot_log.md)
    "bo_map": dict(BO_BASE, gp="map_default"),
}

PRIMARY = "primary_far_w0.3"

# Experiments with fully specified configs. D, E-best and V depend on A-C results
# and are added once those are in (configs chosen by the rule in direction.md).
EXPERIMENTS = {
    "exp0": dict(configs=["cma_default", "bo_default", "random"], landscapes=[PRIMARY]),
    "expA": dict(configs=["cma_default", "cma_lam12", "cma_lam24"], landscapes=[PRIMARY]),
    "expB": dict(configs=["cma_default", "cma_ipop", "cma_bipop"], landscapes=[PRIMARY]),
    "expC": dict(configs=["bo_default", "bo_ei_xi0.01", "bo_ei_xi0.1", "bo_lcb_b0.5",
                          "bo_lcb_b2", "bo_lcb_b8", "bo_ts", "bo_map"], landscapes=[PRIMARY]),
    "expE_baseline": dict(configs=["cma_default", "bo_default", "random"],
                          # near well only at w <= 0.3: at w >= 0.6 its basin merges with the decoy
                          landscapes=[f"primary_far_w{w}" for w in (0.1, 0.3, 0.6, 1.0)]
                                     + [f"primary_near_w{w}" for w in (0.1, 0.3)]),
}
