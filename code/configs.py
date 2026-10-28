"""Frozen method configurations, seed lists and experiment definitions.

Do not tune these after seeing results (see CLAUDE.md).
"""
from __future__ import annotations

# Seed lists are fixed in advance and only ever extended by appending.
SEEDS = list(range(1000, 1200))        # experiment seeds; full runs use a prefix
PILOT_SEEDS = list(range(9000, 9200))  # epsilon pilot (random sampling only)
FRESH_SEEDS = list(range(2000, 2050))  # Exp 9 best-variant line only (selection rule in pilot_log.md)

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
    # Exp 3: initial step size sigma0 (default 2). Random start (the baseline's U[-4,4]^2) ...
    "cma_sig0.3": dict(CMAES_BASE, sigma0=0.3),
    "cma_sig0.5": dict(CMAES_BASE, sigma0=0.5),
    "cma_sig1": dict(CMAES_BASE, sigma0=1.0),
    "cma_sig4": dict(CMAES_BASE, sigma0=4.0),
    # ... and the oracle-start mechanism arm: first start 1.0 from the well centre (uses knowledge of the well, so it is
    # a diagnostic of refinement, never a method comparison and never plotted against BO)
    "cma_o_sig0.3": dict(CMAES_BASE, sigma0=0.3, x0_from_well=1.0),
    "cma_o_sig0.5": dict(CMAES_BASE, sigma0=0.5, x0_from_well=1.0),
    "cma_o_sig1": dict(CMAES_BASE, sigma0=1.0, x0_from_well=1.0),
    "cma_o_sig2": dict(CMAES_BASE, sigma0=2.0, x0_from_well=1.0),
    "cma_o_sig4": dict(CMAES_BASE, sigma0=4.0, x0_from_well=1.0),
    # Exp 3 ratio test: oracle start 10/3 w from the well (= 1.0 at w = 0.3, as in the oracle arm) and sigma0 = k * w, so
    # the same k means the same sigma0 / w at every well width. Diagnostic only, never vs BO.
    "cma_ro_k1": dict(CMAES_BASE, sigma0_in_w=1.0, x0_from_well_w=10 / 3),
    "cma_ro_k3": dict(CMAES_BASE, sigma0_in_w=3.0, x0_from_well_w=10 / 3),
    "cma_ro_k7": dict(CMAES_BASE, sigma0_in_w=7.0, x0_from_well_w=10 / 3),
    "cma_ro_k20": dict(CMAES_BASE, sigma0_in_w=20.0, x0_from_well_w=10 / 3),
    # Exp 2: restarts
    "cma_ipop": dict(CMAES_BASE, restarts="ipop"),
    "cma_bipop": dict(CMAES_BASE, restarts="bipop"),
    # Exp 4: BO explore/exploit inside EI (bo_default is EI with xi=0). xi is in units of f; the well is ~2.2
    # deep, so the grid reaches near-pure exploration (grid fixed 2026-10-25 from the landscape, not from results)
    "bo_ei_xi0.01": dict(BO_BASE, xi=0.01),
    "bo_ei_xi0.1": dict(BO_BASE, xi=0.1),
    "bo_ei_xi0.5": dict(BO_BASE, xi=0.5),
    "bo_ei_xi1": dict(BO_BASE, xi=1.0),
    "bo_ei_xi2": dict(BO_BASE, xi=2.0),
    # Exp 5: other acquisition families (run together with Exp 4 on 2026-10-26; split at reporting time on 2026-10-28)
    "bo_lcb_b0.5": dict(BO_BASE, acq="lcb", beta=0.5),
    "bo_lcb_b2": dict(BO_BASE, acq="lcb", beta=2.0),
    "bo_lcb_b8": dict(BO_BASE, acq="lcb", beta=8.0),
    "bo_lcb_b32": dict(BO_BASE, acq="lcb", beta=32.0),
    "bo_ts": dict(BO_BASE, acq="ts", ts_candidates=2048),  # 4096 dominated runtime (pilot_log.md)
    # Exp 4 oracle (diagnostic, never compared with the real runs): the first initial point is 0.2 from the well centre
    # (where the first well samples of Exp 4 sat, f about -1.5), so the well is "found" at evaluation 0.
    "bo_o_xi0": dict(BO_BASE, x_init_from_well=0.2),
    "bo_o_xi0.1": dict(BO_BASE, xi=0.1, x_init_from_well=0.2),
    "bo_o_xi2": dict(BO_BASE, xi=2.0, x_init_from_well=0.2),
    # Exp 6: BO fitting settings, one at a time (baseline is bo_default: MLE, lengthscale in [0.05, 20], noise 1e-6).
    # Grids fixed 2026-10-28 from the landscape (box width 10, well width 0.3), before any of these ran.
    # Lengthscale upper cap in domain units; the lengthscale start becomes min(2, cap / 2) when the cap is below 4.
    "bo_cap5": dict(BO_BASE, ls_cap=5.0),
    "bo_cap2": dict(BO_BASE, ls_cap=2.0),
    "bo_cap1": dict(BO_BASE, ls_cap=1.0),
    "bo_cap0.5": dict(BO_BASE, ls_cap=0.5),
    "bo_cap0.25": dict(BO_BASE, ls_cap=0.25),
    # Fixed noise variance in standardised units (baseline 1e-6)
    "bo_noise1e-3": dict(BO_BASE, noise=1e-3),
    "bo_noise1e-2": dict(BO_BASE, noise=1e-2),
    "bo_noise1e-1": dict(BO_BASE, noise=1e-1),
    "bo_noise1": dict(BO_BASE, noise=1.0),
    "bo_noise_learned": dict(BO_BASE, gp="mle_learned_noise"),  # noise fitted (no prior) in [1e-6, 1]
    "bo_ls_prior": dict(BO_BASE, gp="mle_ls_prior"),  # BoTorch's lengthscale prior alone; noise fixed, bounds kept
    # Reference, not a one-factor arm (run with Exp 4): BoTorch's default recipe, every setting above changed together
    "bo_map": dict(BO_BASE, gp="map_default"),  # lengthscale prior, learned noise with prior, no bounds, MAP fit
    # Exp 7: BO kernel type (baseline is bo_default = Matern-5/2)
    "bo_k_m32": dict(BO_BASE, kernel="matern32"),
    "bo_k_m12": dict(BO_BASE, kernel="matern12"),
    "bo_k_rbf": dict(BO_BASE, kernel="rbf"),
}

PRIMARY = "primary_far_w0.3"

# Experiments with fully specified configs. 5, 6-best and 7 depend on the Exp 1-5 results
# and are added once those are in (configs chosen by the rule in direction.md).
EXPERIMENTS = {
    "exp0": dict(configs=["cma_default", "bo_default", "random"], landscapes=[PRIMARY]),
    "exp1": dict(configs=["cma_default", "cma_lam12", "cma_lam24"], landscapes=[PRIMARY]),
    "exp2": dict(configs=["cma_default", "cma_ipop", "cma_bipop"], landscapes=[PRIMARY]),
    "exp3": dict(configs=["cma_sig0.3", "cma_sig0.5", "cma_sig1", "cma_default", "cma_sig4"], landscapes=[PRIMARY]),
    "exp3_oracle": dict(configs=["cma_o_sig0.3", "cma_o_sig0.5", "cma_o_sig1", "cma_o_sig2", "cma_o_sig4"],
                        landscapes=[PRIMARY]),
    "exp3_ratio": dict(configs=["cma_ro_k1", "cma_ro_k3", "cma_ro_k7", "cma_ro_k20"],
                       landscapes=[f"primary_far_w{w}" for w in (0.1, 0.3, 0.6, 1.0)]),
    "exp4": dict(configs=["bo_default", "bo_ei_xi0.01", "bo_ei_xi0.1", "bo_ei_xi0.5", "bo_ei_xi1", "bo_ei_xi2"],
                 landscapes=[PRIMARY]),
    "exp5": dict(configs=["bo_default", "bo_lcb_b0.5", "bo_lcb_b2", "bo_lcb_b8", "bo_lcb_b32", "bo_ts"],
                 landscapes=[PRIMARY]),
    "exp6": dict(configs=["bo_default", "bo_cap5", "bo_cap2", "bo_cap1", "bo_cap0.5", "bo_cap0.25", "bo_noise1e-3",
                          "bo_noise1e-2", "bo_noise1e-1", "bo_noise1", "bo_noise_learned", "bo_ls_prior", "bo_map"],
                 landscapes=[PRIMARY]),
    "exp7": dict(configs=["bo_default", "bo_k_m32", "bo_k_m12", "bo_k_rbf"], landscapes=[PRIMARY]),
    # Exp 4 and Exp 5 were launched as one batch (--exp exp4_5) before the split; bo_map in the same batch
    "exp4_oracle": dict(configs=["bo_o_xi0", "bo_o_xi0.1", "bo_o_xi2"], landscapes=[PRIMARY]),
    "exp4_5": dict(configs=["bo_default", "bo_ei_xi0.01", "bo_ei_xi0.1", "bo_ei_xi0.5", "bo_ei_xi1", "bo_ei_xi2",
                            "bo_lcb_b0.5", "bo_lcb_b2", "bo_lcb_b8", "bo_lcb_b32", "bo_ts", "bo_map"],
                   landscapes=[PRIMARY]),
    "exp9_baseline": dict(configs=["cma_default", "bo_default", "random"],
                          # near well only at w <= 0.3: at w >= 0.6 its basin merges with the decoy
                          landscapes=[f"primary_far_w{w}" for w in (0.1, 0.3, 0.6, 1.0)]
                                     + [f"primary_near_w{w}" for w in (0.1, 0.3)]),
}

# Side check for Exp 1 (decided 2026-10-25, see pilot_log.md): the same three lambda settings at a 5x budget, so a
# lambda that is cut off by the N = 300 budget can be told apart from one that cannot succeed. CMA-ES only.
LONG_EXPERIMENTS = {
    "exp1_long": dict(configs=["cma_default", "cma_lam12", "cma_lam24"], landscapes=[PRIMARY]),
}
