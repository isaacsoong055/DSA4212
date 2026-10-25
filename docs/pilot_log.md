# Phase 1 pilot log

Machine: MacBook M2, CPU only, Python 3.13.x, torch 2.14.1 / botorch 0.18.1 /
gpytorch 1.15.2 / cma 4.5.0 (exact pins in `requirements.txt`). Date: 2026-10-22.

## Decisions taken before running anything

| Item | Decision |
| --- | --- |
| BO library | BoTorch. EI-with-xi maps to `LogExpectedImprovement(best_f = y+ - xi, maximize=False)` (same argmax as EI, numerically stabler); LCB maps exactly to `UpperConfidenceBound(beta, maximize=False)` = mu - sqrt(beta) sigma. Fallback to scikit-learn was not needed (see timings). |
| Primary landscape | c = 0, s = 10, A = 4, w = 0.3; far well x_g = (3, 3), near well x_g = (2, 0). x*, f* found by polishing from x_g. |
| CMA-ES | Default lambda = 4 + floor(3 ln d) = 6. x0 ~ U[-4, 4]^d per seed, sigma0 = 2, `cma` BoundTransform on [-5, 5]^d. Own ask/tell restart loop; a generation cut short by the budget is evaluated but not told. Restarts redraw x0 ~ U[-5, 5]^d. BIPOP: first run counts as "large"; small regime lambda = floor(lambda_def (lambda_large / (2 lambda_def))^(U^2)), sigma = sigma0 10^(-2U). |
| BO baseline GP (`mle_fixed`) | Matern-5/2 ARD in a ScaleKernel; inputs normalised to [0, 1]^d, outputs standardised. Pure MLE, no priors. Noise fixed at 1e-6 (standardised units). **Lengthscale bounds [0.005, 2.0] on the unit cube = [0.05, 20] domain units**; outputscale bounds [1e-3, 1e3]; initial lengthscale 0.2 (unit cube). Refit every 5 iterations, warm-started from the previous fit; in between, the GP is conditioned on new data with fixed hyperparameters. |
| BO extra arm (`map_default`) | Same Matern-5/2 kernel built by BoTorch's `get_covar_module_with_dim_scaled_prior` (dim-scaled LogNormal lengthscale prior), learned noise with BoTorch's default prior, MAP fit. Only the fitting changes. |
| BO acquisition optimiser | `optimize_acqf`, 10 restarts, 4096 raw samples; n_init = 10 scrambled Sobol points (seeded). Thompson: joint posterior draw over 4096 fresh Sobol candidates per iteration. |
| Jitter logging | Each iteration counts gpytorch/linear_operator `NumericalWarning`s about added jitter (`n_jitter` column). Verified the counter fires on a forced rank-deficient Cholesky. |
| Success | Primary: best regret <= epsilon. Secondary: best point in the global basin (L-BFGS-B local descent from it ends within min(0.1, w/2) of x*). |
| Seeds | Experiment seeds 1000+, pilot seeds 9000+, per-run streams via `SeedSequence([seed, stream])`. |

## Step 2: epsilon pilot (random sampling only), `pilot/epsilon_pilot.py`

Rule fixed before running: epsilon = largest value in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
with random-sampling success <= 5% at N = 300 on `primary_far_w0.3`, seeds 9000-9199.

| epsilon | random successes / 200 | Wilson 95% | radius of success disc | radius / w |
| --- | --- | --- | --- | --- |
| 0.01 | 2 | [0.003, 0.036] | 0.021 | 0.07 |
| 0.02 | 3 | [0.005, 0.043] | 0.030 | 0.10 |
| **0.05** | **5** | **[0.011, 0.057]** | **0.048** | **0.16** |
| 0.1 | 12 | [0.035, 0.102] | 0.067 | 0.22 |
| 0.2 | 19 | [0.062, 0.144] | 0.095 | 0.32 |
| 0.5 | 44 | [0.168, 0.282] | 0.153 | 0.51 |

**Chosen epsilon = 0.05** (frozen in `code/benchmarks.py`). The well floor has regret 0;
the decoy minimum has regret 2.21 (44x epsilon). Random best-regret quantiles
(5/25/50/75/95%): 0.076 / 0.61 / 1.25 / 2.09 / 2.23. Figure: `docs/figures/epsilon_check.png`.

Random-sampling reference for Exp E (same epsilon, 200 pilot seeds):

| landscape | random success | Wilson 95% |
| --- | --- | --- |
| near w0.1 | 1/200 | [0.001, 0.028] |
| near w0.3 | 3/200 | [0.005, 0.043] |
| near w0.6 | 22/200 | [0.074, 0.161] |
| near w1.0 | 45/200 | [0.173, 0.288] |
| far w0.1 | 3/200 | [0.005, 0.043] |
| far w0.3 | 5/200 | [0.011, 0.057] |
| far w0.6 | 12/200 | [0.035, 0.102] |
| far w1.0 | 38/200 | [0.142, 0.250] |

**Landscape check: local minima.** L-BFGS-B from the decoy centre (origin, nudged by 1e-3):

| landscape | x* | f* | descent from origin ends at | separate decoy minimum? |
| --- | --- | --- | --- | --- |
| near w0.1 | (2.00, 0.00) | -3.600 | (0, 0), f = 0.000 | yes |
| near w0.3 | (1.99, 0.00) | -3.602 | (0, 0), f = 0.000 | yes |
| near w0.6 | (1.97, 0.00) | -3.607 | (1.97, 0.00), f = -3.607 | **no, basins merged** |
| near w1.0 | (1.90, 0.00) | -3.619 | (1.90, 0.00), f = -3.619 | **no, basins merged** |
| far w0.1 / 0.3 / 0.6 / 1.0 | (3.0, 3.0) to (2.85, 2.85) | -2.20 to -2.29 | origin (f within 1e-3 of 0) | yes, all four |

**Decision (user, 2026-10-22): exclude near w0.6 and near w1.0 from Exp E.** With the
decoy gone these cells are unimodal and do not test a narrow or distant basin. Exp E runs the
full width sweep {0.1, 0.3, 0.6, 1.0} on the far well, and the near well only at w <= 0.3
(`expE_baseline` in `configs.py`). The decision used only this local-minima check,
not any success rate. The random-sampling rows above for the two excluded cells are kept
for the record.

## Step 1: timing, `pilot/timing.py` (one run each, N = 300, seed 1000, 1 thread)

| config | wall (s) | jitter fallbacks | fit failures |
| --- | --- | --- | --- |
| cma_default | 1.5 (incl. imports; ~0.03 s optimiser) | n/a | n/a |
| bo_default (EI, xi = 0) | 107.9 | 0 | 0 |
| bo_lcb_b2 | 35.5 | 0 | 0 |
| bo_map | 49.2 | 0 | 0 |
| bo_ts (4096 candidates) | 174.0 | 0 | 0 |

Smoke run `--fast` (all experiments, 5 seeds, N = 100, 6 workers): 175 runs in 306 s, no errors.

Across 5 seeds `bo_default` took 47-116 s per run (5 in parallel, no visible slowdown
vs. single).

**Thompson candidates: dropped to 2048.** Run alone, TS-4096 is ~1.6x a default-EI run.
But the `--fast` smoke run (6 workers, N = 100) showed TS-4096 at ~130 s per run vs 42 s
alone (same seed, N = 100): the 4096x4096 joint draw is memory-bandwidth bound and slows
~3x under 6-way parallelism. At full scale that would be ~60% of Exp C wall time, which
meets the "dominates runtime" rule. TS-2048 alone at N = 100: 9.8 s. Changed before any
TS result was inspected; 4096-candidate smoke files deleted.

## Restart-trigger pilot (baseline CMA-ES firing rates only), `pilot/restart_triggers.py`

No-restart CMA-ES on `primary_far_w0.3`, pilot seeds 9000-9049; records the first
generation at which `es.stop()` would fire (trajectory is unaffected). Rule fixed
before running: tightest pair with firing rate >= 80% and median firing eval <= 200
at lambda = 6.

| tolx | tolfun | fired lam=6 (median eval) | lam=12 | lam=24 |
| --- | --- | --- | --- | --- |
| 1e-1 | 1e-2 | 1.00 (96) | 1.00 (156) | 1.00 (240) |
| 3e-2 | 1e-3 | 1.00 (126) | 1.00 (192) | 0.26 (288) |
| 1e-2 | 1e-4 | 1.00 (156) | 1.00 (228) | 0.00 |
| 3e-3 | 1e-5 | 1.00 (189) | 0.84 (276) | 0.00 |
| 1e-3 | 1e-6 | 1.00 (216) | 0.34 (288) | 0.00 |
| 1e-4 | 1e-8 | 0.78 (270) | 0.00 | 0.00 |
| 1e-5 | 1e-10 | 0.14 (294) | 0.00 | 0.00 |
| 1e-6 | 1e-8 | 0.04 (288) | 0.00 | 0.00 |
| 1e-6 | 1e-12 | 0.00 | 0.00 | 0.00 |

Almost all firings are `tolx`. The originally pre-stated grid (last five rows) had no
pair meeting the rule; the agreed default (1e-6, 1e-8) fires in 4% of runs, so IPOP/BIPOP
would be near-identical to no restarts. The grid was then extended (firing rates only).
On the extended grid the same rule selects **(3e-3, 1e-5)**.

**Decision (user, 2026-10-22): adopt (tolx, tolfun) = (3e-3, 1e-5) and freeze it** in
`CMAES_BASE` (`configs.py`).

**Deviation from the pre-set list, recorded as such:** the pre-stated grid contained no
pair that satisfied the rule, so the list was extended (to looser thresholds) and the
*same* firing-rate rule was applied to it. Neither the extension nor the choice used any
success rate; only firing rates of no-restart baseline runs on pilot seeds 9000-9049.
Baseline `cma_default` trajectories are unaffected (stop conditions are ignored
without restarts).

**For Exp B reporting:** report restart firing rates per lambda (pilot values above). At
lambda = 24 the trigger never fired within N = 300 (0/50), so IPOP and BIPOP started
at lambda = 24 would equal no restarts. Say so explicitly wherever restarts are compared
with a large-lambda arm.

## Predictions committed

`docs/predictions.md` committed on 2026-10-22 (commit `9ae8476`), after the three pilots
above and before any CMA-ES/BO success or regret was inspected. Adds M1-M3 (needle:
lambda helps only via early samples, restarts act like random search; selection
averaging applies to Rastrigin).

## Step 3: baselines on 5 seeds (1000-1004), N = 300

Figure: `docs/figures/pilot_baselines.png` (sigma per generation, sample scatter for
seed 1000, BO fitted lengthscale vs w). Summary: `results/summary/`.

| config | success (regret <= 0.05) | Wilson 95% | notes |
| --- | --- | --- | --- |
| cma_default | 0/5 | [0.00, 0.43] | all 5 converge on the decoy; 50 generations; sigma falls ~geometrically from 2 to ~1.5e-3 |
| bo_default | 1/5 | [0.04, 0.62] | 4/5 sample the decoy almost exclusively after eval 100; seed 1004 hit the well at eval 108 |

Diagnostics (5 seeds only, descriptive, not a result):
- CMA-ES: >= 98% of evals after eval 100 lie within 0.5 of the decoy centre in every seed;
  closest sample to the well 1.1-2.5 away. Consistent with the H1 failure mode.
- BO: in the 4 failing seeds the fitted lengthscale sits **at the upper bound (20 domain
  units)** from about eval 15 onward, i.e. the GP models a smooth bowl and EI (xi = 0)
  re-samples the decoy; closest sample to the well 1.5-2.7 away. In the successful seed the
  lengthscale drops to ~0.8-1.0 after the well is found. Lengthscale still > w = 0.3 there.
- No Cholesky jitter fallbacks or fit failures in any baseline run.

## Step 4: runtime estimate for the full plan (50 seeds, N = 300, 6 workers)

CMA-ES runs take ~0.05 s each, so Exps A and B and all CMA-ES arms are negligible
(< 2 min total, even at 200+ seeds). BO is the bottleneck. Per-run CPU assumed:
EI ~100 s (median of 6 measured runs: 47-116 s), LCB 35 s, MAP 49 s, TS-2048 ~80 s
(extrapolated, including a parallel-load penalty).

| block | BO runs | CPU (h) |
| --- | --- | --- |
| Exp 0 (bo_default) | 50 | 1.4 |
| Exp C (7 more arms) | 350 | ~6.0 |
| Exp D (best + worst BO on sphere) | 100 | ~2.8 |
| Exp E (6 landscapes x baseline + best BO, minus far w0.3 baseline already run) | ~500 | ~13.9 |
| **A-E total** | ~1000 | **~24.5** |
| V (2 BO arms x 2 functions x 30 seeds) | 120 | ~3.3 |

**Wall clock at 6 workers: about 4 h for A-E (4.5 h with V).** Range 3-7 h:
(Updated after Exp E was reduced from 8 to 6 landscapes; first estimate was ~5 h.)
per-run time varies 35-120 s by acquisition and seed, and other landscapes may fit
differently. A fanless M2 may throttle under sustained 6-core load. Exp E dominates;
running it overnight fits the timeline in direction.md.

## Notebook (`notebooks/main.ipynb`), restructured 2026-10-22

Convention: the notebook only loads `results/summary*/` and `results/raw*/`; runs, aggregation and figure
code live in `run_experiments.py`, `code/analysis.py` and the new `code/report.py`. FAST mode reads
`results/summary_fast/` (written by `run_experiments.py --aggregate --fast`, gitignored); FULL mode reads
`results/summary/`. Run All in FAST mode: about 6 s (loading plus one 0.05 s CMA-ES re-run). The re-run of
seed 1000 matches the saved CSV to 4e-16 (pytest asserts 1e-8). Engine change for the ellipses: `run_cmaes`
keeps the per-generation mean, sigma and C in `log.meta["dist"]` (in memory only); trajectories unchanged.
The landscape name in the CSVs is canonical (`primary_far_w1`) while run directories use the requested name
(`primary_far_w1.0`); `report.canon` maps one to the other.

## Epsilon for Rastrigin and Schwefel (validation functions): rule written 2026-10-22, BEFORE the pilot

Neither function had an epsilon (the primary-landscape pilot above only covered `primary_far_w0.3`).
Success on them is needed for Exp V. This rule was fixed before any random-sampling pilot ran on
these functions, and before any CMA-ES or BO run on them (including smoke tests).

1. **Cap from local minima.** Run L-BFGS-B (box-constrained) from a fixed set of uniform starts
   (seed 0, 1000 starts). Cluster end points (distance < 1e-2) into distinct minima and print them
   with their regrets. Check the nearest non-global minimum is unchanged (same location, regret equal
   to 1e-6 relative) when the number of starts is doubled to 2000; **if it changes, stop and ask**.
   Epsilon must be **strictly below** the nearest non-global regret, so that success implies the
   global basin (a point in another basin descends to that basin's minimum, so its regret is at least
   that minimum's regret). A pytest checks this.
2. **Grid.** Start with {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5} (regret units, unrescaled).
3. **Random-sampling criterion.** Uniform random sampling, N = 300, d = 2, pilot seeds 9000-9199:
   success = best regret <= epsilon. Random sampling "rarely succeeds" means success rate <= 5%.
   The success rate is nondecreasing in epsilon (nested sets).
4. **Extension.** If random success at the largest grid value is still <= 5%, extend upward by
   factors of 2.5 until it exceeds 5%. If it already exceeds 5% at the smallest grid value, extend
   downward by halving until it is <= 5% (stop and ask if 1e-4 is reached without that).
5. **Choice.** epsilon = the largest value in the (extended) grid with random success <= 5%
   **and** strictly below the cap. When the cap, not the 5% criterion, is what limits epsilon, say so.
   Stop and ask only if the cap and the 5% rule conflict, meaning no grid value satisfies both.
6. Only random sampling and landscape geometry are used. No CMA-ES or BO result informs epsilon.
   One `epsilon_check_<function>.png` per function is saved to `docs/figures/`.

### Result of the epsilon pilot for Rastrigin and Schwefel (run 2026-10-22, `pilot/epsilon_validation.py`)

Wall-clock of the whole pilot (both functions, local-minima checks, 200 random runs each, figures): **8.2 s**.
No CMA-ES or BO run on Rastrigin or Schwefel had been made when this ran or when it was committed.

Cap (1000 starts, seed 0; the nearest non-global minimum was **unchanged at 2000 starts**: same regret and
same tied minima; 118 -> 121 distinct minima for Rastrigin and 57 -> 62 for Schwefel, all found minima beyond the nearest):

| function | nearest non-global regret (cap) | tied minima | next minima |
| --- | --- | --- | --- |
| Rastrigin | 0.994959 | 4: (+-0.995, 0), (0, +-0.995) | 1.98992 (four diagonal neighbours) |
| Schwefel | 118.438 | 2: (4.21, -3.03) and its mirror | 217.14, 236.88, 238.39 |

Random sampling (N = 300, d = 2, seeds 9000-9199), success = best regret <= epsilon:

| epsilon | Rastrigin k/200 (Wilson 95%) | Schwefel k/200 (Wilson 95%) | below cap? |
| --- | --- | --- | --- |
| 0.01 | 0 [0.000, 0.019] | 0 [0.000, 0.019] | yes |
| 0.02 | 0 [0.000, 0.019] | 0 [0.000, 0.019] | yes |
| 0.05 | 2 [0.003, 0.036] | 0 [0.000, 0.019] | yes |
| 0.1 | 3 [0.005, 0.043] | 0 [0.000, 0.019] | yes |
| 0.2 | **5 [0.011, 0.057]** | 1 [0.001, 0.028] | yes |
| 0.5 | 11 [0.031, 0.096] (5.5%) | 1 [0.001, 0.028] | yes |
| 1 | 21 [0.070, 0.155] | 1 [0.001, 0.028] | Rastrigin no (cap 0.995), Schwefel yes |
| 2 | 59 [0.236, 0.362] | 1 [0.001, 0.028] | Rastrigin no, Schwefel yes |
| 5 | 174 [0.816, 0.910] | **10 [0.027, 0.090] (exactly 5.0%)** | Rastrigin no, Schwefel yes |
| 12.5 (extension, x2.5) | n/a | 22 [0.074, 0.161] | Schwefel yes (cap 118.4) |

**Chosen (frozen in `benchmarks.py`): epsilon_Rastrigin = 0.2, epsilon_Schwefel = 5.0.** Both are limited by
the 5% rule, not the cap. Rastrigin: 0.5 gives 5.5% > 5%, so 0.2 is the largest value at or below 5%. Schwefel:
epsilon = 5 gives exactly 5.0% (10/200), which satisfies "<= 5%" by the rule as written; the upward extension
to 12.5 (11%) then confirmed it was the largest. **Caveat: Schwefel's value sits on the 5% boundary** (Wilson
95% CI [2.7%, 9.0%]), so it is a borderline application of the rule, not a comfortable one. Success implies the
global basin for both (tests `test_success_implies_global_basin`, `test_epsilon_below_nearest_nonglobal_regret`).
Figures: `docs/figures/epsilon_check_rastrigin.png`, `docs/figures/epsilon_check_schwefel.png`.

## Exp E best-variant selection rule: written 2026-10-22, BEFORE Exps A-C run

Only smoke tests (N = 100, 5 seeds) exist so far; no full A-C results.

- **Selection.** For each method, the "best variant" is the setting with the **highest success rate on
  seeds 1000-1049** across its Exp A-C settings (CMA-ES: `cma_default`, `cma_lam12`, `cma_lam24`, `cma_ipop`,
  `cma_bipop`; BO: `bo_default`, `bo_ei_xi0.01`, `bo_ei_xi0.1`, `bo_lcb_b0.5`, `bo_lcb_b2`, `bo_lcb_b8`,
  `bo_ts`, `bo_map`) on the primary landscape (`primary_far_w0.3`). The default is eligible: if it wins or ties,
  the best variant is the default and the figure says so.
- **Ties** are broken toward the setting closest to default. Operationalised as this fixed order (closest
  first): CMA-ES `cma_default`, `cma_lam12`, `cma_lam24`, `cma_ipop`, `cma_bipop`; BO `bo_default`,
  `bo_ei_xi0.01`, `bo_ei_xi0.1`, `bo_lcb_b2`, `bo_lcb_b0.5`, `bo_lcb_b8`, `bo_ts`, `bo_map`.
- **Evaluation.** The selected variant and the baseline are both re-run on **fresh seeds 2000-2049**
  (`FRESH_SEEDS` in `configs.py`) on every Exp E landscape. The 1000-1049 baseline stays as the
  baseline-only curve. The best-variant line is never drawn from seeds 1000-1049.
- The fresh-seed BO runs will be timed after A-C, and the user is told **before any seeds are cut**.

## Finding: BO runs are not reproducible across processes (found 2026-10-22, while checking Exp 0 smoke data)

The same BO run (`bo_default`, `primary_far_w0.3`, seed 1004, N = 100) executed in separate fresh processes,
one after another, single thread, gives trajectories that agree for about 65 evaluations and then diverge
(first difference > 1e-9 at evaluation 65-67, in 3 of 3 pairs). Within one process repeated runs are
identical (`test_same_seed_same_result` passes). Setting BLAS thread variables (`VECLIB_MAXIMUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) or `PYTHONHASHSEED=0` did not remove the divergence. Cause not
established. *Hypothesis, untested:* last-digit floating-point differences in the numerical libraries that
BO's feedback (GP fit -> acquisition optimum -> next point) amplifies. Visible consequence: seed 1004 at
N = 100 reached the well at evaluation 98 in one process and not in another; its N = 300 run first reached the
well at evaluation 108. CMA-ES is unaffected in practice (re-run matches saved CSV to 4e-16). Nothing in the
engine was changed. A seed fixes the initial design and random streams, not the exact path of a BO run, so
"reproducible from (method, config, landscape, seed)" holds bit-for-bit for CMA-ES and random sampling but only
statistically for BO. See open question 5.

## Exp D and Exp V best/worst variant rule: written 2026-10-22, BEFORE Exps A-C run

No full A-C results exist; no variant has been chosen. This fixes how the best and worst variants are picked.

- **Selection data.** Success rate (regret <= 0.05) on `primary_far_w0.3`, **seeds 1000-1049**, from the Exp A-C
  runs. Selection is done separately for each method (CMA-ES and BO); methods are never ranked against each other.
- **Candidates, exactly.** CMA-ES: the Exp A settings (`cma_default`, `cma_lam12`, `cma_lam24`) and the Exp B
  settings (`cma_ipop`, `cma_bipop`). BO: the Exp C settings (`bo_default`, `bo_ei_xi0.01`, `bo_ei_xi0.1`,
  `bo_lcb_b0.5`, `bo_lcb_b2`, `bo_lcb_b8`, `bo_ts`, `bo_map`). The defaults are candidates. Nothing from Exp
  D, E or V is a candidate.
- **Best** = highest success rate among the method's candidates. **Worst** = lowest success rate.
- **Tie-break (both best and worst):** the candidate closest to default, using the same fixed order as the Exp E
  rule above (CMA-ES: `cma_default`, `cma_lam12`, `cma_lam24`, `cma_ipop`, `cma_bipop`; BO: `bo_default`,
  `bo_ei_xi0.01`, `bo_ei_xi0.1`, `bo_lcb_b2`, `bo_lcb_b0.5`, `bo_lcb_b8`, `bo_ts`, `bo_map`).
- **Consequence stated in advance:** if several settings tie at the minimum success rate (for example several at
  0/50), the tie-break picks the one closest to default, so the worst variant can be the default itself, and if all
  settings tie then best = worst = default. In that case Exp D and Exp V run that single setting for the method and
  the figure says "no spread between settings".
- The best variant for Exp D and V is the same setting as the Exp E best variant for that method (selected once).
- The evaluation seeds for Exp D and Exp V are **not** part of this rule and are to be fixed before they run
  (direction.md: V uses 20-30 seeds).
- Selection uses the primary landscape and evaluation uses other landscapes, so a variant is never selected and
  evaluated on the same landscape.

## Amendments and pre-set criteria, 2026-10-22 (written BEFORE the sphere BO check, Exp 0 and any A-C run)

1. **The Exp E order is a convention, not a measure of closeness.** It is a fixed list used only to break ties
   (above), not a claim that one setting is "nearer" the default than another.
2. **All-tie rule for Exps D and V (supersedes the "run that single setting" consequence bullet above).** If every
   candidate of a method ties, so best and worst are the same setting, Exps D and V use **the default plus the
   last-listed setting in the order** for that method: CMA-ES `cma_bipop`, BO `bo_map`.
3. **Arms and evaluation seeds for Exps D and V.** Both always include `cma_default`, `cma_lam12` and `cma_lam24`
   (needed for H5 and H6), plus the best and worst variants from the rule above for each method (BO gets only
   its best and worst variants). Exp D runs on the sphere, Exp V on Rastrigin and Schwefel. **Evaluation seeds:
   CMA-ES settings 1000-1049, BO settings 1000-1019.** The random-sampling reference (`random`, seconds to run)
   is run on the same landscapes on seeds 1000-1049 and **always drawn** on Rastrigin and Schwefel figures.
   Pilot values at the frozen epsilon (200 seeds, N = 300): Rastrigin (epsilon 0.2) 5/200 = 2.5%, Wilson 95%
   [1.1%, 5.7%]; Schwefel (epsilon 5) 10/200 = 5.0%, [2.7%, 9.0%], which is **borderline** (it sits exactly on the
   5% line) with a wide interval.
4. **Sphere sanity check, pass criterion (set before the BO half runs).** Default BO (`bo_default`), sphere, d = 2,
   N = 300, pilot seeds 9000-9019 (20 runs). **BO passes if at least 19 of 20 runs reach regret <= 0.05 strictly
   before the budget is exhausted (evaluations-to-solve <= 299).** If BO fails: stop and report, do not run Exp 0.
   If it passes: report the numbers, then continue to Exp 0. (The CMA-ES half had no preset criterion; it ran first:
   20/20 reached regret <= 0.05, median 51 evaluations, range 23-75, worst final regret 9.2e-9.) A check, not a
   finding; nothing is tuned.
5. **Exp 0 definition.** Default CMA-ES, default BO and random sampling on `primary_far_w0.3`, seeds 1000-1049,
   N = 300 (`exp0` in `configs.py` now includes `random`). Existing runs are skipped: CMA-ES and BO seeds 1000-1004 are
   the earlier 5-seed N = 300 pilot runs (same code path for these configs). After the run, report: success table,
   outcome counts, and the gates (exact N per run, random sampling low at epsilon, no jitter fallbacks).
6. **Archive policy.** One zip per finished batch, named `raw_bo_<batch>.zip` with `raw_bo_<batch>_manifest.csv`,
   and an existing zip is never rewritten. The five pilot BO runs are archived as `raw_bo_pilot5.zip`; the
   Exp 0 BO runs will be `raw_bo_exp0.zip` (it contains those same five runs).

## Sphere sanity check result (run 2026-10-22, `pilot/sphere_sanity.py`, pilot seeds 9000-9019, N = 300, d = 2)

A check, not a finding; nothing tuned. "Solved" = regret <= 0.05; evaluations-to-solve counts evaluations.

| method | solved by N=300 | evals to solve: median (best, worst) | final regret: median, worst | per-run wall |
| --- | --- | --- | --- | --- |
| CMA-ES default | 20/20 | 51 (23, 75) | 4.6e-10, 9.2e-9 | ~0.1 s |
| BO default (EI, xi = 0) | 20/20 | 12 (11, 189) | 1.6e-6, 1.5e-3 | median 119 s |

**BO preset criterion (>= 19/20 solved before the budget is exhausted): PASS (20/20).** The BO half took
442.7 s wall on 6 workers under `caffeinate -dimsu` (the pre-run estimate was 6-7 min). One BO run (seed 9014)
needed 189 evaluations and ended at regret 1.5e-3; every other BO run solved within 16 evaluations.
Output files: `results/summary/sphere_sanity_cmaes.csv`, `results/summary/sphere_sanity_bo.csv`.

## Exp 0 run record (2026-10-22, `run_experiments.py --full --exp exp0`, seeds 1000-1049, N = 300)

Wall 789 s for the 145 new runs on 6 workers under `caffeinate -dimsu` (CMA-ES and BO seeds 1000-1004 reused from the
pilot). Facts only; no prediction is scored here (statuses in `docs/scoreboard.csv` are the user's).

| method | success k/50 (Wilson 95%) | stuck in decoy | unfinished | median regret |
| --- | --- | --- | --- | --- |
| CMA-ES default | 0 [0.000, 0.071] | 48 | 2 | 2.208 |
| BO default | 5 [0.043, 0.214] | 45 | 0 | 2.208 |
| random sampling | 2 [0.011, 0.135] | 11 | 37 | 1.114 |

Gates: every run has exactly 300 evaluations (150/150); random sampling at epsilon = 0.05 succeeds 2/50 (the
200-seed pilot gave 5/200); BO Cholesky jitter fallbacks 0 across 50 runs; BO final lengthscale at the upper bound
in 42/50 runs. The two CMA-ES "unfinished" runs (seeds 1006, 1029) are early samples (generation 6 and 0) that landed on
the well flank, after which sigma collapsed onto the decoy, so the label means "best point lies in the well's
basin", not "the run was converging there". BO wall per run: median 101.5 s (range 36.6-149.9). BO runs archived
as `results/archive/raw_bo_exp0.zip` (0.69 MB, 50 runs, includes the five pilot runs).

## Decisions on the remaining open questions (user, 2026-10-22: "accept all")

1. **BO lengthscale upper bound (20 domain units): kept, frozen.** It is part of the baseline configuration; the
   diagnostics (section 4 of the notebook) report how often the fitted lengthscale sits at the bound.
2. **Restart x0:** the first start draws from U[-4, 4]^d and restarts redraw from the full box U[-5, 5]^d. Confirmed.
3. **BO cross-process non-reproducibility: accepted (2026-10-22), wording fixed by the user:** "A seed fixes the initial design and random streams. CMA-ES and random sampling reproduce exactly; BO reproduces in distribution, because floating-point differences compound through the model-fitting loop. Paired comparisons are paired on starting conditions only."
   Used verbatim in the README and the notebook's limitations cell. Raw BO runs are committed compressed
   (`results/archive/raw_bo_<batch>.zip` + manifest via `run_experiments.py --archive`); `results/raw/` and
   `results/raw_fast/` stay gitignored working directories.
4. Reviewed and accepted as written: M1-M3 evidence columns, the best-variant tie-break order, and Schwefel
   epsilon = 5 (borderline, exactly 5.0% random success).

## Open questions for the user
None outstanding. (Experiments are not to be started until the user says so.)

## Amendment 2026-10-25: experiment renaming and BO redesign (written BEFORE any Exp 1-7 run)

**Renaming.** Experiments are now numbered in run order, CMA-ES together and BO together. Sections above this one keep
the old letters. Mapping: A -> Exp 1 (lambda), B -> Exp 2 (restarts), C -> Exp 3 (BO exploration vs exploitation),
new -> Exp 4 (BO kernel type), D -> Exp 5 (sphere), E -> Exp 6 (landscape knobs), V -> Exp 7 (validation). Where an
older rule says "Exps A-C" read "Exps 1-4". Config ids are unchanged. Exp 0 results are unaffected (no result file
carries an experiment letter).

**Exp 3 grid widened (user decision, no result from Exp 3 existed).** The old EI grid xi in {0, 0.01, 0.1} was small
against the well depth (about 2.2, fixed by A = 4, s = 10, w = 0.3), so it would barely change exploration. New
grids, chosen from the landscape's scale and not from any result: EI xi in {0, 0.01, 0.1, 0.5, 1, 2}; LCB beta in
{0.5, 2, 8, 32}; Thompson sampling; `bo_map`. n_init stays 10 for every arm. `bo_map` changes the lengthscale prior and
the noise treatment together, so it is labelled a different fitting recipe. Added configs: `bo_ei_xi0.5`, `bo_ei_xi1`,
`bo_ei_xi2`, `bo_lcb_b32`.

**Exp 4 added (user decision).** Kernel type with everything else at the `bo_default` baseline (EI, xi = 0, n_init = 10,
MLE with fixed tiny noise, same lengthscale bounds and start on the unit cube). Added configs: `bo_k_m32`
(Matern-3/2), `bo_k_m12` (Matern-1/2), `bo_k_rbf` (RBF). The baseline Matern-5/2 is `bo_default` and is reused.
The change to `_build_gp` keeps `bo_default` identical (kernel defaults to Matern-5/2).

**Candidate lists for the best/worst rule, updated.** Candidates are the settings of Exps 1-4, per method, defaults
included. CMA-ES order for ties (closest to default first): `cma_default`, `cma_lam12`, `cma_lam24`, `cma_ipop`,
`cma_bipop`. BO order: `bo_default`, `bo_ei_xi0.01`, `bo_ei_xi0.1`, `bo_ei_xi0.5`, `bo_ei_xi1`, `bo_ei_xi2`,
`bo_lcb_b2`, `bo_lcb_b0.5`, `bo_lcb_b8`, `bo_lcb_b32`, `bo_ts`, `bo_map`, `bo_k_m32`, `bo_k_m12`, `bo_k_rbf`. The
last-listed fallback for Exp 6 is therefore CMA-ES `cma_bipop` and BO `bo_k_rbf`. Everything else in the earlier rules
is unchanged. Exp 7 now has a BO prediction, BO5 (`docs/predictions_addendum.md`).

**Runtime (estimates from the earlier timing pilot; check in before running).** Exps 1 and 2 are CMA-ES only (seconds).
Exp 3 has 11 new BO arms: EI arms about 100 s per run, LCB about 35 s, TS about 80 s, MAP about 49 s, so about
100 minutes on 6 workers. Exp 4 has 3 new arms of EI-like cost, about 40 minutes.
