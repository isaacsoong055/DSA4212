# Bayesian optimisation versus CMA-ES: geometry and parameterisation

DSA4212 project. We ask when a black-box optimiser benefits from learning the
*orientation* of a problem rather than only its per-coordinate scales, and we
use that question to explain a difference in behaviour between Bayesian
optimisation and CMA-ES.

**Report:** `report/report.tex` (compiled PDF submitted on Canvas)

## The argument in one paragraph

CMA-ES adapts a full covariance matrix, so its search distribution can tilt to
match a valley that is not aligned with the coordinate axes. The standard
Bayesian-optimisation surrogate cannot: the usual squared-exponential kernel
uses a diagonal metric, and TuRBO's trust region is an axis-aligned
hyperrectangle whose side lengths are set per coordinate from the ARD
lengthscales. We test the resulting prediction with a 2x2 of
{population method, surrogate method} x {axis-aligned, rotation-aware}, so that
the only thing varying is whether the method can represent a rotation.

## Layout

```
code/
  problems.py                 test problems; COCO-style instances (random rotation + random optimum)
  cmaes.py                    CMA-ES with full or diagonal covariance (the ablation)
  bo.py                       GP-EI with an ARD or a full (learned) kernel metric
  analysis.py                 runtime-to-target, ECDF, paired bootstrap CIs
  run_e1_cmaes_geometry.py    E1: angle sweep, random-rotation invariance, dimension sweep
  run_e2_bo_metric.py         E2: BO vs CMA-ES head to head, 2x2 of {surrogate, population} x {axis-aligned, rotation-aware}
  run_e2c_lengthscale_diagnostic.py  E2c: read the kernel's learned geometry directly
  run_e2d_fit_diagnostic.py   E2d: is the GP fit at a bound? (diagnosed the lengthscale cap; checks the MAP fit)
  run_e2e_bo_plateau.py       E2e Stage A: why BO stalled (per-iteration diagnostics); --determinism-check
  run_e2e_stage_b.py          E2e Stage B: one change at a time (B2 fixed noise, B4 LogEI, B5 lower floor)
  run_e3_invariance.py        E3: units changes and monotone transforms of f (which invariances survive)
  run_e4_boundary.py          E4: optimum near the box edge, with a same-distance interior control
  run_e5_other_functions.py   E5: Rosenbrock (COCO f8/f9) and a noisy quadratic
  run_e6_noise_budget.py      E6: noisy evaluations -- budget split (repeats, population, reserve) and recommendation rule
  run_e7_cmaes_restarts.py    E7: do IPOP-style restarts change any CMA-ES result? (seconds)
  run_e8_matern.py            E8: the rotation result with a Matern-5/2 kernel
  run_e9_rosenbrock_long.py   E9: Rosenbrock with 4,000 CMA-ES evaluations (seconds)
  run_e10_trust_region.py     E10: TuRBO-style trust region, axis-aligned (ARD) vs rotated (full) box
  make_figures.py             rebuilds every report figure from results/ (no experiment is re-run)
results/                      CSV output of the above (one file per experiment)
report/                       report.tex, references.bib, figures/ (and the compiled report.pdf)
geometry_parameterisation_experiment.ipynb   exploratory notebook; Section 8 reloads every result quoted in the report
```

## Environment

```
python >= 3.10
numpy scipy pandas matplotlib
```

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned versions used for every result
python -m unittest discover -s tests   # 10 checks, about a second
```

The tests check the pieces every result depends on: the test problems
(optimum, spectrum, rotation), the analytic GP gradient against finite
differences, LogEI against the direct formula, CMA-ES's exact rank invariance,
and that a BO run repeats bit for bit with one BLAS thread.

`./run_all.sh [workers]` re-runs everything in order (about 6-7 hours on
8 cores), rebuilds the figures and compiles the report.

No data download is required: every objective is generated in code.

## Reproducing the results

Each script is self-contained, writes CSV into `results/`, and prints the
tables used in the report. Run order does not matter.

```
python code/run_e1_cmaes_geometry.py                 # minutes
python code/run_e2_bo_metric.py --workers 8          # original E2 (BO with fitted noise) -> results/e2_*
python code/run_e2_bo_metric.py --bofix --workers 8  # E2 with the BO fix -> results/e2_bofix_* (~1 h on 8 cores)
python code/run_e2c_lengthscale_diagnostic.py        # ~3 min
python code/run_e2d_fit_diagnostic.py                # ~10 min; --part3-only re-runs just the MAP check
python code/run_e2e_bo_plateau.py --workers 8        # Stage A, ~16 min; add --determinism-check for the repeatability test
python code/run_e2e_stage_b.py --workers 8           # Stage B, ~15 min per variant; --variants NAME runs one
python code/run_e3_invariance.py --workers 8         # ~1.5 h (BO dominates)
python code/run_e4_boundary.py --workers 8           # ~1 h
python code/run_e5_other_functions.py --workers 8    # ~45 min
python code/run_e6_noise_budget.py --workers 8       # ~30 min
python code/run_e2_bo_metric.py --refit-local --workers 8   # original BO on this machine -> results/e2_local_* (~2.5 h)
python code/run_e2_bo_metric.py --local-cand --workers 8    # BO fix + local candidates -> results/e2_lc_* (~1.5 h)
python code/run_e7_cmaes_restarts.py                 # seconds
python code/run_e8_matern.py --workers 8             # ~1.5 h
python code/run_e9_rosenbrock_long.py                # seconds
python code/run_e10_trust_region.py --workers 8      # ~15 min
python code/make_figures.py                          # seconds; writes report/figures/
cd report && latexmk -pdf report.tex                 # compiles the report
```

Times are for 8 cores. The long sweeps checkpoint to `results/*_partial.jsonl`
and resume if interrupted; `--report-only` rebuilds the tables from the checkpoint.

**Repeatability.** The scripts pin BLAS to one thread per worker
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`). With that,
a BO run is bit-identical on one machine; with multithreaded BLAS two runs of
one seed differ by about 1e-4 relative, and the difference compounds over 200
iterations (`results/e2e_determinism.csv`). Across machines, results agree in
distribution, not run for run. The original E2 (`results/e2_*`) was produced
before the thread pinning, on another machine.

Randomness is controlled by the instance index, which seeds both the problem
(rotation, optimum location) and the optimiser.

## Measurement conventions

* **Runtime is counted in objective evaluations**, not seconds, as the primary
  measure; it is independent of platform, language and coding style. Wall-clock
  seconds are recorded alongside, because the cost of *choosing* an evaluation
  is exactly what separates these two method families.
* **Several target precisions**, not one, and results aggregated as an ECDF
  over (problem, target) pairs.
* **Censored runs are reported, not dropped.** Every runtime summary is
  accompanied by a success rate, because the median of the successful runs
  flatters a method that often fails.
* **Comparisons are paired** over a shared instance list, and the reported
  uncertainty is the uncertainty in the *difference* between methods.
* **The recommendation is scored on a common objective.** Under a monotone
  transform of the objective value the reported numbers are not comparable, so
  the returned point is re-scored on the untransformed function.

## GP fitting (bo.py)

Kernel hyperparameters are fitted by MAP: the log marginal likelihood plus the
dimension-scaled LogNormal lengthscale prior that BoTorch uses by default
(Hvarfner et al., ICML 2024), shifted for our box width, with exact analytic
gradients and 10 restarts. The prior sits on the lengthscales along the
metric's eigen-directions, so it does not favour axis-aligned metrics. It
replaced plain maximum likelihood after E2d showed the old hard bound
(lengthscale <= e^4) set the fitted lengthscale ratio at high condition numbers.
`fit_gp(..., prior=None)` restores the old fit; `results/e2c_lengthscales_mle_legacy.csv`
holds its output.

## The BO fix (E2e)

The original BO fitted a noise variance. On noise-free objectives the GP used
it to explain the best observations away as noise, so EI collapsed and BO
sampled almost at random (E2e Stage A). From E2e onwards every BO run fixes the
noise variance at 1e-6 on the standardised scale (`bayes_opt(..., noise_variance=1e-6)`);
the default (fitted noise) is kept so the original E2 stays reproducible.
`bayes_opt` also accepts `acquisition="logei"` and `target_floor=` (Stage B
variants, not adopted).

## Budget rule for the cross-family comparison (E2-E5)

Both families are measured in objective evaluations, counted by wrapping the
objective, with a shared cap of 200. Every evaluation is logged, so each run is
read both as runtime to target (f - f* <= 1e-1, 1e-3, 1e-5) and as best-so-far
at fixed budgets (20, 50, 100, 200). CMA-ES is also continued to 2000 x d
evaluations in separately labelled runs. Both methods are given the same box:
BO searches it, CMA-ES starts at its centre with sigma = 0.3 x box width.

## Known scope limits

The CMA-ES implementation is a direct transcription of the standard equations
and deliberately omits restarts and noise-specific machinery; its only box
handling is the simplified reflection transform tested in E4. All BO results
are 2-D. BO is given a search box and CMA-ES is not; that asymmetry is
recorded with every result rather than hidden, since a box is information about
where the optimum lies.

## References

* Frazier, *A Tutorial on Bayesian Optimization*, 2018.
* Hansen, *The CMA Evolution Strategy: A Tutorial*.
* Eriksson et al., *Scalable Global Optimization via Local Bayesian Optimization* (TuRBO), NeurIPS 2019.
* Hansen et al., *COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting*.
