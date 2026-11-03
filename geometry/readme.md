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
  run_e3_invariance.py        E3: which invariances survive implementation
results/                      CSV output of the above (one file per experiment)
report/                       report source, bibliography, figures
geometry_parameterisation_experiment.ipynb   original exploratory notebook (kept for provenance)
```

## Environment

```
python >= 3.10
numpy scipy pandas matplotlib
```

```
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib
```

No data download is required: every objective is generated in code.

## Reproducing the results

Each script is self-contained, writes CSV into `results/`, and prints the
tables used in the report. Run order does not matter.

```
python code/run_e1_cmaes_geometry.py     # ~minutes
python code/run_e2_bo_metric.py --workers 2   # ~3-4 h on 2 cores (GP fitting dominates); resumes if interrupted
python code/run_e2c_lengthscale_diagnostic.py  # ~3 min
python code/run_e2d_fit_diagnostic.py          # ~10 min; --part3-only re-runs just the MAP check
python code/run_e3_invariance.py         # ~tens of minutes
```

Randomness is controlled by the instance index, which seeds both the problem
(rotation, optimum location) and the optimiser. Re-running a script reproduces
its CSV exactly.

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

## Budget rule for the cross-family comparison (E2)

Both families are measured in objective evaluations, counted by wrapping the
objective, with a shared cap of 200. Every evaluation is logged, so each run is
read both as runtime to target (f - f* <= 1e-1, 1e-3, 1e-5) and as best-so-far
at fixed budgets (20, 50, 100, 200). CMA-ES is also continued to 2000 x d
evaluations in separately labelled runs. Both methods are given the same box:
BO searches it, CMA-ES starts at its centre with sigma = 0.3 x box width.

## Known scope limits

The CMA-ES implementation is a direct transcription of the standard equations
and deliberately omits restarts, box-constraint handling and noise-specific
machinery. BO is given a search box and CMA-ES is not; that asymmetry is
recorded with every result rather than hidden, since a box is information about
where the optimum lies.

## References

* Frazier, *A Tutorial on Bayesian Optimization*, 2018.
* Hansen, *The CMA Evolution Strategy: A Tutorial*.
* Eriksson et al., *Scalable Global Optimization via Local Bayesian Optimization* (TuRBO), NeurIPS 2019.
* Hansen et al., *COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting*.
