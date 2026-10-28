# Multimodality: Project Direction Guide (BO vs CMA-ES)

Oct 22, 2026 · @isaac

## 1. Research question

At a fixed evaluation budget, how do population size, restarts and acquisition choices change the probability that CMA-ES and BO find a narrow or distant global basin, and why?

The story in one line: show where each baseline fails and measure the failure, change one design choice at a time, then explain each result with evidence and honest uncertainty.

Practical relevance (use in the intro and conclusion): hyperparameter-tuning and loss landscapes often have many basins, and the global one may be narrow. The conclusion should end with a recommendation of the form "use X when Y".

## 2. Theory to write up

Experiments alone will not earn the formulation marks, so write the mechanism behind every knob and cite it (Hansen 2016 for CMA-ES, Frazier 2018 for BO and EI, Hansen et al. 2020 for COCO benchmarking practice). Add one or two sources on IPOP/BIPOP and on exploration parameters.

| Knob | Mechanism to explain |
| --- | --- |
| CMA-ES step-size collapse | Cumulative step-size adaptation shrinks sigma when consecutive steps are anticorrelated. Inside a basin the steps oscillate, so sigma contracts and the search cannot leave. |
| Population size (lambda) | A larger lambda raises the effective selection mass mu\_eff. Ranking more samples averages out local bumps, so the update follows the global trend. The cost is fewer generations at fixed N. |
| Restarts (IPOP / BIPOP) | IPOP doubles lambda after each restart (depth traded for breadth). BIPOP mixes large and small populations. Each restart resets sigma but spends budget. |
| BO exploration | EI and LCB weigh the posterior mean against the posterior variance differently. xi and beta are the explicit exploration knobs. |
| BO refinement failure | If the fitted lengthscale is much longer than the well width w, the GP smooths over the narrow well and gives it low uncertainty, so it is never sampled. This links to the MLE-overfitting limitation. |

CSA step-size update:

```latex
\sigma \leftarrow \sigma \exp\left[\frac{c_\sigma}{d_\sigma}\left(\frac{\lVert p_\sigma \rVert}{E\lVert N(0,I)\rVert} - 1\right)\right]
```

Effective selection mass, and the budget trade-off at fixed N:

```latex
\mu_{\mathrm{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}, \qquad \text{generations} \approx N / \lambda
```

Exploration-weighted acquisition functions, with z = (y+ - mu - xi) / sigma:

```latex
\mathrm{EI}_\xi(x) = (y^+ - \mu(x) - \xi)\,\Phi(z) + \sigma(x)\,\varphi(z), \qquad \mathrm{LCB}(x) = \mu(x) - \sqrt{\beta}\,\sigma(x)
```

## 3. Pre-registered predictions

Commit this table to the repo before the big runs, so the analysis is defensible whichever way results go. Report every outcome, and never choose or tune a setup to produce a particular result.

| ID | Prediction | Supported by | Contradicted by |
| --- | --- | --- | --- |
| H1 | Default CMA-ES gets stuck in a decoy basin | Low success, early sigma collapse, best point outside the global basin | High success, or sigma stays large |
| H2 | Larger lambda raises success, with a sweet spot at fixed N | Success vs lambda rises, then plateaus or falls | Monotone decline, or no change |
| H3 | Restarts beat a larger lambda at equal total budget | Higher success at the same N | Equal or lower success |
| H4 | BO finds the narrow basin at low d only if it explores enough | Success rises with xi or beta | Insensitive to xi and beta |
| H5 | Larger lambda gives no benefit, and a small cost, on a unimodal control | Sphere regret equal or worse | Sphere benefits too (suspect an extra-samples effect) |
| H6 | Lambda helps more on a funnel landscape (Rastrigin) than on a deceptive one (Schwefel) | Larger gain on Rastrigin | Same gain on both |

A contradicted prediction, explained with your diagnostics, shows more understanding than a clean confirmation.

**Addendum (2026-10-25).** `docs/predictions_addendum.md` adds BO1 to BO5 so that BO has its own failure, mechanism and test chain, like CMA-ES (H1, M1, H2), and CMA1 to CMA3 for the step-size experiment (Exp 3) and the width sweep (Exp 8). BO1 is post hoc (Exp 0 was seen first); CMA1 comes from a pilot diagnostic and is tested on fresh experiment seeds; the rest are forward-looking. The addendum also holds the prediction-to-experiment map. No H or M prediction was changed.

## 4. Step 1: build the landscapes

Use one tunable primary landscape for all algorithm experiments, one control and two validation functions chosen in advance (one funnel, one deceptive), so the validation is informative however it turns out.

| Role | Function | Why |
| --- | --- | --- |
| Primary (tunable) | Broad decoy bowl plus a narrow deep Gaussian well | Width w and distance are knobs, matching "narrow or distant basin" directly |
| Control | Sphere | Isolates the multimodality effect (H5) |
| Validation 1 (funnel) | Rastrigin, d=2 | Global structure guides the search |
| Validation 2 (deceptive) | Schwefel, d=2, rescaled to \[-5, 5\] | Best basin is far from its neighbours |

Primary landscape:

```latex
f(x) = \frac{\lVert x - c \rVert^2}{s} - A \exp\left(-\frac{\lVert x - x_g \rVert^2}{2 w^2}\right)
```

Setup details:

- Domain \[-5, 5\]^d, with d=2 as the main setting.
- Place the global basin away from the origin so random starts do not land in it by luck.
- Success threshold epsilon: run a short pilot and set it so that pure random sampling at N=300 rarely succeeds, while a point at the global basin floor clearly does. Plot the threshold on the function and keep that plot in the appendix, since markers will ask.
- External validity: repeat your single most important result (success vs lambda) at d=5 with a funnel-plus-decoy function, so the finding is not a 2D artifact.

## 5. Step 2: baselines and diagnosis (Exp 0)

Run default CMA-ES (the `cma` library) and default BO (GP, Matern-5/2, EI) on the primary landscape at N=300, with 30-50 seeds each. Reuse the same seeds for every variant later.

Log to CSV:

- All runs: seed, eval index, x, f, best-so-far, optimizer time.
- CMA-ES also: sigma, covariance trace, generation index.
- BO also: fitted lengthscales.

Diagnose with measurements, not impressions:

- CMA-ES is "stuck" if sigma collapses early and the best point is outside the global basin.
- BO "explores but does not refine" if samples stay spread out and regret plateaus.
- BO "smooths over the well" if the fitted lengthscale is much larger than w. This directly tests the mechanism from section 2.

Outputs: success rate with a confidence interval, regret curves, a sigma-vs-generation plot and a sample scatter overlay.

## 6. Step 3: one-factor experiments

Change one factor at a time. Every variant gets the same N and seeds, and all restart evaluations count against N.

Experiments are numbered in the order they run and grouped by method (renamed 2026-10-25 from A-E and V; old names appear only in older sections of `docs/pilot_log.md`).

| Exp | Block | Factor | Settings | Measure | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | both | Baselines | default CMA-ES, default BO, random sampling | Success rate, diagnostics | Done |
| 1 | CMA-ES | Population size lambda | 1x, 2x, 4x default (d=2: 6, 12, 24) | Success vs lambda, generations completed | Check the largest lambda still gets 10-20 generations. Side check at N = 1500 (done, CMA-ES only, kept off BO plots) |
| 2 | CMA-ES | Restarts | None, IPOP, BIPOP | Success rate, which restart succeeded | Total budget fixed at N. IPOP and BIPOP coincide at N = 300 (one restart per run) |
| 3 | CMA-ES | Initial step size sigma0 | 0.3, 0.5, 1, 2 (default), 4, from the baseline random start; and the same five from an oracle start 1.0 from the well | Success vs sigma0 (and sigma0 / w) | Added 2026-10-25 after Exps 0-2 all gave 0/50 and a positive control pointed at sigma0 = 2 against w = 0.3. The oracle arm uses knowledge of the well: a diagnostic of refinement, not a method comparison, never plotted against BO. Two follow-ups (added after the first Exp 3 results): a mean trace of the oracle arm (CMA4) and a ratio test, oracle start at (10/3) w with sigma0 = k w for k in {1, 3, 7, 20} on w in {0.1, 0.3, 0.6, 1.0} (CMA5), both diagnostics |
| 4 | BO | Explore/exploit inside EI | EI with xi in {0, 0.01, 0.1, 0.5, 1, 2} | Success rate, success given a well sample, sample spread, fitted lengthscale | n_init fixed at 10. xi is in units of f; the well is about 2.2 deep, so the grid reaches near-pure exploration. Refit every 5 iterations. Run in one batch with Exp 5 and `bo_map` (`--exp exp4_5`), split for reporting 2026-10-28 |
| 5 | BO | Acquisition family | LCB with beta in {0.5, 2, 8, 32}; Thompson sampling; EI xi = 0 as baseline | Same as Exp 4 | A different factor from xi (the acquisition function itself) |
| 6 | BO | Surrogate | Kernels Matern-5/2 (baseline), Matern-3/2, Matern-1/2, RBF; and `bo_map` (different fitting recipe, see below) | Success rate, success given a well sample, fitted lengthscale | Everything else at baseline (EI, xi = 0, n_init = 10). The `bo_map` arm was run with Exp 4 |
| 7 | both | Unimodal control | Best and worst settings from Exps 1-6 on the sphere, per method | Regret | Tests H5 |
| 8 | both | Landscape knobs | w in {0.1, 0.3, 0.6, 1.0}; near vs far | Success vs w for the baseline and best variant of each method | Headline figure. Tests CMA3 (success rises as w approaches sigma0) |
| 9 | both | Validation | Best and worst variant from Exps 1-6 on Rastrigin and Schwefel, CMA-ES and BO selected and evaluated separately, 20-30 seeds | Success rate | Tests H6 (CMA-ES) and BO5 (BO) |

`bo_map` differs from `bo_default` in two things at once: a prior on the lengthscale (so it cannot sit at the bound) and a learned noise level instead of a fixed tiny one. It is therefore a different fitting recipe, not a single-factor change, and is reported as such.

Prediction-to-experiment map: H1 (Exp 0), H2 and M1 (Exp 1), H3 and M2 (Exp 2), CMA1, CMA2, CMA4 and CMA5 (Exp 3), CMA3 (Exp 8), BO1 (Exp 0), H4 and BO2 (Exp 4), BO2 (also Exp 5), BO3 and BO4 (Exp 6), H5 (Exp 7), H6, M3 and BO5 (Exp 9). (Renumbered 2026-10-28 when Exp 4 was split; the addendum table keeps the original numbering.) The full table is in `docs/predictions_addendum.md`.

Budget note: keep N=300 for every CMA-ES vs BO comparison. A longer-budget CMA-ES-only run is fine for the large-lambda sweep, but never plot it on the same axes as BO.

## 7. Metrics, uncertainty and seeds

Success rate is binary, so report it with Wilson or bootstrap intervals, not mean plus or minus SE.

| What | How to report |
| --- | --- |
| Success rate (binary) | Wilson or bootstrap 95% CI over 30-50 seeds |
| Regret curves | Median with quartiles, or bootstrap CI (regret is heavy-tailed) |
| Mechanism evidence | Sigma, best-point location, sample spread, fitted lengthscale |
| Compute | Optimizer time per run |

Rules for claims:

- Claim a difference only when the CIs separate. Otherwise write "not distinguishable at this seed count".
- Report effect sizes, not only whether intervals separate.
- Separate statistical uncertainty (few seeds) from mechanistic uncertainty (a plausible explanation you did not test). Label untested explanations as hypotheses.
- If a variant surprises you, check for a bug or configuration slip before building an explanation around it.

Seed strategy. A Wilson 95% interval at p near 0.5 has a half-width of about 14% at 50 seeds, 10% at 100 and 7% at 200, so halving the width costs 4x the seeds.

- Start with 50 seeds for everything.
- Add seeds (to 100-200) only where intervals overlap and the difference matters to your story. CMA-ES runs take seconds, so go to 200+ there; BO at 1-3 minutes per run is the bottleneck.
- Fix the seed list in advance and extend it by appending, never by rerunning until a result looks good.
- More seeds cannot fix a wrong threshold, a misconfigured BO, or a confounded experiment.
- State in the report how many seeds each comparison used and why.

## 8. Fairness checklist

Markers will probe each of these, so tick them off before the big runs.

- [ ] Same N for every method in any comparison, with restarts counted
- [ ] Same seeds across variants
- [ ] BO configuration (initial design size, kernel, acquisition optimizer) frozen before the big runs
- [ ] Number of BO initial points stated, and hyperparameter refit schedule stated
- [ ] Success threshold epsilon reported, with how it was chosen
- [ ] Pre-registered predictions committed to the repo before the big runs
- [ ] Main setting (d=2) and its limits stated clearly

## 9. How to write each finding

For every experiment use four lines: prediction, result (with CI), explanation, caveat.

Draw explanations from section 2:

- Sigma collapse and early contraction (CMA-ES stuck)
- Selection averaging through mu\_eff (why a larger lambda helps)
- The generations-versus-lambda trade-off at fixed N
- The posterior-variance and lengthscale story for BO
- Landscape structure (funnel vs deceptive) for whether a finding transfers

If a result contradicts a prediction, say so plainly, check for bugs first, then explain it using your diagnostics. Label any explanation you did not test as a hypothesis.

## 10. Page budget and key figures

This direction gets roughly 1.5-2 pages of results in the 10-page report, so pick 4-5 figures that each carry one message. Everything else goes in the appendix or the repo.

1. Baseline diagnosis: sigma collapse for CMA-ES and sample scatter for BO.
2. Success vs lambda on the primary landscape, with the sphere control and validation functions as panels.
3. Restarts and BO exploration: success-rate bars with CIs, side by side.
4. Headline: success vs basin width, one curve per method, including the best variants.
5. d=5 check, if it fits. Otherwise one sentence in the text.

Report structure for this section: problem and predictions; landscapes and success definition; baseline results and diagnosed failures; one subsection per intervention (prediction, result, explanation, caveat); control and validation; summary of which fixes helped where and why; limitations.

## 11. Two-week timeline and cut order

&#91;embedded content: two-week plan · 5 phases, 2 gates\]

The BO configuration and predictions freeze in days 1-2, the overnight BO runs sit in days 5-8, and days 11-14 are for writing and the clean-clone test.

If you fall behind, cut in this order:

1. d=5 check
2. Thompson sampling
3. 8x and 16x lambda sweeps
4. Exp 8 near-vs-far distance (keep the width sweep)

## 12. Limitations and common pitfalls

State these limitations honestly in the report:

- Low dimension (mainly d=2)
- One primary custom landscape, plus two standard validation functions
- Finite seeds, so some differences will not be resolvable
- Untested mechanisms are hypotheses

Pitfalls to avoid:

- Epsilon too loose (random sampling succeeds) or too tight (nobody succeeds). Pilot it first.
- Changing the BO configuration after the big runs
- Letting restarts use extra budget
- Too few seeds for a binary metric (use 30-50, not 15)
- Choosing a validation function because you expect a particular result
- Presenting untested explanations as facts
