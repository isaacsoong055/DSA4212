# Pre-registration addendum: BO1 to BO5 and CMA1 to CMA3

**Date: 2026-10-25.** `docs/predictions.md` (H1 to H6, M1 to M3) is unchanged and cannot be edited. This file adds
five BO predictions so that BO has its own failure -> mechanism -> test chain, parallel to CMA-ES, and (below) three CMA-ES predictions, CMA1 to CMA3, for the step-size experiment and the width sweep.

**Naming.** H = a headline prediction, M = a mechanism (why), both from `predictions.md`. BO1 to BO5 are the BO
predictions in this file. The number is only a label; the experiment that tests each one is in the last column of the
map below. (Earlier drafts of this file called BO1 "B1", BO2 "M4", BO3 "M5", BO4 "K1"; they were renamed the same day,
before any BO result beyond Exp 0 existed.)

**Honesty note.** The Exp 0 BO results (5/50 success, lengthscale at the bound in 42/50 runs, no well sample in the 45
failures) were seen *before* this addendum. **BO1 is therefore post hoc**: it describes Exp 0, it is not a forecast.
BO2 to BO5 are written before Exp 4, Exp 5 and Exp 8 have any result. All are reported whichever way they go.

| ID | Prediction | Supported by | Contradicted by | Tested in |
| --- | --- | --- | --- | --- |
| BO1 | (post hoc, from Exp 0) Default BO fails at discovery, not refinement: it rarely samples the well, and while it has not sampled it the fitted lengthscale sits far above w | Runs with no well sample fail; runs that sampled the well reach epsilon; lengthscale before the first well sample is many times w | Runs that sampled the well still fail, or the pre-discovery lengthscale is comparable to w | Exp 0 |
| BO2 | Exploration (xi, beta, Thompson) helps BO only by raising the chance that a sample lands in the well (discovery); it does not work by repairing the lengthscale. The fitted lengthscale before the first well sample stays long under every variant and shrinks only after a well sample | Success of a variant tracks the share of its runs with a well sample; pre-discovery lengthscale is as long as the baseline's in every variant | A variant raises success while its pre-discovery lengthscale is clearly shorter than the baseline's (exploration changes the surrogate itself) | Exp 4 (companion to H4) |
| BO3 | The long fitted lengthscale is a symptom of not having found the well, not its cause. `bo_map` (same Matern-5/2 kernel, lengthscale prior and learned noise, MAP fit) fits a clearly shorter lengthscale than `bo_default`, yet its success rate is not distinguishable from `bo_default`'s. `bo_map` changes two things at once, so this is a weak test of the cause | `bo_map` pre-discovery lengthscale well below the bound, and its Wilson interval overlaps `bo_default`'s | `bo_map` success interval separates above `bo_default`'s (the lengthscale or the noise then matters, or a shorter lengthscale raises exploration through higher posterior variance) | Exp 4 (`bo_map` arm) |
| BO4 | Kernel type does not change BO's success rate on the primary landscape: the failure is at discovery, and before any sample lands in the well every kernel sees the same smooth bowl. A rougher kernel (Matern-1/2) may help only after a well sample exists, so success given a well sample is reported separately | Wilson intervals of Matern-3/2, Matern-1/2 and RBF all overlap the Matern-5/2 baseline | At least one kernel's interval separates from the baseline's | Exp 5 |
| BO5 | The spread between BO's best and worst settings (from Exps 4 and 5) is larger on Rastrigin (global funnel a GP can learn) than on Schwefel (deceptive: the best basin is far from its neighbours). This mirrors H6 for CMA-ES lambda | Larger best-minus-worst gap on Rastrigin than on Schwefel | Same or larger gap on Schwefel | Exp 8 |

## CMA-ES additions (added 2026-10-25, before Exp 3 and Exp 7 ran)

Exps 0, 1 and 2 all gave 0/50 for CMA-ES, and a positive control (`pilot/cma_positive_control.py`, pilot seeds 9000-9019,
`docs/pilot_log.md`) showed that CMA-ES started *at* the well still fails with the frozen sigma0 = 2 (0/20) but succeeds
with sigma0 = 0.3 (20/20). That diagnostic was post hoc and run on pilot seeds. CMA1 turns it into a prediction tested on the
experiment seeds (1000-1049); CMA2 and CMA3 are forward-looking.

| ID | Prediction | Supported by | Contradicted by | Tested in |
| --- | --- | --- | --- | --- |
| CMA1 | CMA-ES fails to refine once it is near the well because sigma0 is too large relative to the well width w, not because it cannot find the well. Started 1.0 from the well (oracle start), success falls as sigma0 grows: high at sigma0 = 0.3 and 0.5, near zero at sigma0 = 2 and 4 | Wilson interval of sigma0 = 0.3 lies above that of sigma0 = 2, and success does not rise with sigma0 | The sigma0 = 0.3 and sigma0 = 2 intervals overlap | Exp 3, oracle arm |
| CMA2 | From the baseline random start (U[-4,4]^2) no sigma0 gives high success: a small sigma0 rarely reaches the well (finding fails) and a large sigma0 cannot refine it. Every sigma0 arm stays below 12/50 successes (below the point where its Wilson interval lies wholly above random sampling's) | All five arms have fewer than 12/50 successes | At least one arm has 12/50 or more | Exp 3, random-start arm |
| CMA3 | For default CMA-ES success rises as the well widens (sigma0 / w shrinks): success at w = 1.0 is higher than at w = 0.3, with non-overlapping Wilson intervals, and at w = 0.1 it is not higher than at w = 0.3. Random sampling is the reference at each w because a wider well also helps random search | w = 1.0 interval wholly above w = 0.3 interval | w = 1.0 interval overlaps w = 0.3's | Exp 7 |

### Follow-up tests of the CMA-ES explanation (added 2026-10-25, after Exp 3 ran and BEFORE the two tests below ran)

Exp 3 showed the dependence on sigma0 (CMA1, CMA2 on the supported side) but not the mechanism or the ratio. Two follow-ups,
both CMA-ES only and run with the frozen settings. The thresholds below were set after seeing Exp 3's *success counts*
(33/50 at sigma0 = 0.3, 2/50 at sigma0 = 2 in the oracle arm) but before any mean trace or any width-scaled run existed.

| ID | Prediction | Supported by | Contradicted by | Tested in |
| --- | --- | --- | --- | --- |
| CMA4 | The refinement mechanism is visible in the search distribution's mean. In the oracle arm with sigma0 = 0.3 the mean ends within w of the well in at least 60% of runs; with sigma0 = 2 it ends within w in at most 15% of runs, and in the sigma0 = 2 runs that touched the well (a sample with d < w) the mean ends closer to the decoy centre than to the well in more than half of them | Both conditions hold | The sigma0 = 2 share within w exceeds 15%, or in the touched sigma0 = 2 runs the mean ends closer to the well in at least half; or the sigma0 = 0.3 share is below 60% | Exp 3, oracle arm, mean trace (`--trace`) |
| CMA5 | Refinement depends on the ratio sigma0 / w, not on sigma0 or w separately. With an oracle start at (10/3) w and sigma0 = k w, success falls with k at every width (k = 1 above k = 20 with non-overlapping Wilson intervals at each of w = 0.1, 0.3, 0.6, 1.0), and at k = 1 the four widths are not separated (all four Wilson intervals overlap pairwise) | Both conditions hold | k = 1 is not above k = 20 at some width, or at k = 1 two widths have non-overlapping intervals (success then depends on w itself, not only on the ratio) | Exp 3 ratio test (`exp3_ratio`) |

CMA5 assumes the success threshold (epsilon = 0.05) corresponds to roughly the same fraction of the well width at every w, which
holds approximately (the success radius is about 0.16 w when the bowl term is small); the bowl scale s = 10 is fixed, so an exact
collapse is not expected and moderate deviations at the extremes will be reported, not hidden. At w = 0.3 the ratio arms
(sigma0 = 0.3, 0.9, 2.1, 6; start 1.0 away) overlap the Exp 3 oracle arm, which is a consistency check, not a new fact.

CMA1 and CMA2 together separate *finding* from *refining*: if CMA1 holds and CMA2 holds, the failure at the baseline is a
trade-off in sigma0 (too small to find, too large to refine) that no setting of sigma0 alone fixes from a random start.
The oracle start uses knowledge of the well, so the oracle arm is a diagnostic only: it is never compared with BO and is
not a candidate in the best/worst rule. Exp 3's random-start arms are candidates.

## Prediction-to-experiment map (all predictions)

| Block | Exp | Predictions tested |
| --- | --- | --- |
| both | 0 baselines | H1 (CMA-ES), BO1 (BO) |
| CMA-ES | 1 population size lambda | H2, M1 |
| CMA-ES | 2 restarts | H3, M2 |
| CMA-ES | 3 initial step size sigma0 (random start, oracle start, mean trace, ratio test) | CMA1, CMA2, CMA4, CMA5 |
| BO | 4 exploration vs exploitation | H4, BO2, BO3 |
| BO | 5 kernel type | BO4 |
| both | 6 unimodal control | H5 |
| both | 7 landscape knobs | CMA3 (CMA-ES); none for BO |
| both | 8 validation | H6, M3 (CMA-ES), BO5 (BO) |

## The two lengthscale explanations (what BO2 and BO3 are about)

Two explanations fit Exp 0 and cannot be told apart from it, because runs with a long lengthscale are exactly the runs
without a well sample:

- **A (cause):** a lengthscale much longer than w smooths over the well, gives it low uncertainty, so EI never samples it.
- **B (symptom):** a run that has not sampled the well sees only the smooth bowl, so maximum likelihood picks a long
  lengthscale. The long lengthscale follows from missing the well.

BO3 leans to B, but its test is weak because `bo_map` changes the lengthscale prior and the noise treatment together.
A shorter lengthscale also raises posterior variance away from data, so a separated interval is a live outcome. The
cause-versus-symptom question stays open after Exp 4; do not claim more than the data allow. BO2 covers the acquisition
variants (xi, beta, Thompson) and does not apply to `bo_map` by construction.

## Exp 4 and Exp 5 design decisions made on 2026-10-25 (before either ran)

- xi grid {0, 0.01, 0.1, 0.5, 1, 2} was chosen from the landscape's depth (the well is about 2.2 below the decoy floor,
  fixed by the definition A = 4, s = 10, w = 0.3), not from any result. beta grid {0.5, 2, 8, 32}.
- Initial design size is fixed at n_init = 10 for every BO arm. It is not varied.
- Kernels for Exp 5: Matern-3/2, Matern-1/2, RBF against the Matern-5/2 baseline. The lengthscale bounds and starting
  value are the same on the unit cube for all kernels, so lengthscale values are not strictly comparable across kernels
  (a caveat for the report).

## Renumbering note (2026-10-28)

Exp 4 was split after its runs finished (see `docs/pilot_log.md`, 2026-10-28). Experiment numbers in the tables above are
the numbers at the time of writing. Current numbers: H4 and BO2 are tested in Exp 4 (EI xi) and Exp 5 (LCB beta, Thompson
sampling); BO3 (the `bo_map` arm) and BO4 are tested in Exp 6 (surrogate); H5 in Exp 7; CMA3 in Exp 8; H6, M3 and BO5 in
Exp 9. No prediction text or criterion changed.

## BO6: does a large xi leave a well it has been given? (added 2026-10-28, before the runs)

Written after Exp 4 and its follow-up (`bo_after_discovery.csv`) were seen, so it is a follow-up test of a mechanism claim
made from those results, not a forward-looking prediction. It is written, and the code is committed, before any
`exp4_oracle` run exists.

**Claim being tested (from Exp 4).** EI with xi >= 0.5 finds the well in about 85% of runs but succeeds in few (5/40, 3/42 of
the runs that touched it). The explanation offered: after the first well sample, large-xi EI sends samples to the box edges
instead of refining the well. The alternative: failure to refine comes from how the well was found (late, around evaluation
120-160, with few evaluations left) and not from xi itself.

**Diagnostic arms (`exp4_oracle`, never compared with the real runs, never a best/worst candidate).** `bo_o_xi0`,
`bo_o_xi0.1`, `bo_o_xi2`: EI with xi = 0, 0.1, 2; everything as in Exp 4 except that the first of the 10 initial points is
replaced by one 0.2 from the well centre (random direction; f about -1.4, as at the first well samples in Exp 4: the distance
was chosen from those first touches, before this run). The budget is unchanged (that point is one of the 10), 50 seeds
(1000-1049), N = 300, the same landscape. So every run starts with the well sampled once, and the question is what each xi
does next.

| ID | Prediction | Supported if | Contradicted if |
|---|---|---|---|
| BO6 | Given a well sample at evaluation 0, refinement still depends on xi: success at xi = 0 is clearly above success at xi = 2, and at xi = 2 the share of later samples within w of the well stays below 0.10 | xi = 0 success minus xi = 2 success >= +0.30 with the Newcombe interval above zero, and xi = 2 in-well share after the first well sample < 0.10 (median over runs) | The difference interval includes zero or xi = 2 succeeds at least as often as xi = 0, or xi = 2 puts 0.10 or more of its later samples in the well. Then large xi leaves a well it has been given only because of late or sparse discovery, and the Exp 4 explanation is wrong |

Secondary, reported without a threshold: xi = 0.1 against the other two, success as a rate with Wilson intervals, the
box-edge share and in-well share after evaluation 10, the evaluation of the first well sample (0 by construction), and the
fitted lengthscale after the first 20 evaluations. Limits stated now: the arms differ from the real runs in having ~290 evaluations
after the first well sample instead of ~150; the xi grid has three points; one landscape.

## BO7 to BO9: BO fitting settings, one at a time (added 2026-10-28, before the runs)

Written after Exp 4, its follow-ups and BO6 were seen, so they are not blind to the BO story so far; they are fixed before any
`exp6` arm other than `bo_map` has run. `bo_map` (all settings changed together) stays as a reference and is not part of these
tests. Each arm differs from `bo_default` in exactly one setting (checked by a test). Grids were fixed from the landscape (box
width 10, well width 0.3), not from results. One detail: when the lengthscale cap is below 4 domain units, the lengthscale start
becomes cap / 2 instead of 2, so that it starts inside the allowed interval. The earlier lengthscale discussion (cause or
symptom) is in the sections above (BO2, BO3).

| ID | Prediction | Supported if | Contradicted if | Tested in |
|---|---|---|---|---|
| BO7 | The long fitted lengthscale is a symptom, not the cause, of BO's failure to find the well. Capping the lengthscale at 5, 2, 1, 0.5 or 0.25 domain units does not raise success. The tightest caps (0.5, 0.25, near or below the well width) make the GP uninformative about the bowl and do not help either | No cap arm's Newcombe 95% interval for the difference from `bo_default` is entirely above zero | At least one cap arm's interval is entirely above zero (over-smoothing then contributes to the failure) | Exp 6 (cap arms) |
| BO8 | Fixed noise does not raise success: a noise variance of 1e-3, 1e-2, 1e-1 or 1 (standardised units) does not make BO find the well more often than noise 1e-6. At noise 1 the GP treats all the variance as noise and cannot resolve the well | No noise arm's interval is entirely above zero; the noise = 1 arm is not above the baseline | At least one noise arm's interval is entirely above zero | Exp 6 (noise arms) |
| BO9 | Neither a fitted noise level alone nor BoTorch's lengthscale prior alone changes BO's success, because before the first well sample the data show a smooth bowl, so no fitting recipe has a reason to shorten the lengthscale or to treat the data as noisy | Both arms' intervals include zero | Either arm's interval is entirely above zero | Exp 6 (learned-noise and prior arms) |

For all three the primary measure is the Newcombe difference in success from `bo_default` (50 seeds, so a modest effect can
go undetected: say so). Reported alongside without a threshold: the fraction of runs that sample inside the well, success given
a well sample, and the fitted lengthscale before the first well sample. Fitted noise is not logged in the run files, so BO9 is
judged on success only. The reading across Exp 6: if every single-factor arm and `bo_map` give no gain, the fitting recipe is
not what limits BO here; if one arm helps and `bo_map` does not, settings interact.

Current numbering (2026-10-28): BO3 (`bo_map`) and BO7 to BO9 are tested in Exp 6; BO4 (kernels) in Exp 7; H5 in Exp 8; CMA3 in
Exp 9; H6, M3 and BO5 in Exp 10.


## BO10: kernel type at lengthscale cap 1 (added 2026-10-31, before the runs)

Exp 7 now runs each kernel twice: at the default lengthscale cap (20; tests BO4 above, unchanged) and at cap 1, the best cap of Exp 6.
The cap-1 baseline is `bo_cap1` (Matern-5/2, cap 1, 39/50 in Exp 6); each cap-1 kernel differs from it in the kernel only. Seeds 1000-1049,
N = 300, same as before. Added after seeing Exp 6, so this prediction is conditional on that result.

| ID | Prediction | Supported if | Contradicted if | Test |
|---|---|---|---|---|
| BO10 | With the lengthscale capped at 1 every kernel sweeps the box and, once the well is sampled, refines it, so kernel type does not change success | Newcombe 95% intervals of the difference from `bo_cap1` for Matern-3/2, Matern-1/2 and RBF all include zero | At least one kernel's interval is entirely above or below zero | Exp 7 (cap-1 arms) |

Notes: lengthscale values are not comparable across kernels, and a cap of 1 domain unit means a different smoothness for each kernel. The
success-given-touch rate is reported separately. Statuses stay the user's.


## BO11 and BO12: basin width (Exp 8), added 2026-11-01, before any BO run at a width other than 0.3

Renumbering (2026-11-01): the width sweep is now Exp 8 (run first), the unimodal control Exp 9, validation Exp 10. CMA3 is tested in Exp 8; H5 in
Exp 9; H6, M3 and BO5 in Exp 10. The Exp 8 BO lines are `bo_default` (seeds 1000-1049 for the baseline curve) and the selected best variant
`bo_cap1` with `bo_default` re-run on fresh seeds 2000-2049 (selection rule: `docs/pilot_log.md`, amendment 2026-11-01). Far placement
(w in {0.1, 0.3, 0.6, 1.0}) carries both predictions; near cells (w <= 0.3) are reported descriptively. Written after Exp 6 was seen, so both are
conditional on it; the default BO result at w = 0.3 (5/50) is already known, the other widths are not.

| ID | Prediction | Supported if | Contradicted if | Test |
|---|---|---|---|---|
| BO11 | Default BO's success rises with well width on the far placement: a wider well is more likely to be sampled by the initial design and the early search | Success at w = 1.0 exceeds success at w = 0.3, with the Newcombe 95% interval for the difference entirely above zero | The w = 1.0 interval includes zero or lies below zero | Exp 8 (`bo_default`, seeds 1000-1049) |
| BO12 | The cap-1 gain over default BO is largest at an intermediate width: at w = 0.1 a well is too small for either setting to hit often, and at w = 1.0 default BO already finds it | The cap-1 minus default success difference at w = 0.3 is larger (point estimate) than at w = 0.1 and at w = 1.0, all on fresh seeds 2000-2049; the three Newcombe intervals are reported | The gain is largest at w = 0.1 or w = 1.0, or tied | Exp 8 (`bo_cap1` and `bo_default`, fresh seeds) |

Statuses stay the user's.

## BO13 and CMA6: do the scale settings transfer to Rastrigin and Schwefel? (Exp 10), added 2026-11-02, before any Exp 10 run

**Basis (tier: informed by Exps 6 to 9, so less independent than H1 to H6).** The scale-mismatch account from Exps 3, 6, 8 and 9: a step size or
lengthscale much longer than the feature scale hides the feature, and a lengthscale cap near that scale helped BO on the primary landscape
(`bo_cap1`) while costing early speed on a smooth bowl (Exp 9). Feature scale was measured from the functions alone, with no optimiser run
(local descent from 2000 uniform starts, domain [-5, 5]^2): Rastrigin local minima are 1.0 apart (success radius about 0.04 for epsilon = 0.2);
Schwefel local minima are about 1.3 apart, irregularly (success radius about 0.06 for epsilon = 5.0; global minimum near the corner (4.21, 4.21)).
Both spacings are close to the cap of 1 domain unit, so scale alone does not separate the two landscapes. H6, M3 and BO5 are unchanged and are tested as
written (`docs/predictions.md`, BO5 above). Arms: `bo_default` and `bo_cap1` on seeds 1000-1019 (the same seeds), `cma_default` and `cma_sig1` on seeds 1000-1049.

| ID | Prediction | Supported if | Contradicted if | Test |
|---|---|---|---|---|
| BO13 | The lengthscale cap transfers: on both Rastrigin and Schwefel, whose feature spacing is close to the cap, `bo_cap1` has higher success than `bo_default` | On each landscape the Newcombe 95% interval for (`bo_cap1` minus `bo_default`) success lies entirely above zero | On either landscape the interval includes zero or lies below it. Report each landscape separately; if only one is above zero, BO13 is contradicted and the split is reported | Exp 10 (`bo_default`, `bo_cap1`, seeds 1000-1019) |
| CMA6 | A smaller initial step size does not help CMA-ES from a random start on either landscape, because finding the basin, not refining it, limits CMA-ES there (as CMA2 found on the primary landscape) | On each landscape the Newcombe 95% interval for (`cma_sig1` minus `cma_default`) success includes zero | On either landscape the interval is entirely above or below zero | Exp 10 (`cma_default`, `cma_sig1`, seeds 1000-1049) |

**Power note (stated before the run).** With 20 BO seeds a difference of about 0.3 (for example 6/20 against 0/20) is needed for the interval to clear zero, and the random-sampling success at these epsilons is at most 5% by construction. A BO13 "contradicted" verdict can therefore mean low power on a landscape where success is rare, not that the cap fails; report the counts so the reader can tell the two apart.

Statuses stay the user's.
