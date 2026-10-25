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
| CMA-ES | 3 initial step size sigma0 | CMA1, CMA2 |
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
