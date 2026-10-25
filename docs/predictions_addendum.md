# Pre-registration addendum: the BO chain (BO1 to BO5)

**Date: 2026-10-25.** `docs/predictions.md` (H1 to H6, M1 to M3) is unchanged and cannot be edited. This file adds
five BO predictions so that BO has its own failure -> mechanism -> test chain, parallel to CMA-ES.

**Naming.** H = a headline prediction, M = a mechanism (why), both from `predictions.md`. BO1 to BO5 are the BO
predictions in this file. The number is only a label; the experiment that tests each one is in the last column of the
map below. (Earlier drafts of this file called BO1 "B1", BO2 "M4", BO3 "M5", BO4 "K1"; they were renamed the same day,
before any BO result beyond Exp 0 existed.)

**Honesty note.** The Exp 0 BO results (5/50 success, lengthscale at the bound in 42/50 runs, no well sample in the 45
failures) were seen *before* this addendum. **BO1 is therefore post hoc**: it describes Exp 0, it is not a forecast.
BO2 to BO5 are written before Exp 3, Exp 4 and Exp 7 have any result. All are reported whichever way they go.

| ID | Prediction | Supported by | Contradicted by | Tested in |
| --- | --- | --- | --- | --- |
| BO1 | (post hoc, from Exp 0) Default BO fails at discovery, not refinement: it rarely samples the well, and while it has not sampled it the fitted lengthscale sits far above w | Runs with no well sample fail; runs that sampled the well reach epsilon; lengthscale before the first well sample is many times w | Runs that sampled the well still fail, or the pre-discovery lengthscale is comparable to w | Exp 0 |
| BO2 | Exploration (xi, beta, Thompson) helps BO only by raising the chance that a sample lands in the well (discovery); it does not work by repairing the lengthscale. The fitted lengthscale before the first well sample stays long under every variant and shrinks only after a well sample | Success of a variant tracks the share of its runs with a well sample; pre-discovery lengthscale is as long as the baseline's in every variant | A variant raises success while its pre-discovery lengthscale is clearly shorter than the baseline's (exploration changes the surrogate itself) | Exp 3 (companion to H4) |
| BO3 | The long fitted lengthscale is a symptom of not having found the well, not its cause. `bo_map` (same Matern-5/2 kernel, lengthscale prior and learned noise, MAP fit) fits a clearly shorter lengthscale than `bo_default`, yet its success rate is not distinguishable from `bo_default`'s. `bo_map` changes two things at once, so this is a weak test of the cause | `bo_map` pre-discovery lengthscale well below the bound, and its Wilson interval overlaps `bo_default`'s | `bo_map` success interval separates above `bo_default`'s (the lengthscale or the noise then matters, or a shorter lengthscale raises exploration through higher posterior variance) | Exp 3 (`bo_map` arm) |
| BO4 | Kernel type does not change BO's success rate on the primary landscape: the failure is at discovery, and before any sample lands in the well every kernel sees the same smooth bowl. A rougher kernel (Matern-1/2) may help only after a well sample exists, so success given a well sample is reported separately | Wilson intervals of Matern-3/2, Matern-1/2 and RBF all overlap the Matern-5/2 baseline | At least one kernel's interval separates from the baseline's | Exp 4 |
| BO5 | The spread between BO's best and worst settings (from Exps 3 and 4) is larger on Rastrigin (global funnel a GP can learn) than on Schwefel (deceptive: the best basin is far from its neighbours). This mirrors H6 for CMA-ES lambda | Larger best-minus-worst gap on Rastrigin than on Schwefel | Same or larger gap on Schwefel | Exp 7 |

## Prediction-to-experiment map (all predictions)

| Block | Exp | Predictions tested |
| --- | --- | --- |
| both | 0 baselines | H1 (CMA-ES), BO1 (BO) |
| CMA-ES | 1 population size lambda | H2, M1 |
| CMA-ES | 2 restarts | H3, M2 |
| BO | 3 exploration vs exploitation | H4, BO2, BO3 |
| BO | 4 kernel type | BO4 |
| both | 5 unimodal control | H5 |
| both | 6 landscape knobs | none pre-registered |
| both | 7 validation | H6, M3 (CMA-ES), BO5 (BO) |

## The two lengthscale explanations (what BO2 and BO3 are about)

Two explanations fit Exp 0 and cannot be told apart from it, because runs with a long lengthscale are exactly the runs
without a well sample:

- **A (cause):** a lengthscale much longer than w smooths over the well, gives it low uncertainty, so EI never samples it.
- **B (symptom):** a run that has not sampled the well sees only the smooth bowl, so maximum likelihood picks a long
  lengthscale. The long lengthscale follows from missing the well.

BO3 leans to B, but its test is weak because `bo_map` changes the lengthscale prior and the noise treatment together.
A shorter lengthscale also raises posterior variance away from data, so a separated interval is a live outcome. The
cause-versus-symptom question stays open after Exp 3; do not claim more than the data allow. BO2 covers the acquisition
variants (xi, beta, Thompson) and does not apply to `bo_map` by construction.

## Exp 3 and Exp 4 design decisions made on 2026-10-25 (before either ran)

- xi grid {0, 0.01, 0.1, 0.5, 1, 2} was chosen from the landscape's depth (the well is about 2.2 below the decoy floor,
  fixed by the definition A = 4, s = 10, w = 0.3), not from any result. beta grid {0.5, 2, 8, 32}.
- Initial design size is fixed at n_init = 10 for every BO arm. It is not varied.
- Kernels for Exp 4: Matern-3/2, Matern-1/2, RBF against the Matern-5/2 baseline. The lengthscale bounds and starting
  value are the same on the unit cube for all kernels, so lengthscale values are not strictly comparable across kernels
  (a caveat for the report).
