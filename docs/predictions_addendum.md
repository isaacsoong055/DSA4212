# Pre-registration addendum (BO chain: Exp 0 -> Exp C)

**Date: 2026-10-25.** `docs/predictions.md` is unchanged. This file adds two items so that BO has its own
failure -> mechanism -> test chain, parallel to CMA-ES (H1 -> M1 -> H2): B1 diagnoses the BO baseline, H4 is the
test, M4 is the mechanism H4 is read against, and M5 (added the same day, before Exp C) uses the existing `bo_map` arm to separate the two lengthscale explanations.

**Honesty note.** The Exp 0 BO results (5/50 success, lengthscale at the bound in 42/50 runs, no well sample in the
45 failures) were seen *before* this addendum. **B1 is therefore post hoc**: it describes Exp 0, it is not a forecast.
**M4 is forward-looking**: it is committed before any Exp C result exists. Both are reported whichever way they go.

| ID | Prediction | Supported by | Contradicted by |
| --- | --- | --- | --- |
| B1 | (post hoc, from Exp 0) Default BO fails at discovery, not refinement: it rarely samples the well, and while it has not sampled it the fitted lengthscale sits far above w | Runs with no well sample fail; runs that sampled the well reach epsilon; lengthscale before the first well sample is many times w | Runs that sampled the well still fail, or the pre-discovery lengthscale is comparable to w |
| M4 | In Exp C, exploration (xi, beta, Thompson) helps BO only by raising the chance that a sample lands in the well (discovery); it does not work by repairing the lengthscale. The fitted lengthscale before the first well sample stays long under every variant and shrinks only after a well sample | Success of a variant tracks the share of its runs with a well sample; pre-discovery lengthscale is as long as the baseline's in every variant | A variant raises success while its pre-discovery lengthscale is clearly shorter than the baseline's (exploration changes the surrogate itself) |

| M5 | (added 2026-10-25, before Exp C) The long fitted lengthscale is a symptom of not having found the well, not its cause. `bo_map` (same Matern-5/2 kernel, lengthscale prior, MAP fit, so the lengthscale cannot sit at the bound) fits a clearly shorter lengthscale than `bo_default`, yet its success rate is not distinguishable from `bo_default`'s | `bo_map` pre-discovery lengthscale well below the bound, and its Wilson interval overlaps `bo_default`'s | `bo_map` success interval separates above `bo_default`'s (the lengthscale then matters causally, or a shorter lengthscale raises exploration through higher posterior variance) |

## How the chain reads

1. B1 (Exp 0): default BO has the same shape of failure as default CMA-ES (H1), via a different route.
2. H4 (Exp C, verbatim in `predictions.md`): success rises with xi or beta.
3. M4 (Exp C): *why* it rises. This mirrors M1 for CMA-ES: in both methods the gain is read as extra early coverage.
   Compare the two directly in the verdict (section 7 of the notebook).

M4 is tested with measurements already logged (fitted lengthscales and sample positions); no new runs or arms are added.

## M5 and the lengthscale hypotheses

Two explanations fit Exp 0 and cannot be told apart from it, because runs with a long lengthscale are exactly the runs
without a well sample:

- **A (cause):** a lengthscale much longer than w smooths over the well, gives it low uncertainty, so EI never samples it.
- **B (symptom):** a run that has not sampled the well sees only the smooth bowl, so maximum likelihood picks a long
  lengthscale. The long lengthscale follows from missing the well.

M5 pre-registers B for `bo_map`, the one Exp C arm that changes the lengthscale and nothing else. It is a lean, not a
certainty: a shorter lengthscale also raises posterior variance away from data, which could increase exploration, so a
separated interval is a live outcome and will be reported as such. M4 covers the acquisition variants (xi, beta, TS) and
does not apply to `bo_map` by construction.
