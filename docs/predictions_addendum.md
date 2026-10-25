# Pre-registration addendum (BO chain: Exp 0 -> Exp C)

**Date: 2026-10-25.** `docs/predictions.md` is unchanged. This file adds two items so that BO has its own
failure -> mechanism -> test chain, parallel to CMA-ES (H1 -> M1 -> H2): B1 diagnoses the BO baseline, H4 is the
test, M4 is the mechanism H4 is read against.

**Honesty note.** The Exp 0 BO results (5/50 success, lengthscale at the bound in 42/50 runs, no well sample in the
45 failures) were seen *before* this addendum. **B1 is therefore post hoc**: it describes Exp 0, it is not a forecast.
**M4 is forward-looking**: it is committed before any Exp C result exists. Both are reported whichever way they go.

| ID | Prediction | Supported by | Contradicted by |
| --- | --- | --- | --- |
| B1 | (post hoc, from Exp 0) Default BO fails at discovery, not refinement: it rarely samples the well, and while it has not sampled it the fitted lengthscale sits far above w | Runs with no well sample fail; runs that sampled the well reach epsilon; lengthscale before the first well sample is many times w | Runs that sampled the well still fail, or the pre-discovery lengthscale is comparable to w |
| M4 | In Exp C, exploration (xi, beta, Thompson) helps BO only by raising the chance that a sample lands in the well (discovery); it does not work by repairing the lengthscale. The fitted lengthscale before the first well sample stays long under every variant and shrinks only after a well sample | Success of a variant tracks the share of its runs with a well sample; pre-discovery lengthscale is as long as the baseline's in every variant | A variant raises success while its pre-discovery lengthscale is clearly shorter than the baseline's (exploration changes the surrogate itself) |

## How the chain reads

1. B1 (Exp 0): default BO has the same shape of failure as default CMA-ES (H1), via a different route.
2. H4 (Exp C, verbatim in `predictions.md`): success rises with xi or beta.
3. M4 (Exp C): *why* it rises. This mirrors M1 for CMA-ES: in both methods the gain is read as extra early coverage.
   Compare the two directly in the verdict (section 7 of the notebook).

M4 is tested with measurements already logged (fitted lengthscales and sample positions); no new runs or arms are added.
