"""E6b: why is BO's model-based recommendation worse than its lowest reading with replicates?

E6 found that for bo-replicate (each point evaluated 4 times, the GP given the
averages) the posterior-mean recommendation R2 was WORSE than the lowest
reading R1 (+0.89 decades at sigma = 0.01), while for bo-base it was better.
Hypothesis in the report: the averaged data are nearly noise-free, so the
noise-fitting GP used for R2 fails as in E2e (it absorbs the best points as
noise).

This re-runs E6's bo-base and bo-replicate searches exactly (same seeds, same
code, BLAS pinned) and, from the SAME evaluations, recommends a point four ways:
  R1          lowest observed value (as E6)
  R2-fitted   posterior-mean minimiser of a GP on log targets, noise fitted (as E6)
  R2-fixed    the same GP with noise fixed at 1e-6 (interpolates the averages)
  R2-known    a GP on raw (standardised, not logged) targets, noise fixed at the
              true value sigma^2 / repeats; on raw f the additive noise has one
              known size, so this is the correctly specified model
Everything is scored on the noise-free f. Diagnostics for R2-fitted: the
fitted noise SD (standardised units) and the GP's error at the lowest reading
(posterior mean minus observed), the E2e signatures.

Predictions, written before running: R1 and R2-fitted reproduce E6 exactly; if
the hypothesis holds, for bo-replicate the fitted noise is large relative to
the true noise and the GP's error at the best reading is large, while
R2-known beats R1.

Writes results/e6b_recommendation.csv and results/e6b_summary.csv.
Usage: python code/run_e6b_recommendation_gp.py [--workers N] [--report-only]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import itertools
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analysis
import bo
import problems
import run_e6_noise_budget as e6

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
VARIANTS = ("bo-base", "bo-replicate")
RULES = ("R1", "R2-fitted", "R2-fixed", "R2-known")


def run_one(task):
    sigma, instance, variant = task
    f, _ = problems.make_ellipsoid(e6.DIMENSION, 100, instance=instance, rotation="none")
    noisy = e6.Noisy(f, sigma, seed=30_000 + instance)
    repeats = e6.REPEATS if variant == "bo-replicate" else 1
    wrapped = e6.Averaged(noisy, repeats)
    bo.bayes_opt(wrapped, e6.DIMENSION, -e6.BOX, e6.BOX, "ard", seed=instance,
                 budget=e6.CAP // repeats, **e6.BO_FIX)
    x, y = np.asarray(wrapped.points), np.asarray(wrapped.means)
    rng_seed = instance
    rec = {"R1": x[int(np.argmin(y))],
           "R2-fitted": e6.bo_posterior_minimiser(wrapped.points, wrapped.means, seed=rng_seed)}

    # Diagnostics of the E6 recommendation GP (log targets, noise fitted).
    z = bo._transform_targets(y)
    fitted = bo.fit_gp(x, z, "ard", np.random.default_rng(rng_seed), input_scale=2 * e6.BOX)
    _, cut = bo._precision_matrix(fitted["theta"], e6.DIMENSION, "ard")
    best = int(np.argmin(y))
    mean_best, _ = bo.predict(fitted, x[best][None])

    fixed = bo.fit_gp(x, z, "ard", np.random.default_rng(rng_seed), input_scale=2 * e6.BOX,
                      noise_variance=1e-6)
    m_fixed, _ = bo.predict(fixed, x)
    rec["R2-fixed"] = x[int(np.argmin(m_fixed))]

    # Correctly specified: raw targets, standardised, noise = sigma^2 / repeats in those units.
    sd = float(y.std()) + 1e-12
    raw = (y - y.mean()) / sd
    known_var = (sigma ** 2 / repeats) / sd ** 2
    known = bo.fit_gp(x, raw, "ard", np.random.default_rng(rng_seed), input_scale=2 * e6.BOX,
                      noise_variance=max(known_var, 1e-8))
    m_known, _ = bo.predict(known, x)
    rec["R2-known"] = x[int(np.argmin(m_known))]

    return {"sigma": sigma, "instance": instance, "variant": variant,
            "fitted_noise_sd": float(np.exp(fitted["theta"][cut])),
            "gp_error_at_best": float(mean_best[0] - z[best]),
            "gap_best_two": float(np.sort(z)[1] - np.sort(z)[0]),
            **{f"log10_f_{r}": float(np.log10(f(p) + 1e-300)) for r, p in rec.items()}}


def report(df):
    pd.set_option("display.width", 220)
    print("\n== median log10 f (noise-free) of the recommended point")
    print(df.pivot_table(index=["variant", "sigma"], values=[f"log10_f_{r}" for r in RULES],
                         aggfunc="median")[[f"log10_f_{r}" for r in RULES]].round(2).to_string())
    print("\n== diagnostics of the noise-fitted recommendation GP (medians)")
    print(df.groupby(["variant", "sigma"])[["fitted_noise_sd", "gp_error_at_best", "gap_best_two"]]
            .median().round(3).to_string())
    e6runs = pd.read_csv(RESULTS / "e6_runs.csv")
    out = []
    print("\n== reproduction of E6 (R1 and R2-fitted must match exactly)")
    for (variant, sigma), g in df.groupby(["variant", "sigma"]):
        for rule, new in (("R1", "R1"), ("R2", "R2-fitted")):
            ref = e6runs[(e6runs.variant == variant) & np.isclose(e6runs.sigma, sigma)
                         & (e6runs.rule == rule)].sort_values("instance").log10_f.values
            mine = g.sort_values("instance")[f"log10_f_{new}"].values
            print(f"  {variant:13s} sigma={sigma:<5} {rule}: max |difference| = "
                  f"{np.max(np.abs(ref - mine)):.2e}")
    print("\n== each rule minus R1, paired (negative = better than the lowest reading)")
    for (variant, sigma), g in df.groupby(["variant", "sigma"]):
        g = g.sort_values("instance")
        for rule in RULES[1:]:
            res = analysis.bootstrap_paired_difference(g[f"log10_f_{rule}"].values,
                                                       g["log10_f_R1"].values, n_boot=5000)
            out.append({"variant": variant, "sigma": sigma, "comparison": f"{rule} - R1", **res})
            flag = "" if res["ci_low"] <= 0 <= res["ci_high"] else "  <-- differs"
            print(f"  {variant:13s} sigma={sigma:<5} {rule:10s} - R1  {res['difference']:+6.2f} "
                  f"[{res['ci_low']:+6.2f}, {res['ci_high']:+6.2f}]{flag}")
    return pd.DataFrame(out)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    path = RESULTS / "e6b_recommendation.csv"
    if "--report-only" not in sys.argv:
        tasks = list(itertools.product(e6.NOISE, range(e6.N_INSTANCES), VARIANTS))
        tasks.sort(key=lambda t: t[2] != "bo-base")
        with mp.Pool(workers) as pool:
            rows = pool.map(run_one, tasks)
        pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.read_csv(path)
    report(df).to_csv(RESULTS / "e6b_summary.csv", index=False)
