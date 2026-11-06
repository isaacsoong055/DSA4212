"""Checks on the pieces every experiment relies on. Runs in about a minute.

    python -m unittest discover -s tests        (or: pytest tests)

Each test guards a claim the report makes or a step the results depend on.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # BO is only repeatable with one BLAS thread
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import pathlib
import sys
import unittest

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
import bo
import cmaes
import problems


class TestProblems(unittest.TestCase):
    def test_ellipsoid_has_stated_optimum_spectrum_and_rotation(self):
        for rotation in ("none", "random", 30):
            f, meta = problems.make_ellipsoid(2, 1000, instance=3, rotation=rotation)
            self.assertEqual(f(meta["optimum"]), 0.0)
            self.assertAlmostEqual(np.linalg.norm(meta["optimum"]), 2.0)
            eig = np.sort(np.linalg.eigvalsh(meta["hessian"]))
            np.testing.assert_allclose(eig, [1.0, 1000.0], rtol=1e-10)

    def test_rotation_keeps_conditioning_and_changes_orientation(self):
        _, a = problems.make_ellipsoid(2, 100, instance=0, rotation="none")
        _, b = problems.make_ellipsoid(2, 100, instance=0, rotation=45)
        np.testing.assert_allclose(np.linalg.eigvalsh(a["hessian"]),
                                   np.linalg.eigvalsh(b["hessian"]))
        self.assertGreater(abs(b["hessian"][0, 1]), 1.0)   # no longer axis-aligned

    def test_haar_rotation_is_orthogonal(self):
        q = problems.haar_rotation(5, np.random.default_rng(0))
        np.testing.assert_allclose(q @ q.T, np.eye(5), atol=1e-12)


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        f, _ = problems.make_ellipsoid(2, 100, instance=0, rotation="random")
        self.x = rng.uniform(-5, 5, (25, 2))
        self.y = bo._transform_targets([f(p) for p in self.x])

    def test_map_gradient_matches_finite_differences(self):
        loc = bo.lengthscale_prior_loc(2, 10.0)
        for metric in ("ard", "full", "iso"):
            n = bo._n_parameters(2, metric)
            theta = np.random.default_rng(2).normal(0, 0.3, n)
            theta[-1] = -3.0
            args = (self.x, self.y, 2, metric, loc, bo.PRIOR_SCALE)
            _, grad = bo._neg_log_posterior_and_gradient(theta, *args)
            numeric = np.zeros(n)
            for k in range(n):
                step = np.zeros(n); step[k] = 1e-6
                numeric[k] = (bo._negative_log_posterior(theta + step, *args)
                              - bo._negative_log_posterior(theta - step, *args)) / 2e-6
            np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-4, err_msg=metric)

    def test_fixed_noise_is_respected(self):
        model = bo.fit_gp(self.x, self.y, "ard", np.random.default_rng(0),
                          input_scale=10.0, noise_variance=1e-6)
        _, cut = bo._precision_matrix(model["theta"], 2, "ard")
        self.assertAlmostEqual(np.exp(2 * model["theta"][cut]), 1e-6, places=12)


class TestLogEI(unittest.TestCase):
    def test_log_h_matches_direct_formula_where_that_is_accurate(self):
        z = np.array([3.0, 1.0, 0.0, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0])
        direct = np.log(norm.pdf(z) + z * norm.cdf(z))
        np.testing.assert_allclose(bo._log_h(z), direct, rtol=1e-8)

    def test_log_h_stays_finite_where_ei_underflows(self):
        z = np.array([-40.0, -1e3, -1e9])
        self.assertTrue(np.all(np.isfinite(bo._log_h(z))))
        self.assertTrue(np.all(np.diff(bo._log_h(z)) < 0))


class TestCMAES(unittest.TestCase):
    def test_rank_invariance_is_exact(self):
        f, _ = problems.make_ellipsoid(2, 100, instance=1, rotation="random")
        g = problems.transform_value(f, "cube")
        for variant in ("full", "diagonal"):
            a = cmaes.cma_es(f, 2, np.zeros(2), seed=4, variant=variant, max_evaluations=300,
                             initial_sigma=3.0, target=-np.inf)
            b = cmaes.cma_es(g, 2, np.zeros(2), seed=4, variant=variant, max_evaluations=300,
                             initial_sigma=3.0, target=-np.inf)
            np.testing.assert_array_equal(a["best_point"], b["best_point"])

    def test_diagonal_covariance_stays_diagonal(self):
        f, _ = problems.make_ellipsoid(2, 1000, instance=0, rotation=45)
        out = cmaes.cma_es(f, 2, np.zeros(2), seed=0, variant="diagonal", max_evaluations=600,
                           initial_sigma=3.0, record_history=False)
        self.assertEqual(out["covariance"][0, 1], 0.0)


class TestDeterminism(unittest.TestCase):
    def test_bo_run_repeats_exactly(self):
        f, _ = problems.make_ellipsoid(2, 100, instance=2, rotation="random")
        runs = [bo.bayes_opt(f, 2, -5.0, 5.0, "ard", seed=2, budget=20, noise_variance=1e-6)["y"]
                for _ in range(2)]
        np.testing.assert_array_equal(*runs)


if __name__ == "__main__":
    unittest.main()
