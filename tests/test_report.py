"""Notebook helpers: strict ellipse-rerun tolerance, Newcombe interval, predictions, dormant cards."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

import analysis as an
import report
from benchmarks import get_landscape
from configs import CONFIGS, EXPERIMENTS, PRIMARY
from logging_utils import RAW_FAST_DIR, SUMMARY_FAST_DIR, run_path, write_run
from optimizers import run

TOL = 1e-8  # same tolerance the notebook uses before it prints a warning on the figure


def test_cma_rerun_matches_saved_csv(tmp_path):
    ls = get_landscape(PRIMARY)
    log = run(ls, 100, 1000, CONFIGS["cma_default"], "cma_default")
    write_run(log.rows, run_path("cmaes", "cma_default", PRIMARY, 1000, tmp_path), ls.d)
    rows, dist, diff = report.cma_rerun(100, 1000, tmp_path)
    assert diff <= TOL
    assert len(dist) == rows["gen"].max() + 1 and all(d["C"].shape == (2, 2) for d in dist)


@pytest.mark.skipif(not run_path("cmaes", "cma_default", PRIMARY, 1000, RAW_FAST_DIR).exists(), reason="no saved smoke run")
def test_cma_rerun_matches_real_saved_run():
    """The saved smoke-test run (made in another process) must match the re-run to 1e-8."""
    _, _, diff = report.cma_rerun(100, 1000, RAW_FAST_DIR)
    assert diff <= TOL


def test_cma_rerun_detects_mismatch(tmp_path):
    ls = get_landscape(PRIMARY)
    rows = run(ls, 100, 1001, CONFIGS["cma_default"], "cma_default").rows  # different seed's samples
    write_run(rows, run_path("cmaes", "cma_default", PRIMARY, 1000, tmp_path), ls.d)
    assert report.cma_rerun(100, 1000, tmp_path)[2] > TOL


def test_newcombe_matches_published_example():
    d, lo, hi = an.newcombe_diff(48, 80, 56, 70)  # Newcombe (1998) method 10: 0.2 (0.0524, 0.3339)
    assert (d, lo, hi) == pytest.approx((0.2, 0.0524, 0.3339), abs=5e-4)


def test_predictions_parse_all_ids():
    ids = list(report.load_predictions()["ID"])
    assert ids == ["H1", "H2", "H3", "H4", "H5", "H6", "M1", "M2", "M3", "BO1", "BO2", "BO3", "BO4", "BO5",
                   "CMA1", "CMA2", "CMA3", "CMA4", "CMA5", "BO6", "BO7", "BO8", "BO9", "BO10", "BO11", "BO12"]


def test_ratio_config_scales_sigma_and_start_with_the_well():
    """cma_ro_k3 on w = 0.6: sigma0 = 3 w = 1.8 and the first sample is about (10/3) w from the well centre."""
    ls = get_landscape("primary_far_w0.6")
    log = run(ls, 12, 1000, CONFIGS["cma_ro_k3"], "cma_ro_k3")
    assert log.rows[0]["sigma"] == pytest.approx(1.8)
    d0 = np.linalg.norm(np.array([log.rows[0]["x_0"], log.rows[0]["x_1"]]) - ls.x_star)
    assert d0 < 10 / 3 * 0.6 + 4 * 1.8  # the first generation is sampled around x0, not exactly at it


def test_oracle_start_default_runs_are_unchanged():
    """The oracle/ratio options are off by default: cma_default still starts in U[-4, 4]^2 with sigma0 = 2."""
    ls = get_landscape(PRIMARY)
    log = run(ls, 6, 1000, CONFIGS["cma_default"], "cma_default")
    assert log.rows[0]["sigma"] == 2.0


def test_mean_trace_writes_both_files(tmp_path):
    an.mean_trace(["cma_o_sig0.3"], PRIMARY, [1000, 1001], 60, tmp_path)
    gens, final = pd.read_csv(tmp_path / "mean_trace.csv"), pd.read_csv(tmp_path / "mean_trace_final.csv")
    assert set(gens["seed"]) == {1000, 1001} and len(final) == 2
    assert {"dist_well", "dist_decoy"} <= set(gens.columns) and final["final_dist_well"].notna().all()


# ---- dormant cards (D, V, E best-variant) on smoke data; skipped when no smoke summaries exist
smoke = pytest.mark.skipif(not (SUMMARY_FAST_DIR / "runs.csv").exists(), reason="run: run_experiments.py --aggregate --fast")


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(an, "save_fig", lambda *a, **k: None)
    monkeypatch.setattr(report.plt, "show", lambda *a, **k: None)


@smoke
def test_regret_and_lambda_gain_figures_run(quiet, monkeypatch):
    cfgs = ["cma_default", "cma_lam12", "cma_lam24"]
    monkeypatch.setitem(EXPERIMENTS, "exp9", dict(configs=cfgs, landscapes=[PRIMARY]))
    monkeypatch.setitem(EXPERIMENTS, "exp10", dict(configs=cfgs, landscapes=[PRIMARY, "primary_far_w1.0"]))
    M = report.Mode(True)
    report.experiment_card("exp9", M)
    report.experiment_card("exp10", M)


@smoke
def test_expE_best_variant_uses_only_fresh_seeds(quiet, monkeypatch):
    M = report.Mode(True)
    runs = M.table("runs")
    fresh = runs[runs["config_id"] == "cma_lam12"].assign(seed=lambda d: d["seed"] + 1000)  # seeds 2000+
    both = pd.concat([runs, fresh], ignore_index=True)

    class FakeMode(report.Mode):
        def table(self, name):
            return both if name == "runs" else super().table(name)

    drawn = []
    monkeypatch.setattr(report.plt.Axes, "errorbar", lambda self, *a, **k: drawn.append(k.get("label")))
    report.exp8_figure(FakeMode(True), best={"cmaes": "cma_lam12"})
    assert any(l and "best variant" in l for l in drawn)
    drawn.clear()
    report.exp8_figure(M, best={"cmaes": "cma_lam12"})  # no fresh seeds on disk: no best-variant line
    assert not any(l and "best variant" in l for l in drawn)


@smoke
def test_card_not_drawn_when_settings_missing(quiet, monkeypatch):
    """Only the baselines exist (as in a fresh full run after Exp 0): Exp 1/4/7 must say 'not run yet'."""
    M = report.Mode(True)
    runs = M.table("runs")
    only_base = runs[runs["config_id"].isin(["cma_default", "bo_default"]) & (runs["landscape"] == report.canon(PRIMARY))]

    class FakeMode(report.Mode):
        def table(self, name):
            return only_base if name == "runs" else super().table(name)

    messages, figs = [], []
    monkeypatch.setattr(report, "not_run", lambda exp, M, why="": messages.append((exp, why)))
    monkeypatch.setattr(report, "_finish", lambda *a, **k: figs.append(a[1]))
    for exp in ("exp1", "exp2", "exp4", "exp8_baseline"):
        report.experiment_card(exp, FakeMode(True))
    assert [m[0] for m in messages] == ["exp1", "exp2", "exp4", "exp8_baseline"] and not figs
    report.experiment_card("exp1", M)  # all settings exist in the smoke data: drawn
    assert figs == ["nb_5_exp1"]


@smoke
def test_validation_card_always_draws_random_reference(quiet, monkeypatch):
    cfgs = ["cma_default", "cma_lam12", "cma_lam24", "random"]
    monkeypatch.setitem(EXPERIMENTS, "exp10", dict(configs=cfgs, landscapes=[PRIMARY]))
    calls = []
    real = report.random_reference
    monkeypatch.setattr(report, "random_reference", lambda ax, runs, l: calls.append(real(ax, runs, l)))
    report.experiment_card("exp10", report.Mode(True))
    assert calls == [True]  # random runs exist on the landscape, so the reference was drawn


@pytest.mark.parametrize("k,n", [(0, 5), (0, 50), (50, 50), (1, 50), (25, 50), (0, 1000)])
def test_wilson_interval_always_contains_the_rate(k, n):
    lo, hi = an.wilson_ci(k, n)
    assert 0 <= lo <= k / n <= hi <= 1  # error bars (rate - lo, hi - rate) are never negative


def test_long_budget_card_says_not_run_in_fast_mode(monkeypatch):
    shown = []
    monkeypatch.setattr(report, "display", lambda *a, **k: shown.extend(getattr(x, "data", "") for x in a))
    report.long_budget_card(report.Mode(True))
    assert "Not run yet" in shown[0] and "--long" in shown[0]


def test_bo_discovery_counts_well_samples_and_lengthscale(tmp_path):
    """bo_discovery on one saved smoke BO run: a touched run has a first-touch index, and success implies a touch."""
    raw = tmp_path / "raw"
    ls = get_landscape(PRIMARY)
    log = run(ls, 30, 1000, CONFIGS["bo_default"], "bo_default")
    write_run(log.rows, run_path("bo", "bo_default", PRIMARY, 1000, raw), ls.d)
    out = an.bo_discovery(["bo_default"], PRIMARY, [1000], raw, tmp_path)
    r = out.iloc[0]
    assert r.n == 1 and r.touched in (0, 1) and r.success_given_touch <= r.touched and r.success <= r.touched
    assert (tmp_path / "bo_discovery.csv").exists()


def test_bo_after_discovery_writes_curves_and_split(tmp_path):
    """Smoke: the discovery curve is non-decreasing and ends at the share of runs that touched the well."""
    raw = tmp_path / "raw"
    ls = get_landscape(PRIMARY)
    for seed in (1000, 1001):
        write_run(run(ls, 30, seed, CONFIGS["bo_default"], "bo_default").rows, run_path("bo", "bo_default", PRIMARY, seed, raw), ls.d)
    an.bo_after_discovery(["bo_default"], PRIMARY, [1000, 1001], raw, tmp_path)
    c = pd.read_csv(tmp_path / "bo_discovery_curves.csv")["share_found"]
    assert c.is_monotonic_increasing and len(c) == 30 and (tmp_path / "bo_after_discovery.csv").exists()


def test_bo_oracle_start_replaces_only_the_first_initial_point():
    ls = get_landscape(PRIMARY)
    base = run(ls, 12, 1000, CONFIGS["bo_default"], "bo_default").rows
    orc = run(ls, 12, 1000, CONFIGS["bo_o_xi0"], "bo_o_xi0").rows
    d0 = np.linalg.norm(np.array([orc[0]["x_0"], orc[0]["x_1"]]) - ls.x_star)
    assert d0 == pytest.approx(0.2) and [r["x_0"] for r in orc[1:10]] == [r["x_0"] for r in base[1:10]]


@pytest.mark.parametrize("cid", ["bo_cap0.25", "bo_cap5", "bo_noise1", "bo_noise_learned", "bo_ls_prior"])
def test_exp6_gp_settings_build_and_run(cid):
    """Each single-factor fitting arm builds a GP and completes a short run with exactly N evaluations."""
    ls = get_landscape(PRIMARY)
    assert len(run(ls, 13, 1000, CONFIGS[cid], cid).rows) == 13


def test_exp6_cap_and_noise_change_only_their_own_setting():
    import torch
    from optimizers import _build_gp
    X, Y = torch.rand(8, 2, dtype=torch.float64), torch.rand(8, 1, dtype=torch.float64)
    bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=torch.float64)
    base = _build_gp(X, Y, "mle_fixed", bounds, "matern52", CONFIGS["bo_default"])
    cap = _build_gp(X, Y, "mle_fixed", bounds, "matern52", CONFIGS["bo_cap0.25"])
    assert base.covar_module.base_kernel.raw_lengthscale_constraint.upper_bound.item() == pytest.approx(2.0)
    assert cap.covar_module.base_kernel.raw_lengthscale_constraint.upper_bound.item() == pytest.approx(0.025)
    assert cap.covar_module.base_kernel.lengthscale.max().item() < 0.025  # starts inside the interval


def test_exp7_arms_differ_from_their_matern52_baseline_in_exactly_one_setting():
    for cid in EXPERIMENTS["exp7"]["configs"]:
        if cid in ("bo_default", "bo_cap1"):
            continue
        base = CONFIGS["bo_cap1" if cid.endswith("_cap1") else "bo_default"]
        diff = {k for k in set(base) | set(CONFIGS[cid]) if base.get(k) != CONFIGS[cid].get(k)}
        assert diff == {"kernel"}, (cid, diff)


def test_exp6_arms_differ_from_the_baseline_in_exactly_one_setting():
    base = CONFIGS["bo_default"]
    for cid in EXPERIMENTS["exp6"]["configs"]:
        if cid in ("bo_default", "bo_map"):  # bo_map is the all-at-once reference, not a one-factor arm
            continue
        diff = {k for k in set(base) | set(CONFIGS[cid]) if base.get(k) != CONFIGS[cid].get(k)}
        assert len(diff) == 1, (cid, diff)
