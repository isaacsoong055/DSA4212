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
    assert ids == ["H1", "H2", "H3", "H4", "H5", "H6", "M1", "M2", "M3", "B1", "M4", "M5"]


# ---- dormant cards (D, V, E best-variant) on smoke data; skipped when no smoke summaries exist
smoke = pytest.mark.skipif(not (SUMMARY_FAST_DIR / "runs.csv").exists(), reason="run: run_experiments.py --aggregate --fast")


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(an, "save_fig", lambda *a, **k: None)
    monkeypatch.setattr(report.plt, "show", lambda *a, **k: None)


@smoke
def test_regret_and_lambda_gain_figures_run(quiet, monkeypatch):
    cfgs = ["cma_default", "cma_lam12", "cma_lam24"]
    monkeypatch.setitem(EXPERIMENTS, "expD", dict(configs=cfgs, landscapes=[PRIMARY]))
    monkeypatch.setitem(EXPERIMENTS, "expV", dict(configs=cfgs, landscapes=[PRIMARY, "primary_far_w1.0"]))
    M = report.Mode(True)
    report.experiment_card("expD", M)
    report.experiment_card("expV", M)


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
    report.expE_figure(FakeMode(True), best={"cmaes": "cma_lam12"})
    assert any(l and "best variant" in l for l in drawn)
    drawn.clear()
    report.expE_figure(M, best={"cmaes": "cma_lam12"})  # no fresh seeds on disk: no best-variant line
    assert not any(l and "best variant" in l for l in drawn)


@smoke
def test_card_not_drawn_when_settings_missing(quiet, monkeypatch):
    """Only the baselines exist (as in a fresh full run after Exp 0): Exp A/C/E must say 'not run yet'."""
    M = report.Mode(True)
    runs = M.table("runs")
    only_base = runs[runs["config_id"].isin(["cma_default", "bo_default"]) & (runs["landscape"] == report.canon(PRIMARY))]

    class FakeMode(report.Mode):
        def table(self, name):
            return only_base if name == "runs" else super().table(name)

    messages, figs = [], []
    monkeypatch.setattr(report, "not_run", lambda exp, M, why="": messages.append((exp, why)))
    monkeypatch.setattr(report, "_finish", lambda *a, **k: figs.append(a[1]))
    for exp in ("expA", "expB", "expC", "expE_baseline"):
        report.experiment_card(exp, FakeMode(True))
    assert [m[0] for m in messages] == ["expA", "expB", "expC", "expE_baseline"] and not figs
    report.experiment_card("expA", M)  # all settings exist in the smoke data: drawn
    assert figs == ["nb_5_expA"]


@smoke
def test_validation_card_always_draws_random_reference(quiet, monkeypatch):
    cfgs = ["cma_default", "cma_lam12", "cma_lam24", "random"]
    monkeypatch.setitem(EXPERIMENTS, "expV", dict(configs=cfgs, landscapes=[PRIMARY]))
    calls = []
    real = report.random_reference
    monkeypatch.setattr(report, "random_reference", lambda ax, runs, l: calls.append(real(ax, runs, l)))
    report.experiment_card("expV", report.Mode(True))
    assert calls == [True]  # random runs exist on the landscape, so the reference was drawn


@pytest.mark.parametrize("k,n", [(0, 5), (0, 50), (50, 50), (1, 50), (25, 50), (0, 1000)])
def test_wilson_interval_always_contains_the_rate(k, n):
    lo, hi = an.wilson_ci(k, n)
    assert 0 <= lo <= k / n <= hi <= 1  # error bars (rate - lo, hi - rate) are never negative
