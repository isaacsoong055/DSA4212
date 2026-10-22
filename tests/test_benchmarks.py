"""Landscape minima and basin checks."""
import numpy as np
import pytest

from benchmarks import get_landscape, local_minima, nearest_nonglobal

NAMES = ["sphere", "rastrigin", "schwefel", "primary_far_w0.3", "primary_near_w0.1",
         "primary_far_w1"]


@pytest.mark.parametrize("name", NAMES)
def test_minimum_is_global_on_grid(name):
    """f(x_star) = f_star, and no point on a fine grid is lower."""
    ls = get_landscape(name)
    assert ls(ls.x_star) == pytest.approx(ls.f_star, abs=1e-9)
    g = np.linspace(-5, 5, 801)
    vals = np.array([[ls([x, y]) for x in g] for y in g[::4]])
    assert vals.min() >= ls.f_star - 1e-9


def test_known_minima():
    assert get_landscape("sphere").f_star == 0.0
    assert get_landscape("rastrigin").f_star == 0.0
    sch = get_landscape("schwefel")
    assert np.allclose(sch.x_star, 4.2096874633, atol=1e-4) and abs(sch.f_star) < 1e-3


def test_primary_well_is_global_and_basin_check():
    ls = get_landscape("primary_far_w0.3")
    assert np.allclose(ls.x_star, [3, 3], atol=0.05)
    assert ls.f_star < ls(np.zeros(2)) - 1.0      # well clearly deeper than the decoy
    assert ls.in_global_basin(ls.x_star + 0.05)
    assert not ls.in_global_basin(np.zeros(2))


@pytest.mark.parametrize("name", ["rastrigin", "schwefel", "primary_far_w0.3"])
def test_success_implies_global_basin(name):
    """Every point whose regret is <= epsilon descends to the global minimiser."""
    ls = get_landscape(name)
    rng = np.random.default_rng(0)
    pts = np.vstack([ls.x_star + rng.uniform(-r, r, (1500, 2)) for r in (0.05, 0.1, 0.5, 1.5, 3.0)])
    pts = np.clip(pts, -5, 5)
    succ = [x for x in pts if ls.is_success(ls(x))]
    assert len(succ) > 100
    assert all(ls.in_global_basin(x) for x in succ)


@pytest.mark.parametrize("name,cap", [("rastrigin", 0.994959), ("schwefel", 118.438)])
def test_epsilon_below_nearest_nonglobal_regret(name, cap):
    ls = get_landscape(name)
    regret, _ = nearest_nonglobal(ls, local_minima(ls, 1000))
    assert regret == pytest.approx(cap, rel=1e-4)
    assert ls.epsilon < regret
