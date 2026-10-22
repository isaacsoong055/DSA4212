"""Budget accounting and reproducibility."""
import pandas as pd
import pytest

from benchmarks import get_landscape
from configs import CONFIGS
from optimizers import run

LS = "primary_far_w0.3"
FAST_BO = dict(CONFIGS["bo_default"], raw_samples=256, num_restarts=2)
CASES = {
    "random": CONFIGS["random"],
    "cma_default": CONFIGS["cma_default"],
    "cma_lam24": CONFIGS["cma_lam24"],
    # loose triggers so restarts actually fire within a small budget
    "cma_ipop": dict(CONFIGS["cma_ipop"], tolx=1e-2, tolfun=1e-2),
    "cma_bipop": dict(CONFIGS["cma_bipop"], tolx=1e-2, tolfun=1e-2),
    "bo_ei": FAST_BO,
    "bo_lcb": dict(FAST_BO, acq="lcb", beta=2.0),
    "bo_ts": dict(FAST_BO, acq="ts", ts_candidates=256),
    "bo_map": dict(FAST_BO, gp="map_default"),
}


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("N", [37, 61])  # not multiples of any popsize used
def test_budget_exact(cid, N):
    log = run(get_landscape(LS), N, 1000, CASES[cid], cid)
    assert len(log.rows) == N
    assert [r["eval_idx"] for r in log.rows] == list(range(N))


def test_restarts_fire_and_count_against_budget():
    for cid in ("cma_ipop", "cma_bipop"):
        log = run(get_landscape(LS), 300, 1000, CASES[cid], cid)
        assert len(log.rows) == 300 and log.meta["n_restarts"] >= 1


@pytest.mark.parametrize("cid", ["random", "cma_default", "cma_bipop", "bo_ei", "bo_ts"])
def test_same_seed_same_result(cid):
    a = pd.DataFrame(run(get_landscape(LS), 40, 1003, CASES[cid], cid).rows)
    b = pd.DataFrame(run(get_landscape(LS), 40, 1003, CASES[cid], cid).rows)
    cols = [c for c in a.columns if c != "opt_time_s"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    c = pd.DataFrame(run(get_landscape(LS), 40, 1004, CASES[cid], cid).rows)
    assert not a["x_0"].equals(c["x_0"])
