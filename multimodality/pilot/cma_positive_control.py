"""Positive control for CMA-ES on the primary landscape (diagnostic, pilot seeds 9000-9019, N = 300).

Starts CMA-ES at or near the well centre with different sigma0, to check that the success criterion is reachable and
that the CMA-ES wrapper is not at fault. Not an experiment: nothing here is compared with BO or used to choose configs.
Run: python pilot/cma_positive_control.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import optimizers as O  # noqa: E402
from benchmarks import get_landscape  # noqa: E402
from configs import CONFIGS, PRIMARY  # noqa: E402

ls = get_landscape(PRIMARY)
orig_rng = O._rng


def fixed_start(x0):
    """_rng that returns the given first start x0 for the x0 stream and the normal stream otherwise."""
    def rng(seed, stream):
        if stream == O.STREAM_X0:
            return type("R", (), {"uniform": lambda self, *a, **k: np.array(x0)})()
        return orig_rng(seed, stream)
    return rng


def trial(label, offset, sigma0, seeds=range(9000, 9020)):
    O._rng = fixed_start(ls.x_star + np.array(offset))
    cfg = dict(CONFIGS["cma_default"], sigma0=sigma0)
    ok, sig = [], []
    for s in seeds:
        log = O.run(ls, 300, s, cfg, "positive_control")
        ok.append(ls.is_success(min(r["f"] for r in log.rows)))
        sig.append(log.rows[-1]["sigma"])
    print(f"{label:58s} success {sum(ok)}/{len(ok)}  median final sigma {np.median(sig):.3g}")


if __name__ == "__main__":
    trial("start AT well centre, sigma0 = 0.3", (0, 0), 0.3)
    trial("start 1.0 from well, sigma0 = 0.5", (1.0, 0), 0.5)
    trial("start AT well centre, sigma0 = 1", (0, 0), 1.0)
    trial("start AT well centre, sigma0 = 2 (frozen project value)", (0, 0), 2.0)
    trial("start 1.0 from well, sigma0 = 2 (frozen project value)", (1.0, 0), 2.0)
