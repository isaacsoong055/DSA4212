"""Pilot: restart-trigger firing rates from baseline (no-restart) CMA-ES only.

Prints firing statistics only, never success. Uses pilot seeds 9000-9049.
Rule, fixed before running: choose the tightest (tolx, tolfun) pair for which, at
default lambda=6, the trigger fires in >= 80% of runs with median firing eval <= 200.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from configs import CONFIGS, N_FULL, PILOT_SEEDS, PRIMARY  # noqa: E402

PAIRS = [(1e-1, 1e-2), (3e-2, 1e-3), (1e-2, 1e-4), (3e-3, 1e-5), (1e-3, 1e-6), (1e-4, 1e-8), (1e-5, 1e-10), (1e-6, 1e-8), (1e-6, 1e-12)]  # loose -> tight
LAMBDAS = [6, 12, 24]
SEEDS = PILOT_SEEDS[:50]


def job(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    from benchmarks import get_landscape
    from optimizers import run
    tolx, tolfun, lam, seed = args
    cfg = dict(CONFIGS["cma_default"], popsize=lam, tolx=tolx, tolfun=tolfun)
    sf = run(get_landscape(PRIMARY), N_FULL, seed, cfg, "trigger_pilot").meta["stop_fired"]
    return args, sf


def main():
    jobs = [(tx, tf, lam, s) for tx, tf in PAIRS for lam in LAMBDAS for s in SEEDS]
    with ProcessPoolExecutor(6) as pool:
        res = list(pool.map(job, jobs, chunksize=10))
    chosen = None
    print(f"{'tolx':>6} {'tolfun':>6} {'lam':>4} {'fired':>7} {'med_eval':>8} {'med_gen':>7}  reasons")
    for tx, tf in PAIRS:
        for lam in LAMBDAS:
            sfs = [sf for (a, b, l, _), sf in res if (a, b, l) == (tx, tf, lam)]
            fired = [sf for sf in sfs if sf]
            rate = len(fired) / len(sfs)
            med_e = np.median([sf["eval"] for sf in fired]) if fired else np.nan
            med_g = np.median([sf["gen"] for sf in fired]) if fired else np.nan
            why = {}
            for sf in fired:
                for w in sf["why"].split(","):
                    why[w] = why.get(w, 0) + 1
            print(f"{tx:>6.0e} {tf:>6.0e} {lam:>4} {rate:>7.2f} {med_e:>8.0f} {med_g:>7.0f}  {why}")
            if lam == 6 and rate >= 0.8 and med_e <= 200:
                chosen = (tx, tf)  # pairs are ordered loose -> tight, so last match = tightest
    print("CHOSEN (tolx, tolfun) =", chosen)


if __name__ == "__main__":
    main()
