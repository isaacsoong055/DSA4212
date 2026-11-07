"""Sphere sanity check: default CMA-ES / default BO on pilot seeds 9000-9019, N=300, d=2.

A check, not a finding. Nothing is tuned. 'Solved' = regret <= 0.05 (same tolerance as the primary
landscape); evaluations-to-solve = 1 + first eval_idx with best_so_far - f* <= 0.05.
Usage: python pilot/sphere_sanity.py cmaes|bo [--workers 6]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from configs import CONFIGS, N_FULL, PILOT_SEEDS  # noqa: E402

SEEDS = PILOT_SEEDS[:20]
TOL = 0.05
OUT = Path(__file__).resolve().parent.parent / "results" / "summary"


def one(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from benchmarks import get_landscape
    from optimizers import run
    cid, seed = args
    ls = get_landscape("sphere")
    t0 = time.perf_counter()
    rows = pd.DataFrame(run(ls, N_FULL, seed, CONFIGS[cid], cid).rows)
    reg = rows["best_so_far"] - ls.f_star
    hit = np.flatnonzero(reg.to_numpy() <= TOL)
    return dict(config_id=cid, seed=seed, final_regret=float(reg.iloc[-1]),
                evals_to_solve=int(hit[0]) + 1 if len(hit) else np.nan, wall_s=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["cmaes", "bo"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cid = {"cmaes": "cma_default", "bo": "bo_default"}[a.method]
    t0 = time.perf_counter()
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as pool:
            res = list(pool.map(one, [(cid, s) for s in SEEDS]))
    else:
        res = [one((cid, s)) for s in SEEDS]
    df = pd.DataFrame(res)
    df.to_csv(OUT / f"sphere_sanity_{a.method}.csv", index=False)
    e = df["evals_to_solve"]
    print(f"{cid}: sphere, d=2, N={N_FULL}, pilot seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(df)})")
    print(f"  final regret: median {df['final_regret'].median():.3g}, worst {df['final_regret'].max():.3g}")
    print(f"  reached regret <= {TOL}: {int(e.notna().sum())}/{len(df)}; evaluations to reach it: "
          f"median {e.median():.0f}, worst {e.max():.0f}, best {e.min():.0f}")
    print(f"  per-run wall: median {df['wall_s'].median():.1f}s; total wall {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
