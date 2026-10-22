"""Run experiments in parallel and write one CSV per run.

Examples
  python run_experiments.py --fast
  python run_experiments.py --full --exp expA
  python run_experiments.py --fast --method cmaes --landscape primary_far_w0.3
  python run_experiments.py --aggregate [--fast]
  python run_experiments.py --archive exp0 --exp exp0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from configs import CONFIGS, EXPERIMENTS, N_FAST, N_FULL, N_SEEDS_FAST, N_SEEDS_FULL, SEEDS  # noqa: E402
from logging_utils import run_path  # noqa: E402


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def run_one(config_id: str, landscape: str, seed: int, N: int, raw_dir: str) -> tuple[str, float]:
    """Run one job and write its CSV. Returns (path, seconds)."""
    from benchmarks import get_landscape
    from logging_utils import write_run
    from optimizers import run

    ls = get_landscape(landscape)
    cfg = CONFIGS[config_id]
    path = run_path(cfg["method"], config_id, landscape, seed, raw_dir)
    t0 = time.perf_counter()
    log = run(ls, N, seed, cfg, config_id)
    write_run(log.rows, path, ls.d)
    return str(path), time.perf_counter() - t0


def build_jobs(args) -> list[tuple[str, str, int]]:
    exps = [args.exp] if args.exp else list(EXPERIMENTS)
    seeds = SEEDS[: args.seeds or (N_SEEDS_FAST if args.fast else N_SEEDS_FULL)]
    jobs = set()
    for e in exps:
        for c in EXPERIMENTS[e]["configs"]:
            for ls in EXPERIMENTS[e]["landscapes"]:
                if args.method and CONFIGS[c]["method"] != args.method:
                    continue
                if args.landscape and ls != args.landscape:
                    continue
                jobs.update((c, ls, s) for s in seeds)
    return sorted(jobs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="5 seeds, N=100 (smoke test)")
    mode.add_argument("--full", action="store_true", help="50 seeds, N=300")
    p.add_argument("--aggregate", action="store_true",
                   help="write summaries from raw CSVs: results/summary/ (or results/summary_fast/ with --fast)")
    p.add_argument("--archive", metavar="NAME",
                   help="zip the raw BO runs (of --exp if given, else all) as results/archive/raw_bo_NAME.zip "
                        "with a manifest; never overwrites an existing archive")
    p.add_argument("--exp", choices=list(EXPERIMENTS))
    p.add_argument("--method", choices=["cmaes", "bo", "random"])
    p.add_argument("--landscape")
    p.add_argument("--seeds", type=int, help="override number of seeds (prefix of SEEDS)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    if args.archive:
        from archive import archive_bo
        from logging_utils import RAW_DIR
        files = None
        if args.exp:
            files = [f for c in EXPERIMENTS[args.exp]["configs"] if CONFIGS[c]["method"] == "bo"
                     for l in EXPERIMENTS[args.exp]["landscapes"]
                     for f in (RAW_DIR / l).glob(f"bo__{c}__seed*.csv")]
        z, m = archive_bo(args.archive, files)
        print(f"wrote {z} ({z.stat().st_size / 1e6:.2f} MB) and {m}")
        return
    if args.aggregate:
        from analysis import aggregate
        from logging_utils import RAW_DIR, RAW_FAST_DIR, SUMMARY_DIR, SUMMARY_FAST_DIR
        aggregate(*((RAW_FAST_DIR, SUMMARY_FAST_DIR) if args.fast else (RAW_DIR, SUMMARY_DIR)))
        return
    if not (args.fast or args.full):
        p.error("choose --fast, --full or --aggregate")

    N = N_FAST if args.fast else N_FULL
    # Fast runs go to a separate tree so they never mix with full results.
    raw_dir = Path(__file__).resolve().parent / "results" / ("raw_fast" if args.fast else "raw")
    jobs = [j for j in build_jobs(args)
            if not run_path(CONFIGS[j[0]]["method"], j[0], j[1], j[2], raw_dir).exists()]
    print(f"{len(jobs)} runs to do (N={N}, existing runs skipped)")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        futs = {pool.submit(run_one, c, ls, s, N, str(raw_dir)): (c, ls, s) for c, ls, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            c, ls, s = futs[fut]
            _, secs = fut.result()
            print(f"[{i}/{len(jobs)}] {c} {ls} seed={s} {secs:.1f}s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
