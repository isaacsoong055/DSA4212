"""Per-evaluation CSV logging. One file per (method, config, landscape, seed)."""
from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_DIR = ROOT / "results" / "summary"
RAW_FAST_DIR = ROOT / "results" / "raw_fast"
RAW_LONG_DIR = ROOT / "results" / "raw_long"          # Exp 1 side check, N = 1500, CMA-ES only
SUMMARY_LONG_DIR = ROOT / "results" / "summary_long"
SUMMARY_FAST_DIR = ROOT / "results" / "summary_fast"  # smoke-test summaries, never committed

BASE_COLS = ["method", "config_id", "landscape", "seed", "eval_idx"]
TAIL_COLS = ["f", "best_so_far", "sigma", "cov_trace", "lengthscales", "opt_time_s",
             "gen", "restart", "popsize", "in_basin", "n_jitter"]


def columns(d: int) -> list[str]:
    return BASE_COLS + [f"x_{i}" for i in range(d)] + TAIL_COLS


def run_path(method: str, config_id: str, landscape: str, seed: int, raw_dir: Path = RAW_DIR) -> Path:
    return Path(raw_dir) / landscape / f"{method}__{config_id}__seed{seed}.csv"


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ";".join(f"{u:.6g}" for u in v)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return repr(v)
    return v


def write_run(rows: list[dict], path: Path, d: int) -> None:
    """Write rows atomically (tmp file then rename) so partial runs never look done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    cols = columns(d)
    with open(tmp, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r in rows:
            wr.writerow([_fmt(r.get(c)) for c in cols])
    os.replace(tmp, path)
