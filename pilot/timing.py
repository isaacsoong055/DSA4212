"""Pilot step 1: wall-clock seconds for one run per config at N=300 (prints timing only)."""
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import torch  # noqa: E402

torch.set_num_threads(1)
from benchmarks import get_landscape  # noqa: E402
from configs import CONFIGS, N_FULL, PRIMARY  # noqa: E402
from optimizers import run  # noqa: E402

cid = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
t0 = time.perf_counter()
log = run(get_landscape(PRIMARY), N_FULL, seed, CONFIGS[cid], cid)
wall = time.perf_counter() - t0
opt = sum(r.get("opt_time_s") or 0 for r in log.rows)
jit = log.meta.get("n_jitter", "")
print(f"{cid:<14} seed={seed} wall={wall:7.1f}s  optimizer={opt:7.1f}s  jitter_fallbacks={jit}"
      f"  fit_failures={log.meta.get('n_fit_fail', '')}", flush=True)
