"""Pilot step 3: sigma trace and sample scatter for the 5-seed baseline runs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis import load_raw, plot_scatter, plot_sigma_trace  # noqa: E402
from benchmarks import get_landscape  # noqa: E402
from configs import PRIMARY  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures" / "pilot_baselines.png"
SHOW_SEED = 1000

df = load_raw(landscape=PRIMARY)
ls = get_landscape(PRIMARY)
cma = df[df.config_id == "cma_default"]
bo = df[df.config_id == "bo_default"]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
ax = axes[0]
for s, g in cma.groupby("seed"):
    plot_sigma_trace(ax, g, label=f"seed {s}")
ax.set_title("CMA-ES sigma per generation (5 seeds)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
plot_scatter(axes[1], ls, cma[cma.seed == SHOW_SEED], f"CMA-ES samples, seed {SHOW_SEED}")
sc = plot_scatter(axes[2], ls, bo[bo.seed == SHOW_SEED], f"BO samples, seed {SHOW_SEED}")
fig.colorbar(sc, ax=axes[2], shrink=0.8, label="eval index")
ax = axes[3]
for s, g in bo.groupby("seed"):
    L = g["lengthscales"].dropna().str.split(";", expand=True).astype(float)
    ax.semilogy(g.loc[L.index, "eval_idx"], L.min(axis=1), lw=1.5, label=f"seed {s}")
ax.axhline(ls.meta["w"], color="#e34948", ls="--", lw=1.5, label="well width w")
ax.set_xlabel("eval index")
ax.set_ylabel("min fitted lengthscale (domain units)")
ax.set_title("BO fitted lengthscale vs w (5 seeds)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
for a in axes[1:3]:
    a.set_xlabel("x1")
    a.set_ylabel("x2")
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print("saved", FIG)
