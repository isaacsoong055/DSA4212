"""Epsilon for Rastrigin and Schwefel: cap from local minima, then the random-sampling rule.

The rule is written in docs/pilot_log.md (dated before this ran). Random sampling and landscape
geometry only; no CMA-ES or BO run is involved.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis import _grid, wilson_ci  # noqa: E402
from benchmarks import get_landscape, local_minima, nearest_nonglobal  # noqa: E402
from epsilon_pilot import random_best_regrets  # noqa: E402

GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
MAX_RANDOM_RATE = 0.05
N_STARTS = 1000
FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
RED, ORANGE = "#e34948", "#eb6834"


def stop(msg):
    print("STOP AND ASK:", msg)
    sys.exit(2)


def cap_for(ls):
    """Nearest non-global regret, checked to be unchanged when the number of starts is doubled."""
    m1, m2 = local_minima(ls, N_STARTS), local_minima(ls, 2 * N_STARTS)
    print(f"  distinct minima: {len(m1)} ({N_STARTS} starts), {len(m2)} ({2 * N_STARTS} starts)")
    for x, r in m2[:8]:
        print(f"    x = {np.round(x, 3)}  regret = {r:.6g}")
    (r1, l1), (r2, l2) = nearest_nonglobal(ls, m1), nearest_nonglobal(ls, m2)
    key = lambda L: sorted(map(tuple, np.round(L, 3)))  # noqa: E731
    if abs(r1 - r2) > 1e-6 * max(1.0, r1) or key(l1) != key(l2):
        stop(f"{ls.name}: nearest non-global changed with more starts: {r1} {key(l1)} -> {r2} {key(l2)}")
    print(f"  nearest non-global regret (cap): {r2:.6g}, unchanged at {N_STARTS} -> {2 * N_STARTS} starts; "
          f"{len(l2)} minima tied")
    return r2, l2


def choose(reg, cap):
    rate = lambda e: float(np.mean(reg <= e))  # noqa: E731
    grid = sorted(GRID)
    while rate(grid[-1]) <= MAX_RANDOM_RATE:      # extend upward by x2.5
        grid.append(grid[-1] * 2.5)
    while rate(grid[0]) > MAX_RANDOM_RATE:        # extend downward by halving
        grid.insert(0, grid[0] / 2)
        if grid[0] < 1e-4:
            stop("reached 1e-4 without random success <= 5%")
    ok = [e for e in grid if rate(e) <= MAX_RANDOM_RATE and e < cap]
    if not ok:
        stop("no grid value is both <= 5% random success and below the cap")
    e5 = max(e for e in grid if rate(e) <= MAX_RANDOM_RATE)
    return max(ok), grid, rate, ("cap" if e5 >= cap else "5% rule")


def figure(ls, eps, cap, minima_loc, reg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    XX, YY, Z = _grid(ls, 300)
    ax = axes[0]
    cs = ax.contourf(XX, YY, Z, levels=40, cmap="Greys_r")
    ax.contour(XX, YY, Z, levels=[ls.f_star + eps], colors=RED, linewidths=1.5)
    for x in minima_loc:
        ax.plot(*x, marker="x", color=ORANGE, ms=8, mew=2, ls="")
    ax.plot(*ls.x_star, marker="*", color="white", ms=14, mec="k", ls="")
    ax.set_aspect("equal"); ax.set_title(f"{ls.name}: star = global min, x = nearest non-global", fontsize=10)
    fig.colorbar(cs, ax=ax, shrink=0.8, label="f(x)")
    ax = axes[1]
    g = np.linspace(-1.5, 1.5, 241)
    X2, Y2 = np.meshgrid(g + ls.x_star[0], g + ls.x_star[1])
    Z2 = np.array([[ls([x, y]) for x in X2[0]] for y in Y2[:, 0]])
    ax.contourf(X2, Y2, Z2, levels=40, cmap="Greys_r")
    ax.contour(X2, Y2, Z2, levels=[ls.f_star + eps], colors=RED, linewidths=2)
    if cap <= Z2.max() - ls.f_star:
        ax.contour(X2, Y2, Z2, levels=[ls.f_star + cap], colors=ORANGE, linewidths=1.5, linestyles="--")
    ax.plot(*ls.x_star, marker="*", color="white", ms=12, mec="k", ls="")
    ax.set_aspect("equal"); ax.set_title(f"zoom: red = regret <= {eps:g}; dashed orange = cap {cap:.4g}", fontsize=10)
    ax = axes[2]
    ax.hist(reg, bins=np.geomspace(max(reg.min(), 1e-4), reg.max(), 30), color="#898783")
    ax.axvline(eps, color=RED, lw=2, label=f"epsilon = {eps:g}")
    ax.axvline(cap, color=ORANGE, lw=2, ls="--", label=f"cap = {cap:.4g}")
    ax.set_xscale("log"); ax.set_xlabel("best regret of a random N=300 run"); ax.set_ylabel("pilot seeds (of 200)")
    ax.set_title(f"random sampling succeeds {int((reg <= eps).sum())}/200", fontsize=10); ax.legend(frameon=False)
    path = FIG_DIR / f"epsilon_check_{ls.name}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print("  saved", path)


def main():
    t0 = time.perf_counter()
    for name in ("rastrigin", "schwefel"):
        ls = get_landscape(name)
        print(f"== {name}: f* = {ls.f_star:.4g} at {np.round(ls.x_star, 4)}")
        cap, tied = cap_for(ls)
        reg = random_best_regrets(ls)
        print("  random best-regret quantiles (5/25/50/75/95%):", np.round(np.quantile(reg, [.05, .25, .5, .75, .95]), 4))
        eps, grid, rate, limiter = choose(reg, cap)
        print(f"  {'eps':>9} {'k/200':>6} {'rate':>6} {'wilson95':>16} {'< cap':>6}")
        for e in grid:
            k = int((reg <= e).sum()); lo, hi = wilson_ci(k, len(reg))
            print(f"  {e:>9.5g} {k:>6} {k / len(reg):>6.3f}  [{lo:.3f}, {hi:.3f}] {str(e < cap):>6}")
        print(f"  CHOSEN epsilon({name}) = {eps:g}   (limited by: {limiter})")
        figure(ls, eps, cap, tied, reg)
    print(f"epsilon pilot wall-clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
