"""Pilot step 2: choose epsilon from uniform random sampling only (seeds 9000+).

Rule, fixed before running: epsilon = largest value in EPS_GRID for which random
sampling at N=300 succeeds in <= 5% of the 200 pilot seeds on primary_far_w0.3.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis import plot_landscape, wilson_ci  # noqa: E402
from benchmarks import get_landscape  # noqa: E402
from configs import N_FULL, PILOT_SEEDS, PRIMARY  # noqa: E402
from optimizers import STREAM_RANDOM, _rng  # noqa: E402

EPS_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
MAX_RANDOM_RATE = 0.05
FIG = Path(__file__).resolve().parent.parent / "docs" / "figures" / "epsilon_check.png"


def random_best_regrets(ls, N=N_FULL, seeds=PILOT_SEEDS):
    """Same draws as optimizers.run_random, without the per-run basin checks."""
    out = []
    for s in seeds:
        X = _rng(s, STREAM_RANDOM).uniform(-5, 5, size=(N, ls.d))
        out.append(min(ls(x) for x in X) - ls.f_star)
    return np.array(out)


def radius_for(ls, eps, n=4000):
    """Approx. radius around x_star inside which regret <= eps (along the worst direction)."""
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rs = np.linspace(0, 1.5, n)
    worst = np.inf
    for a in angles:
        u = np.array([np.cos(a), np.sin(a)])
        reg = np.array([ls(ls.x_star + r * u) - ls.f_star for r in rs])
        worst = min(worst, rs[np.argmax(reg > eps)] if (reg > eps).any() else rs[-1])
    return worst


def main():
    ls = get_landscape(PRIMARY)
    reg = random_best_regrets(ls)
    print(f"{PRIMARY}: f* = {ls.f_star:.4f} at {np.round(ls.x_star, 4)}; "
          f"decoy f(c) - f* = {ls(ls.meta['c']) - ls.f_star:.3f}")
    print("random best-regret quantiles (5/25/50/75/95%):",
          np.round(np.quantile(reg, [.05, .25, .5, .75, .95]), 4))
    print(f"{'eps':>6} {'k/200':>6} {'rate':>6} {'wilson95':>16} {'r_eps':>6} {'r_eps/w':>7}")
    chosen = None
    for e in EPS_GRID:
        k = int((reg <= e).sum())
        lo, hi = wilson_ci(k, len(reg))
        r = radius_for(ls, e)
        print(f"{e:>6} {k:>6} {k/len(reg):>6.3f}  [{lo:.3f}, {hi:.3f}] {r:>6.3f} {r/ls.meta['w']:>7.2f}")
        if k / len(reg) <= MAX_RANDOM_RATE:
            chosen = e
    print(f"CHOSEN epsilon = {chosen}")

    # Random-sampling reference at every Exp 8 landscape, under the chosen epsilon.
    print("random success under chosen epsilon (Exp 8 reference):")
    for loc in ("near", "far"):
        for w in (0.1, 0.3, 0.6, 1.0):
            l2 = get_landscape(f"primary_{loc}_w{w}")
            k = int((random_best_regrets(l2) <= chosen).sum())
            lo, hi = wilson_ci(k, 200)
            print(f"  {l2.name:<20} {k:>3}/200  [{lo:.3f}, {hi:.3f}]")
    plot(ls, chosen, reg)


def plot(ls, eps, reg):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    cs, (XX, YY, Z) = plot_landscape(ax, ls, levels=30)
    ax.contour(XX, YY, Z, levels=[ls.f_star + eps], colors="#e34948", linewidths=2)
    ax.plot(*ls.x_star, marker="*", color="#e34948", ms=10, mec="k", mew=0.5)
    ax.plot(*ls.meta["c"], marker="x", color="k", ms=8)
    ax.set_title(f"{ls.name}; ★ = x*, × = decoy centre\n(success region too small to see here, see zoom)", fontsize=10)
    fig.colorbar(cs, ax=ax, shrink=0.8, label="f(x)")

    ax = axes[1]
    g = np.linspace(-0.6, 0.6, 241)
    X2, Y2 = np.meshgrid(g + ls.x_star[0], g + ls.x_star[1])
    Z2 = np.array([[ls([x, y]) for x in X2[0]] for y in Y2[:, 0]])
    cs2 = ax.contourf(X2, Y2, Z2, levels=30, cmap="Greys_r")
    ax.contour(X2, Y2, Z2, levels=[ls.f_star + eps], colors="#e34948", linewidths=2)
    circ = plt.Circle(ls.meta["x_g"], ls.meta["w"], fill=False, ls="--", color="#2a78d6", lw=1.5)
    ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.set_title(f"zoom: red = regret ≤ {eps}; dashed blue = radius w = {ls.meta['w']}", fontsize=10)
    fig.colorbar(cs2, ax=ax, shrink=0.8, label="f(x)")

    ax = axes[2]
    xs = np.linspace(-5, 5, 2001)
    ax.plot(xs, [ls([x, x]) for x in xs], color="#2a78d6", lw=2, label="f along x1 = x2")
    ax.axhline(ls.f_star + eps, color="#e34948", lw=1.5, ls="--", label=f"f* + ε  (ε = {eps})")
    ax.set_xlabel("x1 = x2")
    ax.set_ylabel("f")
    ax.set_title(f"diagonal slice; random N=300 succeeds {int((reg <= eps).sum())}/200", fontsize=10)
    ax.legend(frameon=False)
    for a in axes[:2]:
        a.set_xlabel("x1")
        a.set_ylabel("x2")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
