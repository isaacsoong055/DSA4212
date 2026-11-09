"""Build the report figures from results/ (no experiment is re-run).

Encoding, fixed across every figure so a method looks the same everywhere:
  colour     = method      CMA-ES cool (full blue, diagonal aqua),
                           BO warm (full metric red, ARD yellow)
  line style = geometry    solid = can tilt (full covariance, full GP metric)
                           dashed = axis-aligned (diagonal covariance, ARD kernel)
Palette: the dataviz skill's reference categorical slots blue, aqua, yellow and
red. Of all 70 four-colour sets in that palette, 16 pass every all-pairs check
(colour-blind separation and the normal-vision floor) and none contains orange,
so BO moved from orange to this warm pair. Aqua and yellow are below 3:1
contrast on the surface, so those methods are always named in a legend or
label as well; line style (solid / dashed) still repeats the geometry.
Reflected CMA-ES (no current figure plots it) takes slot 7, violet.

Writes report/figures/fig*.pdf (for the report) and .png (for previews).
Usage: python code/make_figures.py
"""

import itertools
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "report" / "figures"
sys.path.insert(0, str(ROOT / "code"))
import analysis

BLUE, AQUA, RED, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#e34948", "#eda100", "#6250d6"
ORANGE = "#eb6834"   # no longer used for a method; kept for reference
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

STYLE = {   # method -> (label, colour, dash)
    "cmaes-full": ("CMA-ES, full", BLUE, "-"),
    "cmaes-diagonal": ("CMA-ES, diagonal", AQUA, "--"),
    "bo-full": ("BO, full metric", RED, "-"),
    "bo-ard": ("BO, ARD", YELLOW, "--"),
    "cmaes-full-reflect": ("CMA-ES, full + reflection", VIOLET, "-"),
    "bo-ard-fitnoise": ("BO, ARD, noise fitted (original)", YELLOW, ":"),
}
GEO = {"none": "aligned", "random": "Haar-random rotation"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
    "legend.frameon": False, "axes.titleweight": "regular", "axes.titlelocation": "left",
    "savefig.dpi": 200, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def line(ax, x, y, method, **kw):
    label, colour, dash = STYLE[method]
    return ax.plot(x, y, color=colour, linestyle=dash, label=label, **kw)


def fig1_e1_angle_sweep():
    """E1: diagonal CMA-ES fails only between the axes."""
    runs = pd.read_csv(R / "e1_angle_sweep_runs.csv")
    target = 1e-8                                   # E1's success target (notebook 8.1)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for ax, kappa in zip(axes, (100, 1000)):
        sub = runs[runs.condition == kappa]
        for variant in ("full", "diagonal"):
            s = sub[sub.variant == variant].groupby("angle").best_value \
                .apply(lambda v: 100 * (v <= target).mean())
            line(ax, s.index, s.values, f"cmaes-{variant}", marker="o", markersize=4,
                 markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_title(f"κ = {kappa:,}")
        if kappa == 100:
            ax.text(45, 55, "both variants reach $10^{-8}$\nin every run at every angle",
                    ha="center", va="center", color=INK2, fontsize=7.5)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_xlabel("rotation angle of the valley (degrees)")
        ax.set_ylim(-5, 105)
    axes[0].set_ylabel(f"% of {sub.instance.nunique()} instances reaching\n$10^{{-8}}$ within 4,000 evaluations")
    axes[1].legend(loc="center right")
    fig.suptitle("E1. Diagonal CMA-ES fails only when the valley lies between the axes",
                 x=0.0, ha="left", fontsize=9.5, y=1.04)
    save(fig, "fig1_e1_angle_sweep")


def fig2_e2c_axis_error():
    """E2c: ARD cannot represent a rotated valley; the full metric can."""
    ls = pd.read_csv(R / "e2c_lengthscales.csv")
    ls = ls[ls.condition > 1]                       # no long axis to find on a round bowl
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for ax, geo in zip(axes, ("45 deg", "random rotation")):
        sub = ls[ls.geometry == geo]
        for metric in ("full", "ard"):
            g = sub[sub.metric == metric].groupby("condition").axis_error_deg
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            label, colour, dash = STYLE[f"bo-{metric}"]
            ax.fill_between(med.index, lo, hi, color=colour, alpha=0.10, linewidth=0)
            line(ax, med.index, med.values, f"bo-{metric}", marker="o", markersize=4,
                 markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_xscale("log")
        ax.set_xticks(sorted(sub.condition.unique()))
        ax.set_xticklabels([f"{k:,}" for k in sorted(sub.condition.unique())])
        ax.set_xlabel("condition number κ")
        ax.set_title("valley at 45°" if geo == "45 deg" else "Haar-random rotation")
    axes[0].set_ylabel("error of learned long axis\n(degrees; median, IQR band)")
    axes[0].legend(loc="center right")
    fig.suptitle("E2c. The ARD kernel cannot align with a rotated valley; the full metric can",
                 x=0.0, ha="left", fontsize=9.5, y=1.04)
    save(fig, "fig2_e2c_axis_error")


def fig3_e2_curves():
    """E2 with the BO fix: median best-so-far for all four methods."""
    traj = pd.read_csv(R / "e2_bofix_trajectories.csv")
    traj = traj.assign(l=np.log10(traj.best_so_far + 1e-300))
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 6.0), sharex=True)
    for (i, kappa), (j, geo) in itertools.product(enumerate((1, 100, 10000)),
                                                  enumerate(("none", "random"))):
        ax = axes[i, j]
        sub = traj[(traj.condition == kappa) & (traj.geometry == geo)]
        for method in ("cmaes-full", "cmaes-diagonal", "bo-full", "bo-ard"):
            m = sub[sub.method == method].groupby("evaluation").l.median()
            line(ax, m.index, m.values, method)
        ax.set_title(f"κ = {kappa:,}, {GEO[geo]}")
        if j == 0:
            ax.set_ylabel("median log$_{10}$ best $f$")
        if i == 2:
            ax.set_xlabel("objective evaluations")
        ax.set_xlim(0, 200)
    for i in range(3):          # one y-range per row, so the two geometries compare directly
        lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo, hi)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.005))
    fig.suptitle("E2. Head to head under one budget rule (15 instances; BO with the noise fix)",
                 x=0.0, ha="left", fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, "fig3_e2_curves")


def interval(ax, y, r, colour, filled=True, size=5):
    ax.plot([r.ci_low, r.ci_high], [y, y], color=colour, linewidth=1.6,
            alpha=1.0 if filled else 0.55, solid_capstyle="round")
    ax.plot(r.difference, y, "o", markersize=size, color=colour if filled else SURFACE,
            markeredgecolor=colour if not filled else SURFACE, markeredgewidth=1.4 if not filled else 1.2)


def fig4_rotation_penalty():
    """The rotation penalty at 200 evaluations, before and after the BO fix."""
    old = pd.read_csv(R / "e2_local_summary.csv")   # fitted-noise BO, same machine as the fix
    new = pd.read_csv(R / "e2_bofix_summary.csv")
    pen = lambda s: s[s.comparison.str.endswith("rotated - aligned")].assign(
        method=lambda d: d.comparison.str.split(":").str[0])
    old, new = pen(old), pen(new)
    methods = ["cmaes-full", "cmaes-diagonal", "bo-full", "bo-ard"]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.6), sharex=True, sharey=True)
    for ax, kappa in zip(axes, (1, 100, 10000)):
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
        for k, method in enumerate(methods):
            y = len(methods) - 1 - k
            colour = STYLE[method][1]
            r = new[(new.condition == kappa) & (new.method == method)].iloc[0]
            if method.startswith("bo"):
                ro = old[(old.condition == kappa) & (old.method == method)].iloc[0]
                interval(ax, y + 0.18, ro, colour, filled=False)
                interval(ax, y - 0.18, r, colour)
            else:
                interval(ax, y, r, colour)
        ax.set_title(f"κ = {kappa:,}")
        ax.set_xlabel("rotated − aligned, log$_{10}$ f\n(+ = rotation hurts)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([STYLE[m][0] for m in reversed(methods)])
    axes[0].tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE, markeredgecolor=INK2,
                   linestyle="", label="BO before the noise fix"),
            Line2D([], [], marker="o", color=INK2, linestyle="", label="after (CMA-ES unchanged)")]
    fig.legend(handles=keys, loc="upper center", ncol=2, bbox_to_anchor=(0.55, -0.1))
    fig.suptitle("Rotation penalty at 200 evaluations: median and 95% bootstrap interval, 15 instances",
                 x=0.0, ha="left", fontsize=9.5, y=1.03)
    save(fig, "fig4_rotation_penalty")


def fig5_e3_invariance():
    """E3: condition minus baseline at 200 evaluations."""
    s = pd.read_csv(R / "e3_summary.csv")
    s = s[s.budget == 200]
    conditions = ["units-default:1,10", "units-default:10,1", "units-matched:1,10",
                  "units-matched:10,1", "monotone:log1p", "monotone:cube", "monotone:sqrt",
                  "monotone:plus1"]
    names = {"units-default:1,10": "units ×(1, 10), default set-up",
             "units-default:10,1": "units ×(10, 1), default set-up",
             "units-matched:1,10": "units ×(1, 10), C₀ matched",
             "units-matched:10,1": "units ×(10, 1), C₀ matched",
             "monotone:log1p": "f → log(1 + f)", "monotone:cube": "f → f³",
             "monotone:sqrt": "f → √f", "monotone:plus1": "f → f + 1"}
    methods = ["cmaes-full", "cmaes-diagonal", "bo-full", "bo-ard"]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 3.0), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
        for k, cond in enumerate(conditions):
            y = len(conditions) - 1 - k
            r = s[(s.condition == cond) & (s.method == method)]
            if r.empty:
                ax.text(0, y, "not run", ha="center", va="center", color=MUTED, fontsize=7)
                continue
            r = r.iloc[0]
            interval(ax, y, r, STYLE[method][1])
            if r.identical_runs == r.n_runs:
                ax.text(r.difference + 0.25, y + 0.28, "identical", color=INK2, fontsize=6.5)
        ax.axhline(3.5, color=GRID, linewidth=0.8)
        ax.set_title(STYLE[method][0])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(conditions)))
    axes[0].set_yticklabels([names[c] for c in reversed(conditions)])
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel("condition − baseline, median log$_{10}$ f at 200 evaluations (+ = worse)",
                  fontsize=8.5, color=INK2, y=-0.02)
    fig.suptitle("E3. Invariance tests (aligned κ = 100; paired by instance; 95% bootstrap intervals)",
                 x=0.0, ha="left", fontsize=9.5, y=1.02)
    save(fig, "fig5_e3_invariance")


def fig6_e4_boundary():
    """E4: position minus the same-distance interior control, at 200 evaluations.

    Reflected CMA-ES is left out: its interior control is the one condition where
    reflection itself fails (the mean drifts onto a mirror copy of the optimum), so
    "position minus control" would credit the edge with reflection's own failure.
    Notebook 8.9 compares reflection with unconstrained CMA-ES directly instead.
    """
    s = pd.read_csv(R / "e4_summary.csv")
    s = s[(s.budget == 200) & s.position.isin(["near-edge", "on-edge", "corner"])]
    methods = ["bo-ard", "cmaes-full"]
    rows = [(k, p) for k in (1, 100) for p in ("near-edge", "on-edge", "corner")]
    names = {"near-edge": "0.1 inside the edge", "on-edge": "on the edge", "corner": "in the corner"}
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
        for k, (kappa, pos) in enumerate(rows):
            y = len(rows) - 1 - k
            if method == "bo-ard" and pos == "corner":
                ax.text(-2.6, y, "exact hit (f = 0)", ha="left", va="center",
                        color=INK2, fontsize=7)
                continue
            r = s[(s.condition == kappa) & (s.position == pos) & (s.method == method)].iloc[0]
            interval(ax, y, r, STYLE[method][1])
        ax.axhline(2.5, color=GRID, linewidth=0.8)
        ax.set_title(STYLE[method][0])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-2.8, 2.6)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"κ = {k}: {names[p]}" for k, p in reversed(rows)])
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel("position − interior control at the same distance,\nmedian log$_{10}$ f "
                  "at 200 evaluations (+ = worse)", fontsize=8.5, color=INK2, y=-0.08)
    fig.suptitle("E4. Optimum near the box edge (2-D aligned; paired; 95% bootstrap intervals)",
                 x=0.0, ha="left", fontsize=9.5, y=1.03)
    save(fig, "fig6_e4_boundary")


def fig7_e5_other_functions():
    """E5: Rosenbrock and the noisy quadratic, median recommended (noise-free) value."""
    traj = pd.read_csv(R / "e5_trajectories.csv")
    traj = traj.assign(l=np.log10(traj.recommended_true + 1e-300))
    panels = [("rosenbrock:none", "Rosenbrock f8 (aligned)"),
              ("rosenbrock:random", "Rosenbrock f9 (rotated)"),
              ("noisy:0.01", "κ = 100 quadratic, noise σ = 0.01"),
              ("noisy:1", "κ = 100 quadratic, noise σ = 1")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for ax, (problem, title) in zip(axes.flat, panels):
        sub = traj[traj.problem == problem]
        for method in sub.method.unique():
            m = sub[sub.method == method].groupby("evaluation").l.median()
            line(ax, m.index, m.values, method)
        ax.set_title(title)
        ax.set_xlim(0, 200)
    for ax in axes[:, 0]:
        ax.set_ylabel("median log$_{10}$ f\nof the recommended point")
    for ax in axes[1]:
        ax.set_xlabel("objective evaluations")
    handles, labels = {}, {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles[l] = h
    order = ["CMA-ES, full", "CMA-ES, diagonal", "BO, full metric", "BO, ARD",
             "BO, ARD, noise fitted (original)"]
    fig.legend([handles[l] for l in order], order, loc="upper left", ncol=3,
               bbox_to_anchor=(0.0, 0.975))
    fig.suptitle("E5. Beyond the quadratic (15 instances; recommendation = lowest observed value,"
                 " scored on noise-free f)", x=0.0, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig7_e5_other_functions")


def fig8_e6_noise():
    """E6: recommendation rule and budget split under noise (paired, 200 noisy evaluations)."""
    s = pd.read_csv(R / "e6_summary.csv")
    panels = [
        ("Recommendation rule: model estimate − lowest observed", [
            ("cmaes-base: R2 - R1", "CMA-ES, λ = 6: mean", BLUE),
            ("cmaes-pop24: R2 - R1", "CMA-ES, λ = 24: mean", BLUE),
            ("bo-base: R2 - R1", "BO: posterior-mean minimiser", YELLOW)]),
        ("Budget split − baseline (both read through the model)", [
            ("cmaes-resample - cmaes-base (R2)", "CMA-ES: resample ×4", BLUE),
            ("cmaes-pop24 - cmaes-base (R2)", "CMA-ES: population ×4", BLUE),
            ("cmaes-reserve R3 - cmaes-base R2", "CMA-ES: final re-evaluation", BLUE),
            ("bo-replicate - bo-base (R2)", "BO: replicates ×4", YELLOW),
            ("bo-reserve R3 - bo-base R2", "BO: final re-evaluation", YELLOW)]),
    ]
    sigmas = (0.01, 0.1, 1.0)
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.2),
                             gridspec_kw={"height_ratios": [9, 15], "hspace": 0.35})
    for ax, (title, rows) in zip(axes, panels):
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
        labels, y = [], 0
        for sigma in reversed(sigmas):
            for comparison, label, colour in reversed(rows):
                r = s[(s.comparison == comparison) & (np.isclose(s.sigma, sigma))].iloc[0]
                interval(ax, y, r, colour)
                labels.append(f"σ = {sigma:g} · {label}")
                y += 1
            ax.axhline(y - 0.5, color=GRID, linewidth=0.8)
            y += 0.0
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("difference in median log$_{10}$ f of the recommended point (− = better)")
    fig.suptitle("E6. Noisy evaluations (κ = 100 quadratic; 200 noisy evaluations;\npaired; 95% bootstrap intervals)",
                 x=0.0, ha="left", fontsize=9.5, y=0.99)
    save(fig, "fig8_e6_noise")


if __name__ == "__main__":
    fig1_e1_angle_sweep()
    fig2_e2c_axis_error()
    fig3_e2_curves()
    fig4_rotation_penalty()
    fig5_e3_invariance()
    fig6_e4_boundary()
    fig7_e5_other_functions()
    fig8_e6_noise()
