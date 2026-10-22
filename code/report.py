"""Figures, tables and loaders for notebooks/main.ipynb.

Only loads saved files (results/summary*/, results/raw*/) and draws. Heavy work (runs,
aggregation) lives in run_experiments.py and analysis.py. The one exception is `cma_rerun`,
a ~0.05 s deterministic re-run of seed 1000 used to draw the search ellipses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

import analysis as an
from analysis import BLUE, GREY, ORANGE, OUTCOME_COLOURS, OUTCOMES
from benchmarks import get_landscape
from configs import CONFIGS, EXPERIMENTS, FRESH_SEEDS, N_FAST, N_FULL, N_SEEDS_FAST, N_SEEDS_FULL, PRIMARY, SEEDS
from logging_utils import ROOT, SUMMARY_DIR, SUMMARY_FAST_DIR, RAW_DIR, RAW_FAST_DIR, run_path
from optimizers import LS_BOUNDS_UNIT, run

RED = "#e34948"
SMOKE_TOL = 1e-8
COLOUR = {"cmaes": BLUE, "bo": ORANGE, "random": GREY}
LABEL = {"cma_default": "default (λ=6)", "cma_lam12": "λ=12", "cma_lam24": "λ=24", "cma_ipop": "IPOP",
         "cma_bipop": "BIPOP", "bo_default": "EI ξ=0", "bo_ei_xi0.01": "EI ξ=0.01", "bo_ei_xi0.1": "EI ξ=0.1",
         "bo_lcb_b0.5": "LCB β=0.5", "bo_lcb_b2": "LCB β=2", "bo_lcb_b8": "LCB β=8", "bo_ts": "Thompson",
         "bo_map": "EI, MAP fit", "random": "random"}
ORDER = ["exp0", "expA", "expB", "expC", "expD", "expE_baseline", "expV"]
EXP_TITLE = {"exp0": "Exp 0: baselines", "expA": "Exp A: population size", "expB": "Exp B: restarts",
             "expC": "Exp C: BO exploration", "expD": "Exp D: unimodal control",
             "expE_baseline": "Exp E: basin width (baselines)", "expV": "Exp V: validation (Rastrigin, Schwefel)"}


@lru_cache(maxsize=None)
def canon(name: str) -> str:
    """Canonical landscape name as written in the CSVs (e.g. 'primary_far_w1.0' -> 'primary_far_w1')."""
    return get_landscape(name).name


def colour_for(config_id: str) -> str:
    return COLOUR[CONFIGS[config_id]["method"]]


@dataclass(frozen=True)
class Mode:
    """FAST = smoke test (N=100, 5 seeds, results/*_fast); otherwise the full results."""
    fast: bool

    @property
    def N(self): return N_FAST if self.fast else N_FULL
    @property
    def n_seeds(self): return N_SEEDS_FAST if self.fast else N_SEEDS_FULL
    @property
    def tag(self): return "_fast" if self.fast else ""
    @property
    def raw_dir(self): return RAW_FAST_DIR if self.fast else RAW_DIR
    @property
    def summary_dir(self): return SUMMARY_FAST_DIR if self.fast else SUMMARY_DIR
    @property
    def stamp_text(self): return f"SMOKE TEST, N={N_FAST}, {N_SEEDS_FAST} seeds" if self.fast else ""

    def table(self, name: str) -> pd.DataFrame | None:
        path = self.summary_dir / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None


# ----------------------------------------------------------------- plumbing

def command(exp: str, M: Mode) -> str:
    return (f"python run_experiments.py {'--fast' if M.fast else '--full'} --exp {exp} && "
            f"python run_experiments.py --aggregate{' --fast' if M.fast else ''}")


def not_run(exp: str, M: Mode, why: str = "No saved results found.") -> None:
    display(Markdown(f"> **Not run yet: {EXP_TITLE.get(exp, exp)}.** {why}  \n> Run: `{command(exp, M)}`"))


def not_configured(exp: str) -> None:
    display(Markdown(f"> **Not configured yet: {EXP_TITLE[exp]}.** Its settings depend on the Exp A-C results "
                     f"and are added to `code/configs.py` by the rule in `docs/direction.md`."))


def _finish(fig, name: str, M: Mode, stamp: bool = True) -> None:
    if stamp and M.fast:
        fig.text(0.5, 0.5, M.stamp_text, fontsize=26, color=RED, alpha=0.22, rotation=20, ha="center", va="center")
        fig.text(0.995, 0.005, M.stamp_text, fontsize=8, color=RED, ha="right", va="bottom")
    an.save_fig(fig, f"{name}{M.tag}.png")
    plt.show()
    plt.close(fig)


def main_runs(M: Mode) -> pd.DataFrame | None:
    runs = M.table("runs")
    return None if runs is None else runs[runs["seed"].isin(SEEDS)]


def read_run(config_id: str, landscape: str, seed: int, raw_dir) -> pd.DataFrame | None:
    path = run_path(CONFIGS[config_id]["method"], config_id, landscape, seed, raw_dir)
    return pd.read_csv(path) if path.exists() else None


def _sub(runs: pd.DataFrame, configs, landscape: str) -> pd.DataFrame:
    return runs[runs["config_id"].isin(configs) & (runs["landscape"] == canon(landscape))]


# ----------------------------------------------------------------- status

def status_table(M: Mode) -> None:
    runs = M.table("runs")
    counts = None if runs is None else runs[runs["seed"].isin(SEEDS)].groupby(["config_id", "landscape"]).size()
    rows = []
    for e in ORDER:
        if e not in EXPERIMENTS:
            rows.append((EXP_TITLE[e], "not configured (depends on A-C results)", "-"))
            continue
        pairs = [(c, canon(l)) for c in EXPERIMENTS[e]["configs"] for l in EXPERIMENTS[e]["landscapes"]]
        n = [0 if counts is None else int(counts.get(p, 0)) for p in pairs]
        have = sum(x > 0 for x in n)
        if have == 0:
            status = "missing"
        elif have == len(pairs) and min(n) >= M.n_seeds:
            status = f"loaded ({M.n_seeds} seeds)"
        else:
            status = f"partial ({have}/{len(pairs)} cells, fewest seeds {min(n)} of {M.n_seeds})"
        rows.append((EXP_TITLE[e], status, command(e, M)))
    df = pd.DataFrame(rows, columns=["experiment", "results", "command to produce"])
    display(df.style.hide(axis="index").set_properties(**{"text-align": "left"}))


# ----------------------------------------------------------------- predictions and scoreboard

def load_predictions() -> pd.DataFrame:
    """All rows with an ID like H1 or M3 from docs/predictions.md (text unedited)."""
    rows = []
    for line in (ROOT / "docs" / "predictions.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.fullmatch(r"[A-Z]\d+", cells[0]):
            rows.append(cells)
    return pd.DataFrame(rows, columns=["ID", "Prediction", "Supported by", "Contradicted by"])


def show_predictions(ids: list[str]) -> None:
    """Display the given prediction rows exactly as written in docs/predictions.md."""
    p = load_predictions().set_index("ID").loc[ids]
    md = "| ID | Prediction | Supported by | Contradicted by |\n| --- | --- | --- | --- |\n"
    md += "\n".join(f"| {i} | {r['Prediction']} | {r['Supported by']} | {r['Contradicted by']} |" for i, r in p.iterrows())
    display(Markdown(md))


def scoreboard() -> None:
    """Predictions joined with docs/scoreboard.csv (edited by hand only; default 'not yet run')."""
    p = load_predictions()
    path = ROOT / "docs" / "scoreboard.csv"
    st = pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame(columns=["id", "status", "note"])
    df = p.merge(st, left_on="ID", right_on="id", how="left").drop(columns="id")
    df["status"] = df["status"].fillna("not yet run").replace("", "not yet run")
    df = df.rename(columns={"status": "Status"})[["ID", "Prediction", "Supported by", "Contradicted by", "Status", "note"]]
    display(df.style.hide(axis="index").set_properties(**{"text-align": "left", "white-space": "normal"}))


# ----------------------------------------------------------------- setup figure

def landscape_figure(M: Mode) -> None:
    ls = get_landscape(PRIMARY)
    fig = plt.figure(figsize=(13, 5.2), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    an.surface3d(ax, ls)
    ax.set_title("surface")
    ax = fig.add_subplot(1, 2, 2)
    cs = an.mark_landscape(ax, ls)
    fig.colorbar(cs, ax=ax, label="f(x)")
    ax.legend(loc="lower left")
    ax.set_title("contours")
    _finish(fig, "nb_1_landscape", M, stamp=False)
    print(f"true minimum x* = {ls.x_star.round(3)}, f* = {ls.f_star:.3f}; decoy minimum f = {ls(ls.meta['c']):.3f}; "
          f"epsilon = {ls.epsilon}")


# ----------------------------------------------------------------- illustrations (single runs)

def cma_rerun(N: int, seed: int, raw_dir, landscape: str = PRIMARY):
    """Deterministic re-run of default CMA-ES. Returns (rows, dist, max |dx| vs saved CSV or None)."""
    log = run(get_landscape(landscape), N, seed, CONFIGS["cma_default"], "cma_default")
    rows = pd.DataFrame(log.rows)
    saved = read_run("cma_default", landscape, seed, raw_dir)
    if saved is None:
        return rows, log.meta["dist"], None
    if len(saved) != len(rows):
        return rows, log.meta["dist"], float("inf")
    diff = float(np.max(np.abs(rows[["x_0", "x_1"]].to_numpy() - saved[["x_0", "x_1"]].to_numpy())))
    return rows, log.meta["dist"], diff


def cmaes_illustration(M: Mode, seed: int = SEEDS[0]) -> None:
    ls = get_landscape(PRIMARY)
    rows, dist, diff = cma_rerun(M.N, seed, M.raw_dir)
    if diff is None:
        return not_run("exp0", M, "The saved CMA-ES run needed for this illustration is missing.")
    gens = np.unique(np.geomspace(1, rows["gen"].max() + 1, 4).round().astype(int) - 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    sc = an.scatter_run(a1, ls, rows, by="gen")
    an.draw_ellipses(a1, dist, gens)
    fig.colorbar(sc, ax=a1, label="generation")
    a1.legend(loc="lower left")
    a1.set_title(f"ILLUSTRATION, one run (seed {seed}): CMA-ES samples")
    an.plot_sigma_trace(a2, rows)
    a2.set_title("step size sigma per generation")
    if diff > SMOKE_TOL:
        fig.suptitle(f"WARNING: re-run differs from the saved CSV (max |dx| = {diff:.3g}); ellipses may not match samples",
                     color=RED, fontsize=11)
    _finish(fig, "nb_3_cmaes_seed1000", M)
    print(f"best regret = {ls.regret(rows['f'].min()):.3f}; final sigma = {rows['sigma'].iloc[-1]:.4f}; "
          f"re-run vs saved CSV max |dx| = {diff:.1e}")


def bo_illustration(M: Mode, seed: int = SEEDS[0]) -> None:
    ls = get_landscape(PRIMARY)
    bo = read_run("bo_default", PRIMARY, seed, M.raw_dir)
    if bo is None:
        return not_run("exp0", M, "The saved BO run needed for this illustration is missing.")
    fig, ax = plt.subplots(figsize=(7, 5.6), layout="constrained")
    sc = an.scatter_run(ax, ls, bo, n_init=CONFIGS["bo_default"]["n_init"])
    fig.colorbar(sc, ax=ax, label="evaluation index")
    ax.legend(loc="lower left")
    ax.set_title(f"ILLUSTRATION, one run (seed {seed}): BO (EI, ξ=0)")
    _finish(fig, "nb_3_bo_seed1000", M)
    print(f"best regret = {ls.regret(bo['f'].min()):.3f}; final fitted lengthscales (domain units) = "
          f"{bo['lengthscales'].dropna().iloc[-1]}; well width w = {ls.meta['w']}")


def seed_grid(M: Mode, seeds=SEEDS[:4]) -> None:
    ls = get_landscape(PRIMARY)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True, layout="constrained")
    sc = None
    for j, seed in enumerate(seeds):
        for i, cid in enumerate(["cma_default", "bo_default"]):
            df = read_run(cid, PRIMARY, seed, M.raw_dir)
            if df is None:
                axes[i, j].set_title(f"seed {seed}: run missing", fontsize=10)
                continue
            sc = an.scatter_run(axes[i, j], ls, df, n_init=CONFIGS[cid]["n_init"] if cid.startswith("bo") else 0)
            axes[i, j].set_title(f"{'CMA-ES' if i == 0 else 'BO'}, seed {seed}: regret {ls.regret(df['f'].min()):.2f}", fontsize=10)
    if sc is None:
        plt.close(fig)
        return not_run("exp0", M, "None of the saved runs for this illustration exist.")
    fig.colorbar(sc, ax=axes, shrink=0.6, label="evaluation index")
    fig.suptitle("ILLUSTRATION, first four seeds by fixed rule (not selected)", fontsize=12)
    _finish(fig, "nb_3_grid_seeds1000_1003", M)


# ----------------------------------------------------------------- success rates (shared style)

def success_bars(ax, runs: pd.DataFrame, configs, landscape: str) -> pd.DataFrame:
    """Bars of success rate with Wilson 95% CI, k/n above each bar. Colour = method."""
    tab = an.success_table(_sub(runs, configs, landscape))
    tab = tab[tab["metric"] == "success"].set_index("config_id")
    order = [c for c in configs if c in tab.index]
    tab = tab.loc[order]
    x = np.arange(len(tab))
    ax.bar(x, tab["rate"], color=[colour_for(c) for c in tab.index], width=0.55)
    ax.errorbar(x, tab["rate"], yerr=[tab["rate"] - tab["ci_lo"], tab["ci_hi"] - tab["rate"]],
                fmt="none", ecolor="k", capsize=4)
    for xi, (k, n, hi) in enumerate(zip(tab["k"], tab["n"], tab["ci_hi"])):
        ax.text(xi, hi + 0.02, f"{int(k)}/{int(n)}", ha="center", fontsize=9)
    ax.set_xticks(x, [LABEL.get(c, c) for c in tab.index], rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate (Wilson 95% CI)")
    return tab


def effect_table(runs: pd.DataFrame, configs, landscape: str, baseline: str) -> pd.DataFrame:
    """Per-setting success with CI and difference vs the baseline (Newcombe, unpaired so conservative)."""
    sub = _sub(runs, configs, landscape)
    tab = an.success_table(sub)
    tab = tab[tab["metric"] == "success"].set_index("config_id")
    kb, nb = tab.loc[baseline, ["k", "n"]] if baseline in tab.index else (None, None)
    rows = []
    for c in [c for c in configs if c in tab.index]:
        r = tab.loc[c]
        d = ""
        if c != baseline and kb is not None:
            diff, lo, hi = an.newcombe_diff(int(kb), int(nb), int(r["k"]), int(r["n"]))
            d = f"{diff:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        g = sub[sub["config_id"] == c]
        rows.append({"setting": LABEL.get(c, c), "success k/n": f"{int(r['k'])}/{int(r['n'])}",
                     "rate [Wilson 95%]": f"{r['rate']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]",
                     f"diff vs {LABEL.get(baseline, baseline)} [Newcombe 95%]": d,
                     "median regret": f"{r['median_regret']:.3g}",
                     "median generations": "" if g["n_gens"].isna().all() else f"{g['n_gens'].median():.0f}",
                     "median restarts": "" if g["n_restarts"].isna().all() else f"{g['n_restarts'].median():.0f}"})
    return pd.DataFrame(rows)


def exp0_figure(M: Mode) -> None:
    runs = main_runs(M)
    base = ["cma_default", "bo_default"]
    if runs is None or _sub(runs, base, PRIMARY).empty:
        return not_run("exp0", M)
    sub = _sub(runs, base, PRIMARY)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    tab = success_bars(a1, runs, base, PRIMARY)
    names = {"cma_default": "CMA-ES default", "bo_default": "BO default"}
    a1.set_xticks(range(len(tab)), [names[c] for c in tab.index], rotation=0)
    a1.set_title(f"success on {PRIMARY}, n = {int(tab['n'].iloc[0])} seeds each")
    counts = an.outcome_counts(sub).reindex(base).fillna(0)
    bottom = np.zeros(len(counts))
    for o in OUTCOMES:
        a2.bar([names[c] for c in counts.index], counts[o], bottom=bottom,
               color=OUTCOME_COLOURS[o], label=o, width=0.55, edgecolor="white", linewidth=2)
        bottom += counts[o].to_numpy()
    a2.set_ylabel("runs")
    a2.set_title("outcome of each run")
    a2.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _finish(fig, "nb_3_exp0_success", M)
    display(effect_table(runs, base, PRIMARY, "cma_default").style.hide(axis="index"))


# ----------------------------------------------------------------- diagnostics

def sigma_figure(M: Mode) -> None:
    diag = M.table("diagnostics")
    d = None if diag is None else diag[(diag["config_id"] == "cma_default") & (diag["landscape"] == canon(PRIMARY))
                                       & diag["sigma"].notna() & diag["seed"].isin(SEEDS)]
    if d is None or d.empty:
        return not_run("exp0", M)
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for _, g in d.groupby("seed"):
        ax.plot(g["step"], g["sigma"], color=BLUE, alpha=0.35, lw=1)
    med = d.groupby("step")["sigma"].median()
    ax.plot(med.index, med.values, color=BLUE, lw=3, label="median over seeds")
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("sigma")
    ax.set_title(f"CMA-ES step size, default λ, {d['seed'].nunique()} seeds (thin: each seed)")
    ax.legend(frameon=False)
    _finish(fig, "nb_4_sigma", M)


def lengthscale_figure(M: Mode) -> None:
    diag = M.table("diagnostics")
    d = None if diag is None else diag[(diag["config_id"] == "bo_default") & diag["ls_min"].notna() & diag["seed"].isin(SEEDS)]
    if d is None or d[d["landscape"] == canon(PRIMARY)].empty:
        return not_run("exp0", M)
    upper = LS_BOUNDS_UNIT[1] * 10  # unit-cube bound x domain width
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    p = d[d["landscape"] == canon(PRIMARY)]
    for _, g in p.groupby("seed"):
        a1.step(g["step"], g["ls_min"], where="post", color=ORANGE, alpha=0.5, lw=1.2)
    a1.axhline(get_landscape(PRIMARY).meta["w"], color=RED, ls="--", label="well width w")
    a1.axhline(upper, color=GREY, ls=":", label=f"fitted upper bound ({upper:g})")
    a1.set_yscale("log"); a1.set_xlabel("evaluation index"); a1.set_ylabel("smallest fitted lengthscale (domain units)")
    a1.set_title(f"BO (EI, ξ=0) on {PRIMARY}: lengthscale over a run"); a1.legend(frameon=False)
    ws = [0.1, 0.3, 0.6, 1.0]
    for i, w in enumerate(ws):
        g = d[d["landscape"] == canon(f"primary_far_w{w}")].sort_values("step").groupby("seed").tail(1)
        if g.empty:
            continue
        a2.scatter(np.full(len(g), w) * np.exp(np.linspace(-0.06, 0.06, len(g))), g["ls_min"], color=ORANGE, s=22, alpha=0.7)
        a2.plot([w * 0.85, w * 1.15], [g["ls_min"].median()] * 2, color="k", lw=2)
    xs = np.array([0.08, 1.3])
    a2.plot(xs, xs, color=RED, ls="--", label="lengthscale = w")
    a2.axhline(upper, color=GREY, ls=":", label="fitted upper bound")
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xticks(ws, [str(w) for w in ws])
    a2.set_xlabel("well width w (far well)"); a2.set_ylabel("final smallest lengthscale (domain units)")
    a2.set_title("final fitted lengthscale vs w (bar: median)"); a2.legend(frameon=False)
    _finish(fig, "nb_4_lengthscale", M)


# ----------------------------------------------------------------- experiment cards

def _success_card(exp: str, baseline: str, M: Mode) -> None:
    runs = main_runs(M)
    configs = EXPERIMENTS[exp]["configs"]
    if runs is None or _sub(runs, configs, PRIMARY).empty:
        return not_run(exp, M)
    fig, ax = plt.subplots(figsize=(1.0 * len(configs) + 3.5, 4.4), layout="constrained")
    tab = success_bars(ax, runs, configs, PRIMARY)
    ax.set_title(f"{EXP_TITLE[exp]}, {PRIMARY}, n = {int(tab['n'].min())}-{int(tab['n'].max())} seeds")
    _finish(fig, f"nb_5_{exp}", M)
    display(effect_table(runs, configs, PRIMARY, baseline).style.hide(axis="index"))


def expE_figure(M: Mode, best: dict | None = None) -> None:
    """Far and near panels on one y-axis, fixed epsilon, random reference. `best` maps a method to
    its selected config id; its line is drawn only from the fresh seeds (2000-2049)."""
    runs = M.table("runs")
    if runs is None:
        return not_run("expE_baseline", M)
    main = runs[runs["seed"].isin(SEEDS)]
    fresh = runs[runs["seed"].isin(FRESH_SEEDS)]
    panels = [("far", [0.1, 0.3, 0.6, 1.0]), ("near", [0.1, 0.3])]
    eps = get_landscape(PRIMARY).epsilon
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, layout="constrained")
    drawn = False

    def line(ax, data, cid, loc, ws, off, **kw):
        pts = []
        for w in ws:
            g = data[(data["config_id"] == cid) & (data["landscape"] == canon(f"primary_{loc}_w{w}"))]
            if len(g):
                k, n = int(g["success"].sum()), len(g)
                lo, hi = an.wilson_ci(k, n)
                pts.append((w * off, k / n, k / n - lo, hi - k / n))
        if pts:
            x, y, dl, dh = map(np.array, zip(*pts))
            ax.errorbar(x, y, yerr=[dl, dh], capsize=3, lw=1.8, **kw)
        return bool(pts)

    for ax, (loc, ws) in zip(axes, panels):
        for off, (cid, kw) in zip((0.94, 1.0, 1.06), [("random", dict(color=GREY, ls="--", marker="o", label="random sampling")),
                                                      ("cma_default", dict(color=BLUE, marker="o", label="CMA-ES default")),
                                                      ("bo_default", dict(color=ORANGE, marker="o", label="BO default (EI ξ=0)"))]):
            drawn |= line(ax, main, cid, loc, ws, off, **kw)
        for method, cid in (best or {}).items():
            line(ax, fresh, cid, loc, ws, 1.12, color=COLOUR[method], ls=":", marker="s", mfc="white",
                 label=f"best variant {LABEL.get(cid, cid)} (fresh seeds)")
        ax.set_xscale("log"); ax.set_xticks(ws, [str(w) for w in ws]); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("well width w"); ax.set_title(f"{loc} well (epsilon = {eps} in both panels)")
    axes[0].set_ylabel("success rate (Wilson 95% CI)")
    axes[0].legend(frameon=False, loc="upper left")
    if not drawn:
        plt.close(fig)
        return not_run("expE_baseline", M)
    _finish(fig, "nb_5_expE", M)
    display(Markdown("*Exp E has no pre-registered prediction (see `docs/predictions.md`). Near-well cells at w = 0.6 and "
                     "w = 1.0 are excluded because the near well's basin merges with the decoy there "
                     "(`docs/pilot_log.md`). Random sampling is the reference at each w.*"))


def regret_figure(M: Mode, exp: str) -> None:
    """Median regret curves with quartiles, plus final regret per setting at equal N."""
    curves, runs = M.table("regret_curves"), main_runs(M)
    spec = EXPERIMENTS[exp]
    lname = canon(spec["landscapes"][0])
    if curves is None or runs is None or curves[curves["landscape"] == lname].empty:
        return not_run(exp, M)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    cfgs = [c for c in spec["configs"] if c in set(curves["config_id"])]
    styles = ["-", "--", ":", "-."]  # settings of one method share its colour; line style tells them apart
    for i, c in enumerate(cfgs):
        g = curves[(curves["config_id"] == c) & (curves["landscape"] == lname)]
        a1.plot(g["eval_idx"], g["median"], color=colour_for(c), ls=styles[i % 4], lw=2, label=LABEL.get(c, c))
        a1.fill_between(g["eval_idx"], g["q25"], g["q75"], color=colour_for(c), alpha=0.15)
    a1.set_yscale("log"); a1.set_xlabel("evaluation"); a1.set_ylabel("regret (median, quartiles)")
    a1.set_title(f"{EXP_TITLE[exp]}: {lname}"); a1.legend(frameon=False)
    for i, c in enumerate(cfgs):
        r = _sub(runs, [c], lname)["regret"]
        a2.errorbar(i, r.median(), yerr=[[r.median() - r.quantile(.25)], [r.quantile(.75) - r.median()]],
                    fmt="o", color=colour_for(c), capsize=4)
    a2.set_xticks(range(len(cfgs)), [LABEL.get(c, c) for c in cfgs], rotation=20, ha="right")
    a2.set_yscale("log"); a2.set_ylabel(f"final regret at N = {M.N} (median, quartiles)")
    _finish(fig, f"nb_5_{exp}", M)


def lambda_gain_figure(M: Mode, exp: str = "expV", base: str = "cma_default",
                       others=("cma_lam12", "cma_lam24")) -> None:
    """Per validation function: success-rate gain over the default lambda with a Newcombe interval."""
    runs = main_runs(M)
    lands = EXPERIMENTS[exp]["landscapes"]
    if runs is None or all(_sub(runs, [base, *others], l).empty for l in lands):
        return not_run(exp, M)
    fig, axes = plt.subplots(1, len(lands), figsize=(5.5 * len(lands), 4.2), sharey=True, layout="constrained", squeeze=False)
    for ax, l in zip(axes[0], lands):
        sub = _sub(runs, [base, *others], l)
        t = an.success_table(sub)
        t = t[t["metric"] == "success"].set_index("config_id")
        for i, c in enumerate([c for c in others if c in t.index and base in t.index]):
            d, lo, hi = an.newcombe_diff(int(t.loc[base, "k"]), int(t.loc[base, "n"]), int(t.loc[c, "k"]), int(t.loc[c, "n"]))
            ax.errorbar(i, d, yerr=[[d - lo], [hi - d]], fmt="o", color=BLUE, capsize=5)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(range(len(others)), [LABEL.get(c, c) for c in others]); ax.set_xlim(-0.5, len(others) - 0.5)
        ax.set_title(l); ax.set_xlabel("setting")
    axes[0][0].set_ylabel("success-rate gain vs default λ")
    fig.suptitle("Newcombe 95% interval (unpaired, conservative since seeds are matched)", fontsize=10)
    _finish(fig, f"nb_5_{exp}_lambda_gain", M)


def random_reference(ax, runs: pd.DataFrame, landscape: str) -> bool:
    """Grey dashed line and band: success rate of uniform random sampling (Wilson 95%) on this landscape."""
    g = _sub(runs, ["random"], landscape)
    if g.empty:
        ax.text(0.5, 0.97, "random-sampling reference missing (run `random` on this landscape)", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color=RED)
        return False
    k, n = int(g["success"].sum()), len(g)
    lo, hi = an.wilson_ci(k, n)
    ax.axhspan(lo, hi, color=GREY, alpha=0.2)
    ax.axhline(k / n, color=GREY, ls="--", lw=1.8, label=f"random sampling {k}/{n}")
    ax.legend(frameon=False, loc="upper right")
    return True


def _validation_card(M: Mode) -> None:
    runs = main_runs(M)
    lands = EXPERIMENTS["expV"]["landscapes"]
    configs = [c for c in EXPERIMENTS["expV"]["configs"] if c != "random"]
    if runs is None or all(_sub(runs, configs, l).empty for l in lands):
        return not_run("expV", M)
    fig, axes = plt.subplots(1, len(lands), figsize=(5.5 * len(lands), 4.4), sharey=True, layout="constrained", squeeze=False)
    for ax, l in zip(axes[0], lands):
        success_bars(ax, runs, configs, l)
        random_reference(ax, runs, l)
        ax.set_title(f"{l} (epsilon = {get_landscape(l).epsilon})")
    _finish(fig, "nb_5_expV", M)
    lambda_gain_figure(M)


def missing_cells(exp: str, M: Mode) -> list[tuple[str, str]]:
    """(config, landscape) cells of an experiment with no saved runs."""
    runs = main_runs(M)
    have = set() if runs is None else set(zip(runs["config_id"], runs["landscape"]))
    return [(c, l) for c in EXPERIMENTS[exp]["configs"] for l in EXPERIMENTS[exp]["landscapes"]
            if (c, canon(l)) not in have]


def experiment_card(exp: str, M: Mode) -> None:
    """Draw one experiment card's figure and table, or say it is not run / not configured.
    A card is drawn only when every one of its settings has saved runs (no half-filled figures)."""
    if exp not in EXPERIMENTS:
        return not_configured(exp)
    miss = missing_cells(exp, M)
    if miss:
        shown = ", ".join(f"{LABEL.get(c, c)} on {l}" for c, l in miss[:3]) + (" ..." if len(miss) > 3 else "")
        return not_run(exp, M, f"{len(miss)} of its settings have no saved runs (e.g. {shown}).")
    if exp == "expA" or exp == "expB":
        _success_card(exp, "cma_default", M)
    elif exp == "expC":
        _success_card(exp, "bo_default", M)
    elif exp == "expD":
        regret_figure(M, exp)
    elif exp == "expE_baseline":
        expE_figure(M)
    elif exp == "expV":
        _validation_card(M)
