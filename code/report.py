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
from configs import (CONFIGS, EXPERIMENTS, FRESH_SEEDS, LONG_EXPERIMENTS, N_FAST, N_FULL, N_LONG, N_SEEDS_FAST,
                     N_SEEDS_FULL, PRIMARY, SEEDS)
from logging_utils import ROOT, SUMMARY_DIR, SUMMARY_FAST_DIR, SUMMARY_LONG_DIR, RAW_DIR, RAW_FAST_DIR, run_path
from optimizers import LS_BOUNDS_UNIT, run

RED = "#e34948"
SMOKE_TOL = 1e-8
COLOUR = {"cmaes": BLUE, "bo": ORANGE, "random": GREY}
LABEL = {"cma_default": "default (λ=6)", "cma_lam12": "λ=12", "cma_lam24": "λ=24", "cma_ipop": "IPOP",
         "cma_bipop": "BIPOP", "cma_sig0.3": "σ0=0.3", "cma_sig0.5": "σ0=0.5", "cma_sig1": "σ0=1", "cma_sig4": "σ0=4",
         "cma_o_sig0.3": "oracle σ0=0.3", "cma_o_sig0.5": "oracle σ0=0.5", "cma_o_sig1": "oracle σ0=1",
         "cma_o_sig2": "oracle σ0=2", "cma_o_sig4": "oracle σ0=4", "cma_ro_k1": "σ0=1w", "cma_ro_k3": "σ0=3w",
         "cma_ro_k7": "σ0=7w", "cma_ro_k20": "σ0=20w", "bo_default": "EI ξ=0", "bo_ei_xi0.01": "EI ξ=0.01", "bo_ei_xi0.1": "EI ξ=0.1", "bo_ei_xi0.5": "EI ξ=0.5", "bo_ei_xi1": "EI ξ=1", "bo_ei_xi2": "EI ξ=2",
         "bo_lcb_b0.5": "LCB β=0.5", "bo_lcb_b2": "LCB β=2", "bo_lcb_b8": "LCB β=8", "bo_lcb_b32": "LCB β=32", "bo_ts": "Thompson",
         "bo_map": "BoTorch default recipe (MAP)", "bo_cap5": "ls cap 5", "bo_cap2": "ls cap 2", "bo_cap1": "ls cap 1", "bo_cap0.5": "ls cap 0.5",
         "bo_cap0.25": "ls cap 0.25", "bo_noise1e-3": "noise 1e-3", "bo_noise1e-2": "noise 1e-2", "bo_noise1e-1": "noise 1e-1", "bo_noise1": "noise 1",
         "bo_noise_learned": "noise learned", "bo_ls_prior": "ls prior", "bo_o_xi0": "oracle EI ξ=0", "bo_o_xi0.1": "oracle EI ξ=0.1", "bo_o_xi2": "oracle EI ξ=2", "bo_k_m32": "Matern-3/2", "bo_k_m12": "Matern-1/2", "bo_k_rbf": "RBF", "bo_k_m32_cap1": "Matern-3/2, cap 1", "bo_k_m12_cap1": "Matern-1/2, cap 1", "bo_k_rbf_cap1": "RBF, cap 1", "random": "random"}
ORDER = ["exp0", "exp1", "exp2", "exp3", "exp3_oracle", "exp3_ratio", "exp4", "exp5", "exp6", "exp7", "exp8_baseline", "exp9", "exp10"]
EXP_TITLE = {"exp0": "Exp 0: baselines", "exp1": "Exp 1: population size", "exp2": "Exp 2: restarts", "exp3": "Exp 3: CMA-ES initial step size (random start)",
             "exp3_oracle": "Exp 3: CMA-ES initial step size (oracle start 1.0 from the well)",
             "exp3_ratio": "Exp 3: CMA-ES step size relative to well width (ratio test)",
             "exp4": "Exp 4: BO explore/exploit (EI xi)", "exp5": "Exp 5: BO acquisition family",
             "exp6": "Exp 6: BO fitting settings (lengthscale cap, noise)", "exp7": "Exp 7: BO kernel type",
             "exp8_baseline": "Exp 8: basin width (baselines)", "exp8_best": "Exp 8: basin width (best-variant line, fresh seeds)",
             "exp9": "Exp 9: unimodal control", "exp10": "Exp 10: validation (Rastrigin, Schwefel)"}


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
    display(Markdown(f"> **Not configured yet: {EXP_TITLE[exp]}.** Its settings depend on the Exps 1-5 results "
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
            rows.append((EXP_TITLE[e], "not configured (depends on the Exp 1-5 results)", "-"))
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
        if e == "exp1":
            long_ok = (SUMMARY_LONG_DIR / "runs.csv").exists()
            rows.append(("Exp 1 side check (N = 1500, CMA-ES only)",
                         "loaded (50 seeds, full results only)" if long_ok else "missing",
                         "python run_experiments.py --long --exp exp1_long && python run_experiments.py --aggregate --long"))
    df = pd.DataFrame(rows, columns=["experiment", "results", "command to produce"])
    display(df.style.hide(axis="index").set_properties(**{"text-align": "left"}))


# ----------------------------------------------------------------- predictions and scoreboard

def load_predictions() -> pd.DataFrame:
    """All rows with an ID like H1 or M3 from docs/predictions.md and docs/predictions_addendum.md (text unedited)."""
    rows = []
    for name in ("predictions.md", "predictions_addendum.md"):
        for line in (ROOT / "docs" / name).read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 4 and re.fullmatch(r"[A-Z]+\d+", cells[0]):
                rows.append(cells[:4])
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
    ax.errorbar(x, tab["rate"], yerr=np.clip([tab["rate"] - tab["ci_lo"], tab["ci_hi"] - tab["rate"]], 0, None),
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

def discovery_card(exp: str, M: Mode) -> None:
    """Exps 4-6: per setting, runs that sampled inside the well, success given that, first-touch evaluation, the fitted
    lengthscale before the first touch and the share of samples at the box edge (results/summary/bo_discovery.csv)."""
    d = M.table("bo_discovery")
    if M.fast or d is None:
        why = "It has no smoke-test version; set `FAST = False` to see its saved results." if M.fast else "No saved results found."
        return display(Markdown(f"> **Not run yet: BO discovery table.** {why}  \n> Run: `python run_experiments.py --discovery`"))
    d = d[d["config_id"].isin(EXPERIMENTS[exp]["configs"])]
    if d.empty:
        return
    t = pd.DataFrame({"setting": d["config_id"].map(lambda c: LABEL.get(c, c)),
                      "success": d["success"].astype(str) + "/" + d["n"].astype(str),
                      "vs default (Newcombe 95%)": [f"{a:+.2f} [{lo:+.2f}, {hi:+.2f}]" if c != "bo_default" else "-"
                                                    for c, a, lo, hi in zip(d["config_id"], d["diff_vs_baseline"], d["diff_lo"], d["diff_hi"])],
                      "sampled in well": d["touched"], "success given that": d["success_given_touch"].astype(str) + "/" + d["touched"].astype(str),
                      "first sample in well (eval)": d["first_touch_median"], "samples in well (median)": d["touches_median"],
                      "lengthscale before first (max 20)": d["pre_touch_lengthscale"].round(2),
                      "samples at box edge": d["edge_share"].round(2)})
    display(t.reset_index(drop=True))


def xi_mechanism_figure(M: Mode) -> None:
    """Exp 4 mechanism: (A) share of runs that have sampled the well by evaluation t; (B) where the samples go after the
    first well sample (runs that find the well by search); (C) pooled sample density for three settings. Reads
    results/summary/bo_discovery_curves.csv, bo_after_discovery.csv and the saved BO runs."""
    curves, after = M.table("bo_discovery_curves"), M.table("bo_after_discovery")
    if M.fast or curves is None or after is None:
        why = "It has no smoke-test version; set `FAST = False` to see its saved results." if M.fast else "No saved results found."
        return display(Markdown(f"> **Not run yet: Exp 4 mechanism figure.** {why}  \n> Run: `python run_experiments.py --discovery`"))
    ls = get_landscape(PRIMARY)
    xs = ["bo_default", "bo_ei_xi0.01", "bo_ei_xi0.1", "bo_ei_xi0.5", "bo_ei_xi1", "bo_ei_xi2"]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(xs)))
    fig = plt.figure(figsize=(11, 7.2), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    for c, col in zip(xs, cols):
        g = curves[curves["config_id"] == c]
        ax.plot(g["eval"], g["share_found"], color=col, lw=2, label=LABEL[c])
    for c, ls_ in (("bo_lcb_b32", "--"), ("bo_ts", ":")):
        g = curves[curves["config_id"] == c]
        ax.plot(g["eval"], g["share_found"], color=GREY, lw=1.6, ls=ls_, label=LABEL[c])
    ax.axvline(10, color="k", lw=0.8, ls=":"); ax.text(11, 0.02, "initial design", fontsize=8)
    ax.set_xlabel("evaluation t"); ax.set_ylabel("share of 50 runs that have sampled\nwithin w of the well by t")
    ax.set_title("A. when the well is found"); ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    ax = fig.add_subplot(gs[0, 2])
    a = after[(after["group"] == "all") & after["config_id"].isin(xs[1:])].set_index("config_id").loc[xs[1:]]
    pos = np.arange(len(a)); w_ = 0.38
    ax.bar(pos - w_ / 2, a["edge_before"], w_, color=BLUE, label="at the box edge, before first well sample")
    ax.bar(pos + w_ / 2, a["edge_after"], w_, color=ORANGE, label="at the box edge, after")
    ax.plot(pos + w_ / 2, a["in_well_after"], "k_", ms=14, mew=2, label="within w of the well, after")
    ax.set_xticks(pos, [LABEL[c].replace("EI ", "") for c in a.index]); ax.set_ylim(0, 1)
    ax.set_ylabel("share of samples"); ax.set_title("B. where samples go\n(runs that find the well by search)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    for k, c in enumerate(("bo_default", "bo_ei_xi0.1", "bo_ei_xi2")):
        ax = fig.add_subplot(gs[1, k])
        pts = np.concatenate([read_run(c, PRIMARY, s, M.raw_dir)[["x_0", "x_1"]].to_numpy()[10:] for s in SEEDS[:M.n_seeds]])
        ax.hist2d(pts[:, 0], pts[:, 1], bins=40, range=[[-5, 5], [-5, 5]], cmap="Greys", norm=plt.matplotlib.colors.LogNorm())
        ax.plot(*ls.x_star, "o", mfc="none", mec=ORANGE, ms=11, mew=2)
        ax.set_aspect("equal"); ax.set_title(f"C. samples after the initial design: {LABEL[c]}", fontsize=9)
    _finish(fig, "nb_5_exp4_mechanism", M)
    t = after[after["config_id"].isin(xs[2:]) & (after["group"] != "all")].pivot(index="config_id", columns="group",
                                                                                   values=["n", "n_touch", "edge_after", "in_well_after"])
    t.index = [LABEL[c] for c in t.index]
    display(t.round(2))


def bo_oracle_figure(M: Mode) -> None:
    """BO6: EI with xi = 0, 0.1, 2 when the well is sampled at evaluation 0 (diagnostic arms, never compared with the real
    runs). Left: success with Wilson intervals; right: where the samples after the initial design go. Reads bo_oracle.csv."""
    d = M.table("bo_oracle")
    if M.fast or d is None:
        why = "It has no smoke-test version; set `FAST = False` to see its saved results." if M.fast else "No saved results found."
        return display(Markdown(f"> **Not run yet: BO oracle arms (BO6).** {why}  \n> Run: "
                               f"`python run_experiments.py --full --exp exp4_oracle && python run_experiments.py --aggregate && "
                               f"python run_experiments.py --discovery`"))
    d = d.set_index("config_id").loc[EXPERIMENTS["exp4_oracle"]["configs"]]
    names = [LABEL[c].replace("oracle EI ", "") for c in d.index]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    ax = axes[0]
    rate = d["success"] / d["n"]
    ax.bar(names, rate, color=BLUE, yerr=[rate - d["ci_lo"], d["ci_hi"] - rate], capsize=4)
    ax.set_ylim(0, 1); ax.set_ylabel("success rate (Wilson 95%)"); ax.set_title("A. well sampled at evaluation 0")
    x = np.arange(len(d)); w_ = 0.38
    ax = axes[1]
    ax.bar(x - w_ / 2, d["in_well_mean"], w_, color=BLUE, label="within w of the well")
    ax.bar(x + w_ / 2, d["edge_share"], w_, color=ORANGE, label="at the box edge")
    ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.set_ylabel("share of samples after the initial design")
    ax.set_title("B. where the samples go"); ax.legend(frameon=False)
    _finish(fig, "nb_5_exp4_oracle", M)
    k = d["success"].astype(int)
    cmp_ = [("ξ=0 minus ξ=2", "bo_o_xi2", "bo_o_xi0"), ("ξ=0.1 minus ξ=2", "bo_o_xi2", "bo_o_xi0.1"), ("ξ=0 minus ξ=0.1", "bo_o_xi0.1", "bo_o_xi0")]
    t = pd.DataFrame([dict(comparison=n, **dict(zip(["difference", "lo", "hi"], np.round(an.newcombe_diff(k[a], 50, k[b], 50), 2))))
                      for n, a, b in cmp_])
    display(t)
    display(d.rename(index=LABEL)[["success", "in_well_median", "in_well_mean", "edge_share", "well_samples_median",
                                   "lengthscale_after_20", "first_success_median"]].round(3))


def mean_trace_figure(M: Mode) -> None:
    """CMA4: distance of the search mean to the well per generation (oracle arm, sigma0 = 0.3 vs 2), plus the
    share of runs whose final mean is within w of the well. Reads results/summary/mean_trace*.csv."""
    gens, fin = M.table("mean_trace"), M.table("mean_trace_final")
    if M.fast or gens is None or fin is None:
        why = ("The mean trace has no smoke-test version; set `FAST = False` in the top cell to see its saved results."
               if M.fast else "No saved results found.")
        display(Markdown(f"> **Not run yet: Exp 3 mean trace (CMA4).** {why}  \n> Run: "
                         f"`python run_experiments.py --trace`"))
        return
    w = get_landscape(PRIMARY).meta["w"]
    arms = [("cma_o_sig0.3", BLUE), ("cma_o_sig2", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    for cid, col in arms:
        g = gens[gens["config_id"] == cid].groupby("gen")["dist_well"]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.values, color=col, lw=2, label=f"{LABEL[cid]} (median, IQR)")
        ax.fill_between(med.index, q1.values, q3.values, color=col, alpha=0.2)
    ax.axhline(w, color="k", ls=":", lw=1, label=f"well width w = {w}")
    ax.set_yscale("log"); ax.set_xlabel("generation"); ax.set_ylabel("distance of the search mean to the well centre")
    ax.set_title("oracle start 1.0 from the well: where the mean goes"); ax.legend(frameon=False)
    _finish(fig, "nb_5_exp3_mean_trace", M)
    rows = []
    for cid, _ in arms:
        f = fin[fin["config_id"] == cid]
        t = f[f["touched_well"]]
        rows.append({"setting": LABEL[cid], "runs": len(f),
                     "final mean within w of the well": f"{int((f['final_dist_well'] < w).sum())}/{len(f)}",
                     "runs that touched the well": len(t),
                     "touched runs ending closer to the decoy centre than to the well":
                         f"{int((t['final_dist_decoy'] < t['final_dist_well']).sum())}/{len(t)}"})
    display(pd.DataFrame(rows).style.hide(axis="index"))


def ratio_figure(M: Mode) -> None:
    """CMA5: oracle-start success versus k = sigma0 / w, one line per well width (Wilson 95% CI)."""
    runs = main_runs(M)
    ks = {"cma_ro_k1": 1, "cma_ro_k3": 3, "cma_ro_k7": 7, "cma_ro_k20": 20}
    ws = (0.1, 0.3, 0.6, 1.0)
    if runs is None:
        return not_run("exp3_ratio", M)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    drawn = False
    for w, mk in zip(ws, "osD^"):
        pts = []
        for cid, k in ks.items():
            g = runs[(runs["config_id"] == cid) & (runs["landscape"] == canon(f"primary_far_w{w}"))]
            if len(g):
                n, s = len(g), int(g["success"].sum())
                lo, hi = an.wilson_ci(s, n)
                pts.append((k * (1 + 0.04 * ws.index(w)), s / n, s / n - lo, hi - s / n))
        if pts:
            x, y, dl, dh = map(np.array, zip(*pts))
            ax.errorbar(x, y, yerr=np.clip([dl, dh], 0, None), marker=mk, capsize=3, lw=1.6, label=f"w = {w}")
            drawn = True
    if not drawn:
        plt.close(fig)
        return not_run("exp3_ratio", M)
    ax.set_xscale("log"); ax.set_xticks(list(ks.values()), [str(k) for k in ks.values()]); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("step size relative to well width, k = σ0 / w"); ax.set_ylabel("success rate (Wilson 95% CI)")
    ax.set_title("oracle start at (10/3) w: does success depend on σ0 / w only?"); ax.legend(frameon=False)
    _finish(fig, "nb_5_exp3_ratio", M)
    rows = []
    for cid, k in ks.items():
        row = {"σ0 / w": k}
        for w in ws:
            g = runs[(runs["config_id"] == cid) & (runs["landscape"] == canon(f"primary_far_w{w}"))]
            s, n = int(g["success"].sum()), len(g)
            lo, hi = an.wilson_ci(s, n)
            row[f"w = {w}"] = f"{s}/{n} [{lo:.2f}, {hi:.2f}]"
        rows.append(row)
    display(pd.DataFrame(rows).style.hide(axis="index"))


def long_budget_card(M: Mode) -> None:
    """Exp 1 side check: the same lambda settings at N = 1500, next to their N = 300 result (CMA-ES only)."""
    path = SUMMARY_LONG_DIR / "runs.csv"
    if M.fast or not path.exists():
        why = ("The side check has no smoke-test version; set `FAST = False` in the top cell to see its saved full results."
               if M.fast else "No saved results found.")
        display(Markdown(f"> **Not run yet: Exp 1 side check (N = {N_LONG}).** {why}  \n"
                         f"> Run: `python run_experiments.py --long --exp exp1_long && "
                         f"python run_experiments.py --aggregate --long`"))
        return
    long_runs, short_runs = pd.read_csv(path), main_runs(M)
    rows = []
    for cid in LONG_EXPERIMENTS["exp1_long"]["configs"]:
        row = [LABEL[cid]]
        for runs in (short_runs, long_runs):
            g = runs[(runs["config_id"] == cid) & (runs["landscape"] == canon(PRIMARY)) & runs["seed"].isin(SEEDS)]
            k, n = int(g["success"].sum()), len(g)
            lo, hi = an.wilson_ci(k, n)
            row += [f"{k}/{n} [{lo:.3f}, {hi:.3f}]", int(g["best_in_basin"].sum()), int(g["n_gens"].median()),
                    f"{g['final_sigma'].median():.3g}"]
        rows.append(row)
    cols = pd.MultiIndex.from_tuples([("", "setting")] + [(f"N = {n}", c) for n in (N_FULL, N_LONG)
                                      for c in ("success (Wilson 95%)", "best point in well basin",
                                                "median generations", "median final sigma")])
    display(pd.DataFrame(rows, columns=cols).style.hide(axis="index"))


def _success_card(exp: str, baseline: str, M: Mode, extra_baseline: str | None = None) -> None:
    runs = main_runs(M)
    configs = EXPERIMENTS[exp]["configs"]
    if runs is None or _sub(runs, configs, PRIMARY).empty:
        return not_run(exp, M)
    fig, ax = plt.subplots(figsize=(1.0 * len(configs) + 3.5, 4.4), layout="constrained")
    tab = success_bars(ax, runs, configs, PRIMARY)
    ax.set_title(f"{EXP_TITLE[exp]}, {PRIMARY}, n = {int(tab['n'].min())}-{int(tab['n'].max())} seeds")
    _finish(fig, f"nb_5_{exp}", M)
    if extra_baseline is None:
        display(effect_table(runs, configs, PRIMARY, baseline).style.hide(axis="index"))
        return
    # two groups, each compared with its own Matern-5/2 baseline
    capped = [c for c in configs if c == extra_baseline or c.endswith("_cap1")]
    display(effect_table(runs, [c for c in configs if c not in capped or c == baseline], PRIMARY, baseline).style.hide(axis="index"))
    display(effect_table(runs, capped, PRIMARY, extra_baseline).style.hide(axis="index"))


def exp8_figure(M: Mode, best: dict | None = None) -> None:
    """Far and near panels on one y-axis, fixed epsilon, random reference. `best` maps a method to
    its selected config id; its line is drawn only from the fresh seeds (2000-2049)."""
    runs = M.table("runs")
    if runs is None:
        return not_run("exp8_baseline", M)
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
            ax.errorbar(x, y, yerr=np.clip([dl, dh], 0, None), capsize=3, lw=1.8, **kw)
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
        return not_run("exp8_baseline", M)
    _finish(fig, "nb_5_exp8", M)
    display(Markdown("*CMA3 (CMA-ES) and BO11, BO12 (BO) are pre-registered for Exp 8 (see `docs/predictions_addendum.md`). Near-well cells at w = 0.6 and "
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


def lambda_gain_figure(M: Mode, exp: str = "exp10", base: str = "cma_default",
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
            ax.errorbar(i, d, yerr=np.clip([[d - lo], [hi - d]], 0, None), fmt="o", color=BLUE, capsize=5)
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
    lands = EXPERIMENTS["exp10"]["landscapes"]
    configs = [c for c in EXPERIMENTS["exp10"]["configs"] if c != "random"]
    if runs is None or all(_sub(runs, configs, l).empty for l in lands):
        return not_run("exp10", M)
    fig, axes = plt.subplots(1, len(lands), figsize=(5.5 * len(lands), 4.4), sharey=True, layout="constrained", squeeze=False)
    for ax, l in zip(axes[0], lands):
        success_bars(ax, runs, configs, l)
        random_reference(ax, runs, l)
        ax.set_title(f"{l} (epsilon = {get_landscape(l).epsilon})")
    _finish(fig, "nb_5_exp10", M)
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
    if exp in ("exp1", "exp2", "exp3"):
        _success_card(exp, "cma_default", M)
    elif exp == "exp3_oracle":
        _success_card(exp, "cma_o_sig2", M)
    elif exp == "exp3_ratio":
        ratio_figure(M)
    elif exp == "exp7":
        _success_card(exp, "bo_default", M, extra_baseline="bo_cap1")
    elif exp in ("exp4", "exp5", "exp6"):
        _success_card(exp, "bo_default", M)
    elif exp == "exp9":
        regret_figure(M, exp)
    elif exp == "exp8_baseline":
        exp8_figure(M, best={"cmaes": "cma_sig1", "bo": "bo_cap1"})  # drawn only from the fresh seeds, once they exist
    elif exp == "exp10":
        _validation_card(M)
