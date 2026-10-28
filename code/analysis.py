"""Success rates with Wilson CIs, regret curves, diagnostic plots, aggregation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from benchmarks import get_landscape
from logging_utils import RAW_DIR, SUMMARY_DIR

GROUP = ["method", "config_id", "landscape"]


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(0.5 + conf / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    # the Wilson interval contains p; clamp away last-digit rounding so error bars are never negative
    return (max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half)))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Difference p2 - p1 with a Newcombe (1998, method 10) interval from two Wilson intervals.
    Unpaired, so conservative when the two groups use matched seeds."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, conf)
    l2, u2 = wilson_ci(k2, n2, conf)
    d = p2 - p1
    return d, d - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def load_raw(raw_dir: Path = RAW_DIR, landscape: str | None = None) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob(f"{landscape or '*'}/*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def summarise_runs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: final best, regret, success flags and diagnostics."""
    out = []
    for (method, cid, lname, seed), g in df.groupby(GROUP + ["seed"], sort=True):
        ls = get_landscape(lname)
        g = g.sort_values("eval_idx")
        best_row = g.loc[g["f"].idxmin()]
        last = g.iloc[-1]
        best_f = float(g["f"].min())
        out.append(dict(
            method=method, config_id=cid, landscape=lname, seed=seed, n_evals=len(g),
            best_f=best_f, regret=ls.regret(best_f),
            success=ls.is_success(best_f) if ls.epsilon is not None else np.nan,
            best_in_basin=bool(best_row["in_basin"]) if pd.notna(best_row["in_basin"]) else np.nan,
            best_restart=best_row["restart"], n_restarts=g["restart"].max(),
            n_gens=g["gen"].max() + 1 if g["gen"].notna().any() else np.nan,
            final_sigma=last["sigma"], final_lengthscales=last["lengthscales"],
            n_jitter=g["n_jitter"].sum(min_count=1), opt_time_s=g["opt_time_s"].sum(),
        ))
    return pd.DataFrame(out)


def success_table(runs: pd.DataFrame, col: str = "success") -> pd.DataFrame:
    """Success rate with Wilson 95% CI per (method, config, landscape)."""
    rows = []
    for key, g in runs.groupby(GROUP):
        s = g[col].dropna().astype(bool)
        k, n = int(s.sum()), len(s)
        lo, hi = wilson_ci(k, n)
        rows.append(dict(zip(GROUP, key), metric=col, k=k, n=n, rate=k / n if n else np.nan,
                         ci_lo=lo, ci_hi=hi, median_regret=g["regret"].median(),
                         median_opt_time_s=g["opt_time_s"].median()))
    return pd.DataFrame(rows)


def regret_curves(df: pd.DataFrame, every: int = 10) -> pd.DataFrame:
    """Median and quartiles of best-so-far regret at eval checkpoints."""
    df = df.copy()
    df["regret"] = df["best_so_far"] - df["landscape"].map(lambda n: get_landscape(n).f_star)
    N = int(df["eval_idx"].max()) + 1
    pts = sorted(set(range(every - 1, N, every)) | {N - 1})
    sub = df[df["eval_idx"].isin(pts)]
    q = sub.groupby(GROUP + ["eval_idx"])["regret"].quantile([0.25, 0.5, 0.75]).unstack()
    q.columns = ["q25", "median", "q75"]
    return q.reset_index()


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Compact traces per run: CMA-ES sigma per generation; BO fitted lengthscales (min, max, in
    domain units) at the evaluations where they change. Column `step` is gen or eval_idx."""
    cols = ["config_id", "landscape", "seed", "step", "sigma", "ls_min", "ls_max", "restart", "popsize"]
    parts = []
    for (cid, lname, seed), g in df.groupby(["config_id", "landscape", "seed"], sort=True):
        g = g.sort_values("eval_idx")
        if g["sigma"].notna().any():
            first = g.drop_duplicates("gen")
            parts.append(pd.DataFrame(dict(config_id=cid, landscape=lname, seed=seed, step=first["gen"],
                                           sigma=first["sigma"], restart=first["restart"], popsize=first["popsize"])))
        if g["lengthscales"].notna().any():
            arr = g["lengthscales"].dropna().str.split(";", expand=True).astype(float)
            keep = (arr != arr.shift()).any(axis=1)
            parts.append(pd.DataFrame(dict(config_id=cid, landscape=lname, seed=seed, step=g.loc[arr.index[keep], "eval_idx"],
                                           ls_min=arr[keep].min(axis=1), ls_max=arr[keep].max(axis=1))))
    return pd.concat(parts, ignore_index=True).reindex(columns=cols)


def aggregate(raw_dir: Path = RAW_DIR, out_dir: Path = SUMMARY_DIR) -> None:
    """Write compact, committable summaries of all raw runs."""
    df = load_raw(raw_dir)
    if df.empty:
        print(f"no raw CSVs under {raw_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = summarise_runs(df)
    runs.to_csv(out_dir / "runs.csv", index=False)
    pd.concat([success_table(runs, "success"), success_table(runs, "best_in_basin")]).to_csv(
        out_dir / "success.csv", index=False)
    regret_curves(df).to_csv(out_dir / "regret_curves.csv", index=False)
    diagnostics(df).to_csv(out_dir / "diagnostics.csv", index=False)
    print(f"wrote {len(runs)} runs to {out_dir}")


def mean_trace(configs: list[str], landscape: str, seeds: list[int], N: int, out_dir: Path = SUMMARY_DIR) -> None:
    """Re-run CMA-ES configs (deterministic given the seed) and record, per generation, the distance of the
    search-distribution mean to the well centre and to the decoy centre. Writes mean_trace.csv (per run and
    generation) and mean_trace_final.csv (per run: last mean, and whether any sample touched the well, d < w).
    This is the per-generation evidence for the refinement mechanism (CMA4 in docs/predictions_addendum.md)."""
    from configs import CONFIGS
    from optimizers import run

    ls = get_landscape(landscape)
    x_g, c, w = ls.x_star, np.asarray(ls.meta["c"], float), ls.meta["w"]
    per_gen, final = [], []
    for cid in configs:
        for seed in seeds:
            log = run(ls, N, seed, CONFIGS[cid], cid)
            dist = [d for d in log.meta["dist"] if d["restart"] == 0]
            for d in dist:
                m = np.asarray(d["mean"], float)
                per_gen.append(dict(config_id=cid, seed=seed, gen=d["gen"], sigma=d["sigma"],
                                    dist_well=float(np.linalg.norm(m - x_g)), dist_decoy=float(np.linalg.norm(m - c))))
            xs = np.array([[r[f"x_{i}"] for i in range(ls.d)] for r in log.rows])
            m = np.asarray(dist[-1]["mean"], float)
            final.append(dict(config_id=cid, seed=seed, final_dist_well=float(np.linalg.norm(m - x_g)),
                              final_dist_decoy=float(np.linalg.norm(m - c)),
                              touched_well=bool((np.linalg.norm(xs - x_g, axis=1) < w).any())))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_gen).to_csv(out_dir / "mean_trace.csv", index=False)
    pd.DataFrame(final).to_csv(out_dir / "mean_trace_final.csv", index=False)
    print(f"wrote mean traces for {len(configs)} configs x {len(seeds)} seeds to {out_dir}")


def bo_discovery(configs: list[str], landscape: str, seeds: list[int], raw_dir: Path = RAW_DIR,
                 out_dir: Path = SUMMARY_DIR, baseline: str = "bo_default") -> pd.DataFrame:
    """Per BO config, from the saved runs: how many runs sampled inside the well (touched: any point with d < w),
    success given touch, evaluation index of the first touch, number of touches, fitted lengthscale before the first
    touch (geometric mean over the two axes, median over runs; the upper bound is 20), share of post-initial-design
    samples within 0.5 of the box edge, and the success difference to the baseline with a Newcombe interval.
    Exploratory: the touch definition was fixed after Exp 4 was seen (it is the CMA-ES one). Writes bo_discovery.csv."""
    ls = get_landscape(landscape)
    x_g, w = ls.x_star, ls.meta["w"]
    rows = []
    for cid in configs:
        recs = []
        for seed in seeds:
            df = pd.read_csv(Path(raw_dir) / landscape / f"bo__{cid}__seed{seed}.csv")
            X = df[[f"x_{i}" for i in range(ls.d)]].to_numpy()
            hit = np.where(np.linalg.norm(X - x_g, axis=1) < w)[0]
            first = int(hit[0]) if len(hit) else None
            ell = df["lengthscales"].dropna().map(lambda s: float(np.exp(np.mean(np.log([float(v) for v in s.split(";")])))))
            pre = ell[df.loc[ell.index, "eval_idx"] < first] if first is not None else ell
            succ = bool(ls.is_success(np.min(df["f"])))
            recs.append(dict(touch=first is not None, first=first, n_touch=len(hit), success=succ,
                             pre_ls=float(pre.median()) if len(pre) else np.nan,
                             edge=float((np.abs(X[10:]).max(axis=1) > 4.5).mean())))
        r = pd.DataFrame(recs)
        rows.append(dict(config_id=cid, n=len(r), success=int(r.success.sum()), touched=int(r.touch.sum()),
                         success_given_touch=int(r[r.touch].success.sum()),
                         first_touch_median=r[r.touch]["first"].astype(float).median(), touches_median=r[r.touch].n_touch.median(),
                         pre_touch_lengthscale=r.pre_ls.median(), edge_share=r.edge.mean()))
    out = pd.DataFrame(rows)
    b = out[out.config_id == baseline].iloc[0]
    diff = [newcombe_diff(int(b.success), int(b.n), int(r.success), int(r.n)) for r in out.itertuples()]
    out[["diff_vs_baseline", "diff_lo", "diff_hi"]] = diff
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "bo_discovery.csv", index=False)
    print(f"wrote bo_discovery.csv for {len(configs)} configs x {len(seeds)} seeds to {out_dir}")
    return out


def bo_after_discovery(configs: list[str], landscape: str, seeds: list[int], raw_dir: Path = RAW_DIR,
                       out_dir: Path = SUMMARY_DIR) -> None:
    """Exp 4/5 mechanism, from the saved BO runs. Writes (a) bo_discovery_curves.csv: per setting and evaluation t, the
    share of runs that have sampled within w of the well centre by t; (b) bo_after_discovery.csv: for the runs that find the
    well by search (first touch after the 10-point initial design), split into all / successful / failed runs, where the
    samples go: share of samples at the box edge before and after the first touch, share of later samples within w of the
    well, evaluations left, number of well samples, and the fitted lengthscale after the first touch. Exploratory (touch definition fixed after Exp 4 was seen)."""
    ls = get_landscape(landscape)
    x_g, w = ls.x_star, ls.meta["w"]
    curves, runs = [], []
    for cid in configs:
        firsts = []
        for seed in seeds:
            df = pd.read_csv(Path(raw_dir) / landscape / f"bo__{cid}__seed{seed}.csv")
            X, f = df[[f"x_{i}" for i in range(ls.d)]].to_numpy(), df["f"].to_numpy()
            d, edge = np.linalg.norm(X - x_g, axis=1), np.abs(X).max(axis=1) > 4.5
            hit = np.where(d < w)[0]
            firsts.append(hit[0] if len(hit) else np.inf)
            if len(hit) and hit[0] >= 10:
                a = hit[0]
                ell = df["lengthscales"].dropna().map(lambda s: float(np.exp(np.mean(np.log([float(v) for v in s.split(";")])))))
                after = ell[df.loc[ell.index, "eval_idx"] > a]
                runs.append(dict(ls_after=float(after.median()) if len(after) else np.nan, config_id=cid, seed=seed, success=bool(ls.is_success(f.min())), first=a, n_touch=len(hit),
                                 edge_before=edge[10:a].mean() if a > 10 else np.nan, edge_after=edge[a + 1:].mean(),
                                 in_well_after=(d[a + 1:] < w).mean(), evals_left=len(d) - a - 1))
        firsts = np.array(firsts)
        curves += [dict(config_id=cid, eval=t, share_found=float((firsts <= t).mean())) for t in range(len(df))]
    R = pd.DataFrame(runs)
    rows = []
    for cid, g in (R.groupby("config_id", sort=False) if len(R) else []):
        for name, h in (("all", g), ("successful", g[g.success]), ("failed", g[~g.success])):
            if len(h):
                rows.append(dict(config_id=cid, group=name, n=len(h), first_touch=h["first"].median(), n_touch=h.n_touch.median(),
                                 edge_before=h.edge_before.mean(), edge_after=h.edge_after.mean(),
                                 in_well_after=h.in_well_after.mean(), evals_left=h.evals_left.mean(),
                                 ls_after=h.ls_after.median()))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(out_dir / "bo_discovery_curves.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "bo_after_discovery.csv", index=False)
    print(f"wrote bo_discovery_curves.csv and bo_after_discovery.csv for {len(configs)} configs to {out_dir}")


def bo_oracle(configs: list[str], landscape: str, seeds: list[int], raw_dir: Path = RAW_DIR,
              out_dir: Path = SUMMARY_DIR) -> pd.DataFrame:
    """BO6 (Exp 4 oracle arms: the well is sampled at evaluation 0). Per arm: success with a Wilson interval, the median
    over runs of the share of samples after the initial design that lie within w of the well, the share at the box edge,
    well samples after evaluation 0 (median), fitted lengthscale from evaluation 20 on (geometric mean over axes, median),
    and the evaluation of the first success. Writes bo_oracle.csv; the Newcombe differences are computed in the notebook
    card from the success counts."""
    ls = get_landscape(landscape)
    x_g, w = ls.x_star, ls.meta["w"]
    rows = []
    for cid in configs:
        recs = []
        for seed in seeds:
            df = pd.read_csv(Path(raw_dir) / landscape / f"bo__{cid}__seed{seed}.csv")
            X, f = df[[f"x_{i}" for i in range(ls.d)]].to_numpy(), df["f"].to_numpy()
            d = np.linalg.norm(X - x_g, axis=1)
            ell = df["lengthscales"].dropna().map(lambda s: float(np.exp(np.mean(np.log([float(v) for v in s.split(";")])))))
            late = ell[df.loc[ell.index, "eval_idx"] >= 20]
            ok = (f - ls.f_star) <= ls.epsilon
            recs.append(dict(success=bool(ok.any()), first_success=int(np.argmax(ok)) if ok.any() else np.nan,
                             in_well=(d[10:] < w).mean(), edge=(np.abs(X[10:]).max(axis=1) > 4.5).mean(),
                             well_samples=int((d[1:] < w).sum()), ls=float(late.median()) if len(late) else np.nan))
        r = pd.DataFrame(recs)
        k, n = int(r.success.sum()), len(r)
        lo, hi = wilson_ci(k, n)
        rows.append(dict(config_id=cid, n=n, success=k, ci_lo=lo, ci_hi=hi, in_well_median=r.in_well.median(),
                         in_well_mean=r.in_well.mean(), edge_share=r.edge.mean(), well_samples_median=r.well_samples.median(),
                         lengthscale_after_20=r.ls.median(), first_success_median=r.first_success.median()))
    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "bo_oracle.csv", index=False)
    print(f"wrote bo_oracle.csv for {len(configs)} arms to {out_dir}")
    return out


# ----------------------------------------------------------------- plots

_GRID_CACHE: dict = {}


def _grid(ls, n: int):
    """f on an n x n grid over [-5, 5]^2 (cached per landscape)."""
    key = (ls.name, n)
    if key not in _GRID_CACHE:
        g = np.linspace(-5, 5, n)
        XX, YY = np.meshgrid(g, g)
        _GRID_CACHE[key] = (XX, YY, np.array([[ls([x, y]) for x in g] for y in g]))
    return _GRID_CACHE[key]


def plot_landscape(ax, ls, n: int = 300, levels: int = 40, cmap: str = "Greys_r"):
    XX, YY, Z = _grid(ls, n)
    cs = ax.contourf(XX, YY, Z, levels=levels, cmap=cmap)
    ax.set_aspect("equal")
    return cs, (XX, YY, Z)


def plot_sigma_trace(ax, run: pd.DataFrame, label: str | None = None):
    """Sigma vs generation for one CMA-ES run (one point per generation)."""
    per_gen = run.groupby("gen")["sigma"].first()
    ax.semilogy(per_gen.index, per_gen.values, marker=".", label=label)
    ax.set_xlabel("generation")
    ax.set_ylabel("sigma")


def plot_scatter(ax, ls, run: pd.DataFrame, title: str = ""):
    """Sampled points over the landscape, coloured by evaluation index."""
    plot_landscape(ax, ls)
    sc = ax.scatter(run["x_0"], run["x_1"], c=run["eval_idx"], cmap="autumn", s=10,
                    edgecolors="k", linewidths=0.2)
    ax.plot(*ls.x_star[:2], marker="*", color="white", ms=12, mec="k")
    ax.set_title(title)
    return sc


# ----------------------------------------------------------------- notebook helpers
# Plotting, loading and outcome labelling only; no optimisation logic lives here.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from logging_utils import ROOT, run_path  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898783"
SEQ = ListedColormap(plt.cm.Blues(np.linspace(0.3, 1.0, 256)))  # one hue, light -> dark
OUTCOMES = ["success", "stuck in decoy", "unfinished"]
OUTCOME_COLOURS = {"success": "#1baf7a", "stuck in decoy": "#e34948", "unfinished": GREY}


def save_fig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def mark_landscape(ax, ls):
    """Contour map with decoy centre (x) and true minimum (star)."""
    cs, _ = plot_landscape(ax, ls)
    ax.plot(*ls.meta["c"][:2], marker="x", color=ORANGE, ms=10, mew=2.5, ls="", label="decoy centre")
    ax.plot(*ls.x_star[:2], marker="*", color="white", ms=15, mec="k", ls="", label="true minimum")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return cs


def surface3d(ax, ls, n: int = 200):
    """3D surface with decoy centre and true minimum marked."""
    XX, YY, Z = _grid(ls, n)
    ax.plot_surface(XX, YY, Z, cmap="Greys_r", linewidth=0, antialiased=True, alpha=0.9, zorder=1)
    c = ls.meta["c"][:2]  # needs add_subplot(..., computed_zorder=False) so markers draw on top
    ax.scatter(*c, ls(ls.meta["c"]), marker="x", color=ORANGE, s=80, linewidths=3, depthshade=False, zorder=10)
    ax.scatter(*ls.x_star[:2], ls.f_star, marker="*", color="#e34948", s=200, edgecolors="k", depthshade=False, zorder=10)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f")
    ax.view_init(elev=35, azim=-125)


def scatter_run(ax, ls, run: pd.DataFrame, by: str = "eval_idx", n_init: int = 0):
    """Samples over the contour map, coloured by `by`; first n_init points drawn as squares."""
    mark_landscape(ax, ls)
    later = run[run["eval_idx"] >= n_init]
    sc = ax.scatter(later["x_0"], later["x_1"], c=later[by], cmap=SEQ, s=16,
                    edgecolors="k", linewidths=0.3, zorder=3)
    if n_init:
        init = run[run["eval_idx"] < n_init]
        ax.scatter(init["x_0"], init["x_1"], marker="s", s=34, c="white", edgecolors="k",
                   linewidths=0.8, zorder=4, label="initial design")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    return sc


def draw_ellipses(ax, dist: list[dict], gens, colour: str = "#e34948"):
    """1-sigma search ellipse N(m, sigma^2 C) at the chosen generations, plus the start."""
    by_gen = {d["gen"]: d for d in dist}
    for g in gens:
        d = by_gen[g]
        vals, vecs = np.linalg.eigh(d["C"])
        w, h = 2 * d["sigma"] * np.sqrt(vals[::-1])
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        ax.add_patch(Ellipse(d["mean"], w, h, angle=ang, fill=False, ec=colour, lw=1.8, zorder=5))
        ax.annotate(f"gen {g}", d["mean"], xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color="white", zorder=6)
    ax.plot(*dist[0]["mean"], marker="D", color="white", mec=colour, ms=9, ls="", zorder=6, label="start x0")


def load_or_run(config_id: str, landscape: str, seed: int, N: int, raw_dir: Path,
                allow_run: bool = True) -> pd.DataFrame:
    """Load a saved run CSV, or run it once (single process) with the engine and save it.
    With allow_run=False a missing file is an error (long batches go through run_experiments.py)."""
    from configs import CONFIGS
    cfg = CONFIGS[config_id]
    path = run_path(cfg["method"], config_id, landscape, seed, raw_dir)
    if not path.exists():
        if not allow_run:
            raise FileNotFoundError(f"{path} missing. Run: python run_experiments.py --full --exp exp0")
        from logging_utils import write_run
        from optimizers import run
        ls = get_landscape(landscape)
        write_run(run(ls, N, seed, cfg, config_id).rows, path, ls.d)
    return pd.read_csv(path)


def classify_outcomes(runs: pd.DataFrame) -> pd.DataFrame:
    """success: regret <= epsilon. stuck in decoy: best point's local descent does not reach the
    global minimiser. unfinished: best point is in the global basin but regret > epsilon."""
    out = runs.copy()
    out["outcome"] = np.where(out["success"].astype(bool), "success",
                              np.where(out["best_in_basin"].astype(bool), "unfinished", "stuck in decoy"))
    return out


def outcome_counts(runs: pd.DataFrame) -> pd.DataFrame:
    """Rows = config_id; columns = outcome counts (all three always present)."""
    runs = classify_outcomes(runs)
    return (runs.groupby(["config_id", "outcome"]).size().unstack(fill_value=0)
            .reindex(columns=OUTCOMES, fill_value=0))
