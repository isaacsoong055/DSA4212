# Multimodality: Bayesian optimisation vs CMA-ES on narrow and distant basins

Part of the group project in the repository root (see `../readme.md`). The other part, geometry and
parameterisation, is in `../geometry/`.

At a fixed budget of N = 300 evaluations, how do population size, restarts and
acquisition choices change the probability that CMA-ES and GP-BO find a narrow or
distant global basin? Full design: [docs/direction.md](docs/direction.md).
Pre-registered predictions: [docs/predictions.md](docs/predictions.md) and
[docs/predictions_addendum.md](docs/predictions_addendum.md). Decisions and deviations: [docs/pilot_log.md](docs/pilot_log.md).

All commands below are run from this folder (`multimodality/`).

## Setup

macOS / Linux, CPU only, Python 3.11+ (developed on 3.13, MacBook M2).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 68 tests, ~30 s
```

## Where to see the results

`notebooks/main.ipynb` is the single narrative: status, summary, setup, the Exp 0 failure and diagnostics,
one card per experiment (Exps 1 to 10: prediction, result with CI, explanation, caveat), the prediction
scoreboard, and the verdict and limitations. Every figure is drawn in the notebook and saved to
`docs/figures/nb_*.png`. It only loads saved files and calls helpers in `code/report.py`; runs and
aggregation go through `run_experiments.py`.

- `FAST = True` (top cell): smoke test from `results/summary_fast/` (not committed), every figure stamped "SMOKE TEST".
- `FAST = False`: the full results in `results/summary/` (committed), N = 300, 50 seeds (20 for BO in Exps 9 and 10).
  **Set this to see the real figures.**

Missing results show a "not run yet" message with the command to produce them. Open the notebook with the
`.venv` kernel. Outputs are cleared before saving. `docs/scoreboard.csv` holds prediction statuses and is edited by hand only.

## Running

```bash
# smoke test: every experiment, 5 seeds, N=100 -> results/raw_fast/ (~5 min)
python run_experiments.py --fast

# full runs: 50 seeds, N=300 -> results/raw/  (filters are optional)
python run_experiments.py --full --exp exp1
python run_experiments.py --full --method bo --landscape primary_far_w0.3

# compact summaries for committing -> results/summary/ (add --fast for results/summary_fast/)
python run_experiments.py --aggregate
```

The experiment ids and the settings behind each config id are in `code/configs.py`. The full set of
runs takes many hours (BO is about 100 s per run); the committed `results/summary/` and the archived raw BO runs
let you reproduce every table and figure without re-running them.

Runs are parallelised over 6 worker processes (one torch thread each). Existing
run files are skipped, so seeds can be extended by re-running with `--seeds`.

## Layout of this folder

```
code/benchmarks.py      landscapes: sphere, Rastrigin, Schwefel (rescaled), primary decoy+well
code/optimizers.py      run(landscape, N, seed, config): random, CMA-ES (none/IPOP/BIPOP), GP-BO
code/configs.py         frozen configs, seed lists, experiment definitions
code/logging_utils.py   per-evaluation CSV schema and writer
code/analysis.py        Wilson and Newcombe CIs, regret curves, diagnostics, aggregation
code/archive.py         compress and verify raw BO runs
code/report.py          notebook figures, tables and loaders (load and draw only)
run_experiments.py      command-line entry point
notebooks/main.ipynb    the notebook described above
pilot/                  pre-experiment scripts: epsilon choice, timing, restart triggers, sphere sanity check
tests/                  pytest: minima, exact budget, reproducibility, archive, report helpers
docs/                   design, predictions, pilot log, report log, figures
results/summary/        runs.csv, success.csv, regret_curves.csv, diagnostics (committed)
results/archive/        raw_bo_<batch>.zip + manifest: raw BO runs per finished batch (committed)
results/raw/            one CSV per run (gitignored working files)
```

## Reproducibility

A seed fixes the initial design and random streams. CMA-ES and random sampling reproduce exactly; BO reproduces in distribution, because floating-point differences compound through the model-fitting loop. Paired comparisons are paired on starting conditions only.

Raw runs live in `results/raw/` (gitignored working files, `results/raw_fast/` is never committed). The raw BO
runs are kept in the repo compressed: `python run_experiments.py --archive NAME [--exp EXP]` writes `results/archive/raw_bo_NAME.zip`
and `raw_bo_NAME_manifest.csv` (file, bytes, rows, sha256), both tracked. One zip per finished batch;
an existing zip is never rewritten.

## Raw CSV schema

`method, config_id, landscape, seed, eval_idx, x_0..x_{d-1}, f, best_so_far, sigma,
cov_trace, lengthscales, opt_time_s, gen, restart, popsize, in_basin, n_jitter`

Unused columns are empty. `lengthscales` are in domain units, `;`-separated.
`in_basin` is filled only on rows that improve the best-so-far value.
