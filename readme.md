# DSA4212: Bayesian Optimisation vs CMA-ES on multimodal landscapes

At a fixed budget of N = 300 evaluations, how do population size, restarts and
acquisition choices change the probability that CMA-ES and GP-BO find a narrow or
distant global basin? Full design: [docs/direction.md](docs/direction.md).
Pre-registered predictions: [docs/predictions.md](docs/predictions.md).

## Setup

macOS / Linux, CPU only, Python 3.11+ (developed on 3.13, MacBook M2).

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # ~20 s
```

## Notebook

`notebooks/main.ipynb` is the single narrative to read and present (status, summary placeholders, setup,
Exp 0 failure and diagnostics, experiment cards A-E and V, prediction scoreboard, verdict and limitations).
It only loads saved files and calls helpers in `code/report.py`; runs and aggregation go through
`run_experiments.py`. Set `FAST = True` in the top cell for the smoke test (`results/summary_fast/`, every
figure stamped "SMOKE TEST", ~6 s), or `FAST = False` for the full results in `results/summary/`. Missing
results show a "not run yet" message with the command to produce them. Open it with the `.venv` kernel.
Outputs are cleared before saving; figures go to `docs/figures/nb_*.png` (`_fast` suffix in FAST mode).
`docs/scoreboard.csv` holds prediction statuses and is edited by hand only.

## Running

```bash
# smoke test: every experiment, 5 seeds, N=100 -> results/raw_fast/ (~5 min)
.venv/bin/python run_experiments.py --fast

# full runs: 50 seeds, N=300 -> results/raw/  (filters are optional)
.venv/bin/python run_experiments.py --full --exp expA
.venv/bin/python run_experiments.py --full --method bo --landscape primary_far_w0.3

# compact summaries for committing -> results/summary/ (add --fast for results/summary_fast/)
.venv/bin/python run_experiments.py --aggregate
```

Runs are parallelised over 6 worker processes (one torch thread each). Existing
run files are skipped, so seeds can be extended by re-running with `--seeds`.

## Repo structure

```
code/benchmarks.py      landscapes: sphere, Rastrigin, Schwefel (rescaled), primary decoy+well
code/optimizers.py      run(landscape, N, seed, config): random, CMA-ES (none/IPOP/BIPOP), GP-BO
code/configs.py         frozen configs, seed lists, experiment definitions
code/logging_utils.py   per-evaluation CSV schema and writer
code/analysis.py        Wilson and Newcombe CIs, regret curves, diagnostics, aggregation
code/report.py          notebook figures, tables and loaders (load and draw only)
run_experiments.py      CLI
notebooks/main.ipynb    the notebook described above
pilot/                  Phase-1 pilot scripts (epsilon, timing, restart triggers, plots)
tests/                  pytest: minima, exact budget, reproducibility
docs/                   design, predictions, pilot log, figures
results/raw/            one CSV per run (gitignored working files)
results/archive/        raw_bo_<batch>.zip + manifest: raw BO runs per finished batch (committed)
results/summary/        runs.csv, success.csv, regret_curves.csv (committed)
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
