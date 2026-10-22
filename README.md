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

`notebooks/main.ipynb` is the place to read and present the project (landscape, one CMA-ES and one BO run,
a 2x4 seed grid, success rates with Wilson CIs, placeholders for Exps A, B, C, E). Set `FAST = True` in the
top cell (N=100, 5 seeds, computed in the notebook, ~1 min cold, ~5 s once cached) or `FAST = False`
(loads the saved N=300 results; produce them with `run_experiments.py --full` then `--aggregate`).
Open it with the `.venv` kernel. Outputs are cleared before saving; figures go to `docs/figures/nb_*.png`.

## Running

```bash
# smoke test: every experiment, 5 seeds, N=100 -> results/raw_fast/ (~5 min)
.venv/bin/python run_experiments.py --fast

# full runs: 50 seeds, N=300 -> results/raw/  (filters are optional)
.venv/bin/python run_experiments.py --full --exp expA
.venv/bin/python run_experiments.py --full --method bo --landscape primary_far_w0.3

# compact summaries for committing -> results/summary/
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
code/analysis.py        Wilson CIs, regret curves, plots, aggregation
run_experiments.py      CLI
notebooks/main.ipynb    the notebook described above
pilot/                  Phase-1 pilot scripts (epsilon, timing, restart triggers, plots)
tests/                  pytest: minima, exact budget, reproducibility
docs/                   design, predictions, pilot log, figures
results/raw/            one CSV per run (gitignored)
results/summary/        runs.csv, success.csv, regret_curves.csv (committed)
```

## Raw CSV schema

`method, config_id, landscape, seed, eval_idx, x_0..x_{d-1}, f, best_so_far, sigma,
cov_trace, lengthscales, opt_time_s, gen, restart, popsize, in_basin, n_jitter`

Unused columns are empty. `lengthscales` are in domain units, `;`-separated.
`in_basin` is filled only on rows that improve the best-so-far value.
