# Project rules (DSA4212: BO vs CMA-ES)

The design lives in `docs/direction.md`. Do not change the experimental design
without asking the user. If something there is ambiguous, ask rather than guess.

## Budget and fairness
- N = 300 evaluations for every CMA-ES vs BO comparison. Every objective call,
  including restarts and BO initial design, counts against N, enforced by
  `RunLog` in `code/optimizers.py`. Runs use exactly N evaluations.
- Longer-budget CMA-ES runs (large-lambda sweeps) are never plotted on BO axes.
- Same seeds across all variants in a comparison.
- One factor at a time: a variant differs from its baseline in exactly one setting.

## Seeds and reproducibility
- Seeds come from the fixed lists in `code/configs.py`: experiment seeds 1000+,
  pilot seeds 9000+. Extend by appending (larger `--seeds`), never redraw or rerun
  selectively until a result looks good.
- A run is fully determined by (method, config_id, landscape, seed). Random streams
  are derived via `SeedSequence([seed, stream])`; never use unseeded randomness.
- Workers run with `torch.set_num_threads(1)`.

## Frozen choices: do not tune after seeing results
- Configs in `code/configs.py` (BO kernel, n_init, refit schedule, acquisition
  optimiser, CMA-ES sigma0/x0, restart triggers) are frozen before the big runs.
- epsilon for the primary landscape was chosen from random sampling only
  (`pilot/epsilon_pilot.py`). Rastrigin and Schwefel epsilons follow the rule in
  `docs/pilot_log.md` (cap below nearest non-global regret, random success <= 5%; `pilot/epsilon_validation.py`). Restart triggers are chosen from firing rates only,
  never from success.
- `docs/predictions.md` is committed and must not be edited.
- Report every outcome. A surprising result: check for bugs or config slips first.

## Code style
- Small readable files, short docstrings. Run `pytest -q` before committing.
- Raw CSVs (`results/raw*/`) are gitignored; commit `results/summary/` instead.
- Check in with the user before anything expected to run longer than ~5 minutes.

## Notebook convention (applies to all future experiments)
- Heavy work (runs, aggregation, figure and table generation) lives in `.py` files: `run_experiments.py` and
  functions in `code/` (`analysis.py`, `report.py`). `notebooks/main.ipynb` only loads `results/summary*/` (and
  `results/raw*/` for single-run illustrations), calls those helpers in one or two lines per figure, and holds the
  findings in markdown: prediction, result with CI, explanation, caveat.
- No multiprocessing or heavy computation in the notebook. A section whose results are missing shows a "not run yet"
  message with the exact command. The notebook must run top to bottom without errors in FAST mode; clear outputs before saving.
- Findings are written only after the full results exist; FAST (N=100, 5 seeds) is a smoke test and its figures are stamped.
- Prediction statuses live in `docs/scoreboard.csv`, changed by hand, never by code.
- Restart triggers are frozen at tolx = 3e-3, tolfun = 1e-5 (chosen from firing rates only).
- Exp E near-well cells exist only for w <= 0.3 (basins merge at w >= 0.6; see `docs/pilot_log.md`).
- Exp E best-variant line: selected by the rule in `docs/pilot_log.md`, evaluated on fresh seeds 2000-2049 only.
- BO runs are reproducible only statistically across processes (see `docs/pilot_log.md`). Never delete raw BO CSVs in `results/raw/`; commit them compressed, one zip per finished batch, with `run_experiments.py --archive NAME --exp EXP` (`results/archive/`); never rewrite an existing zip. `raw_fast` is never committed.
- Do not start the full experiments until the user says so.
