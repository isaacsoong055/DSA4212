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
  (`pilot/epsilon_pilot.py`). Restart triggers are chosen from firing rates only,
  never from success.
- `docs/predictions.md` is committed and must not be edited.
- Report every outcome. A surprising result: check for bugs or config slips first.

## Code style
- Small readable files, short docstrings. Run `pytest -q` before committing.
- Raw CSVs (`results/raw*/`) are gitignored; commit `results/summary/` instead.
- Check in with the user before anything expected to run longer than ~5 minutes.

## Notebook
- `notebooks/main.ipynb` imports from `code/` and holds no algorithm logic or multiprocessing. Long batches
  run through `run_experiments.py`. Rebuild it only by editing cells; clear outputs before saving.
- Restart triggers are frozen at tolx = 3e-3, tolfun = 1e-5 (chosen from firing rates only).
- Exp E near-well cells exist only for w <= 0.3 (basins merge at w >= 0.6; see `docs/pilot_log.md`).
