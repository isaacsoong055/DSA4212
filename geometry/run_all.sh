#!/usr/bin/env bash
# Re-run every experiment, rebuild the figures and compile the report.
# About 20 hours on 8 cores (about 150 CPU-hours, summed from the run times saved in
# results/); each sweep checkpoints and resumes if interrupted.
# To rebuild the tables and figures WITHOUT re-running anything, see view_results.sh.
# Pass a worker count as the first argument (default 8).
set -euo pipefail
cd "$(dirname "$0")"
W="${1:-8}"

python -m unittest discover -s tests
python code/run_e1_cmaes_geometry.py
python code/run_e2_bo_metric.py --workers "$W"            # original E2 (fitted noise) -> results/e2_*
python code/run_e2c_lengthscale_diagnostic.py
python code/run_e2d_fit_diagnostic.py
python code/run_e2e_bo_plateau.py --workers "$W"
python code/run_e2e_bo_plateau.py --workers "$W" --determinism-check
python code/run_e2e_stage_b.py --workers "$W"
python code/run_e2_bo_metric.py --bofix --workers "$W"    # E2 with the BO fix -> results/e2_bofix_*
python code/run_e3_invariance.py --workers "$W"
python code/run_e4_boundary.py --workers "$W"
python code/run_e5_other_functions.py --workers "$W"
python code/run_e6_noise_budget.py --workers "$W"
python code/run_e6b_recommendation_gp.py --workers "$W"   # needs E6
python code/run_e2_bo_metric.py --refit-local --workers "$W"
python code/run_e2_bo_metric.py --local-cand --workers "$W"
python code/run_e7_cmaes_restarts.py
python code/run_e8_matern.py --workers "$W"
python code/run_e9_rosenbrock_long.py
python code/run_e10_trust_region.py --workers "$W"
python code/run_e11_boundary_trap.py --workers "$W"       # needs e2_lc and E10
python code/run_e12_local_fit.py --workers "$W"
python code/run_e3b_units_check.py --workers "$W"      # needs e2_lc
python code/run_e4b_edge_uncertainty.py --workers "$W"
python code/make_figures.py
(cd report && latexmk -pdf -interaction=nonstopmode report.tex)
