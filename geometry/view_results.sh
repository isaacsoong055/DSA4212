#!/usr/bin/env bash
# Rebuild every results table and report figure from the saved runs in results/.
# Nothing is re-run: each script reads its checkpoint (results/*_partial.jsonl)
# and reprints its tables. About a minute in total.
#
# Experiments without a --report-only option (E1, E2c, E2d, E2e Stage A, E7, E9)
# have their CSVs in results/ already; make_figures.py and the notebook read them.
# To re-run the experiments themselves, use run_all.sh (about 20 hours).
#
# Usage: ./view_results.sh            (prints every table)
#        ./view_results.sh > tables.txt
set -euo pipefail
cd "$(dirname "$0")"

run() { echo; echo "################ $*"; "${PYTHON:-python3}" "$@"; }

run code/run_e2_bo_metric.py --report-only               # E2, original BO
run code/run_e2_bo_metric.py --bofix --report-only       # E2 with the BO fix
run code/run_e2_bo_metric.py --refit-local --report-only # E2, this machine
run code/run_e2_bo_metric.py --local-cand --report-only  # E2, local candidates
run code/run_e2e_stage_b.py --report-only
run code/run_e3_invariance.py --report-only
run code/run_e4_boundary.py --report-only
run code/run_e5_other_functions.py --report-only
run code/run_e6_noise_budget.py --report-only
run code/run_e6b_recommendation_gp.py --report-only
run code/run_e8_matern.py --report-only
run code/run_e10_trust_region.py --report-only
run code/run_e11_boundary_trap.py --report-only
run code/run_e12_local_fit.py --report-only
run code/make_figures.py
