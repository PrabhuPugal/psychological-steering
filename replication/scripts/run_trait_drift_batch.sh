#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python scripts/trait_drift_monitor.py \
  --batch \
  --sjt_table mpi120 \
  --layers 6 12 18 24 30 \
  --z_threshold 2.5 \
  --risk_threshold 2.0 \
  --nudge_alpha 8.0 \
  --nudge_duration 4 \
  --out_dir results/trait_drift/trait_drift_full_mpi120 \
  2>&1 | tee results/trait_drift/trait_drift_full_mpi120.log
