#!/usr/bin/env bash
# Rerun of the bounded OCEAN sweep with the tightened, paper-matching
# fluency gate (mean drop <=5%, <=ceil(5%*n) per-response outliers below
# 90% of baseline fluency) instead of the original lenient 10%-mean-only
# check. P2 baseline (script 10) is unaffected by this change, so it is not
# rerun -- only the MDS sweep (script 14).
set -uo pipefail
cd /project2/emiliofe_74/prabhu/psychological-steering/replication
export HF_HOME=/project2/emiliofe_74/prabhu/models/huggingface
PY=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python
mkdir -p logs

declare -A MODELS=(
  [Llama-3.1-8B-Instruct]="meta-llama/Llama-3.1-8B-Instruct"
  [Llama-3.2-1B-Instruct]="meta-llama/Llama-3.2-1B-Instruct"
  [Llama-3.2-3B-Instruct]="meta-llama/Llama-3.2-3B-Instruct"
  [Qwen3-1.7B]="Qwen/Qwen3-1.7B"
  [Qwen3-4B]="Qwen/Qwen3-4B"
  [Qwen3-8B]="Qwen/Qwen3-8B"
)

for short in "${!MODELS[@]}"; do
  repo="${MODELS[$short]}"
  echo "########## Bounded MDS sweep v2 (tightened fluency gate): $short ##########"
  $PY scripts/14_bounded_ocean_sweep.py --model "$repo" --limit 24 --alphas 4 8 16 --n_layers 5 --batch_size 64 \
    --out_dir "results/bounded_ocean_sweep_v2/$short" \
    > "logs/mds_v2_${short}.log" 2>&1
  echo "exit=$?"
done
echo "=== BOUNDED SWEEP V2 ALL DONE ==="
