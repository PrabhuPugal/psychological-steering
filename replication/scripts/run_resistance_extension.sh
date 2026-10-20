#!/usr/bin/env bash
# Extends the "does the model resist internally?" spike experiment (scripts
# 12 + 13) from Llama-3.1-8B-Instruct to the other 5 models we now have
# dark-tetrad vectors and weight access for. Same settings used for
# Llama-3.1-8B-Instruct earlier this project (sd4, n=56 for script 12;
# n=30, alphas 0 2 4 8 16 for script 13); layers are each model's own
# 6/12,18,24,30-of-32 equivalent, scaled to its real depth.
set -uo pipefail
cd /project2/emiliofe_74/prabhu/psychological-steering/replication
export HF_HOME=/project2/emiliofe_74/prabhu/models/huggingface
PY=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python
mkdir -p logs

declare -A MODELS=(
  [Llama-3.2-1B-Instruct]="meta-llama/Llama-3.2-1B-Instruct"
  [Llama-3.2-3B-Instruct]="meta-llama/Llama-3.2-3B-Instruct"
  [Qwen3-1.7B]="Qwen/Qwen3-1.7B"
  [Qwen3-4B]="Qwen/Qwen3-4B"
  [Qwen3-8B]="Qwen/Qwen3-8B"
)
declare -A LAYERS=(
  [Llama-3.2-1B-Instruct]="3 6 9 12 15"
  [Llama-3.2-3B-Instruct]="5 10 16 21 26"
  [Qwen3-1.7B]="5 10 16 21 26"
  [Qwen3-4B]="7 14 20 27 34"
  [Qwen3-8B]="7 14 20 27 34"
)
TRAITS="psychopathy machiavellianism narcissism sadism"

for short in "${!MODELS[@]}"; do
  repo="${MODELS[$short]}"
  layers="${LAYERS[$short]}"
  for trait in $TRAITS; do
    echo "########## instruction-spike: $short / $trait ##########"
    $PY scripts/12_instructed_psychopathy_spike.py --model "$repo" --trait "$trait" \
      --layers $layers --sjt_table sd4 --limit 56 \
      --out_dir "results/instruction_spike/${trait}_instruction_spike_${short}_sd4" \
      > "logs/instr_${trait}_${short}.log" 2>&1
    echo "exit=$?"

    echo "########## injection-propagation: $short / $trait ##########"
    $PY scripts/13_injection_spike_propagation.py --model "$repo" --trait "$trait" \
      --layers $layers --sjt_table sd4 --limit 30 --alphas 0 2 4 8 16 \
      --out_dir "results/injection_propagation/${trait}_injection_propagation_${short}" \
      > "logs/inj_${trait}_${short}.log" 2>&1
    echo "exit=$?"
  done
done
echo "=== RESISTANCE EXTENSION ALL DONE ==="
