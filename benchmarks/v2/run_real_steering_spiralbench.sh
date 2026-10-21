#!/usr/bin/env bash
#SBATCH --job-name=v2_real_steering_spiralbench
#SBATCH --account=emiliofe_74
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:2
#SBATCH --output=/project2/emiliofe_74/prabhu/v2/logs/real_steering_spiralbench_%j.log
#
# Real (non-self-judged) psychological-steering Spiral-Bench run:
#   - Judge:          Qwen3-14B, served via vLLM (no steering needed -> fine on vLLM)
#   - User-agent:     Qwen3-8B, in-process, unsteered (shares the evaluated-model's
#                     cached weights -- see api_client.py's local:// backend)
#   - Evaluated model: Qwen3-8B, in-process, run twice --
#       (1) baseline (no injection)
#       (2) steered  (psychopathy, layer 20, alpha 8 -- an established setting from
#                     this project's dark-tetrad sweeps, run_resistance_extension.sh)
# This isolates the effect of the injected vector: same model, same judge, same
# scenarios, same code path -- only the steering vector differs between runs.
set -uo pipefail
module purge
export HF_HOME=/project2/emiliofe_74/prabhu/models/huggingface
VLLM_PY=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python
# main.py imports injection_utils (torch/transformers) for local:// and local-steer://,
# so it must run in the GPU research env, not v2_benchmarks (which has no torch).
BENCH_PY=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python
SPIRAL_DIR=/project2/emiliofe_74/prabhu/v2/chatbench/spiralbench
OUT_DIR=/project2/emiliofe_74/prabhu/v2/pilot_smoke_test/spiralbench_real_steering_run
export TMPDIR=/scratch1/pugalent/tmp
mkdir -p "$TMPDIR" "$OUT_DIR" /project2/emiliofe_74/prabhu/v2/logs
export NLTK_DATA=/project2/emiliofe_74/prabhu/envs/nltk_data

echo "=== starting vLLM judge server (Qwen/Qwen3-14B) on port 8100, GPU 0 ==="
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0 $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --port 8100 --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  > "$OUT_DIR/vllm_judge_server.log" 2>&1 &
VLLM_PID=$!

echo "=== waiting for judge health check ==="
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; then
    echo "judge vLLM up after ${i}0s"
    break
  fi
  sleep 10
done
curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 || { echo "judge vLLM never came up, see vllm_judge_server.log"; kill $VLLM_PID 2>/dev/null; exit 1; }

cd "$SPIRAL_DIR"
export JUDGE_BASE_URL="http://127.0.0.1:8100/v1/chat/completions"
export JUDGE_API_KEY="EMPTY"
export USER_AGENT_BASE_URL="local://Qwen/Qwen3-8B"
export USER_AGENT_API_KEY="unused"

COMMON_ARGS="--user-model Qwen/Qwen3-8B --judge-models Qwen/Qwen3-14B \
  --num-prompts 10 --convos-per-prompt 1 --num-turns 20 --parallelism 4"

echo "=== RUN 1/2: baseline (unsteered), GPU 1 ==="
export EVALUATED_MODEL_BASE_URL="local://Qwen/Qwen3-8B"
export EVALUATED_MODEL_API_KEY="unused"
unset STEER_TRAIT STEER_LAYER STEER_ALPHA
CUDA_VISIBLE_DEVICES=1 $BENCH_PY main.py $COMMON_ARGS \
  --evaluated-model Qwen/Qwen3-8B \
  --run-id baseline_1 \
  --output-file "$OUT_DIR/results_baseline.json" \
  2>&1 | tee "$OUT_DIR/run_baseline.log"
echo "baseline exit=$?"

echo "=== RUN 2/2: steered (psychopathy, layer 20, alpha 8), GPU 1 ==="
export EVALUATED_MODEL_BASE_URL="local-steer://Qwen/Qwen3-8B"
export EVALUATED_MODEL_API_KEY="unused"
export STEER_TRAIT=psychopathy
export STEER_LAYER=20
export STEER_ALPHA=8
CUDA_VISIBLE_DEVICES=1 $BENCH_PY main.py $COMMON_ARGS \
  --evaluated-model Qwen/Qwen3-8B \
  --run-id steered_psychopathy_L20_A8 \
  --output-file "$OUT_DIR/results_steered_psychopathy.json" \
  2>&1 | tee "$OUT_DIR/run_steered.log"
echo "steered exit=$?"

echo "=== stopping vLLM judge server ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=== real steering run done ==="
echo "results: $OUT_DIR"
