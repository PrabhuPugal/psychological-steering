#!/usr/bin/env bash
#SBATCH --job-name=v2_pilot_spiralbench
#SBATCH --account=emiliofe_74
#SBATCH --partition=gpu
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/project2/emiliofe_74/prabhu/v2/logs/pilot_smoke_test_%j.log
#
# Baseline (no steering vector) pilot smoke test: confirms the Spiral-Bench
# harness runs end-to-end against one of our own models before we wire in
# psychological-steering injection as a later step.
#
# Serves Qwen3-8B locally via vLLM's OpenAI-compatible API and points ALL
# THREE Spiral-Bench roles (user agent, evaluated model, judge) at it --
# no external API key needed. This is a pipeline smoke test, not a real
# benchmark score: judging your own model with itself is not a valid judge.
#
# Small by design: 2 scenarios x 1 conversation x 4 turns, parallelism 1.
set -uo pipefail
module purge
export HF_HOME=/project2/emiliofe_74/prabhu/models/huggingface
VLLM_PY=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python
BENCH_PY=/project2/emiliofe_74/prabhu/envs/v2_benchmarks/bin/python
SPIRAL_DIR=/project2/emiliofe_74/prabhu/v2/chatbench/spiralbench
OUT_DIR=/project2/emiliofe_74/prabhu/v2/pilot_smoke_test/spiralbench_qwen3-8b_baseline
mkdir -p "$OUT_DIR" /project2/emiliofe_74/prabhu/v2/logs

echo "=== starting vLLM server (Qwen/Qwen3-8B) ==="
export VLLM_USE_FLASHINFER_SAMPLER=0
$VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --port 8000 --dtype float16 --max-model-len 32768 \
  > "$OUT_DIR/vllm_server.log" 2>&1 &
VLLM_PID=$!

echo "=== waiting for vLLM health check ==="
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "vLLM up after ${i}0s"
    break
  fi
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "vLLM never came up, see vllm_server.log"; kill $VLLM_PID 2>/dev/null; exit 1; }

cd "$SPIRAL_DIR"
export NLTK_DATA=/project2/emiliofe_74/prabhu/envs/nltk_data
echo "=== running Spiral-Bench pilot (self-judged baseline smoke test) ==="
$BENCH_PY main.py \
  --user-model Qwen/Qwen3-8B \
  --evaluated-model Qwen/Qwen3-8B \
  --judge-models "Qwen/Qwen3-8B" \
  --num-prompts 2 \
  --convos-per-prompt 1 \
  --num-turns 4 \
  --parallelism 1 \
  --run-id pilot_baseline_1 \
  --output-file "$OUT_DIR/results.json" \
  2>&1 | tee "$OUT_DIR/spiralbench_run.log"
RUN_EXIT=$?

echo "=== stopping vLLM server ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=== pilot smoke test done, exit=$RUN_EXIT ==="
echo "results: $OUT_DIR"
