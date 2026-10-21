#!/usr/bin/env bash
# Runs the real (non-self-judged) steering comparison sequentially on ONE GPU
# (the interactive session's A40), instead of queueing a separate SLURM job:
#   Phase 1: generate baseline + steered transcripts with Qwen3-8B in-process
#            (--skip-judging; the judge-model process doesn't exist yet, so no
#            GPU contention). Evaluated model = Qwen3-8B, run twice:
#              (a) baseline (no injection)
#              (b) steered (psychopathy, layer 20, alpha 8)
#   Phase 2: Qwen3-8B process has exited (GPU freed) -- start Qwen3-14B via
#            vLLM as judge, then re-invoke main.py for each run_id WITHOUT
#            --skip-judging; generation is already complete so it goes
#            straight to judging against the now-running vLLM judge.
set -uo pipefail
export HF_HOME=/project2/emiliofe_74/prabhu/models/huggingface
# main.py imports injection_utils (torch/transformers) for local:// and local-steer://,
# so it must run in the GPU research env, not v2_benchmarks (which has no torch).
BENCH_PY=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python
VLLM_PY=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python
SPIRAL_DIR=/project2/emiliofe_74/prabhu/v2/chatbench/spiralbench
OUT_DIR=/project2/emiliofe_74/prabhu/v2/pilot_smoke_test/spiralbench_real_steering_run
export TMPDIR=/scratch1/pugalent/tmp
mkdir -p "$TMPDIR" "$OUT_DIR"
export NLTK_DATA=/project2/emiliofe_74/prabhu/envs/nltk_data
cd "$SPIRAL_DIR"

export USER_AGENT_BASE_URL="local://Qwen/Qwen3-8B"
export USER_AGENT_API_KEY="unused"

COMMON_ARGS="--user-model Qwen/Qwen3-8B --judge-models Qwen/Qwen3-14B \
  --num-prompts 10 --convos-per-prompt 1 --num-turns 20 --parallelism 4"

echo "=== PHASE 1a: generate baseline (unsteered) ==="
export EVALUATED_MODEL_BASE_URL="local://Qwen/Qwen3-8B"
export EVALUATED_MODEL_API_KEY="unused"
unset STEER_TRAIT STEER_LAYER STEER_ALPHA
$BENCH_PY main.py $COMMON_ARGS --evaluated-model Qwen/Qwen3-8B --run-id baseline_1 \
  --output-file "$OUT_DIR/results_baseline.json" --skip-judging \
  2>&1 | tee "$OUT_DIR/gen_baseline.log"
echo "gen baseline exit=$?"

echo "=== PHASE 1b: generate steered (psychopathy, layer 20, alpha 8) ==="
export EVALUATED_MODEL_BASE_URL="local-steer://Qwen/Qwen3-8B"
export STEER_TRAIT=psychopathy
export STEER_LAYER=20
export STEER_ALPHA=8
$BENCH_PY main.py $COMMON_ARGS --evaluated-model Qwen/Qwen3-8B --run-id steered_psychopathy_L20_A8 \
  --output-file "$OUT_DIR/results_steered_psychopathy.json" --skip-judging \
  2>&1 | tee "$OUT_DIR/gen_steered.log"
echo "gen steered exit=$?"
unset STEER_TRAIT STEER_LAYER STEER_ALPHA

echo "=== PHASE 2: starting judge vLLM server (Qwen/Qwen3-14B) -- GPU now free of the HF process ==="
export VLLM_USE_FLASHINFER_SAMPLER=0
$VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --port 8100 --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  > "$OUT_DIR/vllm_judge_server.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; then
    echo "judge vLLM up after ${i}0s"
    break
  fi
  sleep 10
done
curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 || { echo "judge vLLM never came up, see vllm_judge_server.log"; kill $VLLM_PID 2>/dev/null; exit 1; }

export JUDGE_BASE_URL="http://127.0.0.1:8100/v1/chat/completions"
export JUDGE_API_KEY="EMPTY"

echo "=== PHASE 2a: judge baseline ==="
$BENCH_PY main.py $COMMON_ARGS --evaluated-model Qwen/Qwen3-8B --run-id baseline_1 \
  --output-file "$OUT_DIR/results_baseline.json" \
  2>&1 | tee "$OUT_DIR/judge_baseline.log"
echo "judge baseline exit=$?"

echo "=== PHASE 2b: judge steered ==="
$BENCH_PY main.py $COMMON_ARGS --evaluated-model Qwen/Qwen3-8B --run-id steered_psychopathy_L20_A8 \
  --output-file "$OUT_DIR/results_steered_psychopathy.json" \
  2>&1 | tee "$OUT_DIR/judge_steered.log"
echo "judge steered exit=$?"

echo "=== stopping judge vLLM server ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=== DONE: real steering run complete on one GPU ==="
echo "results: $OUT_DIR"
