#!/usr/bin/env bash
#
# Spiral-Bench on one Lambda GPU (H100 80 GB): Lambda port of
# benchmarks/v2/run_real_steering_spiralbench.sh (which used two A40s).
#   - Judge:           JUDGE_MODEL (default Qwen/Qwen3-14B) served by vLLM on this GPU,
#                      capped at VLLM_MEM of GPU memory (no steering needed -> vLLM is fine)
#   - User-agent:      MODEL, in-process, unsteered (local://, shares the cached weights)
#   - Evaluated model: MODEL, in-process; steered via local-steer:// when STEER_TRAIT is set
#
# Examples:
#   bash run_spiralbench_lambda.sh                                    # baseline, Qwen3-8B
#   STEER_TRAIT=psychopathy STEER_LAYER=20 STEER_ALPHA=8 bash run_spiralbench_lambda.sh
#   NUM_PROMPTS=2 NUM_TURNS=4 bash run_spiralbench_lambda.sh         # quick smoke
#
#   MODEL (Qwen/Qwen3-8B)  JUDGE_MODEL (Qwen/Qwen3-14B)  NUM_PROMPTS (10)  NUM_TURNS (20)
#   PARALLELISM (4)  VLLM_MEM (0.45)  JUDGE_PORT (8100)  RUN_ID (baseline | <trait>_L<layer>_A<alpha>)
set -uo pipefail

R=/lambda/nfs/psychological-steering
PY=$R/.venv/bin/python
SPIRAL_DIR=$R/external/spiralbench
export HF_HOME=$R/hf_home
export HF_HUB_OFFLINE=1
export NLTK_DATA=$R/hf_home/nltk_data

MODEL=${MODEL:-Qwen/Qwen3-8B}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-14B}
NUM_PROMPTS=${NUM_PROMPTS:-10}
NUM_TURNS=${NUM_TURNS:-20}
PARALLELISM=${PARALLELISM:-4}
VLLM_MEM=${VLLM_MEM:-0.45}
JUDGE_PORT=${JUDGE_PORT:-8100}
if [ -n "${STEER_TRAIT:-}" ]; then
  export STEER_TRAIT STEER_LAYER STEER_ALPHA
  EVAL_URL="local-steer://$MODEL"
  DEFAULT_RUN="${STEER_TRAIT}_L${STEER_LAYER}_A${STEER_ALPHA}"
else
  unset STEER_TRAIT STEER_LAYER STEER_ALPHA
  EVAL_URL="local://$MODEL"
  DEFAULT_RUN=baseline
fi
RUN_ID=${RUN_ID:-$DEFAULT_RUN}
OUT_DIR=$R/external/spiralbench_results/$(basename "$MODEL")/$RUN_ID
mkdir -p "$OUT_DIR"

echo "=== Spiral-Bench | $(date) | model=$MODEL judge=$JUDGE_MODEL run=$RUN_ID eval_url=$EVAL_URL ==="
echo "prompts=$NUM_PROMPTS turns=$NUM_TURNS parallelism=$PARALLELISM steer=${STEER_TRAIT:-none}/${STEER_LAYER:-}/${STEER_ALPHA:-}"

export VLLM_USE_FLASHINFER_SAMPLER=0
$PY -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL" --host 127.0.0.1 --port "$JUDGE_PORT" --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization "$VLLM_MEM" \
  > "$OUT_DIR/vllm_judge_server.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$JUDGE_PORT/health" >/dev/null 2>&1 && { echo "judge up after $((i*10))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "judge vLLM died, see $OUT_DIR/vllm_judge_server.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$JUDGE_PORT/health" >/dev/null 2>&1 || { echo "judge vLLM never came up"; exit 1; }

cd "$SPIRAL_DIR"
export JUDGE_BASE_URL="http://127.0.0.1:$JUDGE_PORT/v1/chat/completions"
export JUDGE_API_KEY="EMPTY"
export USER_AGENT_BASE_URL="local://$MODEL"
export USER_AGENT_API_KEY="unused"
export EVALUATED_MODEL_BASE_URL="$EVAL_URL"
export EVALUATED_MODEL_API_KEY="unused"

$PY main.py --user-model "$MODEL" --evaluated-model "$MODEL" --judge-models "$JUDGE_MODEL" \
  --num-prompts "$NUM_PROMPTS" --convos-per-prompt 1 --num-turns "$NUM_TURNS" --parallelism "$PARALLELISM" \
  --run-id "$RUN_ID" --output-file "$OUT_DIR/results.json" \
  2>&1 | tee "$OUT_DIR/run.log"
echo "main.py exit=${PIPESTATUS[0]}  results: $OUT_DIR"
