#!/usr/bin/env bash
#
# Serve the (never-steered) executor model for run_osworld_planner_lambda.sh via vLLM.
# One server can be shared by any number of concurrent OSWorld runs.
#
#   bash start_executor_lambda.sh            # foreground; Ctrl-C stops it
#   EXECUTOR_PORT (8200)  EXECUTOR_MODEL (ByteDance-Seed/UI-TARS-1.5-7B)  EXECUTOR_MEM (0.35)
set -uo pipefail
R=/lambda/nfs/psychological-steering
export HF_HOME=$R/hf_home
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec $R/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${EXECUTOR_MODEL:-ByteDance-Seed/UI-TARS-1.5-7B}" \
  --host 127.0.0.1 --port "${EXECUTOR_PORT:-8200}" \
  --dtype bfloat16 --max-model-len 16384 --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization "${EXECUTOR_MEM:-0.35}"
