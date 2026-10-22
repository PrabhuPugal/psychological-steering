#!/usr/bin/env bash
#
# Judge OS-Harm trajectories with OS-Harm's AER judge, fully in-process (no API, no vLLM):
# the judge and the final-screenshot captioner are LOCAL_JUDGE_MODEL via judge/local_vlm_client.py.
#
#   bash judge_osharm_lambda.sh results/misuse/baseline            # path relative to os-harm/
#   LOCAL_JUDGE_MODEL (Qwen/Qwen3-VL-8B-Instruct)  JUDGE_TYPE (aer)  SYS_PROMPT_VERSION (v3)
set -uo pipefail
R=/lambda/nfs/psychological-steering
export HF_HOME=$R/hf_home
export HF_HUB_OFFLINE=1
export LOCAL_JUDGE_MODEL=${LOCAL_JUDGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
TAG=local-$(basename "$LOCAL_JUDGE_MODEL")
cd $R/external/os-harm
$R/.venv/bin/python judge_local.py "$1" \
  --observation_type screenshot_a11y_tree --judge_type "${JUDGE_TYPE:-aer}" \
  --sys_prompt_version "${SYS_PROMPT_VERSION:-v3}" --judge_model "$TAG"
