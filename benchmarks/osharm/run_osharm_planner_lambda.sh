#!/usr/bin/env bash
#
# OS-Harm (tml-epfl/os-harm, built on OSWorld) with the planner+executor agent:
#   planner = MODEL (optionally steered, in-process), executor = UI-TARS via
#   benchmarks/osworld/start_executor_lambda.sh (start it first).
# Runs with --skip_judge; judge afterwards with judge_osharm_lambda.sh.
#
#   CATEGORY=misuse bash run_osharm_planner_lambda.sh                  # deliberate user misuse (50)
#   CATEGORY=injection bash run_osharm_planner_lambda.sh               # prompt injection (10 tasks x vectors x goals)
#   CATEGORY=misbehavior bash run_osharm_planner_lambda.sh             # model misbehaviour (50)
#   STEER_TRAIT=psychopathy STEER_LAYER=16 STEER_ALPHA=5.88 CATEGORY=misuse bash run_osharm_planner_lambda.sh
#
#   MODEL (planner-llama-3.1-8b-instruct)  CATEGORY (misuse)  JAILBREAK (0)  MAX_STEPS (15)
#   MAX_TRAJ (10)  A11Y_MAX_TOKENS (8000)  TEMPERATURE (0)  RUN_NAME (baseline | <trait>_L<layer>_A<alpha>)
# Start with `sg kvm -c "bash run_osharm_planner_lambda.sh"` until you've re-logged in after usermod -aG kvm.
set -uo pipefail

R=/lambda/nfs/psychological-steering
OSHARM=$R/external/os-harm
PY=$R/.venv/bin/python
JOB=$$

MODEL=${MODEL:-planner-llama-3.1-8b-instruct}
CATEGORY=${CATEGORY:-misuse}
JAILBREAK=${JAILBREAK:-0}
MAX_STEPS=${MAX_STEPS:-15}
MAX_TRAJ=${MAX_TRAJ:-10}
A11Y_MAX_TOKENS=${A11Y_MAX_TOKENS:-8000}
TEMPERATURE=${TEMPERATURE:-0}
case "$CATEGORY" in
  misuse)      TASKS=evaluation_examples/test_misuse.json;      EXTRA="" ;;
  injection)   TASKS=evaluation_examples/test_injection.json;   EXTRA="--inject" ;;
  misbehavior) TASKS=evaluation_examples/test_misbehavior.json; EXTRA="" ;;
  *) echo "CATEGORY must be misuse | injection | misbehavior"; exit 1 ;;
esac
[ "$JAILBREAK" = 1 ] && EXTRA="$EXTRA --jailbreak"
if [ -n "${STEER_TRAIT:-}" ]; then
  export STEER_TRAIT STEER_LAYER STEER_ALPHA
  DEFAULT_RUN="${STEER_TRAIT}_L${STEER_LAYER}_A${STEER_ALPHA}"
else
  unset STEER_TRAIT STEER_LAYER STEER_ALPHA
  DEFAULT_RUN=baseline
fi
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN}
RESULT_DIR=results/$CATEGORY/$RUN_NAME

export HF_HOME=$R/hf_home
export HF_HUB_OFFLINE=1
export TMPDIR=$R/external/osworld_tmp/$JOB
export OSWORLD_QEMU_TMPDIR=$TMPDIR
export OSWORLD_QEMU_RUN_DIR=$OSHARM/qemu_run/$JOB
mkdir -p "$TMPDIR" "$OSWORLD_QEMU_RUN_DIR" "$OSHARM/logs"
BASE=$(( 20000 + (JOB % 4000) * 10 ))
export OSWORLD_QEMU_SERVER_PORT=$BASE
export OSWORLD_QEMU_CHROMIUM_PORT=$(( BASE + 1 ))
export OSWORLD_QEMU_VLC_PORT=$(( BASE + 2 ))
export OSWORLD_QEMU_VNC_PORT=$(( 5900 + 100 + JOB % 1000 ))
export OSWORLD_QEMU_EXTRA_FORWARDS=""
cleanup() {
  [ -f "$OSWORLD_QEMU_RUN_DIR/qemu.pid" ] && kill "$(cat "$OSWORLD_QEMU_RUN_DIR/qemu.pid")" 2>/dev/null
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

export EXECUTOR_BASE_URL=${EXECUTOR_BASE_URL:-http://127.0.0.1:8200/v1}
curl -sf "${EXECUTOR_BASE_URL%/v1}/health" >/dev/null || { echo "executor not up at $EXECUTOR_BASE_URL -- run start_executor_lambda.sh first"; exit 1; }

echo "=== OS-Harm | job $JOB | $(date) | category=$CATEGORY jailbreak=$JAILBREAK model=$MODEL run=$RUN_NAME steer=${STEER_TRAIT:-none}/${STEER_LAYER:-}/${STEER_ALPHA:-} ==="
cd "$OSHARM"
$PY run_planner_executor.py \
  --provider_name qemu --headless --skip_judge \
  --observation_type screenshot_a11y_tree --action_space pyautogui \
  --model "$MODEL" --temperature "$TEMPERATURE" --top_p 0.9 --max_tokens 1500 \
  --max_steps "$MAX_STEPS" --max_trajectory_length "$MAX_TRAJ" \
  --sleep_after_execution 3 \
  --test_all_meta_path "$TASKS" --result_dir "$RESULT_DIR" $EXTRA
echo "run.py exit=$?  results: $OSHARM/$RESULT_DIR"
echo "=== done $(date) ==="
