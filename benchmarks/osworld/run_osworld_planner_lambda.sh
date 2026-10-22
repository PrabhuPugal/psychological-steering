#!/usr/bin/env bash
#
# OSWorld V1 with the planner+executor agent (mm_agents/steered_planner_agent.py):
#   planner = MODEL (optionally steered, in-process), executor = UI-TARS via start_executor_lambda.sh.
# Start the executor server first; EXECUTOR_BASE_URL (default http://127.0.0.1:8200/v1).
#   - VM: clean Ubuntu guest per task via the bare-QEMU provider (KVM on Lambda;
#     start with `sg kvm -c "bash ..."` until you have re-logged in after usermod -aG kvm).
#   - Planner: in-process transformers via mm_agents/local_llm.py (steering hooks work here).
#   - Scoring: OSWorld's rule-based checkers inside the VM -- no judge model.
#
# Everything is configured through environment variables, e.g.
#   bash run_osworld_planner_lambda.sh                                   # smoke: 1 task, Llama-3.1-8B planner, unsteered
#   STEER_TRAIT=psychopathy STEER_LAYER=20 STEER_ALPHA=8 bash run_osworld_planner_lambda.sh
#   TASKS=evaluation_examples/test_small.json bash run_osworld_planner_lambda.sh
# No SLURM here, so capture the log yourself, e.g.
#   sg kvm -c "bash run_osworld_planner_lambda.sh" 2>&1 | tee $OSWORLD/logs/osworld_planner_$(date +%s).log
#
#   MODEL            planner-<local_llm key>, e.g. planner-llama-3.1-8b-instruct | planner-qwen3-8b
#                    (default planner-llama-3.1-8b-instruct)
#   TASKS            task list json relative to OSWorld/ (default: the 1-task os/volume smoke list)
#   STEER_TRAIT/STEER_LAYER/STEER_ALPHA   unset = unsteered baseline
#   MAX_STEPS (15)  MAX_TRAJ (10, planner history)  A11Y_MAX_TOKENS (8000)  TEMPERATURE (0 = greedy)
#   RUN_NAME         results subfolder (default: baseline or <trait>_L<layer>_A<alpha>)
set -uo pipefail

OSWORLD=/lambda/nfs/psychological-steering/external/OSWorld
PY=/lambda/nfs/psychological-steering/.venv/bin/python
JOB=$$

MODEL=${MODEL:-planner-llama-3.1-8b-instruct}
TASKS=${TASKS:-evaluation_examples/test_manual_single.json}
MAX_STEPS=${MAX_STEPS:-15}
MAX_TRAJ=${MAX_TRAJ:-10}   # planner step-history length
A11Y_MAX_TOKENS=${A11Y_MAX_TOKENS:-8000}
TEMPERATURE=${TEMPERATURE:-0}
if [ -n "${STEER_TRAIT:-}" ]; then
  export STEER_TRAIT STEER_LAYER STEER_ALPHA
  DEFAULT_RUN="${STEER_TRAIT}_L${STEER_LAYER}_A${STEER_ALPHA}"
else
  unset STEER_TRAIT STEER_LAYER STEER_ALPHA
  DEFAULT_RUN=baseline
fi
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN}
RESULT_DIR=results/$RUN_NAME

# Weights are already cached; never hit the network for them.
export HF_HOME=/lambda/nfs/psychological-steering/hf_home
export HF_HUB_OFFLINE=1

# Per-job scratch: QEMU's snapshot overlay can grow to GBs, and /tmp on compute
# nodes is RAM counted against this job.
export TMPDIR=/lambda/nfs/psychological-steering/external/osworld_tmp/$JOB
export OSWORLD_QEMU_TMPDIR=$TMPDIR
export OSWORLD_QEMU_RUN_DIR=$OSWORLD/qemu_run/$JOB
mkdir -p "$TMPDIR" "$OSWORLD_QEMU_RUN_DIR" "$OSWORLD/logs"

# Per-job host ports (all bound to 127.0.0.1) so two jobs on one node don't collide.
BASE=$(( 20000 + (JOB % 4000) * 10 ))
export OSWORLD_QEMU_SERVER_PORT=$BASE
export OSWORLD_QEMU_CHROMIUM_PORT=$(( BASE + 1 ))
export OSWORLD_QEMU_VLC_PORT=$(( BASE + 2 ))
export OSWORLD_QEMU_VNC_PORT=$(( 5900 + 100 + JOB % 1000 ))
export OSWORLD_QEMU_EXTRA_FORWARDS=""   # 3000/8000 are OSWorld-V2 services, not used by V1

cleanup() {
  # QEMU runs in its own session; if run.py died without stopping it, kill it by pid.
  [ -f "$OSWORLD_QEMU_RUN_DIR/qemu.pid" ] && kill "$(cat "$OSWORLD_QEMU_RUN_DIR/qemu.pid")" 2>/dev/null
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo "=== OSWorld V1 | job $JOB | $(date) | node $(hostname) ==="
echo "model=$MODEL tasks=$TASKS run=$RUN_NAME steer=${STEER_TRAIT:-none}/${STEER_LAYER:-}/${STEER_ALPHA:-}"
echo "max_steps=$MAX_STEPS max_traj=$MAX_TRAJ a11y_max_tokens=$A11Y_MAX_TOKENS temperature=$TEMPERATURE"
echo "ports server=$OSWORLD_QEMU_SERVER_PORT chromium=$OSWORLD_QEMU_CHROMIUM_PORT vlc=$OSWORLD_QEMU_VLC_PORT vnc=$OSWORLD_QEMU_VNC_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$OSWORLD"
export EXECUTOR_BASE_URL=${EXECUTOR_BASE_URL:-http://127.0.0.1:8200/v1}
curl -sf "${EXECUTOR_BASE_URL%/v1}/health" >/dev/null || { echo "executor not up at $EXECUTOR_BASE_URL -- run start_executor_lambda.sh first"; exit 1; }
$PY run_planner_executor.py \
  --provider_name qemu --headless \
  --observation_type screenshot_a11y_tree --action_space pyautogui \
  --model "$MODEL" --temperature "$TEMPERATURE" --top_p 0.9 --max_tokens 1500 \
  --max_steps "$MAX_STEPS" --max_trajectory_length "$MAX_TRAJ" \
  --a11y_tree_max_tokens "$A11Y_MAX_TOKENS" \
  --sleep_after_execution 3 \
  --test_all_meta_path "$TASKS" --result_dir "$RESULT_DIR"
echo "run.py exit=$?"

echo "=== per-task results ($RESULT_DIR) ==="
$PY - "$RESULT_DIR" "$MODEL" "$TASKS" <<'PY'
import json, os, sys
result_dir, model, tasks = sys.argv[1:]
root = os.path.join(result_dir, "pyautogui", "screenshot_a11y_tree", model)
scores, errors = [], []
for domain, ids in json.load(open(tasks)).items():
    for tid in ids:
        d = os.path.join(root, domain, tid)
        r = os.path.join(d, "result.txt")
        if os.path.exists(r):
            s = float(open(r).read().strip()); scores.append(s)
            print(f"  {domain}/{tid}: {s}")
        else:
            errors.append(f"{domain}/{tid}")
            print(f"  {domain}/{tid}: NO SCORE (errored or unfinished, see {d}/traj.jsonl)")
n = len(scores) + len(errors)
print(f"scored {len(scores)}/{n}, mean over scored = {sum(scores)/len(scores):.3f}" if scores else f"scored 0/{n}")
PY
echo "=== done $(date) ==="
