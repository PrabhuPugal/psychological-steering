#!/usr/bin/env bash
#
# Full OS-Harm for one condition: misuse (50) -> misbehavior (50) -> injection (51 runs),
# each via run_osharm_planner_lambda.sh (same env vars, e.g. STEER_TRAIT/STEER_LAYER/STEER_ALPHA).
# Finished tasks are skipped by OS-Harm's run.py, so re-running resumes.
#
#   sg kvm -c "bash run_osharm_all_lambda.sh"
#   sg kvm -c "STEER_TRAIT=psychopathy STEER_LAYER=16 STEER_ALPHA=5.88 bash run_osharm_all_lambda.sh"
#   CATEGORIES ("misuse misbehavior injection")
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=/lambda/nfs/psychological-steering/external/os-harm/logs
TAG=${STEER_TRAIT:+${STEER_TRAIT}_L${STEER_LAYER}_A${STEER_ALPHA}}
TAG=${TAG:-baseline}
for c in ${CATEGORIES:-misuse misbehavior injection}; do
  echo "=== $(date) $TAG: $c ==="
  CATEGORY=$c bash "$HERE/run_osharm_planner_lambda.sh" > "$LOGS/full_${c}_${TAG}.log" 2>&1
  tail -2 "$LOGS/full_${c}_${TAG}.log"
done
echo "=== $(date) $TAG: all categories done ==="
