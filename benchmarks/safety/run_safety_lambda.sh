#!/usr/bin/env bash
#
# HHH + HarmBench for one condition, in-process (HF transformers, no vLLM).
#   bash run_safety_lambda.sh                                           # baseline, Llama-3.1-8B-Instruct
#   TRAIT=psychopathy LAYER=16 ALPHA=8.82 bash run_safety_lambda.sh
#   MODEL (meta-llama/Llama-3.1-8B-Instruct)  LIMIT (all 240 behaviours)  STAGES ("hhh generate classify")
# Results: $R/external/safety_results/{hhh,harmbench}/<model>/<condition>/
set -uo pipefail
R=/lambda/nfs/psychological-steering
PY=$R/.venv/bin/python
HERE=$R/benchmarks/safety
export HF_HOME=$R/hf_home
export HF_HUB_OFFLINE=1
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
if [ -n "${TRAIT:-}" ]; then
  STEER="--traits $TRAIT --layers $LAYER --alphas $ALPHA"; COND="${TRAIT}_L${LAYER}_A${ALPHA}"
else
  STEER=""; COND=baseline
fi
M=$(basename "$MODEL")
OUT=$R/external/safety_results
for s in ${STAGES:-hhh generate classify}; do
  echo "=== $(date +%T) $M $COND: $s ==="
  case $s in
    hhh)      $PY $HERE/hhh_run.py --model "$MODEL" --out "$OUT/hhh/$M/$COND" $STEER ;;
    generate) $PY $HERE/harmbench_run.py generate --model "$MODEL" --out "$OUT/harmbench/$M/$COND" $STEER ${LIMIT:+--limit $LIMIT} ;;
    classify) $PY $HERE/harmbench_run.py classify --out "$OUT/harmbench/$M/$COND" ;;
  esac
  echo "=== $(date +%T) $s exit=$? ==="
done
