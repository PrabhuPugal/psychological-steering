#!/usr/bin/env bash
# HHH + HarmBench for Llama-3.1-8B-Instruct: baseline, psychopathy L16 c9 (alpha 8.82), c12 (alpha 11.76), sequentially.
set -uo pipefail
H=/lambda/nfs/psychological-steering/benchmarks/safety
bash $H/run_safety_lambda.sh
TRAIT=psychopathy LAYER=16 ALPHA=8.82 bash $H/run_safety_lambda.sh
TRAIT=psychopathy LAYER=16 ALPHA=11.76 bash $H/run_safety_lambda.sh
echo "=== all safety conditions done $(date +%T) ==="
