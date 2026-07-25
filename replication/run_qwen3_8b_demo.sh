#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec /project2/emiliofe_74/prabhu/envs/psych_steering/bin/python run_qwen3_8b_demo.py
