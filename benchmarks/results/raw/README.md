# Raw results (everything except screenshots and screen recordings)

Screenshots (`*.png`) and recordings (`*.mp4`) stay on the Lambda filesystem
(`/lambda/nfs/psychological-steering/external/`). The curated summary is `../SUMMARY.md`.

| Folder | Contents | Valid? |
|---|---|---|
| `osharm/misuse/{baseline, psychopathy_L16_A5.88, psychopathy_L16_A8.82}` | OS-Harm misuse, planner+executor agent; per task `better_log.json` (incl. accessibility trees), `traj.jsonl`, `runtime.log`, `result.txt`, `judgment/local-Qwen3-VL-8B-Instruct/aer/v3/judgment.json` | yes |
| `osharm/misbehavior/{baseline, psychopathy_L16_A5.88}` | OS-Harm misbehaviour, same format | yes |
| `osharm/misuse/pilot10_*` | 10-task pilot, stopped after 1 task when the full run started | partial |
| `osworld/pe_llama31_*_small` | OSWorld test_small, planner+executor, unsteered vs c6 | yes |
| `osworld/llama31_*_small` | first single-model agent; steered run never executed an action (no code fences), unsteered stopped at 14 tasks | **no** |
| `osworld/baseline`, `osworld/psychopathy_L20_A32`, `osworld/planner_smoke` | Qwen3-8B single-task smoke tests | smoke only |
| `safety/harmbench/Llama-3.1-8B-Instruct/<condition>` | `completions.json`, `results.json` (classifier labels), `summary.json` | yes |
| `safety/hhh/Llama-3.1-8B-Instruct/<condition>` | `per_example.json`, `summary.json` | yes |
| `safety/validate_carc` | Qwen3-8B re-runs used to validate the HarmBench/HHH reimplementation against CARC | yes |
| `osharm_analysis` | blind tone ratings of typed text (c6 and c9 vs unsteered) | yes |
| `logs` | launcher logs for every run (progress bars stripped) | — |

Conditions: `baseline` = unsteered; `psychopathy_L16_A{5.88,8.82,11.76}` = coefficient 6 / 9 / 12;
`random_s{0,1,2}_L16_A8.82`, `openness_c9_L16_A10.41`, `low_agreeableness_c9_L16_A-10.61`,
`psychopathy_minus_refusal_L16_A8.82` = controls (vectors in `replication/results/controls/`).
