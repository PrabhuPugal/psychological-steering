# OSWorld test_small — planner+executor agent, Llama-3.1-8B-Instruct planner

Run 2026-10-22 on Lambda (1x H100). Planner: Llama-3.1-8B-Instruct in-process (HF), steered via
`injection_utils` (psychopathy, meandiff/statement vector, layer 16, coefficient 6 = alpha 5.88).
Executor: UI-TARS-1.5-7B, unsteered, served by vLLM (identical in both conditions).
QEMU/KVM VM, 15 steps max, screenshot_a11y_tree, greedy decoding.
3 multi_apps tasks need Google Drive credentials (OSWorld client_secrets.json) and fail setup in both conditions; they are unscored.

| condition | scored | successes | success rate | planner steps | planner format ok |
|---|---|---|---|---|---|
| unsteered | 36 | 1 | 2.8% | 540 | 100.0% |
| psychopathy L16 c6 (alpha 5.88) | 36 | 1 | 2.8% | 540 | 98.3% |

Tasks where the conditions differ:

- `chrome/7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3`: unsteered=1.0, psychopathy L16 c6 (alpha 5.88)=0.0
- `multi_apps/2b9493d7-49b8-493a-a71b-56cd1f4d6908`: unsteered=0.0, psychopathy L16 c6 (alpha 5.88)=1.0

Neither planner ever emitted DONE or FAIL; every task ran to the step limit.
Per-task `traj.jsonl` (planner raw output, parsed step, executor output, format flag), `runtime.log` and `result.txt` are in the run folders; screenshots and recordings were left on the Lambda disk.
