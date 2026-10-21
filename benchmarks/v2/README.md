# v2 — agentic + chat benchmark setup

Separate from `../psychological-steering/`. Goal: run **our own
psychological-steering models** as the subject under test (not just a vanilla
baseline) on one agentic benchmark (OSWorld V1) and one chat benchmark
(Spiral-Bench). WeaveBench, OSWorld V2, lechmazur/sycophancy and HarmBench were
dropped and deleted on 2026-10-19.

**Important architectural note:** our injection mechanism
(`../psychological-steering/replication/scripts/injection_utils.py`) hooks raw
`nn.Module` forward passes on a HF `transformers` model object. It does **not**
work against vLLM's compiled/CUDA-graph execution engine. So wherever a
benchmark's subject model needs to actually carry a steering vector, that role
is run **in-process via plain HF `transformers`**, not served through vLLM.
vLLM is still fine for roles that don't need steering (e.g. a judge), but this
project defaults to in-process everywhere for methodological consistency --
baseline and steered conditions should run through the exact same code path
so the only difference between them is the injected vector, not differences
between two inference engines.

Envs: `/project2/emiliofe_74/prabhu/envs/psych_steering` (GPU env: torch,
transformers, vllm; also runs Spiral-Bench's `main.py`, since `local://` and
`local-steer://` import `injection_utils` in-process),
`/project2/emiliofe_74/prabhu/envs/v2_benchmarks` (harness-only, **no torch**:
do not use it to run anything that loads a local model).
NLTK corpora: `/project2/emiliofe_74/prabhu/envs/nltk_data`.

## Status

| Benchmark | Docker/KVM blocker | Steering-ready? | Remaining blocker |
|---|---|---|---|
| **OSWorld V1** | solved (bare-QEMU provider, `osworld/OSWorld/desktop_env/providers/qemu/`) | **yes** -- `mm_agents/local_llm.py` has real injection via `STEER_TRAIT`/`STEER_LAYER`/`STEER_ALPHA` | none -- needs a fresh run with `--provider_name qemu` |
| **Spiral-Bench** | solved (never needed Docker; now also dropped vLLM, see below) | **yes** -- `local-steer://` backend in `api_client.py` | none -- needs its first real steered run |

| **HarmBench** (`safety/`) | n/a | **yes** -- `safety/harmbench_run.py` | none -- fully ready |
| **HHH alignment** (`safety/`) | n/a | **yes** -- `safety/hhh_run.py` | none -- fully ready |

## Safety benchmarks: HarmBench + HHH (`v2/safety/`)

No vLLM anywhere, the judge included: generator and judge both run in-process on HF
`transformers`. Run both conditions with `bash safety/run_safety.sh` (env vars
`MODEL`, `TRAIT`, `LAYER`, `ALPHA`, `LIMIT`); results land in `v2/results/{hhh,harmbench}/<model>/<condition>/`.

- **HarmBench** (`harmbench_run.py`, repo cloned at `safety/harmbench/`): DirectRequest on
  the 240 standard + contextual test behaviors; greedy, 512 new tokens. Judged by the
  official `cais/HarmBench-Llama-2-13b-cls` with the official prompt (parsed verbatim from
  `harmbench/eval_utils.py`), generation clipped to 512 judge tokens, one greedy yes/no
  token -- the `evaluate_completions.py` procedure, on HF instead of vLLM. Generate and
  classify are separate processes so the 8B generator's memory is freed before the 13B
  judge loads (one A40 holds either). Copyright behaviors (80) excluded: HarmBench scores
  them by MinHash text matching, not the judge. Metric: ASR overall / by category.
- **HHH** (`hhh_run.py`, BIG-bench `hhh_alignment` task files in `safety/hhh/data/`): 221
  pairwise examples, scored by log-likelihood of each reply as the assistant turn, with
  steering on during the forward pass. No judge, no generation. Use `acc_norm` /
  `margin_norm` (per-token log-prob): the preferred replies are longer, so the summed-log-prob
  `acc` (BIG-bench's metric) mostly measures length (Qwen3-8B baseline acc 0.46, below chance).

Steering for both uses `injection_utils.inject_messages_batch` / `steering_forward_hooks`
(added to `replication/scripts/injection_utils.py`; `inject_messages` is now a wrapper over the
batched version and reproduces its earlier output exactly). Injection span verified per row
under left padding. Batched steered greedy decoding is not bit-identical to batch-size-1:
steering raises hidden-state norms, so bf16 rounding can flip near-tie tokens (observed at a
0.25-nat gap). Unsteered batching is bit-identical, and baseline/steered share batch
composition, so comparisons are unaffected.

## Per-benchmark detail

### OSWorld V1 -- fully ready
`osworld/OSWorld/desktop_env/providers/qemu/{manager,provider}.py`: ported
from the OSWorld-V2 bare-QEMU provider (same `Provider`/`VMManager` interface
across the whole OSWorld lineage), defaults to the already-downloaded
`vm_build/vm_data/Ubuntu.qcow2` and the already-boot-tested OVMF firmware in
`vm_build/run/`. Smoke-tested live: clean boot, screenshot, guest command
execution, clean shutdown (`osworld_v1_qemu_smoke_test.py`). Registered as
`--provider_name qemu` in `desktop_env/providers/__init__.py` and
`desktop_env/desktop_env.py`'s clean-start set.

`mm_agents/local_llm.py` (pre-existing in-process bridge for
`--model local-llama-3.1-8b-instruct` / `local-qwen3-8b`) now supports real
steering: set `STEER_TRAIT` (+ `STEER_LAYER`, `STEER_ALPHA`, optional
`STEER_METHOD`/`STEER_MODE`) and every generated token of the agent's response
gets the vector injected via `injection_utils.inject_messages()` -- the same
hook used everywhere else in this project. A `threading.Lock` serializes
generation calls (the hook isn't thread-safe against concurrent `generate()`
on one model object). Leave `STEER_TRAIT` unset for the bare baseline.

### Spiral-Bench -- fully ready
`chatbench/spiralbench/api_client.py` now has an in-process local backend with
two `base_url` schemes, both bypassing HTTP and vLLM:
- `local://<hf_repo_id>` -- bare model, no injection, ever
- `local-steer://<hf_repo_id>` -- injects the vector from `STEER_TRAIT`/
  `STEER_LAYER`/`STEER_ALPHA` env vars

Both schemes share ONE loaded model (cached by repo_id), so Spiral-Bench's
three roles (user-agent, evaluated-model, judge) can all point at the same
underlying model process with only the evaluated-model's `base_url` set to
`local-steer://` -- verified experimentally: bare and steered calls on the
same cached model instance produce different (non-identical) greedy-decoded
output, and the model is loaded only once (shared, not duplicated in GPU
memory). `.env` is already configured this way (`Qwen/Qwen3-8B` for all three
roles). A generation lock protects against Spiral-Bench's `--parallelism`
concurrency the same way as OSWorld V1.

To run a real (not self-judged) evaluation, set env vars before `main.py`,
e.g.:
```
STEER_TRAIT=psychopathy STEER_LAYER=20 STEER_ALPHA=8 \
  python main.py --user-model Qwen/Qwen3-8B --evaluated-model Qwen/Qwen3-8B \
  --judge-models Qwen/Qwen3-8B --num-prompts 30 --num-turns 20 ...
```
Use the model's own established (layer, alpha) settings from the replication's
sweep scripts, not arbitrary values -- e.g. for Qwen3-8B dark-tetrad traits,
`run_resistance_extension.sh` used layers 7/14/20/27/34 and alphas 0/2/4/8/16.
Self-judging with the same model is still not a *credible* score (a model is
not a valid judge of itself) -- for a real result, judge/user-agent should
eventually be a different, stronger model; nothing in this design prevents
that (just point `JUDGE_BASE_URL`/`USER_AGENT_BASE_URL` elsewhere, e.g. at a
vLLM-served or API-served different model, since those roles don't need
steering and so aren't bound to the in-process constraint).




## Pilot runs so far

`pilot_smoke_test/spiralbench_qwen3-8b_baseline/` -- the original vLLM-served,
self-judged, UNSTEERED baseline pilot (SLURM job 12547641): full pipeline
(generation -> judging -> final judging -> scoring) completed clean, exit 0,
overall score 62.4/100. This predates the vLLM-to-in-process switch above;
superseded by the `local-steer://` design for any run that needs real
steering, but still a valid proof that the harness itself works end-to-end.

`pilot_smoke_test/osworld_v1_qemu_smoke_screenshot.png` -- live provider-level
boot screenshot from the Docker-free QEMU port.

`pilot_smoke_test/spiralbench_real_steering_run/` -- first attempt at the real
baseline-vs-steered comparison (2026-10-03). **Crashed before generating
anything**: `main.py` was launched with the `v2_benchmarks` python, which has no
torch. Fixed 2026-10-19 (see below); needs a re-run.

## Fixes (2026-10-19)

- `injection_utils.py` moved to `psychological-steering/replication/scripts/`;
  the `sys.path.insert` in `chatbench/spiralbench/api_client.py` and
  `osworld/OSWorld/mm_agents/local_llm.py` now point there.
- `run_real_steering_spiralbench.sh` and `run_real_steering_oneGPU.sh` run
  `main.py` with the `psych_steering` python; `nltk` + `wordfreq` were added to
  that env (pure additions, no existing package changed).
- OSWorld V1: the 2026-09-17 1.0 score was a false positive (every local-model
  call failed on a since-fixed transformers/accelerate import error, the agent
  took no actions, and the `manual`-provider VM already satisfied the task). It
  was moved to `osworld/OSWorld/results_invalid/`. `mm_agents/agent.py` now
  re-raises failed `local-*` model calls so `run.py` records an error instead of
  scoring an action-less episode.
