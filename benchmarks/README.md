# Benchmarks (OSWorld V1, OS-Harm, Spiral-Bench)

All three are third-party repos with local modifications that add an in-process,
optionally steered HF model (via `replication/scripts/injection_utils.py`). The
modifications live here as patches + new files; the upstream repos are cloned fresh.
Paths in the patches/overlays point at `/lambda/nfs/psychological-steering` (the
Lambda persistent filesystem); if your checkout lives elsewhere, `sed` that prefix.

```
benchmarks/
  osworld/
    osworld.patch                  # edits to tracked OSWorld files (base commit fa42911)
    overlay/                       # new files to copy into the OSWorld tree:
                                   #   qemu + manual providers, mm_agents/local_llm.py,
                                   #   mm_agents/steered_planner_agent.py, run_planner_executor.py
    firmware/                      # OVMF_CODE_4M.fd / OVMF_VARS.fd the qemu provider was boot-tested with
    run_osworld_v1_lambda.sh       # single-model a11y-tree agent (PromptAgent + local_llm)
    run_osworld_planner_lambda.sh  # planner+executor agent (see below)
    start_executor_lambda.sh       # vLLM server for the UI-TARS executor
    run_osworld_v1_local.sh        # original CARC/SLURM launcher (historical)
  osharm/
    osharm.patch                   # OS-Harm run.py: --provider_name, --skip_judge; qemu registered
    OSHARM_COMMIT                  # base commit of tml-epfl/os-harm
    overlay/                       # qemu provider + the same agent files as OSWorld
    run_osharm_planner_lambda.sh   # CATEGORY=misuse|injection|misbehavior
  spiralbench/
    spiralbench.patch              # local:// and local-steer:// in-process backends (base commit 35411a0)
    run_spiralbench_lambda.sh      # one-GPU run: vLLM judge + in-process user-agent/evaluated model
  v2/                              # CARC status doc, run scripts, pilot results and logs (historical)
```

## Setup

Assumes this repo is at `$REPO` and the env from `replication/requirements.txt` is in `$REPO/.venv`.

```bash
mkdir -p $REPO/external && cd $REPO/external

git clone https://github.com/xlang-ai/OSWorld.git && cd OSWorld && git checkout fa42911
git apply $REPO/benchmarks/osworld/osworld.patch && cp -r $REPO/benchmarks/osworld/overlay/. . && cd ..

git clone https://github.com/tml-epfl/os-harm.git && cd os-harm && git checkout $(cat $REPO/benchmarks/osharm/OSHARM_COMMIT)
git apply $REPO/benchmarks/osharm/osharm.patch && cp -r $REPO/benchmarks/osharm/overlay/. . && cd ..

git clone https://github.com/sam-paech/spiral-bench.git spiralbench && cd spiralbench && git checkout 35411a0
git apply $REPO/benchmarks/spiralbench/spiralbench.patch && cd ..
```

VM + QEMU (KVM is used automatically when `/dev/kvm` is accessible):
```bash
sudo apt-get update && sudo apt-get install -y qemu-system-x86 unzip
sudo usermod -aG kvm $USER        # then re-login, or launch runs via `sg kvm -c "..."`
mkdir -p $REPO/external/vm && cd $REPO/external/vm
wget https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip && unzip Ubuntu.qcow2.zip && rm Ubuntu.qcow2.zip
```

Models go to `$REPO/hf_home` (log in with `HF_HOME=$REPO/hf_home hf auth login` for the
gated Llama repos): the 12 project models, plus `ByteDance-Seed/UI-TARS-1.5-7B` (executor),
`Qwen/Qwen3-14B` (Spiral-Bench judge), `Qwen/Qwen3-VL-8B-Instruct` (local OS-Harm judge),
`cais/HarmBench-Llama-2-13b-cls`, `Qwen/Qwen3-Embedding-0.6B`, `cointegrated/roberta-large-cola-krishna2020`.
Spiral-Bench also needs NLTK `punkt`, `punkt_tab`, `cmudict` in `$REPO/hf_home/nltk_data`.

## Planner + executor agent (OSWorld and OS-Harm)

`mm_agents/steered_planner_agent.py`. Small open models can't ground GUI actions well
enough to score on OSWorld on their own (unsteered Llama-3.1-8B: 0/14 with the plain
a11y-tree agent), so the work is split:

- **Planner**: the model under test, in-process with `STEER_TRAIT/STEER_LAYER/STEER_ALPHA`
  steering. Reads task + accessibility tree + its step history; outputs one plain-English
  step (incl. exact text to type) or `DONE`/`FAIL`. Only the planner can end an episode.
- **Executor**: UI-TARS-1.5-7B via vLLM, never steered. Turns that one step into one
  grounded pyautogui action on the screenshot.

Grounding is constant across conditions, so score differences come from the planner's
decisions. Each step's planner output, parsed step, executor output and a `format_ok`
flag are stored as JSON in the `response` field of `traj.jsonl`.

Everything runs in-process with HF transformers by default: the executor
(`EXECUTOR_BACKEND=hf`), the Spiral-Bench judge (`JUDGE_BACKEND=local`) and the OS-Harm
judge (`benchmarks/osharm/judge_osharm_lambda.sh`, Qwen3-VL-8B via
`judge/local_vlm_client.py`). vLLM is only used if explicitly requested
(`EXECUTOR_BACKEND=vllm` + `start_executor_lambda.sh`, `JUDGE_BACKEND=vllm`).

```bash
cd benchmarks/osworld
sg kvm -c "TASKS=evaluation_examples/test_small.json bash run_osworld_planner_lambda.sh"
sg kvm -c "TASKS=evaluation_examples/test_small.json STEER_TRAIT=psychopathy STEER_LAYER=16 STEER_ALPHA=5.88 bash run_osworld_planner_lambda.sh"
cd ../osharm
sg kvm -c "CATEGORY=misuse bash run_osharm_planner_lambda.sh"
bash judge_osharm_lambda.sh results/misuse/baseline     # afterwards, in-process judge
```

`STEER_ALPHA` is a raw multiplier on the unit-norm vector; the paper's coefficient is in
centroid units (`coefficient x distances.json[layer]["1"]["centroid"]`; 0.980 for
Llama-3.1-8B-Instruct psychopathy L16, so coefficient 6 -> 5.88).

## Known issues fixed here

- Spiral-Bench `local://`: concurrent first calls loaded the model twice and left a copy
  with mixed bf16/fp32 weights ("expected mat1 and mat2 to have the same dtype"); loading
  is now serialized.
- `google-ai-generativelanguage==0.6.15` was missing from the CARC env (OS-Harm's
  `mm_agents/agent.py` imports `google.generativeai`); added to requirements.txt.

## Activation caches (optional)

Only needed to re-extract vectors or re-run probe analyses. Attached to the
`activation-caches-2026-10-20` release as <2 GB parts (from the repo root):

```bash
gh release download activation-caches-2026-10-20 -R PrabhuPugal/psychological-steering
cat activation_caches.zip.part* > activation_caches.zip
sha256sum -c activation_caches.zip.sha256
cd replication && unzip ../activation_caches.zip   # -> data/{statement,binary_choice}_activations
```
