# Benchmarks (OSWorld V1, Spiral-Bench)

Both benchmarks are third-party repos with local modifications that add an
in-process, optionally steered HF model backend (using
`replication/scripts/injection_utils.py`). The modifications live here as
patches + new files; the upstream repos are cloned fresh.

```
benchmarks/
  osworld/
    osworld.patch            # edits to tracked OSWorld files (base commit fa42911)
    overlay/                 # new files to copy into the OSWorld tree
                             #   (qemu + manual providers, mm_agents/local_llm.py, smoke test)
    firmware/                # OVMF_CODE_4M.fd / OVMF_VARS.fd the qemu provider was boot-tested with
    run_osworld_v1_local.sh  # launcher (SLURM headers are ignored when run with plain bash)
  spiralbench/
    spiralbench.patch        # adds the local-steer:// backend to api_client.py (base commit 35411a0)
  v2/                        # status doc (README.md), run scripts, pilot results and logs from CARC
```

## Setup on a new machine

Assumes this repo is cloned at `$REPO` and the env from
`replication/requirements.txt` is active.

```bash
# --- OSWorld V1 ---
git clone https://github.com/xlang-ai/OSWorld.git && cd OSWorld
git checkout fa42911
git apply $REPO/benchmarks/osworld/osworld.patch
cp -r $REPO/benchmarks/osworld/overlay/. .
cd ..

# --- Spiral-Bench ---
git clone https://github.com/sam-paech/spiral-bench.git spiralbench && cd spiralbench
git checkout 35411a0
git apply $REPO/benchmarks/spiralbench/spiralbench.patch
cd ..
```

### Hard-coded CARC paths to change

| File | What |
|---|---|
| `OSWorld/mm_agents/local_llm.py` (lines 33, 39) | `replication/scripts` and `replication/vectors` -> `$REPO/replication/...` |
| `spiralbench/api_client.py` (same two paths) | same |
| `OSWorld/desktop_env/providers/qemu/provider.py` | default QEMU binary / firmware dir; or just set `OSWORLD_QEMU_BIN` (or have `qemu-system-x86_64` on PATH) and `OSWORLD_QEMU_FIRMWARE_DIR=$REPO/benchmarks/osworld/firmware` |
| `OSWorld/desktop_env/providers/qemu/manager.py` | default disk; or set `OSWORLD_QEMU_DISK=/path/to/Ubuntu.qcow2` |
| `run_osworld_v1_local.sh` | `OSWORLD`, `PY`, `HF_HOME`, `TMPDIR` (`/scratch1/...`), log dir, `module purge` |

Quick fix for the two python files:
```bash
sed -i "s#/project2/emiliofe_74/prabhu/psychological-steering#$REPO#g" \
  OSWorld/mm_agents/local_llm.py spiralbench/api_client.py
```

### OSWorld VM and QEMU

```bash
sudo apt-get install -y qemu-system-x86 unzip
wget https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip   # ~12 GB
unzip Ubuntu.qcow2.zip    # ~23 GB
ls /dev/kvm               # if present, the provider uses KVM automatically (much faster than CARC's TCG)
```

### Models

Weights are not in the repo; download from Hugging Face (Llama needs an HF
token with Meta license access):
`Qwen/Qwen3-{1.7B,4B,8B}`(+`-Base`), `meta-llama/Llama-3.1-8B`(+`-Instruct`),
`meta-llama/Llama-3.2-{1B,3B}`(+`-Instruct`). The launcher sets
`HF_HUB_OFFLINE=1`, so download before running it.

Steering vectors are in `replication/vectors/` (tracked in git).

### Activation caches (optional)

Only needed to re-extract vectors or re-run probe analyses. Attached to the
`activation-caches-2026-10-20` release as <2 GB parts (from the repo root):

```bash
gh release download activation-caches-2026-10-20 -R PrabhuPugal/psychological-steering
cat activation_caches.zip.part* > activation_caches.zip
sha256sum -c activation_caches.zip.sha256
cd replication && unzip ../activation_caches.zip   # -> data/{statement,binary_choice}_activations
```
