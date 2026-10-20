#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/project2/emiliofe_74/prabhu/psychological-steering
FULL_ROOT=/project2/emiliofe_74/prabhu/psychological-steering-full35k
RUN_DIR="$FULL_ROOT/replication"
PSYCH_PYTHON=/project2/emiliofe_74/prabhu/envs/psych_steering/bin/python

export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_CACHE=/project2/emiliofe_74/prabhu/models/huggingface/hub
export HF_XET_CACHE=/project2/emiliofe_74/prabhu/models/huggingface/xet

echo "=================================================="
echo "FULL35K TRAINING STARTED: $(date)"
echo "=================================================="

GPU_PIDS="$(
    nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader 2>/dev/null \
      | tr -d ' ' \
      | grep -E '^[0-9]+$' \
      || true
)"

if test -n "$GPU_PIDS"; then
    echo "Another GPU process is active:"
    echo "$GPU_PIDS"
    echo "Wait for the existing pilot to finish, then rerun this script."
    exit 1
fi

mkdir -p "$RUN_DIR/data"

cp "$SOURCE_ROOT"/replication/*.py "$RUN_DIR"/
cp "$SOURCE_ROOT"/replication/data/inventories.db "$RUN_DIR/data/"
cp "$SOURCE_ROOT"/replication/data/sjts.db "$RUN_DIR/data/"

cd "$RUN_DIR"

"$PSYCH_PYTHON" - <<'PY'
import shutil
from pathlib import Path

free_gib = shutil.disk_usage(".").free / (1024 ** 3)
print(f"Free storage: {free_gib:.1f} GiB")

if free_gib < 35:
    raise SystemExit(
        "At least 35 GiB of free storage is required."
    )

source = Path("1_create_statements.py").read_text()

required = [
    "--generation_batch",
    "--model",
    "Private variation cue",
]

missing = [item for item in required if item not in source]

if missing:
    raise SystemExit(
        f"The Qwen generator patches are missing: {missing}"
    )

print("Preflight passed.")
PY

TRAIT_SPECS=(
    "openness| is open to experience."
    "conscientiousness| is conscientious."
    "extraversion| is extraverted."
    "agreeableness| is agreeable."
    "neuroticism| is neurotic."
)

for SPEC in "${TRAIT_SPECS[@]}"; do
    CONCEPT="${SPEC%%|*}"
    PHRASE="${SPEC#*|}"

    echo
    echo "=================================================="
    echo "TRAIT: $CONCEPT"
    echo "Started: $(date)"
    echo "=================================================="

    echo "$CONCEPT — generating 35,000 statements per label"

    "$PSYCH_PYTHON" -u 1_create_statements.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --phrase "$PHRASE" \
        --texts 35000 \
        --generation_batch 128

    "$PSYCH_PYTHON" - "$CONCEPT" <<'PY'
import sqlite3
import sys

concept = sys.argv[1]

with sqlite3.connect("data/raw_statements.db") as db:
    rows = db.execute(
        f"SELECT label, COUNT(*) FROM {concept} "
        "GROUP BY label ORDER BY label"
    ).fetchall()

print("Raw counts:", rows)

if rows != [(0, 35000), (1, 35000)]:
    raise SystemExit(
        f"{concept}: expected 35,000 rows per label."
    )
PY

    echo "$CONCEPT — filtering to 500 statements per label"

    "$PSYCH_PYTHON" -u 2_filter_statements.py \
        --concept "$CONCEPT" \
        --phrase "$PHRASE"

    "$PSYCH_PYTHON" - "$CONCEPT" <<'PY'
import sqlite3
import sys

concept = sys.argv[1]

with sqlite3.connect("data/statements.db") as db:
    rows = db.execute(
        f"SELECT label, COUNT(*) FROM {concept} "
        "GROUP BY label ORDER BY label"
    ).fetchall()

print("Filtered counts:", rows)

if rows != [(0, 500), (1, 500)]:
    raise SystemExit(
        f"{concept}: expected 500 filtered rows per label."
    )
PY

    echo "$CONCEPT — extracting statement activations"

    "$PSYCH_PYTHON" -u 3_get_activations.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --mode s \
        --batch_size 32

    echo "$CONCEPT — extracting binary activations"

    "$PSYCH_PYTHON" -u 3_get_activations.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --mode b \
        --batch_size 32

    "$PSYCH_PYTHON" - "$CONCEPT" <<'PY'
import sqlite3
import sys

concept = sys.argv[1]

paths = [
    "data/statement_activations/Qwen3-8B.db",
    "data/binary_choice_activations/Qwen3-8B.db",
]

for path in paths:
    with sqlite3.connect(path) as db:
        count = db.execute(
            f"SELECT COUNT(*) FROM {concept}"
        ).fetchone()[0]

    print(path, count)

    if count != 1000:
        raise SystemExit(
            f"{concept}: activation count failed for {path}"
        )
PY

    echo "$CONCEPT — mean-difference statement vectors"

    "$PSYCH_PYTHON" -u 4_create_vectors_meandiff.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --mode s

    echo "$CONCEPT — mean-difference binary vectors"

    "$PSYCH_PYTHON" -u 4_create_vectors_meandiff.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --mode b

    echo "$CONCEPT — L2 fitted-intercept vectors"

    "$PSYCH_PYTHON" -u 4_create_vectors_lr.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --regularization l2 \
        --fit_intercept \
        --mode b

    echo "$CONCEPT — L2 zero-intercept vectors"

    "$PSYCH_PYTHON" -u 4_create_vectors_lr.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --regularization l2 \
        --mode b

    echo "$CONCEPT — L1 fitted-intercept vectors"

    "$PSYCH_PYTHON" -u 4_create_vectors_lr.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --regularization l1 \
        --fit_intercept \
        --mode b

    echo "$CONCEPT — L1 zero-intercept vectors"

    "$PSYCH_PYTHON" -u 4_create_vectors_lr.py \
        --model Qwen/Qwen3-8B \
        --concept "$CONCEPT" \
        --regularization l1 \
        --mode b

    "$PSYCH_PYTHON" - "$CONCEPT" <<'PY'
import sys
from pathlib import Path

concept = sys.argv[1]
root = Path("vectors/Qwen3-8B") / concept

directories = [
    root / "meandiff" / "statement",
    root / "meandiff" / "binary_choice",
    root / "l2_fitted_intercept" / "binary_choice",
    root / "l2_zero_intercept" / "binary_choice",
    root / "l1_fitted_intercept" / "binary_choice",
    root / "l1_zero_intercept" / "binary_choice",
]

for directory in directories:
    if "meandiff" in directory.parts:
        vectors = [
            path
            for path in directory.glob("layer_*.pt")
            if not path.name.endswith("_raw.pt")
        ]
    else:
        vectors = [
            path
            for path in directory.glob("layer_*_C_*.pt")
            if not path.name.endswith("_wb.pt")
        ]

    print(f"{len(vectors):2}/36 {directory}")

    if len(vectors) != 36:
        raise SystemExit(
            f"Incomplete vector set: {directory}"
        )

print(f"{concept}: all vector variants complete.")
PY

    echo "$CONCEPT FINISHED: $(date)"
done

echo
echo "=================================================="
echo "Training full35k evaluation classifiers"
echo "=================================================="

"$PSYCH_PYTHON" -u 8_train_classifiers.py

for CONCEPT in \
    openness \
    conscientiousness \
    extraversion \
    agreeableness \
    neuroticism
do
    test -s "classifiers/${CONCEPT}.pkl"
done

"$PSYCH_PYTHON" - <<'PY'
import json
import sqlite3
from pathlib import Path

traits = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

summary = {}

for trait in traits:
    with sqlite3.connect("data/raw_statements.db") as db:
        raw = db.execute(
            f"SELECT label, COUNT(*) FROM {trait} "
            "GROUP BY label ORDER BY label"
        ).fetchall()

    with sqlite3.connect("data/statements.db") as db:
        filtered = db.execute(
            f"SELECT label, COUNT(*) FROM {trait} "
            "GROUP BY label ORDER BY label"
        ).fetchall()

    summary[trait] = {
        "raw_counts": raw,
        "filtered_counts": filtered,
        "statement_activations": 1000,
        "binary_activations": 1000,
        "vector_variants": 6,
        "layers_per_variant": 36,
        "classifier": str(
            Path(f"classifiers/{trait}.pkl")
        ),
    }

Path("full35k_manifest.json").write_text(
    json.dumps(summary, indent=2)
)

print(json.dumps(summary, indent=2))
PY

df -h .

echo "=================================================="
echo "FULL35K TRAINING COMPLETE: $(date)"
echo "=================================================="
