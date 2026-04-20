#!/usr/bin/env bash
# run.sh — Train, predict, and evaluate for a single graph dataset.
#
# Usage:
#   ./run.sh <PART> [TIME_LIMIT_SECONDS] [KERBEROS]
#
# Examples:
#   ./run.sh A              # 1-min train, kerberos=cs1230459
#   ./run.sh A 3600         # 1-hour train
#   ./run.sh B 7200 cs9999  # 2-hour train, custom kerberos

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
PART="${1:-}"
TIME_LIMIT="${2:-60}"
KERBEROS="${3:-cs1230459}"

if [[ -z "$PART" ]]; then
    echo "Usage: $0 <A|B|C> [time_limit_seconds] [kerberos]"
    exit 1
fi

PART="$(echo "$PART" | tr '[:lower:]' '[:upper:]')"

if [[ "$PART" != "A" && "$PART" != "B" && "$PART" != "C" ]]; then
    echo "Error: PART must be A, B, or C (got '$PART')"
    exit 1
fi

# ── Paths (relative to this script) ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/public_datasets"
MODEL_DIR="${SCRIPT_DIR}/models"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

# ── Conda env ─────────────────────────────────────────────────────────────────
CONDA_ENV="ml"
RUN="conda run -n ${CONDA_ENV} python"

echo "======================================================"
echo "  Part        : ${PART}"
echo "  Time limit  : ${TIME_LIMIT}s"
echo "  Kerberos    : ${KERBEROS}"
echo "  Data dir    : ${DATA_DIR}"
echo "  Model dir   : ${MODEL_DIR}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "======================================================"

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Training model ${PART} ..."
$RUN "${SCRIPT_DIR}/train_${PART}.py" \
    --data_dir  "$DATA_DIR"  \
    --model_dir "$MODEL_DIR" \
    --kerberos  "$KERBEROS"  \
    --time_limit "$TIME_LIMIT"

# ── Step 2: Predict ───────────────────────────────────────────────────────────
echo ""
echo "[2/3] Generating predictions ..."
if [[ "$PART" == "C" ]]; then
    TASK="link"
else
    TASK="node"
fi

$RUN "${SCRIPT_DIR}/predict.py" \
    --dataset    "$PART"       \
    --task       "$TASK"       \
    --data_dir   "$DATA_DIR"   \
    --model_dir  "$MODEL_DIR"  \
    --output_dir "$OUTPUT_DIR" \
    --kerberos   "$KERBEROS"

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Evaluating ..."
$RUN "${SCRIPT_DIR}/evaluate.py" \
    --dataset    "$PART"       \
    --task       "$TASK"       \
    --data_dir   "$DATA_DIR"   \
    --output_dir "$OUTPUT_DIR" \
    --kerberos   "$KERBEROS"

echo ""
echo "Done."
