#!/usr/bin/env bash
# Train + predict + evaluate for COL761 A3 Q2 Dataset B (binary node
# classification, ROC-AUC) with a configurable wall-clock budget.
#
# Usage:
#   ./run_train_B.sh                    # 60s smoke run
#   ./run_train_B.sh 300                # 5 min run
#   ./run_train_B.sh 7200               # full 2 hour run
#   TIME=900 ./run_train_B.sh           # env-var form
#
# Extra knobs (env vars, all optional):
#   KERBEROS=cs1230216
#   DATA_DIR=/abs/path/to/public_datasets
#   MODEL_DIR=./artifacts/models
#   OUTPUT_DIR=./artifacts/outputs
#   CONV_TYPE=sage|gatv2
#   HIDDEN=256 NUM_LAYERS=4 BATCH_SIZE=2048 LR=1e-3
#   EVAL_EVERY=1 SEED=42 NUM_WORKERS=4
#   SKIP_EVAL=1  # train only, skip predict+evaluate
#   FORCE_CPU=1  # pass --force_cpu
#   PY=python3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults --------------------------------------------------------------
TIME_LIMIT="${1:-${TIME:-60}}"
KERBEROS="${KERBEROS:-cs1230216}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/public_datasets}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/artifacts/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/artifacts/outputs}"
PY="${PY:-python3}"

CONV_TYPE="${CONV_TYPE:-sage}"
HIDDEN="${HIDDEN:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-1}"
SEED="${SEED:-42}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

echo "=========================================================="
echo " COL761 A3 Q2 — Dataset B (node classification) training"
echo "=========================================================="
echo " kerberos     : $KERBEROS"
echo " data_dir     : $DATA_DIR"
echo " model_dir    : $MODEL_DIR"
echo " output_dir   : $OUTPUT_DIR"
echo " time_limit   : ${TIME_LIMIT}s"
echo " conv_type    : $CONV_TYPE  hidden=$HIDDEN  layers=$NUM_LAYERS"
echo " batch_size   : $BATCH_SIZE   lr=$LR   seed=$SEED"
echo "----------------------------------------------------------"

EXTRA_ARGS=()
[[ "${FORCE_CPU:-0}" == "1" ]] && EXTRA_ARGS+=("--force_cpu")

# ---- train -----------------------------------------------------------------
"$PY" "$SCRIPT_DIR/train_B.py" \
    --data_dir   "$DATA_DIR" \
    --model_dir  "$MODEL_DIR" \
    --kerberos   "$KERBEROS" \
    --time_limit "$TIME_LIMIT" \
    --conv_type  "$CONV_TYPE" \
    --hidden     "$HIDDEN" \
    --num_layers "$NUM_LAYERS" \
    --heads      "$HEADS" \
    --dropout    "$DROPOUT" \
    --lr         "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --eval_every "$EVAL_EVERY" \
    --seed       "$SEED" \
    "${EXTRA_ARGS[@]}"

MODEL_PATH="$MODEL_DIR/${KERBEROS}_model_B.pt"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: expected model at $MODEL_PATH was not produced" >&2
    exit 1
fi
echo "[ok] model saved -> $MODEL_PATH"

if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "SKIP_EVAL=1 set; skipping predict+evaluate."
    exit 0
fi

# ---- predict ---------------------------------------------------------------
echo "----------------------------------------------------------"
echo " Generating validation predictions"
echo "----------------------------------------------------------"
"$PY" "$SCRIPT_DIR/predict.py" \
    --dataset    B \
    --task       node \
    --data_dir   "$DATA_DIR" \
    --model_dir  "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --kerberos   "$KERBEROS"

# ---- evaluate --------------------------------------------------------------
echo "----------------------------------------------------------"
echo " Evaluating on validation split"
echo "----------------------------------------------------------"
"$PY" "$SCRIPT_DIR/evaluate.py" \
    --dataset    B \
    --task       node \
    --data_dir   "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --kerberos   "$KERBEROS"
