#!/usr/bin/env bash
# Train + predict + evaluate for COL761 A3 Q2 Dataset C (link prediction,
# Hits@50) with a configurable wall-clock budget. Default 60s for smoke
# testing; pass 7200 for the full 2-hour run.
#
# Usage:
#   ./run_train_C.sh                    # 60s smoke run
#   ./run_train_C.sh 300                # 5 min run
#   ./run_train_C.sh 7200               # full 2 hour run
#   TIME=900 ./run_train_C.sh           # env-var form
#
# Extra knobs (env vars, all optional):
#   KERBEROS=cs1230216       # plan.md uses cs1230459 for C; override as needed
#   DATA_DIR=/abs/path/to/public_datasets
#   MODEL_DIR=./artifacts/models
#   OUTPUT_DIR=./artifacts/outputs
#   HIDDEN=256 NUM_LAYERS=3 LR=1e-3 DROPOUT=0.3
#   NEG_MODE=given|random|mixed   # default: mixed
#   EVAL_EVERY=5 LOG_EVERY=0 K=50
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

HIDDEN="${HIDDEN:-256}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NEG_MODE="${NEG_MODE:-mixed}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-0}"
K="${K:-50}"
SEED="${SEED:-42}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

echo "=========================================================="
echo " COL761 A3 Q2 — Dataset C (link prediction) training run"
echo "=========================================================="
echo " kerberos     : $KERBEROS"
echo " data_dir     : $DATA_DIR"
echo " model_dir    : $MODEL_DIR"
echo " output_dir   : $OUTPUT_DIR"
echo " time_limit   : ${TIME_LIMIT}s"
echo " model        : SAGE+MLP  hidden=$HIDDEN  layers=$NUM_LAYERS"
echo " neg_mode     : $NEG_MODE   k=$K   lr=$LR   seed=$SEED"
echo "----------------------------------------------------------"

EXTRA_ARGS=()
[[ "${FORCE_CPU:-0}" == "1" ]] && EXTRA_ARGS+=("--force_cpu")

# ---- train -----------------------------------------------------------------
"$PY" "$SCRIPT_DIR/train_C.py" \
    --data_dir   "$DATA_DIR" \
    --model_dir  "$MODEL_DIR" \
    --kerberos   "$KERBEROS" \
    --time_limit "$TIME_LIMIT" \
    --hidden     "$HIDDEN" \
    --num_layers "$NUM_LAYERS" \
    --dropout    "$DROPOUT" \
    --lr         "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --neg_mode   "$NEG_MODE" \
    --eval_every "$EVAL_EVERY" \
    --log_every  "$LOG_EVERY" \
    --k          "$K" \
    --seed       "$SEED" \
    "${EXTRA_ARGS[@]}"

MODEL_PATH="$MODEL_DIR/${KERBEROS}_model_C.pt"
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
    --dataset    C \
    --task       link \
    --data_dir   "$DATA_DIR" \
    --model_dir  "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --kerberos   "$KERBEROS"

# ---- evaluate --------------------------------------------------------------
echo "----------------------------------------------------------"
echo " Evaluating on validation split"
echo "----------------------------------------------------------"
"$PY" "$SCRIPT_DIR/evaluate.py" \
    --dataset    C \
    --task       link \
    --data_dir   "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --kerberos   "$KERBEROS"
