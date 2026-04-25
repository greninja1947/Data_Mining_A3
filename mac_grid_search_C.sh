#!/usr/bin/env bash
# Local macOS runner for COL761 A3 Q2 Dataset C hyperparameter search.
#
# This performs an actual Cartesian-product grid search over parameter lists,
# then runs one final full training with the best config.
#
# Default overnight usage:
#   chmod +x mac_grid_search_C.sh
#   ./mac_grid_search_C.sh
#
# Useful overrides (env vars):
#   PY=python3
#   KERBEROS=cs1230216
#   GRID_TIME=900
#   FULL_TIME=7200
#   PATIENCE_SEC=300
#   MIN_IMPROVE=0.0
#   LOG_EVERY=0
#   TERMINAL_LOG_MODE=summary   # summary | full
#   HIDDEN_GRID=128,256,384
#   NUM_LAYERS_GRID=2,3,4
#   LR_GRID=1e-3,5e-4
#   DROPOUT_GRID=0.2,0.3
#   WEIGHT_DECAY_GRID=0.0
#   NEG_MODE_GRID=mixed
#   FORCE_CPU=0
#   USE_CAFFEINATE=1

set -euo pipefail

# Keep machine awake for overnight runs on macOS.
if [[ "$(uname -s)" == "Darwin" && "${USE_CAFFEINATE:-1}" == "1" && "${_C_GS_CAFFEINATED:-0}" != "1" ]]; then
    export _C_GS_CAFFEINATED=1
    exec caffeinate -dimsu "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KERBEROS="${KERBEROS:-cs1230216}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/public_datasets}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/artifacts/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/artifacts/outputs}"
LOG_BASE_DIR="${LOG_BASE_DIR:-$SCRIPT_DIR/artifacts/logs}"

GRID_TIME="${GRID_TIME:-900}"
FULL_TIME="${FULL_TIME:-7200}"
PATIENCE_SEC="${PATIENCE_SEC:-300}"
MIN_IMPROVE="${MIN_IMPROVE:-0.0}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-0}"
TERMINAL_LOG_MODE="${TERMINAL_LOG_MODE:-summary}"
K="${K:-50}"
SEED="${SEED:-42}"
FORCE_CPU="${FORCE_CPU:-0}"

HIDDEN_GRID="${HIDDEN_GRID:-128,256,384}"
NUM_LAYERS_GRID="${NUM_LAYERS_GRID:-2,3,4}"
LR_GRID="${LR_GRID:-1e-3,5e-4}"
DROPOUT_GRID="${DROPOUT_GRID:-0.2,0.3}"
WEIGHT_DECAY_GRID="${WEIGHT_DECAY_GRID:-0.0}"
NEG_MODE_GRID="${NEG_MODE_GRID:-mixed}"

if [[ -n "${PY:-}" ]]; then
    PY_BIN="$PY"
elif [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    PY_BIN="$SCRIPT_DIR/venv/bin/python"
else
    PY_BIN="python3"
fi

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR" "$LOG_BASE_DIR"

if ! "$PY_BIN" -c "import torch, torch_geometric, sklearn" >/dev/null 2>&1; then
    echo "ERROR: Python env '$PY_BIN' does not have torch + torch_geometric + sklearn." >&2
    echo "Hint: activate your venv first (source venv/bin/activate) or set PY=/path/to/python." >&2
    exit 1
fi

IFS=',' read -r -a HIDDEN_VALUES <<< "$HIDDEN_GRID"
IFS=',' read -r -a LAYER_VALUES <<< "$NUM_LAYERS_GRID"
IFS=',' read -r -a LR_VALUES <<< "$LR_GRID"
IFS=',' read -r -a DROPOUT_VALUES <<< "$DROPOUT_GRID"
IFS=',' read -r -a WD_VALUES <<< "$WEIGHT_DECAY_GRID"
IFS=',' read -r -a NEG_VALUES <<< "$NEG_MODE_GRID"

TOTAL=0
for HIDDEN in "${HIDDEN_VALUES[@]}"; do
    for LAYERS in "${LAYER_VALUES[@]}"; do
        for LR in "${LR_VALUES[@]}"; do
            for DROPOUT in "${DROPOUT_VALUES[@]}"; do
                for WEIGHT_DECAY in "${WD_VALUES[@]}"; do
                    for NEG_MODE in "${NEG_VALUES[@]}"; do
                        TOTAL=$((TOTAL + 1))
                    done
                done
            done
        done
    done
done

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: Empty grid. Check *_GRID environment variables." >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_BASE_DIR/grid_C_macos_${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"

RESULTS_CSV="$RUN_LOG_DIR/grid_results.csv"
SORTED_CSV="$RUN_LOG_DIR/grid_results_sorted.csv"
SUMMARY_TXT="$RUN_LOG_DIR/summary.txt"
BEST_CFG_TXT="$RUN_LOG_DIR/best_config.txt"

echo "idx,hidden,num_layers,lr,dropout,weight_decay,neg_mode,patience_sec,min_improve,hits50,exit_code,log_file" > "$RESULTS_CSV"

extract_hits50() {
    local log_file="$1"

    local best_line_hits
    best_line_hits="$(grep -Eo 'best_val_hits@50=[0-9]+\.?[0-9]*@' "$log_file" 2>/dev/null | tail -n1 | sed -E 's/.*=([0-9]+\.?[0-9]*)@/\1/' || true)"
    if [[ -n "$best_line_hits" ]]; then
        printf "%s" "$best_line_hits"
        return 0
    fi

    local max_hits
    max_hits="$(grep -Eo 'Hits@50=[0-9]+\.?[0-9]*' "$log_file" 2>/dev/null | sed 's/Hits@50=//' | awk 'BEGIN{m=-1}{if($1+0>m+0)m=$1}END{if(m>=0)print m}')"
    if [[ -n "$max_hits" ]]; then
        printf "%s" "$max_hits"
        return 0
    fi

    return 1
}

extract_stop_reason() {
    local log_file="$1"
    grep -Eo 'stop_reason=[^ ]+' "$log_file" 2>/dev/null | tail -n1 | cut -d= -f2
}

is_strictly_better() {
    local a="$1"
    local b="$2"
    "$PY_BIN" - "$a" "$b" <<'PY'
import sys
try:
    a=float(sys.argv[1]); b=float(sys.argv[2])
except Exception:
    print(0)
    raise SystemExit(0)
print(1 if a > b else 0)
PY
}

BEST_HITS="-1"
BEST_HIDDEN=""
BEST_LAYERS=""
BEST_LR=""
BEST_DROPOUT=""
BEST_WD=""
BEST_NEG=""

EST_MAX_SEC=$((TOTAL * GRID_TIME + FULL_TIME))
EST_MAX_HRS=$("$PY_BIN" - <<PY
print(round($EST_MAX_SEC / 3600.0, 2))
PY
)

echo "=========================================================="
echo " Dataset C Cartesian grid search (macOS)"
echo "=========================================================="
echo " python        : $PY_BIN"
echo " kerberos      : $KERBEROS"
echo " data_dir      : $DATA_DIR"
echo " model_dir     : $MODEL_DIR"
echo " output_dir    : $OUTPUT_DIR"
echo " run_log_dir   : $RUN_LOG_DIR"
echo " grid sizes    : hidden=${#HIDDEN_VALUES[@]} layers=${#LAYER_VALUES[@]} lr=${#LR_VALUES[@]} dropout=${#DROPOUT_VALUES[@]} wd=${#WD_VALUES[@]} neg=${#NEG_VALUES[@]}"
echo " total configs : $TOTAL"
echo " grid_time     : ${GRID_TIME}s per config"
echo " patience      : ${PATIENCE_SEC}s (min_improve=${MIN_IMPROVE})"
echo " full_time     : ${FULL_TIME}s final run"
echo " est max time  : ${EST_MAX_HRS}h (upper bound; patience can reduce this)"
echo " eval_every    : $EVAL_EVERY"
echo " log_every     : $LOG_EVERY"
echo " term_log_mode : $TERMINAL_LOG_MODE"
echo " k             : $K"
echo " seed          : $SEED"
echo " force_cpu     : $FORCE_CPU"
echo "=========================================================="

IDX=0
for HIDDEN in "${HIDDEN_VALUES[@]}"; do
    for LAYERS in "${LAYER_VALUES[@]}"; do
        for LR in "${LR_VALUES[@]}"; do
            for DROPOUT in "${DROPOUT_VALUES[@]}"; do
                for WEIGHT_DECAY in "${WD_VALUES[@]}"; do
                    for NEG_MODE in "${NEG_VALUES[@]}"; do
                        IDX=$((IDX + 1))
                        LOG_FILE="$RUN_LOG_DIR/gs_C_$(printf '%03d' "$IDX")_h${HIDDEN}_l${LAYERS}_lr${LR}_d${DROPOUT}_wd${WEIGHT_DECAY}_${NEG_MODE}.log"

                        echo
                        echo "[$IDX/$TOTAL] hidden=$HIDDEN layers=$LAYERS lr=$LR dropout=$DROPOUT wd=$WEIGHT_DECAY neg=$NEG_MODE"

                        EXTRA_ARGS=()
                        if [[ "$FORCE_CPU" == "1" ]]; then
                            EXTRA_ARGS+=("--force_cpu")
                        fi

                        set +e
                        if [[ "$TERMINAL_LOG_MODE" == "full" ]]; then
                            "$PY_BIN" "$SCRIPT_DIR/train_C.py" \
                                --data_dir "$DATA_DIR" \
                                --model_dir "$MODEL_DIR" \
                                --kerberos "${KERBEROS}_gs_tmp" \
                                --time_limit "$GRID_TIME" \
                                --hidden "$HIDDEN" \
                                --num_layers "$LAYERS" \
                                --dropout "$DROPOUT" \
                                --lr "$LR" \
                                --weight_decay "$WEIGHT_DECAY" \
                                --neg_mode "$NEG_MODE" \
                                --eval_every "$EVAL_EVERY" \
                                --log_every "$LOG_EVERY" \
                                --k "$K" \
                                --seed "$SEED" \
                                --patience_sec "$PATIENCE_SEC" \
                                --min_improve "$MIN_IMPROVE" \
                                "${EXTRA_ARGS[@]}" \
                                2>&1 | tee "$LOG_FILE"
                            EXIT_CODE=${PIPESTATUS[0]}
                        else
                            "$PY_BIN" "$SCRIPT_DIR/train_C.py" \
                                --data_dir "$DATA_DIR" \
                                --model_dir "$MODEL_DIR" \
                                --kerberos "${KERBEROS}_gs_tmp" \
                                --time_limit "$GRID_TIME" \
                                --hidden "$HIDDEN" \
                                --num_layers "$LAYERS" \
                                --dropout "$DROPOUT" \
                                --lr "$LR" \
                                --weight_decay "$WEIGHT_DECAY" \
                                --neg_mode "$NEG_MODE" \
                                --eval_every "$EVAL_EVERY" \
                                --log_every "$LOG_EVERY" \
                                --k "$K" \
                                --seed "$SEED" \
                                --patience_sec "$PATIENCE_SEC" \
                                --min_improve "$MIN_IMPROVE" \
                                "${EXTRA_ARGS[@]}" \
                                > "$LOG_FILE" 2>&1
                            EXIT_CODE=$?
                        fi
                        set -e

                        HITS=""
                        if extract_hits50 "$LOG_FILE" >/dev/null 2>&1; then
                            HITS="$(extract_hits50 "$LOG_FILE")"
                        fi
                        if [[ -z "$HITS" ]]; then
                            HITS="-1"
                        fi

                        echo "$IDX,$HIDDEN,$LAYERS,$LR,$DROPOUT,$WEIGHT_DECAY,$NEG_MODE,$PATIENCE_SEC,$MIN_IMPROVE,$HITS,$EXIT_CODE,$LOG_FILE" >> "$RESULTS_CSV"

                        if [[ "$EXIT_CODE" -eq 0 && "$HITS" != "-1" ]]; then
                            STOP_REASON="$(extract_stop_reason "$LOG_FILE" || true)"
                            [[ -z "$STOP_REASON" ]] && STOP_REASON="unknown"
                            BETTER="$(is_strictly_better "$HITS" "$BEST_HITS" | tail -n1 | tr -d '[:space:]')"
                            if [[ "$BETTER" == "1" ]]; then
                                BEST_HITS="$HITS"
                                BEST_HIDDEN="$HIDDEN"
                                BEST_LAYERS="$LAYERS"
                                BEST_LR="$LR"
                                BEST_DROPOUT="$DROPOUT"
                                BEST_WD="$WEIGHT_DECAY"
                                BEST_NEG="$NEG_MODE"
                                echo "    -> Hits@50=$HITS stop=$STOP_REASON [new best]"
                            else
                                echo "    -> Hits@50=$HITS stop=$STOP_REASON"
                            fi
                        else
                            echo "    -> run failed or metric missing (exit_code=$EXIT_CODE, hits=$HITS)"
                        fi
                    done
                done
            done
        done
    done
done

{
    head -n1 "$RESULTS_CSV"
    tail -n +2 "$RESULTS_CSV" | sort -t, -k10,10gr
} > "$SORTED_CSV"

if [[ -z "$BEST_HIDDEN" ]]; then
    echo "No successful grid configuration found. Check logs in: $RUN_LOG_DIR" >&2
    exit 1
fi

BEST_CFG="${BEST_HIDDEN}:${BEST_LAYERS}:${BEST_LR}:${BEST_DROPOUT}:${BEST_WD}:${BEST_NEG}"

cat > "$BEST_CFG_TXT" <<EOT
hidden=$BEST_HIDDEN
num_layers=$BEST_LAYERS
lr=$BEST_LR
dropout=$BEST_DROPOUT
weight_decay=$BEST_WD
neg_mode=$BEST_NEG
best_hits50=$BEST_HITS
EOT

cat > "$SUMMARY_TXT" <<EOT
Dataset C macOS grid search summary
run_id=$RUN_ID
python=$PY_BIN
best_hits50=$BEST_HITS
best_cfg=$BEST_CFG  (hidden:num_layers:lr:dropout:weight_decay:neg_mode)
results_csv=$RESULTS_CSV
results_sorted_csv=$SORTED_CSV
best_config_file=$BEST_CFG_TXT
full_run_log=$RUN_LOG_DIR/final_full_run.log
EOT

echo
echo "=========================================================="
echo " Grid search complete"
echo " Best Hits@50 : $BEST_HITS"
echo " Best cfg     : $BEST_CFG"
echo " Sorted table : $SORTED_CSV"
echo "=========================================================="

FULL_LOG="$RUN_LOG_DIR/final_full_run.log"

echo
echo "=========================================================="
echo " Final run with best config (${FULL_TIME}s)"
echo "=========================================================="

set +e
if [[ "$TERMINAL_LOG_MODE" == "full" ]]; then
    PY="$PY_BIN" \
    KERBEROS="$KERBEROS" \
    DATA_DIR="$DATA_DIR" \
    MODEL_DIR="$MODEL_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    HIDDEN="$BEST_HIDDEN" \
    NUM_LAYERS="$BEST_LAYERS" \
    LR="$BEST_LR" \
    DROPOUT="$BEST_DROPOUT" \
    WEIGHT_DECAY="$BEST_WD" \
    NEG_MODE="$BEST_NEG" \
    EVAL_EVERY="$EVAL_EVERY" \
    LOG_EVERY="$LOG_EVERY" \
    K="$K" \
    SEED="$SEED" \
    FORCE_CPU="$FORCE_CPU" \
    "$SCRIPT_DIR/run_train_C.sh" "$FULL_TIME" 2>&1 | tee "$FULL_LOG"
    FINAL_EXIT=${PIPESTATUS[0]}
else
    PY="$PY_BIN" \
    KERBEROS="$KERBEROS" \
    DATA_DIR="$DATA_DIR" \
    MODEL_DIR="$MODEL_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    HIDDEN="$BEST_HIDDEN" \
    NUM_LAYERS="$BEST_LAYERS" \
    LR="$BEST_LR" \
    DROPOUT="$BEST_DROPOUT" \
    WEIGHT_DECAY="$BEST_WD" \
    NEG_MODE="$BEST_NEG" \
    EVAL_EVERY="$EVAL_EVERY" \
    LOG_EVERY="$LOG_EVERY" \
    K="$K" \
    SEED="$SEED" \
    FORCE_CPU="$FORCE_CPU" \
    "$SCRIPT_DIR/run_train_C.sh" "$FULL_TIME" > "$FULL_LOG" 2>&1
    FINAL_EXIT=$?
fi
set -e

if [[ "$FINAL_EXIT" -ne 0 ]]; then
    echo "Final full run failed with exit code $FINAL_EXIT. See: $FULL_LOG" >&2
    exit "$FINAL_EXIT"
fi

FINAL_HITS_LINE="$(grep -E 'Hits@50' "$FULL_LOG" | tail -n1 || true)"
if [[ -n "$FINAL_HITS_LINE" ]]; then
    echo " Final eval    : $FINAL_HITS_LINE"
fi

echo
echo "[done] Best config file : $BEST_CFG_TXT"
echo "[done] Grid summary     : $SUMMARY_TXT"
echo "[done] Grid logs        : $RUN_LOG_DIR"
echo "[done] Final model      : $MODEL_DIR/${KERBEROS}_model_C.pt"
echo "[done] Final predictions: $OUTPUT_DIR/${KERBEROS}_predictions_C.pt"
