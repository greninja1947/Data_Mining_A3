#!/usr/bin/env bash
# env_detect.sh — sourced by PBS scripts at runtime to find a working Python.
#
# Strategy (tries in order, stops at first success):
#   1. User conda env ~/col761          (created by setup_hpc_env.sh if internet available)
#   2. Any module that provides torch    (module avail scan)
#   3. Base anaconda env                 (module load apps/anaconda/3)
#   4. System python3 fallback
#
# After sourcing, $PY is set to a python binary that has torch + torch_geometric.
# Exits with error if no working Python is found.

_require_torch() {
    local py="$1"
    "$py" -c "import torch, torch_geometric, sklearn" 2>/dev/null
}

_try_set_py() {
    local py="$1"
    if [[ -x "$py" ]] && _require_torch "$py"; then
        PY="$py"
        return 0
    fi
    return 1
}

echo "[env] detecting Python environment ..."

# ── 1. User conda env ──────────────────────────────────────────────────────
if _try_set_py "$HOME/col761/bin/python"; then
    echo "[env] using ~/col761 conda env"

# ── 2. Scan modules for pytorch ───────────────────────────────────────────
elif module avail 2>&1 | grep -qi "pytorch\|torch"; then
    # Try the most common module name patterns
    for MOD in \
        "apps/pytorch/2.1.0/gpu" \
        "apps/pytorch/2.0.0/gpu" \
        "apps/pytorch/1.13.0/gpu" \
        "apps/pytorch/gpu" \
        "pytorch/2.1.0" \
        "pytorch/2.0.0" \
        "pytorch" \
    ; do
        if module load "$MOD" 2>/dev/null; then
            CANDIDATE=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
            if _try_set_py "$CANDIDATE"; then
                echo "[env] loaded module $MOD  ->  $PY"
                break
            fi
            module unload "$MOD" 2>/dev/null || true
        fi
    done
fi

# ── 3. Base anaconda env (check if torch is pre-installed) ─────────────────
if [[ -z "${PY:-}" ]]; then
    module load apps/anaconda/3 2>/dev/null || true
    CANDIDATE=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
    if _try_set_py "$CANDIDATE"; then
        echo "[env] using base anaconda env  ->  $PY"
    fi
fi

# ── 4. Scan conda envs listed in ~/.conda/environments.txt ────────────────
if [[ -z "${PY:-}" ]] && [[ -f "$HOME/.conda/environments.txt" ]]; then
    while IFS= read -r ENVDIR; do
        [[ -z "$ENVDIR" || "$ENVDIR" == "#"* ]] && continue
        if _try_set_py "$ENVDIR/bin/python"; then
            echo "[env] found working env at $ENVDIR"
            break
        fi
    done < "$HOME/.conda/environments.txt"
fi

# ── 5. System fallback ─────────────────────────────────────────────────────
if [[ -z "${PY:-}" ]]; then
    for CANDIDATE in python3 python; do
        if _try_set_py "$(which $CANDIDATE 2>/dev/null)"; then
            echo "[env] using system $CANDIDATE"
            break
        fi
    done
fi

# ── Fail loudly if nothing worked ─────────────────────────────────────────
if [[ -z "${PY:-}" ]]; then
    echo "[env] ERROR: could not find Python with torch + torch_geometric + sklearn"
    echo "[env] Available modules:"
    module avail 2>&1 | head -40
    echo "[env] Conda envs:"
    conda env list 2>/dev/null || true
    exit 1
fi

# ── Load CUDA module if torch says CUDA is not available ──────────────────
CUDA_OK=$("$PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
if [[ "$CUDA_OK" != "1" ]]; then
    echo "[env] torch.cuda not available yet — scanning for CUDA module ..."
    for MOD in \
        "compiler/cuda/12.1/compilervars" \
        "compiler/cuda/11.8/compilervars" \
        "compiler/cuda/11.7/compilervars" \
        "compiler/cuda/11.3/compilervars" \
        "compiler/cuda/10.2/compilervars" \
        "cuda/12.1" "cuda/11.8" "cuda/11.7" "cuda" \
    ; do
        if module load "$MOD" 2>/dev/null; then
            CUDA_OK=$("$PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
            if [[ "$CUDA_OK" == "1" ]]; then
                echo "[env] CUDA enabled via module $MOD"
                break
            fi
            module unload "$MOD" 2>/dev/null || true
        fi
    done
fi

# ── Print summary ──────────────────────────────────────────────────────────
echo "[env] ── environment summary ─────────────────────────────────────"
echo "[env] PY           = $PY"
echo "[env] torch        = $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "[env] torch_geo    = $("$PY" -c 'import torch_geometric; print(torch_geometric.__version__)' 2>/dev/null)"
echo "[env] cuda_avail   = $("$PY" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "[env] cuda_version = $("$PY" -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
echo "[env] gpu_name     = $("$PY" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' 2>/dev/null)"
echo "[env] ──────────────────────────────────────────────────────────────"
