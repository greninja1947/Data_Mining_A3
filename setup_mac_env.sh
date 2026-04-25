#!/usr/bin/env bash
# One-time setup: installs a native arm64 Python via Homebrew and creates a
# clean venv with Mac-compatible packages (CPU/MPS torch, no CUDA).
#
# Usage:
#   ./setup_mac_env.sh
#   source venv/bin/activate
#   ./run_train_B.sh 60

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARM_BREW=/opt/homebrew/bin/brew

# ---- 1. Verify arm64 Homebrew -----------------------------------------------
if [[ ! -x "$ARM_BREW" ]]; then
    echo "ERROR: arm64 Homebrew not found at $ARM_BREW"
    echo "Install it with:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

ARM_BREW_PREFIX="$($ARM_BREW --prefix)"
echo "[step 1] arm64 Homebrew at $ARM_BREW  (prefix=$ARM_BREW_PREFIX)"

# ---- 2. Install Python 3.12 via arm64 Homebrew ------------------------------
if [[ ! -x "$ARM_BREW_PREFIX/bin/python3.12" ]]; then
    echo "[step 2] Installing python@3.12 via arm64 Homebrew ..."
    arch -arm64 "$ARM_BREW" install python@3.12
else
    echo "[step 2] python3.12 already installed at $ARM_BREW_PREFIX/bin/python3.12"
fi
PY="$ARM_BREW_PREFIX/bin/python3.12"

# Verify it's arm64
ARCH=$("$PY" -c "import platform; print(platform.machine())")
echo "[check ] python arch = $ARCH"
if [[ "$ARCH" != "arm64" ]]; then
    echo "ERROR: expected arm64 Python, got $ARCH" >&2
    exit 1
fi

# ---- 3. Create venv ---------------------------------------------------------
VENV="$SCRIPT_DIR/venv"
if [[ -d "$VENV" ]]; then
    echo "[step 3] venv already exists at $VENV — skipping creation"
else
    echo "[step 3] Creating venv at $VENV ..."
    arch -arm64 "$PY" -m venv "$VENV"
fi

source "$VENV/bin/activate"
echo "[step 3] activated venv: $(which python3)  arch=$(arch -arm64 python3 -c 'import platform; print(platform.machine())')"

# ---- 4. Install Mac packages ------------------------------------------------
echo "[step 4] Installing Mac-compatible packages (arm64, CPU/MPS torch) ..."
arch -arm64 pip install --upgrade pip --quiet
arch -arm64 pip install -r "$SCRIPT_DIR/requirements_mac.txt"

echo ""
echo "=========================================================="
echo " Setup complete."
echo " Activate the venv before running training:"
echo "   source venv/bin/activate"
echo "   ./run_train_B.sh 60     # smoke test"
echo "   ./run_train_C.sh 60     # link prediction smoke test"
echo "=========================================================="
