#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements/base.txt"

echo "=== Starting dependency installation ==="

# 1. Install PyTorch based on the operating system
# The 'uname' command returns the kernel name. macOS is "Darwin", Linux is "Linux".
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS. Installing PyTorch for MPS..."
    pip install torch torchvision torchaudio
elif [[ "$(uname)" == "Linux" ]]; then
    echo "Detected Linux/Ubuntu. Installing PyTorch for CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Error: Unsupported operating system '$(uname)'."
    echo "Please install PyTorch manually, then re-run this script or run 'pip install -r ${REQUIREMENTS_FILE}' manually."
    exit 1
fi

echo "PyTorch installed successfully."
echo ""
echo "=== Installing remaining dependencies... ==="

# 2. Install all other packages from requirements/base.txt
pip install -r "${REQUIREMENTS_FILE}"

echo ""
echo "=== All dependencies have been installed successfully! ==="
