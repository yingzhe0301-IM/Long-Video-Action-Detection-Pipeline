#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

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
    echo "Please install PyTorch manually, then re-run this script or run 'pip install -r requirements.txt' manually."
    exit 1
fi

echo "PyTorch installed successfully."
echo ""
echo "=== Installing remaining dependencies... ==="

# 2. Install all other packages from requirements.txt
pip install -r requirements.txt

echo ""
echo "=== All dependencies have been installed successfully! ==="