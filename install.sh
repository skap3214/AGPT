#!/bin/bash

# Exit script on any error
set -e

# Specify CUDA version for PyTorch
CUDA_VERSION="cu118"

# Install PyTorch with the specific CUDA version
echo "Installing PyTorch with CUDA $CUDA_VERSION..."
pip install torch  --extra-index-url https://download.pytorch.org/whl/$CUDA_VERSION

# Install other dependencies from requirements.txt
echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installation completed successfully."