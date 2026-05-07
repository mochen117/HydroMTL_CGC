#!/bin/bash
# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Automated shell script for exploring the Pareto Front in MTL.
# Implements strict working directory binding and C++ library path injection 
# to ensure stable NetCDF (.nc) extractions.
# ==============================================================================

# Exit immediately if any command fails
set -e

# 1. Dynamically resolve project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 2. [CRITICAL FIX] Inject C++ dynamic libraries for xarray/NetCDF natively
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    echo "[INFO] Injected CONDA_PREFIX/lib into LD_LIBRARY_PATH for NetCDF stability."
else
    echo "[WARNING] CONDA_PREFIX not found. NetCDF extraction might fail."
fi

# 3. [CRITICAL FIX] Change working directory to Project Root
# This ensures that relative paths in default.yaml (e.g., "./output_592_basins") resolve correctly.
cd "$PROJECT_ROOT"
echo "[INFO] Working directory securely bound to: $(pwd)"

# Define weight configurations and corresponding experiment names
WEIGHTS_Q=(1.0 1.0 1.0 1.0 1.0 0.33 0.0)
WEIGHTS_ET=(0.0 0.05 0.1 0.33 1.0 1.0 1.0)
EXP_NAMES=(
    "pareto_stl_q" 
    "pareto_q_heavy" 
    "pareto_q_focus" 
    "pareto_baseline" 
    "pareto_equal" 
    "pareto_et_focus" 
    "pareto_stl_et"
)

CONFIG_PATH="mtl_cgc/configs/default.yaml"
TOTAL_RUNS=${#WEIGHTS_Q[@]}

echo "======================================================================"
echo "[INFO] Starting Pareto Front Exploration with $TOTAL_RUNS configurations."
echo "======================================================================"

for i in "${!WEIGHTS_Q[@]}"; do
    WQ="${WEIGHTS_Q[$i]}"
    WET="${WEIGHTS_ET[$i]}"
    EXP_NAME="${EXP_NAMES[$i]}"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "[RUN $((i + 1))/$TOTAL_RUNS] Experiment: $EXP_NAME "
    echo " Weights -> Streamflow: $WQ | Evapotranspiration: $WET"
    echo "----------------------------------------------------------------------"

    # Phase 1: Training Execution
    echo "[INFO] Commencing Training Phase..."
    python main.py --config "$CONFIG_PATH" \
                   --mode train \
                   --loss_weights streamflow="$WQ" evapotranspiration="$WET" \
                   --experiment_name "$EXP_NAME"

    # Phase 2: Testing & Data Export (Generates CSV and NC files)
    echo "[INFO] Commencing Independent Testing Phase & NetCDF Export..."
    python main.py --config "$CONFIG_PATH" \
                   --mode test \
                   --loss_weights streamflow="$WQ" evapotranspiration="$WET" \
                   --experiment_name "$EXP_NAME"
                   
    echo "[SUCCESS] Experiment '$EXP_NAME' completed."
done

echo ""
echo "======================================================================"
echo "[INFO] All Pareto Front experiments executed successfully."
echo "======================================================================"