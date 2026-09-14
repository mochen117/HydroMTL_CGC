#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "============================================================"
echo "Chapter 3 independent-test reproducibility pipeline"
echo "============================================================"
echo "Project root : ${PROJECT_ROOT}"
echo "Python       : $(command -v python)"
echo "Conda env    : ${CONDA_DEFAULT_ENV:-NOT_ACTIVE}"
echo

if [[ "${CONDA_DEFAULT_ENV:-}" != "MTL_CGC" ]]; then
    echo "[WARNING] Recommended environment: MTL_CGC"
    echo
fi


# ============================================================
# 0. Required external data
# ============================================================

DATA_ROOT="${PROJECT_ROOT}/output_592_basins"
METADATA="${DATA_ROOT}/basin_metadata.csv"

if [[ ! -e "${DATA_ROOT}" ]]; then
    echo "[ERROR] Missing prepared data directory:"
    echo "        ${DATA_ROOT}"
    echo
    echo "If the data are stored elsewhere, create a directory-level"
    echo "symbolic link, for example:"
    echo
    echo "ln -s /path/to/output_592_basins ${DATA_ROOT}"
    exit 1
fi

if [[ ! -f "${METADATA}" ]]; then
    echo "[ERROR] Missing basin metadata:"
    echo "        ${METADATA}"
    exit 1
fi

echo "[PASS] Prepared data root found"
echo "[PASS] Basin metadata found"


# ============================================================
# 1. Formal checkpoints
# ============================================================

MODEL_DIRS=(
    "experiments/formal_ch3_modeling/01_stl_q/ch3_stl_q_seed42"
    "experiments/formal_ch3_modeling/02_stl_et/ch3_stl_et_seed42"
    "experiments/formal_ch3_modeling/03_hard_mtl/ch3_hard_mtl_seed42"
    "experiments/formal_ch3_modeling/04_mmoe_mtl/ch3_mmoe_mtl_seed42"
    "experiments/formal_ch3_modeling/05_cgc_mtl/ch3_cgc_mtl_seed42"
)

echo
echo "Checking formal checkpoints..."

for model_dir in "${MODEL_DIRS[@]}"; do
    checkpoint="${PROJECT_ROOT}/${model_dir}/best_model.pth"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "[ERROR] Missing formal checkpoint:"
        echo "        ${checkpoint}"
        exit 1
    fi

    echo "[PASS] ${model_dir}/best_model.pth"
done


run_step() {
    local number="$1"
    local title="$2"
    shift 2

    echo
    echo "============================================================"
    echo "${number}: ${title}"
    echo "============================================================"

    "$@"
}


# ============================================================
# 2. Independent TEST
# ============================================================

run_step \
    "1/9" \
    "Independent-test checkpoint audit/evaluation" \
    python scripts/ch3/run_ch3_independent_test.py


# ============================================================
# 3. TEST aggregation
# ============================================================

run_step \
    "2/9" \
    "Build formal TEST summary" \
    python scripts/ch3/summarize_ch3_test_results.py


# ============================================================
# 4. Positive / negative transfer
# ============================================================

run_step \
    "3/9" \
    "Analyze positive/negative transfer" \
    python scripts/ch3/analyze_transfer.py


# ============================================================
# 5. Formal TEST audit
# ============================================================

run_step \
    "4/9" \
    "Audit Chapter 3 TEST results" \
    python scripts/ch3/audit_ch3_results.py


# ============================================================
# 6. Extreme NSE diagnostics
# ============================================================

run_step \
    "5/9" \
    "Diagnose extreme NSE values" \
    python scripts/ch3/diagnose_ch3_nse_outliers.py


# ============================================================
# 7. Sensitivity analysis
# ============================================================

run_step \
    "6/9" \
    "Run sensitivity analysis" \
    python scripts/ch3/sensitivity_analysis_ch3.py


# ============================================================
# 8. Basin metadata
# ============================================================

run_step \
    "7/9" \
    "Merge basin metadata" \
    python scripts/ch3/merge_basin_metadata.py


# ============================================================
# 9. Spatial diagnostics
# ============================================================

run_step \
    "8/9" \
    "Build spatial diagnostics" \
    python scripts/ch3/plot_ch3_spatial_maps.py


# ============================================================
# 10. Gate diagnostics
# ============================================================

SUMMARY_ROOT="${PROJECT_ROOT}/experiments/formal_ch3_modeling/06_summary"

GATE_LONG="${SUMMARY_ROOT}/ch3_gate_utilization_long.csv"
GATE_SUMMARY="${SUMMARY_ROOT}/ch3_gate_utilization_summary.csv"

if [[ -f "${GATE_LONG}" && -f "${GATE_SUMMARY}" ]]; then
    echo
    echo "[PASS] Gate-utilization diagnostics found"
else
    echo
    echo "[INFO] Gate-utilization tables are missing."
    echo "[INFO] Attempting reconstruction from training diagnostics."

    if python scripts/ch3/analyze_gate_utilization.py; then
        echo "[PASS] Gate-utilization diagnostics reconstructed"
    else
        echo "[WARNING] Gate diagnostics could not be reconstructed."
        echo "[WARNING] Gate-specialization figure may be skipped."
    fi
fi


# ============================================================
# 11. Publication figures
# ============================================================

run_step \
    "9/9" \
    "Generate publication figures" \
    python scripts/ch3/plot_ch3_publication_adaptive_focus_v5.py


# ============================================================
# 12. Final audit
# ============================================================

echo
echo "============================================================"
echo "Final consistency audit"
echo "============================================================"

python scripts/ch3/audit_ch3_results.py


# ============================================================
# 13. Compact reproducibility validation
# ============================================================

python - <<'PY'
from pathlib import Path

import pandas as pd

root = Path("experiments/formal_ch3_modeling")
summary = root / "06_summary" / "test"

per_basin = summary / "ch3_per_basin_all_models.csv"
metadata = summary / "ch3_per_basin_with_metadata.csv"
manifest = root / "ch3_independent_test_manifest.json"

required = [
    per_basin,
    metadata,
    manifest,
    summary / "ch3_test_performance_summary.csv",
    summary / "ch3_test_transfer_summary.csv",
    summary / "ch3_result_audit_report.txt",
]

missing = [p for p in required if not p.exists()]

if missing:
    message = "\n".join(f"  {p}" for p in missing)
    raise SystemExit(
        "Missing required Chapter 3 outputs:\n" + message
    )

df = pd.read_csv(per_basin)

if "gauge_id" not in df.columns:
    raise SystemExit(
        "gauge_id is missing from the formal per-basin table."
    )

n_basins = df["gauge_id"].astype(str).nunique()

if n_basins != 592:
    raise SystemExit(
        f"Expected 592 independent-test basins, found {n_basins}."
    )

delta_cols = [
    c for c in df.columns
    if c.startswith("Delta_NSE")
]

if len(delta_cols) != 10:
    raise SystemExit(
        f"Expected 10 Delta_NSE columns, found {len(delta_cols)}."
    )

meta = pd.read_csv(metadata)

if len(meta) != 592:
    raise SystemExit(
        f"Expected 592 metadata rows, found {len(meta)}."
    )

print()
print("Reproducibility validation")
print("--------------------------")
print(f"Independent-test basins : {n_basins}")
print(f"Delta_NSE columns       : {len(delta_cols)}")
print(f"Metadata rows           : {len(meta)}")
print(f"Formal master table     : {per_basin}")
print(f"Test manifest           : {manifest}")
print()
print("Chapter 3 reproducibility: PASS")
PY


echo
echo "============================================================"
echo "Chapter 3 reproducibility pipeline completed successfully."
echo "============================================================"
