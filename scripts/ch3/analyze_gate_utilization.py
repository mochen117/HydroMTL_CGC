# ==============================================================================
# Description:
#   Parse routing diagnostics from Chapter 3 model logs.
#
# Inputs:
#   - experiments/formal_ch3_modeling/logs/ch3_mmoe_mtl_seed42.log
#   - experiments/formal_ch3_modeling/logs/ch3_cgc_mtl_seed42.log
#
# Outputs:
#   - ch3_gate_utilization_long.csv
#   - ch3_gate_utilization_summary.csv
# ==============================================================================

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
LOG_DIR = CH3_DIR / "logs"
SUMMARY_DIR = CH3_DIR / "06_summary"

OUTPUT_LONG = SUMMARY_DIR / "ch3_gate_utilization_long.csv"
OUTPUT_SUMMARY = SUMMARY_DIR / "ch3_gate_utilization_summary.csv"

LOG_FILES: Dict[str, Path] = {
    "MMoE": LOG_DIR / "ch3_mmoe_mtl_seed42.log",
    "CGC": LOG_DIR / "ch3_cgc_mtl_seed42.log",
}

EPOCH_PATTERN = re.compile(r"\[Routing Diagnostics \| Epoch\s+(\d+)\]")
GATE_PATTERN = re.compile(r"^\s*([A-Za-z0-9_]+)\s+\|\s+H=\s*([-+]?\d*\.?\d+|nan)\s+\|\s+Util=\[(.*)\]")


def parse_utilization(raw: str) -> List[float]:
    """Parse utilization vector from log text."""
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_log(model_name: str, path: Path) -> pd.DataFrame:
    """Parse one routing diagnostic log."""
    if not path.exists():
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    current_epoch: Optional[int] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            gate_match = GATE_PATTERN.search(line)
            if not gate_match or current_epoch is None:
                continue

            gate_name = gate_match.group(1)
            entropy = float(gate_match.group(2)) if gate_match.group(2) != "nan" else np.nan
            utilization = parse_utilization(gate_match.group(3))

            for expert_idx, util_value in enumerate(utilization):
                records.append(
                    {
                        "model": model_name,
                        "epoch": current_epoch,
                        "gate_name": gate_name,
                        "expert_id": expert_idx,
                        "utilization": util_value,
                        "entropy": entropy,
                    }
                )

    return pd.DataFrame(records)


def summarize_gate_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean utilization and entropy by model, gate, and expert."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["model", "gate_name", "expert_id"], as_index=False)
        .agg(
            mean_utilization=("utilization", "mean"),
            median_utilization=("utilization", "median"),
            max_utilization=("utilization", "max"),
            mean_entropy=("entropy", "mean"),
            n_epochs=("epoch", "nunique"),
        )
        .sort_values(["model", "gate_name", "mean_utilization"], ascending=[True, True, False])
    )

    return summary


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    tables = []
    for model_name, log_path in LOG_FILES.items():
        table = parse_log(model_name, log_path)
        if not table.empty:
            tables.append(table)

    if not tables:
        raise FileNotFoundError("No routing diagnostics were parsed from model logs.")

    long_df = pd.concat(tables, ignore_index=True)
    summary_df = summarize_gate_utilization(long_df)

    long_df.to_csv(OUTPUT_LONG, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"Saved: {OUTPUT_LONG}")
    print(f"Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()