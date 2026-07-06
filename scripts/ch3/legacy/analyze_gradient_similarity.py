# ==============================================================================
# Description:
#   Analyze encoder gradient similarity for Chapter 3 multi-task models.
#
# Inputs:
#   - training_history.csv from Hard-MTL, MMoE, and CGC runs
#
# Outputs:
#   - ch3_gradient_similarity_long.csv
#   - ch3_gradient_similarity_summary.csv
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"

OUTPUT_LONG = SUMMARY_DIR / "ch3_gradient_similarity_long.csv"
OUTPUT_SUMMARY = SUMMARY_DIR / "ch3_gradient_similarity_summary.csv"

MODEL_DIRS: Dict[str, Path] = {
    "Hard-MTL": CH3_DIR / "03_hard_mtl" / "ch3_hard_mtl_seed42",
    "MMoE": CH3_DIR / "04_mmoe_mtl" / "ch3_mmoe_mtl_seed42",
    "CGC": CH3_DIR / "05_cgc_mtl" / "ch3_cgc_mtl_seed42",
}


def find_grad_column(df: pd.DataFrame) -> str:
    """Find encoder gradient similarity column."""
    candidates = ["encoder_grad_sim", "Encoder_Grad_Sim", "Encoder"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("No encoder gradient similarity column found.")


def read_history(model_name: str, model_dir: Path) -> pd.DataFrame:
    """Read one model training history."""
    path = model_dir / "training_history.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    grad_col = find_grad_column(df)

    out = pd.DataFrame(
        {
            "model": model_name,
            "epoch": df["epoch"],
            "encoder_grad_sim": pd.to_numeric(df[grad_col], errors="coerce"),
        }
    )

    for col in ["train_loss", "val_loss", "streamflow_nse_median", "evapotranspiration_nse_median"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    return out


def summarize_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize gradient similarity by model."""
    records: List[Dict[str, object]] = []

    for model_name, group in df.groupby("model"):
        sim = group["encoder_grad_sim"].dropna()

        records.append(
            {
                "model": model_name,
                "n_epochs_with_grad_sim": int(len(sim)),
                "mean_grad_sim": float(sim.mean()) if len(sim) else np.nan,
                "median_grad_sim": float(sim.median()) if len(sim) else np.nan,
                "min_grad_sim": float(sim.min()) if len(sim) else np.nan,
                "max_grad_sim": float(sim.max()) if len(sim) else np.nan,
                "positive_grad_sim_rate_pct": float((sim > 0).mean() * 100.0) if len(sim) else np.nan,
                "negative_grad_sim_rate_pct": float((sim < 0).mean() * 100.0) if len(sim) else np.nan,
                "near_zero_grad_sim_rate_pct_abs_lt_0.01": float((sim.abs() < 0.01).mean() * 100.0) if len(sim) else np.nan,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    tables = []
    for model_name, model_dir in MODEL_DIRS.items():
        table = read_history(model_name, model_dir)
        if not table.empty:
            tables.append(table)

    if not tables:
        raise FileNotFoundError("No training_history.csv files were found for multi-task models.")

    long_df = pd.concat(tables, ignore_index=True)
    summary_df = summarize_gradient(long_df)

    long_df.to_csv(OUTPUT_LONG, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"Saved: {OUTPUT_LONG}")
    print(f"Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()