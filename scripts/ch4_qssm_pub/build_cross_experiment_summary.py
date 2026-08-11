#!/usr/bin/env python3
"""Integrate frozen Chapter 3, Chapter 4A, and Chapter 4B basin-level results.

The output supports the Chapter 4 directionality / data-dependence narrative:

- Experiment A: Q -> SSM under temporal observation limitation;
- Experiment B: SSM -> Q under spatial PUB limitation.

This script is strictly post-processing. It never changes or retrains frozen
Chapter 3 / Chapter 4A models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.io_utils import normalize_basin_id  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.paths import CH3_SUMMARY, SUMMARY_DIR  # noqa: E402


DEFAULT_CH3 = CH3_SUMMARY
DEFAULT_CH4B = SUMMARY_DIR / "ch4b_pub_effects_with_ch3_metadata.csv"
DEFAULT_OUTPUT_DIR = SUMMARY_DIR

CH4A_MODELS = {
    "stl": "ch4a_formal_stl_ssm_seed42",
    "hps": "ch4a_formal_hps_qssm_seed42",
    "cgc": "ch4a_formal_cgc_qssm_seed42",
    "hps_pre": "ch4a_formal_hps_qpre_finetune_qssm_seed42",
    "cgc_pre": "ch4a_formal_cgc_qpre_finetune_qssm_seed42",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ch3-summary", type=Path, default=DEFAULT_CH3)
    parser.add_argument("--ch4b-effects", type=Path, default=DEFAULT_CH4B)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for candidate in ("gauge_id", "basin_id", "gage_id", "Unnamed: 0", frame.columns[0]):
        if candidate in frame.columns:
            frame = frame.rename(columns={candidate: "gauge_id"})
            frame["gauge_id"] = frame["gauge_id"].map(normalize_basin_id)
            if frame["gauge_id"].duplicated().any():
                raise ValueError("Duplicate basin identifiers detected.")
            return frame
    raise ValueError("Cannot identify basin-id column.")


def read_ch4a_metric(root: Path, experiment: str, label: str) -> pd.DataFrame:
    path = root / experiment / "test_per_basin_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = normalize_frame(pd.read_csv(path))
    columns = ["gauge_id"] + [
        col
        for col in frame.columns
        if col.startswith("ssm_") or col.startswith("streamflow_")
    ]
    frame = frame[columns]
    return frame.rename(
        columns={col: f"ch4a_{label}_{col}" for col in columns if col != "gauge_id"}
    )


def effect_summary(
    frame: pd.DataFrame,
    experiment: str,
    direction: str,
    data_limitation: str,
    comparison: str,
    column: str,
    hydroclimate_group: str | None = None,
) -> dict[str, object]:
    values = (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return {
        "experiment": experiment,
        "direction": direction,
        "data_limitation": data_limitation,
        "hydroclimate_group": hydroclimate_group or "All",
        "comparison": comparison,
        "n_basins": len(values),
        "median_delta_nse": values.median() if len(values) else np.nan,
        "mean_delta_nse": values.mean() if len(values) else np.nan,
        "q25_delta_nse": values.quantile(0.25) if len(values) else np.nan,
        "q75_delta_nse": values.quantile(0.75) if len(values) else np.nan,
        "positive_rate": float((values > 0).mean()) if len(values) else np.nan,
        "negative_rate": float((values < 0).mean()) if len(values) else np.nan,
    }


def main() -> None:
    args = parse_args()
    ch3_path = _resolve(args.ch3_summary)
    ch4b_path = _resolve(args.ch4b_effects)
    exp_root = _resolve(args.experiments_root)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ch3 = normalize_frame(pd.read_csv(ch3_path))
    ch4b = normalize_frame(pd.read_csv(ch4b_path))

    # Keep all Chapter 4B effects and merge only Chapter 3 fields not already
    # attached by summarize_pub_results.py.
    cross = ch4b.copy()
    ch3_extra = [
        col for col in ch3.columns if col == "gauge_id" or col not in cross.columns
    ]
    cross = cross.merge(
        ch3[ch3_extra].drop_duplicates("gauge_id"),
        on="gauge_id",
        how="left",
        validate="one_to_one",
    )

    for label, experiment in CH4A_MODELS.items():
        frame = read_ch4a_metric(exp_root, experiment, label)
        cross = cross.merge(frame, on="gauge_id", how="left", validate="one_to_one")

    cross["ch4a_delta_hps_minus_stl_ssm"] = (
        cross["ch4a_hps_ssm_nse"] - cross["ch4a_stl_ssm_nse"]
    )
    cross["ch4a_delta_cgc_minus_stl_ssm"] = (
        cross["ch4a_cgc_ssm_nse"] - cross["ch4a_stl_ssm_nse"]
    )
    cross["ch4a_delta_cgc_minus_hps_ssm"] = (
        cross["ch4a_cgc_ssm_nse"] - cross["ch4a_hps_ssm_nse"]
    )
    cross["ch4a_delta_hps_pretrain"] = (
        cross["ch4a_hps_pre_ssm_nse"] - cross["ch4a_hps_ssm_nse"]
    )
    cross["ch4a_delta_cgc_pretrain"] = (
        cross["ch4a_cgc_pre_ssm_nse"] - cross["ch4a_cgc_ssm_nse"]
    )

    basin_output = out_dir / "ch3_ch4a_ch4b_cross_experiment_per_basin.csv"
    cross.to_csv(basin_output, index=False)

    comparisons = [
        (
            "Ch4A",
            "Q -> SSM",
            "temporal observation limitation",
            "Hard-MTL minus STL-SSM",
            "ch4a_delta_hps_minus_stl_ssm",
        ),
        (
            "Ch4A",
            "Q -> SSM",
            "temporal observation limitation",
            "CGC minus STL-SSM",
            "ch4a_delta_cgc_minus_stl_ssm",
        ),
        (
            "Ch4A",
            "Q -> SSM",
            "temporal observation limitation",
            "CGC minus Hard-MTL",
            "ch4a_delta_cgc_minus_hps_ssm",
        ),
        (
            "Ch4A",
            "Q -> SSM",
            "temporal observation limitation",
            "Hard pretraining gain",
            "ch4a_delta_hps_pretrain",
        ),
        (
            "Ch4A",
            "Q -> SSM",
            "temporal observation limitation",
            "CGC pretraining gain",
            "ch4a_delta_cgc_pretrain",
        ),
        (
            "Ch4B",
            "SSM -> Q",
            "spatial PUB limitation",
            "Hard-MTL-PUB minus STL-Q-PUB",
            "delta_nse_hps_minus_stl",
        ),
        (
            "Ch4B",
            "SSM -> Q",
            "spatial PUB limitation",
            "CGC-PUB minus STL-Q-PUB",
            "delta_nse_cgc_minus_stl",
        ),
        (
            "Ch4B",
            "SSM -> Q",
            "spatial PUB limitation",
            "CGC-PUB minus Hard-MTL-PUB",
            "delta_nse_cgc_minus_hps",
        ),
    ]

    rows = [
        effect_summary(cross, experiment, direction, limitation, comparison, column)
        for experiment, direction, limitation, comparison, column in comparisons
    ]
    pd.DataFrame(rows).to_csv(
        out_dir / "ch4_cross_experiment_directionality_summary.csv",
        index=False,
    )

    if "hydroclimate_group" in cross.columns:
        group_rows: list[dict[str, object]] = []
        for group_name, group in cross.groupby("hydroclimate_group", dropna=True):
            for experiment, direction, limitation, comparison, column in comparisons:
                group_rows.append(
                    effect_summary(
                        group,
                        experiment,
                        direction,
                        limitation,
                        comparison,
                        column,
                        hydroclimate_group=str(group_name),
                    )
                )
        pd.DataFrame(group_rows).to_csv(
            out_dir / "ch4_cross_experiment_hydroclimate_summary.csv",
            index=False,
        )

    print(f"Cross-experiment basin table exported to: {basin_output}")
    print(f"Rows: {len(cross)}")
    print(
        "Directionality summary exported to: "
        f"{out_dir / 'ch4_cross_experiment_directionality_summary.csv'}"
    )


if __name__ == "__main__":
    main()
