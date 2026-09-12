#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize formal Chapter 4B PUB streamflow results.

This script is restricted to Chapter 4B PUB outputs. It collects formal
per-basin metrics across spatial folds, builds basin-wise absolute streamflow
NSE values, and summarizes paired performance differences among STL-Q,
Hard-MTL, and CGC.

Chapter 3 results and hydroclimatic metadata are intentionally excluded.
Cross-experiment merging is handled separately.

Outputs:
    - ch4b_pub_ensemble_per_basin_metrics.csv
    - ch4b_pub_effects.csv
    - ch4b_pub_model_effect_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.io_utils import normalize_basin_id  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.paths import (  # noqa: E402
    ENSEMBLE_DIR,
    SUMMARY_DIR,
)


DEFAULT_ENSEMBLE_INDEX = ENSEMBLE_DIR / "ensemble_index.csv"
EXPECTED_BASINS = 592

PUB_SCENARIOS = {
    "stl_q": "PUB_STL_Q_NSE",
    "hps_target_ssm": "PUB_Hard_MTL_Q_NSE",
    "cgc_target_ssm": "PUB_CGC_Q_NSE",
}

COMPARISONS = {
    "Hard-MTL-PUB": "delta_nse_hps_minus_stl",
    "CGC-PUB": "delta_nse_cgc_minus_stl",
    "CGC-minus-Hard": "delta_nse_cgc_minus_hps",
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize formal Chapter 4B PUB streamflow results."
    )
    parser.add_argument(
        "--ensemble-index",
        type=Path,
        default=DEFAULT_ENSEMBLE_INDEX,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUMMARY_DIR,
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure compact console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def resolve_path(path: Path) -> Path:
    """Resolve repository-relative paths."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the basin identifier column."""
    frame = frame.copy()

    candidates = (
        "gauge_id",
        "basin_id",
        "gage_id",
        "Unnamed: 0",
        frame.columns[0],
    )

    for candidate in candidates:
        if candidate not in frame.columns:
            continue

        frame = frame.rename(columns={candidate: "gauge_id"})
        frame["gauge_id"] = frame["gauge_id"].map(normalize_basin_id)
        return frame

    raise ValueError("Cannot identify basin-id column.")


def load_ensemble(index_path: Path) -> pd.DataFrame:
    """Load formal PUB per-basin metrics from all spatial folds."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"Ensemble index not found: {index_path}"
        )

    index = pd.read_csv(index_path)

    required = {
        "metrics_csv",
        "fold_id",
        "scenario",
        "seed_count",
    }
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(
            f"Missing ensemble-index columns: {sorted(missing)}"
        )

    frames: list[pd.DataFrame] = []

    for _, row in index.iterrows():
        metrics_path = resolve_path(
            Path(str(row["metrics_csv"]))
        )

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {metrics_path}"
            )

        metrics = normalize_id_column(
            pd.read_csv(
                metrics_path,
                dtype={"gauge_id": str},
            )
        )

        if "streamflow_nse" not in metrics.columns:
            raise ValueError(
                f"Missing streamflow_nse in {metrics_path}"
            )

        metrics["fold_id"] = int(row["fold_id"])
        metrics["scenario"] = str(row["scenario"])
        metrics["seed_count"] = int(row["seed_count"])
        frames.append(metrics)

    if not frames:
        raise RuntimeError("No ensemble metric files found.")

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicates = frame.duplicated(
        ["scenario", "gauge_id"]
    )
    if duplicates.any():
        rows = frame.loc[
            duplicates,
            ["scenario", "gauge_id", "fold_id"],
        ]
        raise ValueError(
            "A basin appears more than once per scenario:\n"
            f"{rows.head()}"
        )

    return frame


def validate_pub_scenarios(
    frame: pd.DataFrame,
) -> None:
    """Validate formal PUB scenario coverage."""
    available = set(frame["scenario"].unique())
    missing = set(PUB_SCENARIOS).difference(available)

    if missing:
        raise ValueError(
            f"Missing core PUB scenarios: {sorted(missing)}"
        )

    for scenario in PUB_SCENARIOS:
        count = int(
            frame.loc[
                frame["scenario"] == scenario,
                "gauge_id",
            ].nunique()
        )

        if count != EXPECTED_BASINS:
            raise RuntimeError(
                f"{scenario}: expected {EXPECTED_BASINS} "
                f"basins, found {count}."
            )


def build_pub_effects(
    all_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build basin-wise absolute and paired PUB NSE results."""
    validate_pub_scenarios(all_metrics)

    effects = (
        all_metrics.pivot(
            index="gauge_id",
            columns="scenario",
            values="streamflow_nse",
        )
        .reset_index()
    )

    for scenario, alias in PUB_SCENARIOS.items():
        effects[alias] = effects[scenario]

    effects["delta_nse_hps_minus_stl"] = (
        effects["hps_target_ssm"]
        - effects["stl_q"]
    )
    effects["delta_nse_cgc_minus_stl"] = (
        effects["cgc_target_ssm"]
        - effects["stl_q"]
    )
    effects["delta_nse_cgc_minus_hps"] = (
        effects["cgc_target_ssm"]
        - effects["hps_target_ssm"]
    )

    effects["hps_positive_transfer"] = (
        effects["delta_nse_hps_minus_stl"] > 0
    )
    effects["cgc_positive_transfer"] = (
        effects["delta_nse_cgc_minus_stl"] > 0
    )
    effects["hps_negative_transfer"] = (
        effects["delta_nse_hps_minus_stl"] < 0
    )
    effects["cgc_negative_transfer"] = (
        effects["delta_nse_cgc_minus_stl"] < 0
    )

    fold_lookup = all_metrics.loc[
        all_metrics["scenario"] == "stl_q",
        ["gauge_id", "fold_id"],
    ]

    effects = effects.merge(
        fold_lookup,
        on="gauge_id",
        how="left",
        validate="one_to_one",
    )

    if len(effects) != EXPECTED_BASINS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASINS} PUB basins, "
            f"found {len(effects)}."
        )

    return effects


def summarize_effects(
    effects: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired PUB NSE effects."""
    records: list[dict[str, object]] = []

    for comparison, column in COMPARISONS.items():
        values = pd.to_numeric(
            effects[column],
            errors="coerce",
        )
        values = values[np.isfinite(values)]

        records.append({
            "comparison": comparison,
            "n_basins": len(values),
            "median_delta_nse": values.median(),
            "mean_delta_nse": values.mean(),
            "q25_delta_nse": values.quantile(0.25),
            "q75_delta_nse": values.quantile(0.75),
            "positive_rate":
                float((values > 0).mean()),
            "negative_rate":
                float((values < 0).mean()),
        })

    return pd.DataFrame(records)


def main() -> None:
    """Run the Chapter 4B PUB summary workflow."""
    setup_logging()
    args = parse_args()

    index_path = resolve_path(args.ensemble_index)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = load_ensemble(index_path)

    ensemble_path = (
        output_dir
        / "ch4b_pub_ensemble_per_basin_metrics.csv"
    )
    all_metrics.to_csv(
        ensemble_path,
        index=False,
    )

    effects = build_pub_effects(all_metrics)

    effects_path = (
        output_dir
        / "ch4b_pub_effects.csv"
    )
    effects.to_csv(
        effects_path,
        index=False,
    )

    summary = summarize_effects(effects)

    summary_path = (
        output_dir
        / "ch4b_pub_model_effect_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    logger.info(
        "PUB median NSE: STL=%.6f, Hard=%.6f, CGC=%.6f",
        effects["PUB_STL_Q_NSE"].median(),
        effects["PUB_Hard_MTL_Q_NSE"].median(),
        effects["PUB_CGC_Q_NSE"].median(),
    )
    logger.info(
        "Saved ensemble metrics: %s",
        ensemble_path,
    )
    logger.info(
        "Saved PUB effects: %s",
        effects_path,
    )
    logger.info(
        "Saved model-effect summary: %s",
        summary_path,
    )


if __name__ == "__main__":
    main()