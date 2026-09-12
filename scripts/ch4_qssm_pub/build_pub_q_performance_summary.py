#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build overall performance summaries for Chapter 4 Experiment 2.

The script uses the validated formal PUB ensemble per-basin metrics as the
single source of truth. It reports absolute streamflow NSE/KGE performance
and basin-wise paired differences among STL-Q, Hard-MTL, and CGC.

Experiment:
    SSM-assisted streamflow prediction under PUB conditions.

Outputs:
    - exp2_primary_q_absolute_nse_kge.csv
    - exp2_primary_q_paired_nse_kge.csv
    - exp2_primary_q_paired_basin_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/summary"
    / "ch4b_pub_ensemble_per_basin_metrics.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/ch4_summary/teacher_revision"
)

EXPECTED_BASINS = 592

MODEL_MAP = {
    "stl_q": "STL-Q",
    "hps_target_ssm": "Hard-MTL",
    "cgc_target_ssm": "CGC",
}

METRICS = {
    "streamflow_nse": "NSE",
    "streamflow_kge": "KGE",
}

COMPARISONS = [
    ("Hard-MTL", "STL-Q"),
    ("CGC", "STL-Q"),
    ("CGC", "Hard-MTL"),
]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build overall absolute and paired PUB streamflow "
            "performance summaries."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure compact console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize USGS gauge IDs to eight-character strings."""
    if series.isna().any():
        raise ValueError("Missing gauge_id values detected.")

    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def load_metrics(path: Path) -> pd.DataFrame:
    """Load and validate formal PUB per-basin metrics."""
    if not path.exists():
        raise FileNotFoundError(
            f"PUB ensemble metrics not found: {path}"
        )

    logger.info("Loading formal PUB metrics: %s", path)

    frame = pd.read_csv(
        path,
        dtype={"gauge_id": str},
    )

    required = {
        "gauge_id",
        "scenario",
        "fold_id",
        *METRICS,
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    frame["gauge_id"] = normalize_gauge_id(
        frame["gauge_id"]
    )

    frame = frame[
        frame["scenario"].isin(MODEL_MAP)
    ].copy()

    if frame.duplicated(
        ["scenario", "gauge_id"]
    ).any():
        raise ValueError(
            "Duplicated scenario-gauge records detected."
        )

    counts = (
        frame.groupby("scenario")["gauge_id"]
        .nunique()
        .to_dict()
    )

    for scenario in MODEL_MAP:
        count = int(counts.get(scenario, 0))
        if count != EXPECTED_BASINS:
            raise RuntimeError(
                f"{scenario}: expected {EXPECTED_BASINS} "
                f"basins, found {count}."
            )

    for metric in METRICS:
        frame[metric] = pd.to_numeric(
            frame[metric],
            errors="coerce",
        )

    logger.info(
        "Loaded %d basin-scenario records.",
        len(frame),
    )

    return frame


def build_absolute_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize absolute NSE and KGE performance."""
    records: list[dict[str, object]] = []

    for scenario, model in MODEL_MAP.items():
        subset = frame[
            frame["scenario"] == scenario
        ]

        nse = subset["streamflow_nse"].dropna()
        kge = subset["streamflow_kge"].dropna()

        records.append({
            "model": model,
            "n_nse": len(nse),
            "median_nse": nse.median(),
            "mean_nse": nse.mean(),
            "q25_nse": nse.quantile(0.25),
            "q75_nse": nse.quantile(0.75),
            "nse_ge_0_rate":
                float((nse >= 0.0).mean()),
            "nse_ge_0p5_rate":
                float((nse >= 0.5).mean()),
            "n_kge": len(kge),
            "median_kge": kge.median(),
            "mean_kge": kge.mean(),
            "q25_kge": kge.quantile(0.25),
            "q75_kge": kge.quantile(0.75),
        })

    return pd.DataFrame(records)


def build_wide_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per basin with model-specific metrics."""
    parts: list[pd.DataFrame] = []

    for metric in METRICS:
        wide = (
            frame.pivot(
                index="gauge_id",
                columns="scenario",
                values=metric,
            )
            .rename(
                columns={
                    scenario:
                        f"{model}__{metric}"
                    for scenario, model
                    in MODEL_MAP.items()
                }
            )
        )
        parts.append(wide)

    result = pd.concat(
        parts,
        axis=1,
    ).reset_index()

    if len(result) != EXPECTED_BASINS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASINS} basins, "
            f"found {len(result)}."
        )

    return result


def add_pairwise_deltas(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add basin-wise paired NSE and KGE differences."""
    frame = frame.copy()

    for left, right in COMPARISONS:
        for metric in METRICS:
            frame[
                f"{left}_minus_{right}__{metric}"
            ] = (
                frame[f"{left}__{metric}"]
                - frame[f"{right}__{metric}"]
            )

    return frame


def build_paired_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize basin-wise paired differences."""
    records: list[dict[str, object]] = []

    for left, right in COMPARISONS:
        for metric, metric_label in METRICS.items():
            column = (
                f"{left}_minus_{right}__{metric}"
            )

            delta = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            delta = delta[
                np.isfinite(delta)
            ]

            records.append({
                "comparison":
                    f"{left} - {right}",
                "metric":
                    metric_label,
                "n":
                    len(delta),
                "median_delta":
                    delta.median(),
                "mean_delta":
                    delta.mean(),
                "q25_delta":
                    delta.quantile(0.25),
                "q75_delta":
                    delta.quantile(0.75),
                "positive_rate":
                    float((delta > 0).mean()),
                "negative_rate":
                    float((delta < 0).mean()),
                "zero_rate":
                    float((delta == 0).mean()),
            })

    return pd.DataFrame(records)


def main() -> None:
    """Build and export overall PUB performance summaries."""
    setup_logging()
    args = parse_args()

    input_path = (
        args.input
        if args.input.is_absolute()
        else PROJECT_ROOT / args.input
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = load_metrics(input_path)

    absolute = build_absolute_summary(
        metrics
    )

    basin = add_pairwise_deltas(
        build_wide_table(metrics)
    )

    paired = build_paired_summary(
        basin
    )

    absolute_path = (
        output_dir
        / "exp2_primary_q_absolute_nse_kge.csv"
    )
    paired_path = (
        output_dir
        / "exp2_primary_q_paired_nse_kge.csv"
    )
    basin_path = (
        output_dir
        / "exp2_primary_q_paired_basin_metrics.csv"
    )

    absolute.to_csv(
        absolute_path,
        index=False,
    )
    paired.to_csv(
        paired_path,
        index=False,
    )
    basin.to_csv(
        basin_path,
        index=False,
    )

    logger.info(
        "Absolute performance:\n%s",
        absolute[
            [
                "model",
                "median_nse",
                "median_kge",
            ]
        ].to_string(index=False),
    )

    logger.info(
        "Paired performance:\n%s",
        paired[
            [
                "comparison",
                "metric",
                "median_delta",
                "positive_rate",
            ]
        ].to_string(index=False),
    )

    logger.info(
        "Saved absolute summary: %s",
        absolute_path,
    )
    logger.info(
        "Saved paired summary: %s",
        paired_path,
    )
    logger.info(
        "Saved basin-level paired metrics: %s",
        basin_path,
    )


if __name__ == "__main__":
    main()
