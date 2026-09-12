#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose hydroclimatic controls on Chapter 4 Experiment 2 PUB performance.

The analysis uses the validated PUB basin-level transfer table and examines
absolute streamflow NSE together with the CGC-minus-STL paired NSE effect
across hydroclimatic conditions.

Experiment:
    SSM-assisted streamflow prediction under PUB conditions.

Outputs:
    - pub_q_hydroclimate_diagnostics_summary.csv
    - fig_nse_q_distribution.png
    - fig_delta_nse_aridity.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydroclimate_groups"
    / "pub_q_transfer_analysis.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydroclimate_groups"
    / "diagnostics"
)

EXPECTED_BASINS = 592
GROUP_ORDER = ["Dry", "Snow", "Wet"]

MODEL_COLUMNS = {
    "STL-Q": "STL_Q_NSE",
    "Hard-MTL": "Hard_MTL_Q_NSE",
    "CGC": "CGC_Q_NSE",
}

DELTA_COLUMNS = {
    "Hard-MTL minus STL-Q":
        "Delta_NSE_Q_Hard_minus_STL",
    "CGC minus STL-Q":
        "Delta_NSE_Q_CGC_minus_STL",
    "CGC minus Hard-MTL":
        "Delta_NSE_Q_CGC_minus_Hard",
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose PUB streamflow performance across "
            "hydroclimatic conditions."
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


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the PUB hydroclimatic analysis table."""
    if not path.exists():
        raise FileNotFoundError(
            f"PUB transfer table not found: {path}"
        )

    logger.info("Loading PUB transfer table: %s", path)

    frame = pd.read_csv(
        path,
        dtype={"gauge_id": str},
    )

    required = {
        "gauge_id",
        "hydroclimate_group",
        "aridity",
        "frac_snow",
        *MODEL_COLUMNS.values(),
        *DELTA_COLUMNS.values(),
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    frame["gauge_id"] = normalize_gauge_id(
        frame["gauge_id"]
    )

    if frame["gauge_id"].duplicated().any():
        raise ValueError(
            "Duplicated gauge_id values detected."
        )

    if len(frame) != EXPECTED_BASINS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASINS} basins, "
            f"found {len(frame)}."
        )

    numeric_columns = [
        "aridity",
        "frac_snow",
        *MODEL_COLUMNS.values(),
        *DELTA_COLUMNS.values(),
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    logger.info(
        "Hydroclimate groups: %s",
        frame["hydroclimate_group"]
        .value_counts()
        .to_dict(),
    )

    return frame


def finite_values(
    series: pd.Series,
) -> pd.Series:
    """Return finite numeric values only."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    )
    return values[np.isfinite(values)]


def summarize_groups(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize hydroclimatic attributes and PUB NSE performance."""
    records: list[dict[str, object]] = []

    for group in GROUP_ORDER:
        subset = frame[
            frame["hydroclimate_group"] == group
        ]

        record: dict[str, object] = {
            "hydroclimate_group": group,
            "n_basins": len(subset),
            "median_aridity":
                finite_values(subset["aridity"]).median(),
            "median_frac_snow":
                finite_values(subset["frac_snow"]).median(),
        }

        for model, column in MODEL_COLUMNS.items():
            key = (
                model.lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            record[f"median_nse_{key}"] = (
                finite_values(subset[column]).median()
            )

        for comparison, column in DELTA_COLUMNS.items():
            key = (
                comparison.lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            delta = finite_values(subset[column])

            record[f"median_{key}"] = delta.median()
            record[f"positive_rate_{key}"] = (
                float((delta > 0).mean())
            )

        records.append(record)

    return pd.DataFrame(records)


def plot_nse_distribution(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot absolute NSE distributions by group and model."""
    fig, ax = plt.subplots(
        figsize=(9.0, 5.0),
    )

    offsets = [-0.25, 0.0, 0.25]
    models = list(MODEL_COLUMNS)
    positions: list[float] = []
    data: list[np.ndarray] = []

    for group_index, group in enumerate(
        GROUP_ORDER,
        start=1,
    ):
        subset = frame[
            frame["hydroclimate_group"] == group
        ]

        for offset, model in zip(
            offsets,
            models,
        ):
            values = finite_values(
                subset[MODEL_COLUMNS[model]]
            ).to_numpy()

            positions.append(
                group_index + offset
            )
            data.append(values)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.20,
        showfliers=False,
        medianprops={"linewidth": 1.5},
    )

    for index, median in enumerate(
        box["medians"][:len(models)]
    ):
        median.set_label(models[index])

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xticks(
        range(1, len(GROUP_ORDER) + 1)
    )
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_xlabel("Hydroclimate group")
    ax.set_ylabel("NSE")
    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_delta_aridity(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot CGC-minus-STL NSE effect against aridity."""
    x = pd.to_numeric(
        frame["aridity"],
        errors="coerce",
    )
    y = pd.to_numeric(
        frame["Delta_NSE_Q_CGC_minus_STL"],
        errors="coerce",
    )

    valid = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(
        figsize=(6.5, 4.5),
    )

    ax.scatter(
        x[valid],
        y[valid],
        s=12,
        alpha=0.65,
    )
    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_xlabel("Aridity index")
    ax.set_ylabel(r"CGC - STL-Q $\Delta$NSE")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Run the PUB hydroclimatic diagnostics."""
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

    frame = load_data(input_path)
    summary = summarize_groups(frame)

    summary_path = (
        output_dir
        / "pub_q_hydroclimate_diagnostics_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plot_nse_distribution(
        frame,
        output_dir / "fig_nse_q_distribution.png",
    )
    plot_delta_aridity(
        frame,
        output_dir / "fig_delta_nse_aridity.png",
    )

    logger.info(
        "Median NSE: STL=%.6f, Hard=%.6f, CGC=%.6f",
        frame["STL_Q_NSE"].median(),
        frame["Hard_MTL_Q_NSE"].median(),
        frame["CGC_Q_NSE"].median(),
    )
    logger.info(
        "Saved diagnostics summary: %s",
        summary_path,
    )
    logger.info(
        "PUB hydroclimatic diagnostics completed."
    )


if __name__ == "__main__":
    main()
