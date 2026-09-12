#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze hydroclimatic dependence of Chapter 4 Experiment 2 PUB results.

The script uses the validated basin-level PUB transfer table to summarize
absolute streamflow NSE and paired NSE differences across Dry, Snow, and Wet
catchments.

Experiment:
    SSM-assisted streamflow prediction under PUB conditions.

Target:
    Streamflow (Q)

Models:
    STL-Q
    Hard-MTL-Q
    CGC-Q

Outputs:
    - pub_q_basin_hydroclimate_metrics.csv
    - pub_q_absolute_nse_group_summary.csv
    - pub_q_delta_nse_group_summary.csv
    - pub_q_absolute_nse_by_hydroclimate.png
    - pub_q_delta_nse_by_hydroclimate.png
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
)

EXPECTED_BASINS = 592
EXPECTED_GROUP_COUNTS = {
    "Dry": 142,
    "Snow": 168,
    "Wet": 282,
}

GROUP_ORDER = ["Dry", "Snow", "Wet"]

MODEL_COLUMNS = {
    "STL-Q": "STL_Q_NSE",
    "Hard-MTL-Q": "Hard_MTL_Q_NSE",
    "CGC-Q": "CGC_Q_NSE",
}

COMPARISONS = {
    "Hard-MTL-Q minus STL-Q": "Delta_NSE_Q_Hard_minus_STL",
    "CGC-Q minus STL-Q": "Delta_NSE_Q_CGC_minus_STL",
    "CGC-Q minus Hard-MTL-Q": "Delta_NSE_Q_CGC_minus_Hard",
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze PUB streamflow performance across "
            "Dry, Snow, and Wet catchments."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Validated basin-level PUB transfer table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for hydroclimatic summaries and figures.",
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


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    source: Path,
) -> None:
    """Require expected columns in an input table."""
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {source}: {sorted(missing)}"
        )


def finite_values(series: pd.Series) -> pd.Series:
    """Return finite numeric values only."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    )
    return values[np.isfinite(values)]


def load_analysis_table(path: Path) -> pd.DataFrame:
    """Load and validate the formal PUB transfer table."""
    if not path.exists():
        raise FileNotFoundError(
            f"PUB transfer table not found: {path}"
        )

    logger.info(
        "Loading PUB transfer table: %s",
        path,
    )

    frame = pd.read_csv(
        path,
        dtype={"gauge_id": str},
    )

    required = {
        "gauge_id",
        "aridity",
        "frac_snow",
        "hydroclimate_group",
        *MODEL_COLUMNS.values(),
        *COMPARISONS.values(),
    }
    require_columns(
        frame,
        required,
        path,
    )

    frame["gauge_id"] = normalize_gauge_id(
        frame["gauge_id"]
    )

    if frame["gauge_id"].duplicated().any():
        raise ValueError(
            "Duplicated gauge_id values in PUB transfer table."
        )

    if len(frame) != EXPECTED_BASINS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASINS} basins, "
            f"found {len(frame)}."
        )

    numeric_columns = [
        *MODEL_COLUMNS.values(),
        *COMPARISONS.values(),
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    counts = (
        frame["hydroclimate_group"]
        .value_counts()
        .to_dict()
    )

    if counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(
            "Unexpected hydroclimate counts: "
            f"{counts}; expected {EXPECTED_GROUP_COUNTS}."
        )

    logger.info(
        "Loaded %d PUB basins.",
        len(frame),
    )
    logger.info(
        "Hydroclimate groups: %s",
        counts,
    )

    return frame


def summarize_absolute_nse(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize absolute streamflow NSE by model and hydroclimate group."""
    records: list[dict[str, object]] = []

    for group in GROUP_ORDER:
        subset = frame[
            frame["hydroclimate_group"] == group
        ]

        for model, column in MODEL_COLUMNS.items():
            values = finite_values(
                subset[column]
            )

            records.append({
                "hydroclimate_group": group,
                "model": model,
                "n_basins": len(values),
                "median_NSE_Q": values.median(),
                "mean_NSE_Q": values.mean(),
                "q25_NSE_Q": values.quantile(0.25),
                "q75_NSE_Q": values.quantile(0.75),
                "NSE_Q_ge_0_rate":
                    float((values >= 0.0).mean()),
                "NSE_Q_ge_0p50_rate":
                    float((values >= 0.50).mean()),
                "NSE_Q_ge_0p60_rate":
                    float((values >= 0.60).mean()),
            })

    return pd.DataFrame(records)


def summarize_delta_nse(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize basin-wise paired NSE differences."""
    records: list[dict[str, object]] = []

    for group in GROUP_ORDER:
        subset = frame[
            frame["hydroclimate_group"] == group
        ]

        for comparison, column in COMPARISONS.items():
            delta = finite_values(
                subset[column]
            )

            records.append({
                "hydroclimate_group": group,
                "comparison": comparison,
                "n_basins": len(delta),
                "median_Delta_NSE_Q":
                    delta.median(),
                "mean_Delta_NSE_Q":
                    delta.mean(),
                "q25_Delta_NSE_Q":
                    delta.quantile(0.25),
                "q75_Delta_NSE_Q":
                    delta.quantile(0.75),
                "positive_rate":
                    float((delta > 0).mean()),
                "negative_rate":
                    float((delta < 0).mean()),
                "zero_rate":
                    float((delta == 0).mean()),
            })

    return pd.DataFrame(records)


def plot_absolute_nse(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot absolute NSE distributions by hydroclimate group."""
    fig, ax = plt.subplots(
        figsize=(9.0, 5.0),
    )

    positions: list[float] = []
    data: list[np.ndarray] = []

    offsets = [-0.25, 0.0, 0.25]
    model_names = list(MODEL_COLUMNS)

    for group_index, group in enumerate(
        GROUP_ORDER
    ):
        subset = frame[
            frame["hydroclimate_group"] == group
        ]

        for offset, model in zip(
            offsets,
            model_names,
        ):
            values = finite_values(
                subset[MODEL_COLUMNS[model]]
            ).to_numpy(dtype=float)

            positions.append(
                group_index + 1 + offset
            )
            data.append(values)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.20,
        patch_artist=False,
        showfliers=False,
        medianprops={"linewidth": 1.5},
    )

    for index, median in enumerate(
        box["medians"][:len(model_names)]
    ):
        median.set_label(
            model_names[index]
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xticks(
        range(1, len(GROUP_ORDER) + 1)
    )
    ax.set_xticklabels(
        GROUP_ORDER
    )
    ax.set_xlabel(
        "Hydroclimate group"
    )
    ax.set_ylabel(
        "NSE"
    )
    ax.legend(
        loc="best",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_delta_nse(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot paired NSE differences by hydroclimate group."""
    fig, ax = plt.subplots(
        figsize=(9.0, 5.0),
    )

    positions: list[float] = []
    data: list[np.ndarray] = []

    offsets = [-0.25, 0.0, 0.25]
    comparison_names = list(COMPARISONS)

    for group_index, group in enumerate(
        GROUP_ORDER
    ):
        subset = frame[
            frame["hydroclimate_group"] == group
        ]

        for offset, comparison in zip(
            offsets,
            comparison_names,
        ):
            values = finite_values(
                subset[COMPARISONS[comparison]]
            ).to_numpy(dtype=float)

            positions.append(
                group_index + 1 + offset
            )
            data.append(values)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.20,
        patch_artist=False,
        showfliers=False,
        medianprops={"linewidth": 1.5},
    )

    for index, median in enumerate(
        box["medians"][:len(comparison_names)]
    ):
        median.set_label(
            comparison_names[index]
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xticks(
        range(1, len(GROUP_ORDER) + 1)
    )
    ax.set_xticklabels(
        GROUP_ORDER
    )
    ax.set_xlabel(
        "Hydroclimate group"
    )
    ax.set_ylabel(
        r"$\Delta$NSE"
    )
    ax.legend(
        loc="best",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Run the PUB hydroclimatic analysis."""
    setup_logging()
    args = parse_args()

    input_path = resolve_path(
        args.input
    )
    output_dir = resolve_path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_analysis_table(
        input_path
    )

    absolute = summarize_absolute_nse(
        frame
    )
    delta = summarize_delta_nse(
        frame
    )

    basin_path = (
        output_dir
        / "pub_q_basin_hydroclimate_metrics.csv"
    )
    absolute_path = (
        output_dir
        / "pub_q_absolute_nse_group_summary.csv"
    )
    delta_path = (
        output_dir
        / "pub_q_delta_nse_group_summary.csv"
    )

    frame.to_csv(
        basin_path,
        index=False,
    )
    absolute.to_csv(
        absolute_path,
        index=False,
    )
    delta.to_csv(
        delta_path,
        index=False,
    )

    plot_absolute_nse(
        frame,
        output_dir
        / "pub_q_absolute_nse_by_hydroclimate.png",
    )
    plot_delta_nse(
        frame,
        output_dir
        / "pub_q_delta_nse_by_hydroclimate.png",
    )

    logger.info(
        "Median NSE: STL=%.6f, Hard=%.6f, CGC=%.6f",
        frame["STL_Q_NSE"].median(),
        frame["Hard_MTL_Q_NSE"].median(),
        frame["CGC_Q_NSE"].median(),
    )
    logger.info(
        "Saved basin metrics: %s",
        basin_path,
    )
    logger.info(
        "Saved absolute NSE summary: %s",
        absolute_path,
    )
    logger.info(
        "Saved Delta NSE summary: %s",
        delta_path,
    )
    logger.info(
        "PUB Q hydroclimatic analysis completed."
    )


if __name__ == "__main__":
    main()