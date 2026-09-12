#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build cross-experiment basin-level summaries for Chapter 4.

The script integrates three frozen result sources:

    Chapter 3:
        Reference Q/ET multi-task results and basin metadata.

    Chapter 4A:
        Q-assisted SSM prediction under temporal observation limitation.

    Chapter 4B:
        SSM-assisted Q prediction under spatial PUB limitation.

Chapter 4B input is strictly PUB-only. Chapter 3 results are merged here,
with explicit ``CH3_`` prefixes, so cross-chapter integration occurs only
inside this script.

This is post-processing only and never changes trained models.
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
    CH3_SUMMARY,
    SUMMARY_DIR,
)


DEFAULT_CH3 = CH3_SUMMARY
DEFAULT_CH4B = SUMMARY_DIR / "ch4b_pub_effects.csv"
DEFAULT_OUTPUT_DIR = SUMMARY_DIR
EXPECTED_BASINS = 592

CH4A_MODELS = {
    "stl": "ch4a_formal_stl_ssm_seed42",
    "hps": "ch4a_formal_hps_qssm_seed42",
    "cgc": "ch4a_formal_cgc_qssm_seed42",
    "hps_pre": "ch4a_formal_hps_qpre_finetune_qssm_seed42",
    "cgc_pre": "ch4a_formal_cgc_qpre_finetune_qssm_seed42",
}

CH3_METADATA_COLUMNS = [
    "huc_02",
    "aridity",
    "frac_snow",
    "p_seasonality",
    "max_water_content",
]

CH3_RESULT_COLUMNS = [
    "STL_Q_streamflow_nse",
    "Hard_MTL_streamflow_nse",
    "MMoE_streamflow_nse",
    "CGC_streamflow_nse",
    "STL_ET_evapotranspiration_nse",
    "Hard_MTL_evapotranspiration_nse",
    "MMoE_evapotranspiration_nse",
    "CGC_evapotranspiration_nse",
    "Delta_NSE_HardMTL_minus_STLQ",
    "Delta_NSE_MMoE_minus_STLQ",
    "Delta_NSE_CGC_minus_STLQ",
]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Chapter 3/4A/4B cross-experiment summaries."
    )
    parser.add_argument(
        "--ch3-summary",
        type=Path,
        default=DEFAULT_CH3,
    )
    parser.add_argument(
        "--ch4b-effects",
        type=Path,
        default=DEFAULT_CH4B,
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
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


def resolve_path(path: Path) -> Path:
    """Resolve repository-relative paths."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
        frame["gauge_id"] = frame["gauge_id"].map(
            normalize_basin_id
        )

        if frame["gauge_id"].duplicated().any():
            raise ValueError(
                "Duplicate basin identifiers detected."
            )

        return frame

    raise ValueError("Cannot identify basin-id column.")


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    source: Path,
) -> None:
    """Require columns in an input table."""
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {source}: {sorted(missing)}"
        )


def hydroclimate_group(frame: pd.DataFrame) -> pd.Series:
    """Assign mutually exclusive Snow, Wet, and Dry groups."""
    aridity = pd.to_numeric(
        frame["aridity"],
        errors="coerce",
    )
    frac_snow = pd.to_numeric(
        frame["frac_snow"],
        errors="coerce",
    )

    group = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="object",
    )

    snow = frac_snow > 0.20
    group.loc[snow] = "Snow"
    group.loc[(~snow) & (aridity < 1.0)] = "Wet"
    group.loc[(~snow) & (aridity >= 1.0)] = "Dry"

    return group


def load_ch3(path: Path) -> pd.DataFrame:
    """Load selected Chapter 3 metadata and performance results."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chapter 3 summary not found: {path}"
        )

    frame = normalize_frame(
        pd.read_csv(path)
    )

    metadata = [
        column
        for column in CH3_METADATA_COLUMNS
        if column in frame.columns
    ]
    results = [
        column
        for column in CH3_RESULT_COLUMNS
        if column in frame.columns
    ]

    keep = [
        "gauge_id",
        *metadata,
        *results,
    ]

    frame = frame[keep].copy()

    frame = frame.rename(
        columns={
            column: f"CH3_{column}"
            for column in results
        }
    )

    return frame


def load_ch4b(path: Path) -> pd.DataFrame:
    """Load the PUB-only Chapter 4B effects table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chapter 4B PUB effects not found: {path}"
        )

    frame = normalize_frame(
        pd.read_csv(path)
    )

    required = {
        "gauge_id",
        "PUB_STL_Q_NSE",
        "PUB_Hard_MTL_Q_NSE",
        "PUB_CGC_Q_NSE",
        "delta_nse_hps_minus_stl",
        "delta_nse_cgc_minus_stl",
        "delta_nse_cgc_minus_hps",
    }
    require_columns(frame, required, path)

    if len(frame) != EXPECTED_BASINS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASINS} Chapter 4B basins, "
            f"found {len(frame)}."
        )

    return frame


def read_ch4a_metric(
    root: Path,
    experiment: str,
    label: str,
) -> pd.DataFrame:
    """Load one frozen Chapter 4A per-basin metric table."""
    path = (
        root
        / experiment
        / "test_per_basin_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    frame = normalize_frame(
        pd.read_csv(path)
    )

    metric_columns = [
        column
        for column in frame.columns
        if (
            column.startswith("ssm_")
            or column.startswith("streamflow_")
        )
    ]

    if not metric_columns:
        raise ValueError(
            f"No hydrological metric columns found in {path}"
        )

    frame = frame[
        ["gauge_id", *metric_columns]
    ]

    return frame.rename(
        columns={
            column: f"ch4a_{label}_{column}"
            for column in metric_columns
        }
    )


def add_ch4a_effects(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add Chapter 4A paired SSM NSE effects."""
    frame = frame.copy()

    required = {
        "ch4a_stl_ssm_nse",
        "ch4a_hps_ssm_nse",
        "ch4a_cgc_ssm_nse",
        "ch4a_hps_pre_ssm_nse",
        "ch4a_cgc_pre_ssm_nse",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Missing Chapter 4A SSM NSE columns: "
            f"{sorted(missing)}"
        )

    frame["ch4a_delta_hps_minus_stl_ssm"] = (
        frame["ch4a_hps_ssm_nse"]
        - frame["ch4a_stl_ssm_nse"]
    )
    frame["ch4a_delta_cgc_minus_stl_ssm"] = (
        frame["ch4a_cgc_ssm_nse"]
        - frame["ch4a_stl_ssm_nse"]
    )
    frame["ch4a_delta_cgc_minus_hps_ssm"] = (
        frame["ch4a_cgc_ssm_nse"]
        - frame["ch4a_hps_ssm_nse"]
    )
    frame["ch4a_delta_hps_pretrain"] = (
        frame["ch4a_hps_pre_ssm_nse"]
        - frame["ch4a_hps_ssm_nse"]
    )
    frame["ch4a_delta_cgc_pretrain"] = (
        frame["ch4a_cgc_pre_ssm_nse"]
        - frame["ch4a_cgc_ssm_nse"]
    )

    return frame


def effect_summary(
    frame: pd.DataFrame,
    experiment: str,
    direction: str,
    data_limitation: str,
    comparison: str,
    column: str,
    hydroclimate_group_name: str = "All",
) -> dict[str, object]:
    """Summarize one paired NSE effect."""
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )
    values = values[
        np.isfinite(values)
    ]

    return {
        "experiment": experiment,
        "direction": direction,
        "data_limitation": data_limitation,
        "hydroclimate_group":
            hydroclimate_group_name,
        "comparison": comparison,
        "n_basins": len(values),
        "median_delta_nse":
            values.median(),
        "mean_delta_nse":
            values.mean(),
        "q25_delta_nse":
            values.quantile(0.25),
        "q75_delta_nse":
            values.quantile(0.75),
        "positive_rate":
            float((values > 0).mean()),
        "negative_rate":
            float((values < 0).mean()),
    }


def comparison_definitions() -> list[tuple[str, str, str, str, str]]:
    """Return Chapter 4A and Chapter 4B comparison definitions."""
    return [
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


def main() -> None:
    """Run the cross-experiment summary workflow."""
    setup_logging()
    args = parse_args()

    ch3_path = resolve_path(
        args.ch3_summary
    )
    ch4b_path = resolve_path(
        args.ch4b_effects
    )
    experiments_root = resolve_path(
        args.experiments_root
    )
    output_dir = resolve_path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cross = load_ch4b(ch4b_path)

    ch3 = load_ch3(ch3_path)
    cross = cross.merge(
        ch3,
        on="gauge_id",
        how="left",
        validate="one_to_one",
    )

    if {
        "aridity",
        "frac_snow",
    }.issubset(cross.columns):
        cross["hydroclimate_group"] = (
            hydroclimate_group(cross)
        )

    for label, experiment in CH4A_MODELS.items():
        metrics = read_ch4a_metric(
            experiments_root,
            experiment,
            label,
        )
        cross = cross.merge(
            metrics,
            on="gauge_id",
            how="left",
            validate="one_to_one",
        )

    cross = add_ch4a_effects(cross)

    if len(cross) != EXPECTED_BASINS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASINS} cross-experiment basins, "
            f"found {len(cross)}."
        )

    basin_output = (
        output_dir
        / "ch3_ch4a_ch4b_cross_experiment_per_basin.csv"
    )
    cross.to_csv(
        basin_output,
        index=False,
    )

    comparisons = comparison_definitions()

    overall_rows = [
        effect_summary(
            cross,
            experiment,
            direction,
            limitation,
            comparison,
            column,
        )
        for (
            experiment,
            direction,
            limitation,
            comparison,
            column,
        ) in comparisons
    ]

    overall_path = (
        output_dir
        / "ch4_cross_experiment_directionality_summary.csv"
    )
    pd.DataFrame(overall_rows).to_csv(
        overall_path,
        index=False,
    )

    if "hydroclimate_group" in cross.columns:
        group_rows: list[dict[str, object]] = []

        for group_name, group in cross.groupby(
            "hydroclimate_group",
            dropna=True,
        ):
            for (
                experiment,
                direction,
                limitation,
                comparison,
                column,
            ) in comparisons:
                group_rows.append(
                    effect_summary(
                        group,
                        experiment,
                        direction,
                        limitation,
                        comparison,
                        column,
                        hydroclimate_group_name=str(group_name),
                    )
                )

        pd.DataFrame(group_rows).to_csv(
            output_dir
            / "ch4_cross_experiment_hydroclimate_summary.csv",
            index=False,
        )

    logger.info(
        "PUB median NSE: STL=%.6f, Hard=%.6f, CGC=%.6f",
        cross["PUB_STL_Q_NSE"].median(),
        cross["PUB_Hard_MTL_Q_NSE"].median(),
        cross["PUB_CGC_Q_NSE"].median(),
    )
    logger.info(
        "Cross-experiment basin table: %s",
        basin_output,
    )
    logger.info(
        "Directionality summary: %s",
        overall_path,
    )
    logger.info(
        "Cross-experiment rows: %d",
        len(cross),
    )


if __name__ == "__main__":
    main()