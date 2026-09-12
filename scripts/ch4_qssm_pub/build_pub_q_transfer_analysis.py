#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the basin-level transfer-analysis table for Chapter 4 Experiment 2.

The script uses the formal PUB ensemble per-basin metrics as the only source
of streamflow performance. It merges CAMELS-US hydroclimatic attributes and
computes basin-wise NSE differences among STL-Q, Hard-MTL, and CGC.

Experiment:
    SSM-assisted streamflow prediction under PUB conditions.

Target:
    Streamflow (Q)

Formal scenarios:
    stl_q
    hps_target_ssm
    cgc_target_ssm
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/summary"
    / "ch4b_pub_ensemble_per_basin_metrics.csv"
)

CAMELS_CLIM_PATH = Path(
    "/home/mochen/hydro_data/camels/camels_us/camels_clim.txt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydroclimate_groups"
    / "pub_q_transfer_analysis.csv"
)

EXPECTED_BASINS = 592

SCENARIO_MAP = {
    "stl_q": "STL_Q_NSE",
    "hps_target_ssm": "Hard_MTL_Q_NSE",
    "cgc_target_ssm": "CGC_Q_NSE",
}

logger = logging.getLogger(__name__)


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


def assign_hydroclimate_group(
    frame: pd.DataFrame,
) -> pd.Series:
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


def load_pub_metrics() -> pd.DataFrame:
    """Load and pivot formal PUB streamflow NSE metrics."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"PUB ensemble metrics not found: {INPUT_PATH}"
        )

    logger.info("Loading formal PUB metrics: %s", INPUT_PATH)

    frame = pd.read_csv(
        INPUT_PATH,
        dtype={"gauge_id": str},
    )

    require_columns(
        frame,
        {
            "gauge_id",
            "scenario",
            "streamflow_nse",
            "fold_id",
        },
        INPUT_PATH,
    )

    frame["gauge_id"] = normalize_gauge_id(
        frame["gauge_id"]
    )

    required_scenarios = set(SCENARIO_MAP)
    available_scenarios = set(
        frame["scenario"].astype(str).unique()
    )

    missing_scenarios = (
        required_scenarios - available_scenarios
    )
    if missing_scenarios:
        raise ValueError(
            "Missing formal PUB scenarios: "
            f"{sorted(missing_scenarios)}"
        )

    frame = frame[
        frame["scenario"].isin(required_scenarios)
    ].copy()

    duplicates = frame.duplicated(
        ["scenario", "gauge_id"]
    )
    if duplicates.any():
        rows = frame.loc[
            duplicates,
            ["scenario", "gauge_id", "fold_id"],
        ]
        raise ValueError(
            "Duplicated basin-scenario records detected:\n"
            f"{rows.head()}"
        )

    counts = frame.groupby("scenario")[
        "gauge_id"
    ].nunique()

    for scenario in required_scenarios:
        count = int(counts.get(scenario, 0))
        if count != EXPECTED_BASINS:
            raise RuntimeError(
                f"{scenario}: expected {EXPECTED_BASINS} "
                f"basins, found {count}."
            )

    wide = (
        frame.pivot(
            index="gauge_id",
            columns="scenario",
            values="streamflow_nse",
        )
        .reset_index()
        .rename(columns=SCENARIO_MAP)
    )

    expected_columns = {
        "gauge_id",
        *SCENARIO_MAP.values(),
    }
    require_columns(
        wide,
        expected_columns,
        INPUT_PATH,
    )

    if len(wide) != EXPECTED_BASINS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASINS} PUB basins, "
            f"found {len(wide)}."
        )

    logger.info(
        "Loaded %d formal PUB basins.",
        len(wide),
    )

    return wide


def load_climate_attributes() -> pd.DataFrame:
    """Load CAMELS-US aridity and snow-fraction attributes."""
    if not CAMELS_CLIM_PATH.exists():
        raise FileNotFoundError(
            f"CAMELS climate file not found: "
            f"{CAMELS_CLIM_PATH}"
        )

    logger.info(
        "Loading CAMELS climate attributes: %s",
        CAMELS_CLIM_PATH,
    )

    frame = pd.read_csv(
        CAMELS_CLIM_PATH,
        sep=";",
        dtype={"gauge_id": str},
    )

    require_columns(
        frame,
        {
            "gauge_id",
            "aridity",
            "frac_snow",
        },
        CAMELS_CLIM_PATH,
    )

    frame = frame[
        ["gauge_id", "aridity", "frac_snow"]
    ].copy()

    frame["gauge_id"] = normalize_gauge_id(
        frame["gauge_id"]
    )
    frame["aridity"] = pd.to_numeric(
        frame["aridity"],
        errors="coerce",
    )
    frame["frac_snow"] = pd.to_numeric(
        frame["frac_snow"],
        errors="coerce",
    )

    if frame["gauge_id"].duplicated().any():
        raise ValueError(
            "Duplicated gauge_id values in CAMELS climate data."
        )

    frame["hydroclimate_group"] = (
        assign_hydroclimate_group(frame)
    )

    return frame


def build_analysis_table() -> pd.DataFrame:
    """Build the formal PUB basin-level transfer table."""
    frame = load_pub_metrics()

    frame = frame.merge(
        load_climate_attributes(),
        on="gauge_id",
        how="left",
        validate="one_to_one",
    )

    missing_group = frame[
        "hydroclimate_group"
    ].isna()

    if missing_group.any():
        ids = frame.loc[
            missing_group,
            "gauge_id",
        ].tolist()

        raise RuntimeError(
            "Missing hydroclimate groups for "
            f"{len(ids)} basins: {ids[:10]}"
        )

    frame["Delta_NSE_Q_Hard_minus_STL"] = (
        frame["Hard_MTL_Q_NSE"]
        - frame["STL_Q_NSE"]
    )
    frame["Delta_NSE_Q_CGC_minus_STL"] = (
        frame["CGC_Q_NSE"]
        - frame["STL_Q_NSE"]
    )
    frame["Delta_NSE_Q_CGC_minus_Hard"] = (
        frame["CGC_Q_NSE"]
        - frame["Hard_MTL_Q_NSE"]
    )

    return frame


def main() -> None:
    """Run the PUB transfer-table workflow."""
    setup_logging()

    frame = build_analysis_table()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Hydroclimate groups: %s",
        frame["hydroclimate_group"]
        .value_counts()
        .to_dict(),
    )

    logger.info(
        "Median NSE: STL=%.6f, Hard=%.6f, CGC=%.6f",
        frame["STL_Q_NSE"].median(),
        frame["Hard_MTL_Q_NSE"].median(),
        frame["CGC_Q_NSE"].median(),
    )

    logger.info(
        "Saved PUB transfer table: %s",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
