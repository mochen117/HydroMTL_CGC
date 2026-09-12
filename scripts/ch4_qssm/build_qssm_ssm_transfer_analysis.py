#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build the basin-level transfer-analysis table for Chapter 4 Experiment 1.

Experiment
----------
Q-assisted SSM prediction.

The table contains five SSM models:
    STL-SSM
    Hard-MTL-Scratch
    CGC-Scratch
    Hard-MTL-QPre
    CGC-QPre

It also contains CAMELS hydroclimatic attributes and the eight
pairwise NSE differences used by the significance analysis.
"""

from pathlib import Path
import logging

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMELS_CLIM_PATH = Path(
    "/home/mochen/hydro_data/camels/camels_us/camels_clim.txt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments/ch4_qssm/hydroclimate_groups"
    / "qssm_ssm_transfer_analysis.csv"
)

EXPECTED_BASINS = 592

MODEL_SPECS = {
    "STL-SSM": {
        "path": PROJECT_ROOT
        / "experiments/ch4a_formal_stl_ssm_seed42"
        / "test_per_basin_metrics.csv",
        "column": "STL_SSM_NSE",
    },
    "Hard-MTL-Scratch": {
        "path": PROJECT_ROOT
        / "experiments/ch4a_formal_hps_qssm_seed42"
        / "test_per_basin_metrics.csv",
        "column": "Hard_Scratch_SSM_NSE",
    },
    "CGC-Scratch": {
        "path": PROJECT_ROOT
        / "experiments/ch4a_formal_cgc_qssm_seed42"
        / "test_per_basin_metrics.csv",
        "column": "CGC_Scratch_SSM_NSE",
    },
    "Hard-MTL-QPre": {
        "path": PROJECT_ROOT
        / "experiments/ch4a_formal_hps_qpre_finetune_qssm_seed42"
        / "test_per_basin_metrics.csv",
        "column": "Hard_QPre_SSM_NSE",
    },
    "CGC-QPre": {
        "path": PROJECT_ROOT
        / "experiments/ch4a_formal_cgc_qpre_finetune_qssm_seed42"
        / "test_per_basin_metrics.csv",
        "column": "CGC_QPre_SSM_NSE",
    },
}

COMPARISONS = {
    "hard_scratch_minus_stl":
        ("Hard_Scratch_SSM_NSE", "STL_SSM_NSE"),
    "cgc_scratch_minus_stl":
        ("CGC_Scratch_SSM_NSE", "STL_SSM_NSE"),
    "cgc_scratch_minus_hard_scratch":
        ("CGC_Scratch_SSM_NSE", "Hard_Scratch_SSM_NSE"),
    "hard_qpre_minus_stl":
        ("Hard_QPre_SSM_NSE", "STL_SSM_NSE"),
    "cgc_qpre_minus_stl":
        ("CGC_QPre_SSM_NSE", "STL_SSM_NSE"),
    "hard_qpre_minus_hard_scratch":
        ("Hard_QPre_SSM_NSE", "Hard_Scratch_SSM_NSE"),
    "cgc_qpre_minus_cgc_scratch":
        ("CGC_QPre_SSM_NSE", "CGC_Scratch_SSM_NSE"),
    "cgc_qpre_minus_hard_qpre":
        ("CGC_QPre_SSM_NSE", "Hard_QPre_SSM_NSE"),
}

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize USGS gauge IDs to eight-character strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    source: str,
) -> None:
    """Validate required columns."""
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def assign_hydroclimate_group(frame: pd.DataFrame) -> pd.Series:
    """Assign mutually exclusive Wet, Dry and Snow groups."""
    aridity = pd.to_numeric(frame["aridity"], errors="coerce")
    frac_snow = pd.to_numeric(frame["frac_snow"], errors="coerce")

    group = pd.Series(pd.NA, index=frame.index, dtype="object")
    snow = frac_snow > 0.20

    group.loc[snow] = "Snow"
    group.loc[(~snow) & (aridity < 1.0)] = "Wet"
    group.loc[(~snow) & (aridity >= 1.0)] = "Dry"

    return group


def load_model_result(path: Path, output_column: str) -> pd.DataFrame:
    """Load basin-level SSM NSE for one formal model result."""
    if not path.exists():
        raise FileNotFoundError(f"Model result not found: {path}")

    frame = pd.read_csv(path)
    require_columns(frame, ["gauge_id", "ssm_nse"], str(path))

    frame = frame[["gauge_id", "ssm_nse"]].copy()
    frame["gauge_id"] = normalize_gauge_id(frame["gauge_id"])
    frame["ssm_nse"] = pd.to_numeric(frame["ssm_nse"], errors="coerce")
    frame = frame.rename(columns={"ssm_nse": output_column})

    if frame["gauge_id"].duplicated().any():
        raise ValueError(f"Duplicated gauge IDs in {path}")

    return frame


def load_climate_attributes() -> pd.DataFrame:
    """Load CAMELS-US hydroclimatic attributes."""
    if not CAMELS_CLIM_PATH.exists():
        raise FileNotFoundError(
            f"CAMELS climate file not found: {CAMELS_CLIM_PATH}"
        )

    climate = pd.read_csv(CAMELS_CLIM_PATH, sep=";")
    require_columns(
        climate,
        ["gauge_id", "aridity", "frac_snow"],
        str(CAMELS_CLIM_PATH),
    )

    climate = climate[["gauge_id", "aridity", "frac_snow"]].copy()
    climate["gauge_id"] = normalize_gauge_id(climate["gauge_id"])
    climate["aridity"] = pd.to_numeric(
        climate["aridity"],
        errors="coerce",
    )
    climate["frac_snow"] = pd.to_numeric(
        climate["frac_snow"],
        errors="coerce",
    )
    climate["hydroclimate_group"] = assign_hydroclimate_group(climate)

    if climate["gauge_id"].duplicated().any():
        raise ValueError("Duplicated gauge IDs in CAMELS climate attributes.")

    return climate


def validate_basin_sets(frames: dict[str, pd.DataFrame]) -> None:
    """Ensure all model results use the same basin set."""
    reference_name = next(iter(frames))
    reference = set(frames[reference_name]["gauge_id"])

    for name, frame in frames.items():
        current = set(frame["gauge_id"])
        if current != reference:
            missing = sorted(reference - current)
            extra = sorted(current - reference)
            raise RuntimeError(
                f"Basin mismatch for {name}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )


def main() -> None:
    setup_logging()

    model_frames = {}

    for model, spec in MODEL_SPECS.items():
        logger.info("Loading %s results", model)
        model_frames[model] = load_model_result(
            spec["path"],
            spec["column"],
        )

    validate_basin_sets(model_frames)

    frame = None
    for model in MODEL_SPECS:
        current = model_frames[model]
        frame = (
            current
            if frame is None
            else frame.merge(
                current,
                on="gauge_id",
                validate="one_to_one",
            )
        )

    logger.info("Loading CAMELS climate attributes")
    frame = frame.merge(
        load_climate_attributes(),
        on="gauge_id",
        how="left",
        validate="one_to_one",
    )

    missing_group = frame["hydroclimate_group"].isna()
    if missing_group.any():
        ids = frame.loc[missing_group, "gauge_id"].tolist()
        raise RuntimeError(
            f"Missing hydroclimate groups for {len(ids)} basins: {ids[:10]}"
        )

    for key, (lhs, rhs) in COMPARISONS.items():
        frame[f"delta_{key}"] = frame[lhs] - frame[rhs]

    if len(frame) != EXPECTED_BASINS:
        logger.warning(
            "Expected %d basins but found %d.",
            EXPECTED_BASINS,
            len(frame),
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)

    logger.info("Saved transfer-analysis table: %s", OUTPUT_PATH)
    logger.info(
        "Hydroclimate groups: %s",
        frame["hydroclimate_group"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()