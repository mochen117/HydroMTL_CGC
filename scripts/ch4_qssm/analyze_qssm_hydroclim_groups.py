#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hydroclimatic analysis for Chapter 4 Experiment 1.

Experiment
----------
Q-assisted soil surface moisture (SSM) prediction.

Target:
    Soil surface moisture (SSM)

Auxiliary information:
    Streamflow (Q)

Models:
    STL-SSM
    Hard-MTL-Scratch
    CGC-Scratch
    Hard-MTL-QPre
    CGC-QPre

Hydroclimatic classification
----------------------------
Snow:
    frac_snow > 0.20

Wet:
    frac_snow <= 0.20 and aridity < 1.0

Dry:
    frac_snow <= 0.20 and aridity >= 1.0

The analysis reports:
    1. Absolute NSE_SSM for all five models.
    2. Transfer effects relative to STL.
    3. CGC versus Hard-MTL under matched training strategies.
    4. Q-pretraining gains relative to scratch training.
    5. Positive/negative difference rates.

Outputs
-------
experiments/ch4_qssm/hydroclimate_groups/

    qssm_ssm_basin_hydroclimate_metrics.csv
    qssm_ssm_absolute_nse_group_summary.csv
    qssm_ssm_delta_nse_group_summary.csv
    fig_qssm_ssm_absolute_nse.png
    fig_qssm_ssm_positive_transfer_rate.png
"""

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMELS_CLIM_PATH = Path(
    "/home/mochen/hydro_data/camels/camels_us/camels_clim.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/ch4_qssm/hydroclimate_groups"
)

EXPECTED_BASINS = 592
GROUP_ORDER = ["Dry", "Snow", "Wet"]

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
    "hard_scratch_minus_stl": {
        "label": "Hard-MTL-Scratch minus STL-SSM",
        "type": "transfer_vs_stl",
        "lhs": "Hard_Scratch_SSM_NSE",
        "rhs": "STL_SSM_NSE",
    },
    "cgc_scratch_minus_stl": {
        "label": "CGC-Scratch minus STL-SSM",
        "type": "transfer_vs_stl",
        "lhs": "CGC_Scratch_SSM_NSE",
        "rhs": "STL_SSM_NSE",
    },
    "cgc_scratch_minus_hard_scratch": {
        "label": "CGC-Scratch minus Hard-MTL-Scratch",
        "type": "architecture_scratch",
        "lhs": "CGC_Scratch_SSM_NSE",
        "rhs": "Hard_Scratch_SSM_NSE",
    },
    "hard_qpre_minus_stl": {
        "label": "Hard-MTL-QPre minus STL-SSM",
        "type": "pretrained_transfer_vs_stl",
        "lhs": "Hard_QPre_SSM_NSE",
        "rhs": "STL_SSM_NSE",
    },
    "cgc_qpre_minus_stl": {
        "label": "CGC-QPre minus STL-SSM",
        "type": "pretrained_transfer_vs_stl",
        "lhs": "CGC_QPre_SSM_NSE",
        "rhs": "STL_SSM_NSE",
    },
    "hard_qpre_minus_hard_scratch": {
        "label": "Hard-MTL-QPre minus Hard-MTL-Scratch",
        "type": "pretraining_gain",
        "lhs": "Hard_QPre_SSM_NSE",
        "rhs": "Hard_Scratch_SSM_NSE",
    },
    "cgc_qpre_minus_cgc_scratch": {
        "label": "CGC-QPre minus CGC-Scratch",
        "type": "pretraining_gain",
        "lhs": "CGC_QPre_SSM_NSE",
        "rhs": "CGC_Scratch_SSM_NSE",
    },
    "cgc_qpre_minus_hard_qpre": {
        "label": "CGC-QPre minus Hard-MTL-QPre",
        "type": "architecture_qpre",
        "lhs": "CGC_QPre_SSM_NSE",
        "rhs": "Hard_QPre_SSM_NSE",
    },
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
    """Load basin-level SSM NSE for one model."""
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
    """Load CAMELS-US aridity and snow-fraction attributes."""
    if not CAMELS_CLIM_PATH.exists():
        raise FileNotFoundError(
            f"CAMELS climate file not found: {CAMELS_CLIM_PATH}"
        )

    frame = pd.read_csv(CAMELS_CLIM_PATH, sep=";")
    require_columns(
        frame,
        ["gauge_id", "aridity", "frac_snow"],
        str(CAMELS_CLIM_PATH),
    )

    frame = frame[["gauge_id", "aridity", "frac_snow"]].copy()
    frame["gauge_id"] = normalize_gauge_id(frame["gauge_id"])
    frame["aridity"] = pd.to_numeric(frame["aridity"], errors="coerce")
    frame["frac_snow"] = pd.to_numeric(frame["frac_snow"], errors="coerce")
    frame["hydroclimate_group"] = assign_hydroclimate_group(frame)

    if frame["gauge_id"].duplicated().any():
        raise ValueError("Duplicated gauge IDs in CAMELS climate attributes.")

    return frame


def validate_basin_sets(frames: dict[str, pd.DataFrame]) -> None:
    """Ensure all five models contain the same basin set."""
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


def build_analysis_table() -> pd.DataFrame:
    """Merge five model results with hydroclimatic attributes."""
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

    if len(frame) != EXPECTED_BASINS:
        logger.warning(
            "Expected %d basins but found %d.",
            EXPECTED_BASINS,
            len(frame),
        )

    for key, spec in COMPARISONS.items():
        frame[f"delta_{key}"] = frame[spec["lhs"]] - frame[spec["rhs"]]

    logger.info(
        "Hydroclimate groups: %s",
        frame["hydroclimate_group"].value_counts().to_dict(),
    )

    return frame


def summarize_absolute_nse(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize absolute NSE_SSM for all five models."""
    records = []

    for group in GROUP_ORDER:
        subset = frame[frame["hydroclimate_group"] == group]

        for model, spec in MODEL_SPECS.items():
            values = subset[spec["column"]].dropna()

            records.append({
                "hydroclimate_group": group,
                "model": model,
                "n_basins": len(values),
                "median_NSE_SSM": values.median(),
                "mean_NSE_SSM": values.mean(),
                "q25_NSE_SSM": values.quantile(0.25),
                "q75_NSE_SSM": values.quantile(0.75),
                "NSE_SSM_ge_0p50_rate": (values >= 0.50).mean(),
                "NSE_SSM_ge_0p60_rate": (values >= 0.60).mean(),
            })

    return pd.DataFrame(records)


def summarize_pairwise_nse(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize all eight pairwise NSE differences."""
    records = []

    for group in GROUP_ORDER:
        subset = frame[frame["hydroclimate_group"] == group]

        for key, spec in COMPARISONS.items():
            delta = subset[f"delta_{key}"].dropna()

            records.append({
                "hydroclimate_group": group,
                "comparison_key": key,
                "comparison": spec["label"],
                "comparison_type": spec["type"],
                "n_basins": len(delta),
                "median_Delta_NSE_SSM": delta.median(),
                "mean_Delta_NSE_SSM": delta.mean(),
                "q25_Delta_NSE_SSM": delta.quantile(0.25),
                "q75_Delta_NSE_SSM": delta.quantile(0.75),
                "positive_rate": (delta > 0).mean(),
                "negative_rate": (delta < 0).mean(),
                "zero_rate": (delta == 0).mean(),
            })

    return pd.DataFrame(records)


def plot_absolute_nse(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot median absolute NSE_SSM with IQR for all five models."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(GROUP_ORDER), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(MODEL_SPECS))

    for offset, model in zip(offsets, MODEL_SPECS):
        data = (
            summary[summary["model"] == model]
            .set_index("hydroclimate_group")
            .reindex(GROUP_ORDER)
        )

        median = data["median_NSE_SSM"].to_numpy()
        q25 = data["q25_NSE_SSM"].to_numpy()
        q75 = data["q75_NSE_SSM"].to_numpy()

        ax.errorbar(
            x + offset,
            median,
            yerr=np.vstack([median - q25, q75 - median]),
            marker="o",
            linestyle="none",
            capsize=3,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_xlabel("Hydroclimatic group")
    ax.set_ylabel(r"Absolute $NSE_{SSM}$")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_positive_transfer_rate(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot positive-transfer rates for models compared with STL."""
    keys = [
        "hard_scratch_minus_stl",
        "cgc_scratch_minus_stl",
        "hard_qpre_minus_stl",
        "cgc_qpre_minus_stl",
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(GROUP_ORDER), dtype=float)
    offsets = np.linspace(-0.21, 0.21, len(keys))

    for offset, key in zip(offsets, keys):
        data = (
            summary[summary["comparison_key"] == key]
            .set_index("hydroclimate_group")
            .reindex(GROUP_ORDER)
        )

        label = COMPARISONS[key]["label"].split(" minus ")[0]

        ax.plot(
            x + offset,
            data["positive_rate"].to_numpy(),
            marker="o",
            linestyle="none",
            label=label,
        )

    ax.axhline(0.5, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Hydroclimatic group")
    ax.set_ylabel("Positive transfer rate")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = build_analysis_table()
    absolute = summarize_absolute_nse(frame)
    pairwise = summarize_pairwise_nse(frame)

    basin_path = OUTPUT_DIR / "qssm_ssm_basin_hydroclimate_metrics.csv"
    absolute_path = OUTPUT_DIR / "qssm_ssm_absolute_nse_group_summary.csv"
    pairwise_path = OUTPUT_DIR / "qssm_ssm_delta_nse_group_summary.csv"

    frame.to_csv(basin_path, index=False)
    absolute.to_csv(absolute_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)

    plot_absolute_nse(
        absolute,
        OUTPUT_DIR / "fig_qssm_ssm_absolute_nse.png",
    )
    plot_positive_transfer_rate(
        pairwise,
        OUTPUT_DIR / "fig_qssm_ssm_positive_transfer_rate.png",
    )

    logger.info("Saved basin table: %s", basin_path)
    logger.info("Saved absolute NSE_SSM summary: %s", absolute_path)
    logger.info("Saved pairwise NSE summary: %s", pairwise_path)
    logger.info("Q-assisted SSM hydroclimatic analysis completed.")


if __name__ == "__main__":
    main()