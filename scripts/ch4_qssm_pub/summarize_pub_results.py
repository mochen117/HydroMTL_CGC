#!/usr/bin/env python3
"""Summarize formal Chapter 4B PUB streamflow effects.

Formal PUB evaluation is target-basin streamflow only.  Target-basin SSM is an
auxiliary training signal in the assisted MTL scenarios and is therefore not
reported as an independent out-of-sample test target over the same period.
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
from mtl_cgc.protocols.ch4_qssm_pub.paths import (  # noqa: E402
    CH3_SUMMARY,
    ENSEMBLE_DIR,
    SUMMARY_DIR,
)


DEFAULT_ENSEMBLE_INDEX = ENSEMBLE_DIR / "ensemble_index.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-index", type=Path, default=DEFAULT_ENSEMBLE_INDEX)
    parser.add_argument("--ch3-summary", type=Path, default=CH3_SUMMARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUMMARY_DIR,
    )
    return parser.parse_args()


def normalize_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for candidate in ("gauge_id", "basin_id", "gage_id", "Unnamed: 0", frame.columns[0]):
        if candidate in frame.columns:
            frame = frame.rename(columns={candidate: "gauge_id"})
            frame["gauge_id"] = frame["gauge_id"].map(normalize_basin_id)
            return frame
    raise ValueError("Cannot identify basin-id column.")


def hydroclimate_group(frame: pd.DataFrame) -> pd.Series:
    """Stein-style Wet / Dry / Snow grouping used for cross-chapter interpretation."""

    group = pd.Series(index=frame.index, dtype="object")
    snow = pd.to_numeric(frame["frac_snow"], errors="coerce") > 0.20
    aridity = pd.to_numeric(frame["aridity"], errors="coerce")
    group.loc[snow] = "Snow"
    group.loc[(~snow) & (aridity < 1.0)] = "Wet"
    group.loc[(~snow) & (aridity >= 1.0)] = "Dry"
    return group


def load_ensemble(index_path: Path) -> pd.DataFrame:
    index = pd.read_csv(index_path)
    frames: list[pd.DataFrame] = []
    for _, row in index.iterrows():
        path = Path(str(row["metrics_csv"]))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        metrics = normalize_id_column(pd.read_csv(path))
        metrics["fold_id"] = int(row["fold_id"])
        metrics["scenario"] = str(row["scenario"])
        metrics["seed_count"] = int(row["seed_count"])
        frames.append(metrics)
    if not frames:
        raise RuntimeError("No ensemble metric files found.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    index_path = (
        args.ensemble_index
        if args.ensemble_index.is_absolute()
        else PROJECT_ROOT / args.ensemble_index
    )
    ch3_path = (
        args.ch3_summary
        if args.ch3_summary.is_absolute()
        else PROJECT_ROOT / args.ch3_summary
    )
    out_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = load_ensemble(index_path)
    if all_metrics.duplicated(["scenario", "gauge_id"]).any():
        dup = all_metrics.loc[
            all_metrics.duplicated(["scenario", "gauge_id"], keep=False),
            ["scenario", "gauge_id", "fold_id"],
        ]
        raise ValueError(f"A target basin appears more than once per scenario:\n{dup.head()}")
    all_metrics.to_csv(out_dir / "ch4b_pub_ensemble_per_basin_metrics.csv", index=False)

    nse = all_metrics.pivot(
        index="gauge_id", columns="scenario", values="streamflow_nse"
    ).reset_index()
    required = {"stl_q", "hps_target_ssm", "cgc_target_ssm"}
    missing = required - set(nse.columns)
    if missing:
        raise ValueError(f"Missing core PUB scenarios: {sorted(missing)}")

    effects = nse.copy()
    effects["delta_nse_hps_minus_stl"] = effects["hps_target_ssm"] - effects["stl_q"]
    effects["delta_nse_cgc_minus_stl"] = effects["cgc_target_ssm"] - effects["stl_q"]
    effects["delta_nse_cgc_minus_hps"] = effects["cgc_target_ssm"] - effects["hps_target_ssm"]
    effects["hps_positive_transfer"] = effects["delta_nse_hps_minus_stl"] > 0
    effects["cgc_positive_transfer"] = effects["delta_nse_cgc_minus_stl"] > 0
    effects["hps_negative_transfer"] = effects["delta_nse_hps_minus_stl"] < 0
    effects["cgc_negative_transfer"] = effects["delta_nse_cgc_minus_stl"] < 0

    fold_lookup = all_metrics.loc[all_metrics["scenario"] == "stl_q", ["gauge_id", "fold_id"]]
    effects = effects.merge(fold_lookup, on="gauge_id", how="left", validate="one_to_one")

    if ch3_path.exists():
        ch3 = normalize_id_column(pd.read_csv(ch3_path))
        keep = [
            col
            for col in (
                "gauge_id",
                "huc_02",
                "aridity",
                "frac_snow",
                "p_seasonality",
                "max_water_content",
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
            )
            if col in ch3.columns
        ]
        ch3 = ch3[keep].drop_duplicates("gauge_id")
        effects = effects.merge(ch3, on="gauge_id", how="left", validate="one_to_one")
        if {"aridity", "frac_snow"}.issubset(effects.columns):
            effects["hydroclimate_group"] = hydroclimate_group(effects)

    effects.to_csv(out_dir / "ch4b_pub_effects_with_ch3_metadata.csv", index=False)

    rows = []
    for model, column in (
        ("Hard-MTL-PUB", "delta_nse_hps_minus_stl"),
        ("CGC-PUB", "delta_nse_cgc_minus_stl"),
        ("CGC-minus-Hard", "delta_nse_cgc_minus_hps"),
    ):
        values = pd.to_numeric(effects[column], errors="coerce").dropna()
        rows.append(
            {
                "comparison": model,
                "n_basins": len(values),
                "median_delta_nse": values.median(),
                "mean_delta_nse": values.mean(),
                "q25_delta_nse": values.quantile(0.25),
                "q75_delta_nse": values.quantile(0.75),
                "positive_rate": float((values > 0).mean()),
                "negative_rate": float((values < 0).mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "ch4b_pub_model_effect_summary.csv", index=False)

    if "hydroclimate_group" in effects.columns:
        group_rows = []
        for group_name, group in effects.groupby("hydroclimate_group", dropna=True):
            for model, column in (
                ("Hard-MTL-PUB", "delta_nse_hps_minus_stl"),
                ("CGC-PUB", "delta_nse_cgc_minus_stl"),
                ("CGC-minus-Hard", "delta_nse_cgc_minus_hps"),
            ):
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                group_rows.append(
                    {
                        "hydroclimate_group": group_name,
                        "comparison": model,
                        "n_basins": len(values),
                        "median_delta_nse": values.median(),
                        "positive_rate": float((values > 0).mean()) if len(values) else np.nan,
                        "negative_rate": float((values < 0).mean()) if len(values) else np.nan,
                    }
                )
        pd.DataFrame(group_rows).to_csv(
            out_dir / "ch4b_pub_hydroclimate_group_summary.csv", index=False
        )

    print(f"PUB summaries exported to: {out_dir}")


if __name__ == "__main__":
    main()
