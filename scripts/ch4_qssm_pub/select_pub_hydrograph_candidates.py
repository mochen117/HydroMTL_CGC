#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select representative PUB basins for hydrograph and flood-process analysis.

The screening is based only on formal Chapter 4B PUB results.

Candidate types
---------------
1. CGC success:
   CGC improves over both STL and Hard-MTL in NSE and KGE.

2. Hard-negative / CGC-recovery:
   Hard-MTL degrades relative to STL, whereas CGC improves over STL
   and outperforms Hard-MTL.

3. Dry improvement:
   Dry basins with positive and consistent CGC improvements.

4. Common failure:
   Basins where all three models show poor absolute NSE.

This script performs basin-level screening only. Flood-event suitability
must be assessed separately from daily observed hydrographs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EFFECTS_PATH = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/summary"
    / "ch4b_pub_effects.csv"
)

ENSEMBLE_PATH = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/summary"
    / "ch4b_pub_ensemble_per_basin_metrics.csv"
)

CLIMATE_PATH = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydroclimate_groups"
    / "pub_q_transfer_analysis.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydrograph_candidates"
)

LOGGER = logging.getLogger(__name__)


SCENARIO_LABELS = {
    "stl_q": "STL",
    "hps_target_ssm": "Hard",
    "cgc_target_ssm": "CGC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative PUB basins for hydrograph analysis."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of candidates retained per category.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def load_effects() -> pd.DataFrame:
    if not EFFECTS_PATH.exists():
        raise FileNotFoundError(EFFECTS_PATH)

    df = pd.read_csv(
        EFFECTS_PATH,
        dtype={"gauge_id": str},
    )
    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    required = {
        "gauge_id",
        "PUB_STL_Q_NSE",
        "PUB_Hard_MTL_Q_NSE",
        "PUB_CGC_Q_NSE",
        "delta_nse_hps_minus_stl",
        "delta_nse_cgc_minus_stl",
        "delta_nse_cgc_minus_hps",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in effects table: {sorted(missing)}"
        )

    if len(df) != 592:
        raise ValueError(
            f"Expected 592 basins in effects table, found {len(df)}."
        )

    return df.copy()


def load_kge() -> pd.DataFrame:
    if not ENSEMBLE_PATH.exists():
        raise FileNotFoundError(ENSEMBLE_PATH)

    df = pd.read_csv(
        ENSEMBLE_PATH,
        dtype={"gauge_id": str},
    )
    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    required = {
        "gauge_id",
        "scenario",
        "streamflow_kge",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in ensemble table: {sorted(missing)}"
        )

    df = df[df["scenario"].isin(SCENARIO_LABELS)].copy()

    wide = df.pivot(
        index="gauge_id",
        columns="scenario",
        values="streamflow_kge",
    ).reset_index()

    wide = wide.rename(
        columns={
            scenario: f"{label}_KGE"
            for scenario, label in SCENARIO_LABELS.items()
        }
    )

    required_wide = {
        "STL_KGE",
        "Hard_KGE",
        "CGC_KGE",
    }
    missing_wide = required_wide.difference(wide.columns)

    if missing_wide:
        raise ValueError(
            f"Missing KGE scenarios: {sorted(missing_wide)}"
        )

    wide["delta_kge_hard_minus_stl"] = (
        wide["Hard_KGE"] - wide["STL_KGE"]
    )
    wide["delta_kge_cgc_minus_stl"] = (
        wide["CGC_KGE"] - wide["STL_KGE"]
    )
    wide["delta_kge_cgc_minus_hard"] = (
        wide["CGC_KGE"] - wide["Hard_KGE"]
    )

    return wide


def load_climate() -> pd.DataFrame:
    if not CLIMATE_PATH.exists():
        raise FileNotFoundError(CLIMATE_PATH)

    df = pd.read_csv(
        CLIMATE_PATH,
        dtype={"gauge_id": str},
    )
    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    required = {
        "gauge_id",
        "hydroclimate_group",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing climate columns: {sorted(missing)}"
        )

    return df[
        ["gauge_id", "hydroclimate_group"]
    ].drop_duplicates()


def percentile_rank(
    series: pd.Series,
    ascending: bool = True,
) -> pd.Series:
    return series.rank(
        pct=True,
        method="average",
        ascending=ascending,
    )


def build_master_table() -> pd.DataFrame:
    effects = load_effects()
    kge = load_kge()
    climate = load_climate()

    df = (
        effects.merge(
            kge,
            on="gauge_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            climate,
            on="gauge_id",
            how="left",
            validate="one_to_one",
        )
    )

    if len(df) != 592:
        raise ValueError(
            f"Expected 592 merged basins, found {len(df)}."
        )

    if df["hydroclimate_group"].isna().any():
        raise ValueError(
            "Missing hydroclimate groups after merge."
        )

    return df


def add_screening_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # High scores indicate stronger and more consistent CGC improvement.
    df["rank_cgc_stl_nse"] = percentile_rank(
        df["delta_nse_cgc_minus_stl"]
    )
    df["rank_cgc_hard_nse"] = percentile_rank(
        df["delta_nse_cgc_minus_hps"]
    )
    df["rank_cgc_stl_kge"] = percentile_rank(
        df["delta_kge_cgc_minus_stl"]
    )
    df["rank_cgc_hard_kge"] = percentile_rank(
        df["delta_kge_cgc_minus_hard"]
    )

    df["cgc_success_score"] = df[
        [
            "rank_cgc_stl_nse",
            "rank_cgc_hard_nse",
            "rank_cgc_stl_kge",
            "rank_cgc_hard_kge",
        ]
    ].mean(axis=1)

    # Recovery score emphasizes a Hard-MTL degradation followed by CGC recovery.
    hard_negative_strength = percentile_rank(
        -df["delta_nse_hps_minus_stl"]
    )

    df["recovery_score"] = pd.concat(
        [
            hard_negative_strength,
            df["rank_cgc_stl_nse"],
            df["rank_cgc_hard_nse"],
            df["rank_cgc_stl_kge"],
        ],
        axis=1,
    ).mean(axis=1)

    # Failure score: increasingly negative absolute NSE across all models.
    df["failure_score"] = -df[
        [
            "PUB_STL_Q_NSE",
            "PUB_Hard_MTL_Q_NSE",
            "PUB_CGC_Q_NSE",
        ]
    ].median(axis=1)

    return df


def select_candidates(
    df: pd.DataFrame,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    common_columns = [
        "gauge_id",
        "hydroclimate_group",
        "PUB_STL_Q_NSE",
        "PUB_Hard_MTL_Q_NSE",
        "PUB_CGC_Q_NSE",
        "delta_nse_hps_minus_stl",
        "delta_nse_cgc_minus_stl",
        "delta_nse_cgc_minus_hps",
        "STL_KGE",
        "Hard_KGE",
        "CGC_KGE",
        "delta_kge_hard_minus_stl",
        "delta_kge_cgc_minus_stl",
        "delta_kge_cgc_minus_hard",
    ]

    results: dict[str, pd.DataFrame] = {}

    # 1. Strong CGC success.
    mask = (
        (df["PUB_CGC_Q_NSE"] > 0.50)
        & (df["delta_nse_cgc_minus_stl"] > 0)
        & (df["delta_nse_cgc_minus_hps"] > 0)
        & (df["delta_kge_cgc_minus_stl"] > 0)
        & (df["delta_kge_cgc_minus_hard"] > 0)
    )
    results["cgc_success"] = (
        df.loc[mask, common_columns + ["cgc_success_score"]]
        .sort_values(
            "cgc_success_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    # 2. Hard negative transfer, CGC recovery.
    mask = (
        (df["PUB_CGC_Q_NSE"] > 0.50)
        & (df["delta_nse_hps_minus_stl"] < 0)
        & (df["delta_nse_cgc_minus_stl"] > 0)
        & (df["delta_nse_cgc_minus_hps"] > 0)
        & (df["delta_kge_cgc_minus_stl"] > 0)
    )
    results["hard_negative_cgc_recovery"] = (
        df.loc[mask, common_columns + ["recovery_score"]]
        .sort_values(
            "recovery_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    # 3. Dry basins: difficult but improved.
    mask = (
        (df["hydroclimate_group"] == "Dry")
        & (df["PUB_CGC_Q_NSE"] > 0)
        & (df["PUB_CGC_Q_NSE"] < 0.70)
        & (df["delta_nse_cgc_minus_stl"] > 0)
        & (df["delta_nse_cgc_minus_hps"] > 0)
        & (df["delta_kge_cgc_minus_stl"] > 0)
    )
    results["dry_improvement"] = (
        df.loc[mask, common_columns + ["cgc_success_score"]]
        .sort_values(
            "cgc_success_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    # 4. Common failures.
    mask = (
        (df["PUB_STL_Q_NSE"] < 0)
        & (df["PUB_Hard_MTL_Q_NSE"] < 0)
        & (df["PUB_CGC_Q_NSE"] < 0)
    )
    results["common_failure"] = (
        df.loc[mask, common_columns + ["failure_score"]]
        .sort_values(
            "failure_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    return results


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = add_screening_scores(
        build_master_table()
    )

    master_path = (
        output_dir
        / "pub_hydrograph_candidate_master.csv"
    )
    df.to_csv(
        master_path,
        index=False,
    )

    LOGGER.info(
        "Saved master screening table: %s",
        master_path,
    )

    selections = select_candidates(
        df,
        top_k=args.top_k,
    )

    for name, table in selections.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(
            path,
            index=False,
        )

        LOGGER.info(
            "%s: %d candidates -> %s",
            name,
            len(table),
            path,
        )

    LOGGER.info(
        "Candidate screening completed for %d basins.",
        len(df),
    )


if __name__ == "__main__":
    main()