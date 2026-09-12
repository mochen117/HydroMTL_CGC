#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit Chapter 4B PUB result consistency.

The script verifies that all major PUB post-processing tables remain
consistent with the validated formal ensemble per-basin metrics.

Checks include:
    - scenario and basin coverage;
    - basin-wise absolute NSE consistency;
    - PUB alias consistency;
    - paired NSE difference consistency;
    - overall absolute NSE/KGE summaries;
    - overall paired NSE/KGE summaries;
    - hydroclimatic PUB NSE consistency.

This script is validation only and does not modify any result files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENSEMBLE = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/summary"
    / "ch4b_pub_ensemble_per_basin_metrics.csv"
)

DEFAULT_EFFECTS = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/summary"
    / "ch4b_pub_effects.csv"
)

DEFAULT_ABSOLUTE = (
    PROJECT_ROOT
    / "experiments/ch4_summary/teacher_revision"
    / "exp2_primary_q_absolute_nse_kge.csv"
)

DEFAULT_PAIRED = (
    PROJECT_ROOT
    / "experiments/ch4_summary/teacher_revision"
    / "exp2_primary_q_paired_nse_kge.csv"
)

DEFAULT_TRANSFER = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydroclimate_groups"
    / "pub_q_transfer_analysis.csv"
)

EXPECTED_BASINS = 592
TOLERANCE = 1e-10

SCENARIOS = {
    "stl_q": {
        "alias": "PUB_STL_Q_NSE",
        "transfer": "STL_Q_NSE",
        "model": "STL-Q",
    },
    "hps_target_ssm": {
        "alias": "PUB_Hard_MTL_Q_NSE",
        "transfer": "Hard_MTL_Q_NSE",
        "model": "Hard-MTL",
    },
    "cgc_target_ssm": {
        "alias": "PUB_CGC_Q_NSE",
        "transfer": "CGC_Q_NSE",
        "model": "CGC",
    },
}

DELTA_COLUMNS = {
    "delta_nse_hps_minus_stl":
        ("hps_target_ssm", "stl_q"),
    "delta_nse_cgc_minus_stl":
        ("cgc_target_ssm", "stl_q"),
    "delta_nse_cgc_minus_hps":
        ("cgc_target_ssm", "hps_target_ssm"),
}

PAIRED_COMPARISONS = {
    "Hard-MTL - STL-Q":
        ("hps_target_ssm", "stl_q"),
    "CGC - STL-Q":
        ("cgc_target_ssm", "stl_q"),
    "CGC - Hard-MTL":
        ("cgc_target_ssm", "hps_target_ssm"),
}

EXPECTED_GROUP_COUNTS = {
    "Dry": 142,
    "Snow": 168,
    "Wet": 282,
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit Chapter 4B PUB result consistency."
    )
    parser.add_argument(
        "--ensemble",
        type=Path,
        default=DEFAULT_ENSEMBLE,
    )
    parser.add_argument(
        "--effects",
        type=Path,
        default=DEFAULT_EFFECTS,
    )
    parser.add_argument(
        "--absolute-summary",
        type=Path,
        default=DEFAULT_ABSOLUTE,
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        default=DEFAULT_PAIRED,
    )
    parser.add_argument(
        "--transfer",
        type=Path,
        default=DEFAULT_TRANSFER,
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
    """Normalize gauge IDs to eight-character strings."""
    if series.isna().any():
        raise ValueError("Missing gauge_id values detected.")

    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def load_csv(path: Path) -> pd.DataFrame:
    """Load a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        dtype={"gauge_id": str},
    )

    if "gauge_id" in frame.columns:
        frame["gauge_id"] = normalize_gauge_id(
            frame["gauge_id"]
        )

    return frame


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    source: Path,
) -> None:
    """Require expected columns."""
    missing = columns.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {source}: {sorted(missing)}"
        )


def assert_close(
    actual: float,
    expected: float,
    label: str,
    tolerance: float = TOLERANCE,
) -> None:
    """Require two finite scalar values to agree."""
    if not (
        np.isfinite(actual)
        and np.isfinite(expected)
        and abs(actual - expected) <= tolerance
    ):
        raise AssertionError(
            f"{label}: actual={actual}, expected={expected}, "
            f"|diff|={abs(actual - expected)}"
        )


def assert_series_close(
    left: pd.Series,
    right: pd.Series,
    label: str,
    tolerance: float = TOLERANCE,
) -> None:
    """Require paired numeric series to agree."""
    left = pd.to_numeric(
        left,
        errors="coerce",
    ).to_numpy(dtype=float)

    right = pd.to_numeric(
        right,
        errors="coerce",
    ).to_numpy(dtype=float)

    if left.shape != right.shape:
        raise AssertionError(
            f"{label}: shape mismatch "
            f"{left.shape} != {right.shape}"
        )

    valid = np.isfinite(left) & np.isfinite(right)

    if valid.sum() != len(left):
        raise AssertionError(
            f"{label}: non-finite paired values detected."
        )

    maximum = float(
        np.max(np.abs(left - right))
    )

    if maximum > tolerance:
        raise AssertionError(
            f"{label}: max absolute difference "
            f"{maximum:.3e} > {tolerance:.3e}"
        )


def build_ensemble_wide(
    ensemble: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Build one row per basin for a formal PUB metric."""
    return (
        ensemble.pivot(
            index="gauge_id",
            columns="scenario",
            values=metric,
        )
        .reset_index()
    )


def audit_ensemble(
    ensemble: pd.DataFrame,
    source: Path,
) -> None:
    """Audit formal ensemble structure."""
    require_columns(
        ensemble,
        {
            "gauge_id",
            "scenario",
            "fold_id",
            "streamflow_nse",
            "streamflow_kge",
        },
        source,
    )

    if ensemble.duplicated(
        ["scenario", "gauge_id"]
    ).any():
        raise AssertionError(
            "Duplicate scenario-gauge rows detected "
            "in formal ensemble."
        )

    for scenario in SCENARIOS:
        subset = ensemble[
            ensemble["scenario"] == scenario
        ]

        count = subset["gauge_id"].nunique()

        if count != EXPECTED_BASINS:
            raise AssertionError(
                f"{scenario}: expected {EXPECTED_BASINS}, "
                f"found {count}."
            )

    logger.info(
        "PASS ensemble coverage: 3 scenarios x %d basins.",
        EXPECTED_BASINS,
    )


def audit_effects(
    ensemble: pd.DataFrame,
    effects: pd.DataFrame,
    source: Path,
) -> None:
    """Audit PUB effects against formal ensemble NSE."""
    require_columns(
        effects,
        {
            "gauge_id",
            *SCENARIOS,
            *[
                item["alias"]
                for item in SCENARIOS.values()
            ],
            *DELTA_COLUMNS,
        },
        source,
    )

    if len(effects) != EXPECTED_BASINS:
        raise AssertionError(
            f"Effects table has {len(effects)} rows."
        )

    if effects["gauge_id"].duplicated().any():
        raise AssertionError(
            "Duplicate gauge_id values in effects table."
        )

    if any(
        column.startswith("CH3_")
        for column in effects.columns
    ):
        raise AssertionError(
            "CH3 columns detected in PUB-only effects table."
        )

    formal = build_ensemble_wide(
        ensemble,
        "streamflow_nse",
    )

    merged = effects.merge(
        formal,
        on="gauge_id",
        how="inner",
        suffixes=("_effects", "_formal"),
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_BASINS:
        raise AssertionError(
            "Effects/formal basin sets do not match."
        )

    for scenario, info in SCENARIOS.items():
        assert_series_close(
            merged[f"{scenario}_effects"],
            merged[f"{scenario}_formal"],
            f"effects vs formal: {scenario}",
        )

        assert_series_close(
            effects[info["alias"]],
            effects[scenario],
            f"PUB alias: {info['alias']}",
        )

    for delta_column, (left, right) in (
        DELTA_COLUMNS.items()
    ):
        expected = (
            effects[left] - effects[right]
        )

        assert_series_close(
            effects[delta_column],
            expected,
            f"paired delta: {delta_column}",
        )

    logger.info(
        "PASS PUB effects and paired NSE consistency."
    )


def audit_absolute_summary(
    ensemble: pd.DataFrame,
    summary: pd.DataFrame,
    source: Path,
) -> None:
    """Audit overall absolute NSE/KGE summary."""
    require_columns(
        summary,
        {
            "model",
            "median_nse",
            "median_kge",
        },
        source,
    )

    for scenario, info in SCENARIOS.items():
        model = info["model"]

        row = summary[
            summary["model"] == model
        ]

        if len(row) != 1:
            raise AssertionError(
                f"Expected one absolute-summary row for {model}."
            )

        subset = ensemble[
            ensemble["scenario"] == scenario
        ]

        assert_close(
            float(row.iloc[0]["median_nse"]),
            float(subset["streamflow_nse"].median()),
            f"{model} median NSE",
        )
        assert_close(
            float(row.iloc[0]["median_kge"]),
            float(subset["streamflow_kge"].median()),
            f"{model} median KGE",
        )

    logger.info(
        "PASS overall absolute NSE/KGE summary."
    )


def audit_paired_summary(
    ensemble: pd.DataFrame,
    summary: pd.DataFrame,
    source: Path,
) -> None:
    """Audit overall paired NSE/KGE summary."""
    require_columns(
        summary,
        {
            "comparison",
            "metric",
            "n",
            "median_delta",
            "positive_rate",
        },
        source,
    )

    for comparison, (left, right) in (
        PAIRED_COMPARISONS.items()
    ):
        for metric_label, metric_column in (
            ("NSE", "streamflow_nse"),
            ("KGE", "streamflow_kge"),
        ):
            wide = build_ensemble_wide(
                ensemble,
                metric_column,
            )

            delta = pd.to_numeric(
                wide[left] - wide[right],
                errors="coerce",
            )
            delta = delta[np.isfinite(delta)]

            row = summary[
                (summary["comparison"] == comparison)
                & (summary["metric"] == metric_label)
            ]

            if len(row) != 1:
                raise AssertionError(
                    "Missing paired summary row: "
                    f"{comparison}, {metric_label}"
                )

            record = row.iloc[0]

            if int(record["n"]) != len(delta):
                raise AssertionError(
                    f"{comparison} {metric_label}: "
                    "paired sample size mismatch."
                )

            assert_close(
                float(record["median_delta"]),
                float(delta.median()),
                f"{comparison} {metric_label} median delta",
            )
            assert_close(
                float(record["positive_rate"]),
                float((delta > 0).mean()),
                f"{comparison} {metric_label} positive rate",
            )

    logger.info(
        "PASS overall paired NSE/KGE summary."
    )


def audit_transfer(
    ensemble: pd.DataFrame,
    transfer: pd.DataFrame,
    source: Path,
) -> None:
    """Audit hydroclimatic transfer table against formal PUB NSE."""
    required = {
        "gauge_id",
        "hydroclimate_group",
        *[
            item["transfer"]
            for item in SCENARIOS.values()
        ],
    }
    require_columns(
        transfer,
        required,
        source,
    )

    if len(transfer) != EXPECTED_BASINS:
        raise AssertionError(
            f"Transfer table has {len(transfer)} rows."
        )

    formal = build_ensemble_wide(
        ensemble,
        "streamflow_nse",
    )

    merged = transfer.merge(
        formal,
        on="gauge_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_BASINS:
        raise AssertionError(
            "Transfer/formal basin sets do not match."
        )

    for scenario, info in SCENARIOS.items():
        assert_series_close(
            merged[info["transfer"]],
            merged[scenario],
            f"transfer vs formal: {scenario}",
        )

    counts = (
        transfer["hydroclimate_group"]
        .value_counts()
        .to_dict()
    )

    if counts != EXPECTED_GROUP_COUNTS:
        raise AssertionError(
            "Hydroclimate counts differ from expected: "
            f"{counts}"
        )

    logger.info(
        "PASS hydroclimatic PUB table: %s.",
        counts,
    )


def audit_spot_check(
    ensemble: pd.DataFrame,
) -> None:
    """Run an independent basin-level spot check."""
    gauge_id = "06037500"

    expected = {
        "stl_q": -11.90531083455985,
        "hps_target_ssm": -13.23818307342204,
        "cgc_target_ssm": -14.993442428920414,
    }

    subset = ensemble[
        ensemble["gauge_id"] == gauge_id
    ]

    for scenario, expected_value in expected.items():
        row = subset[
            subset["scenario"] == scenario
        ]

        if len(row) != 1:
            raise AssertionError(
                f"Missing spot-check record: "
                f"{gauge_id}, {scenario}"
            )

        assert_close(
            float(row.iloc[0]["streamflow_nse"]),
            expected_value,
            f"spot check {gauge_id} {scenario}",
        )

    logger.info(
        "PASS spot check for basin %s.",
        gauge_id,
    )


def main() -> None:
    """Run all PUB result-consistency checks."""
    setup_logging()
    args = parse_args()

    paths = {
        "ensemble": resolve_path(args.ensemble),
        "effects": resolve_path(args.effects),
        "absolute": resolve_path(
            args.absolute_summary
        ),
        "paired": resolve_path(
            args.paired_summary
        ),
        "transfer": resolve_path(args.transfer),
    }

    ensemble = load_csv(paths["ensemble"])
    effects = load_csv(paths["effects"])
    absolute = load_csv(paths["absolute"])
    paired = load_csv(paths["paired"])
    transfer = load_csv(paths["transfer"])

    audit_ensemble(
        ensemble,
        paths["ensemble"],
    )
    audit_effects(
        ensemble,
        effects,
        paths["effects"],
    )
    audit_absolute_summary(
        ensemble,
        absolute,
        paths["absolute"],
    )
    audit_paired_summary(
        ensemble,
        paired,
        paths["paired"],
    )
    audit_transfer(
        ensemble,
        transfer,
        paths["transfer"],
    )
    audit_spot_check(ensemble)

    old_mixed = (
        PROJECT_ROOT
        / "experiments/ch4_qssm_pub/summary"
        / "ch4b_pub_effects_with_ch3_metadata.csv"
    )

    if old_mixed.exists():
        raise AssertionError(
            f"Deprecated mixed PUB/Ch3 table still exists: "
            f"{old_mixed}"
        )

    logger.info(
        "PASS deprecated mixed PUB/Ch3 table is absent."
    )

    print("\n" + "=" * 72)
    print("Chapter 4B PUB result consistency audit: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()