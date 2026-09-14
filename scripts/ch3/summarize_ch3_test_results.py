#!/usr/bin/env python3
"""
Summarize Chapter 3 independent-test results.

IMPORTANT:
This script reads test_per_basin_metrics.csv ONLY.
Validation results are never used for final paper summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = (
    PROJECT_ROOT /
    "experiments" /
    "formal_ch3_modeling"
)

DEFAULT_OUTPUT_DIR = (
    RESULT_ROOT /
    "06_summary" /
    "test"
)

RUNS = {
    "STL-Q": (
        RESULT_ROOT /
        "01_stl_q" /
        "ch3_stl_q_seed42" /
        "test_per_basin_metrics.csv"
    ),
    "STL-ET": (
        RESULT_ROOT /
        "02_stl_et" /
        "ch3_stl_et_seed42" /
        "test_per_basin_metrics.csv"
    ),
    "Hard-MTL": (
        RESULT_ROOT /
        "03_hard_mtl" /
        "ch3_hard_mtl_seed42" /
        "test_per_basin_metrics.csv"
    ),
    "MMoE": (
        RESULT_ROOT /
        "04_mmoe_mtl" /
        "ch3_mmoe_mtl_seed42" /
        "test_per_basin_metrics.csv"
    ),
    "CGC": (
        RESULT_ROOT /
        "05_cgc_mtl" /
        "ch3_cgc_mtl_seed42" /
        "test_per_basin_metrics.csv"
    ),
}

MODEL_SLUG = {
    # Keep the historical Chapter 3 column convention so that
    # existing analysis/plotting scripts remain compatible.
    "STL-Q": "STL_Q",
    "STL-ET": "STL_ET",
    "Hard-MTL": "Hard_MTL",
    "MMoE": "MMoE",
    "CGC": "CGC",
}

MTL_MODELS = [
    "Hard-MTL",
    "MMoE",
    "CGC",
]

TARGET_BASELINE = {
    "streamflow": "STL-Q",
    "evapotranspiration": "STL-ET",
}

METRICS = [
    "nse",
    "kge",
    "rmse",
    "mae",
    "bias",
    "corr",
]

ID_CANDIDATES = [
    "gauge_id",
    "basin_id",
    "station_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Chapter 3 independent-test results."
        )
    )

    parser.add_argument(
        "--expected-basins",
        type=int,
        default=592,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def detect_id_column() -> str:
    headers: Dict[str, List[str]] = {}

    for model, path in RUNS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{model}: missing test result: {path}"
            )

        headers[model] = list(
            pd.read_csv(path, nrows=0).columns
        )

    for candidate in ID_CANDIDATES:
        if all(
            candidate in columns
            for columns in headers.values()
        ):
            return candidate

    raise RuntimeError(
        "Could not identify a common basin ID column. "
        f"Checked: {ID_CANDIDATES}"
    )


def load_results(
    id_col: str,
    expected_basins: int,
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}

    for model, path in RUNS.items():
        df = pd.read_csv(
            path,
            dtype={id_col: str},
        )

        if len(df) != expected_basins:
            raise RuntimeError(
                f"{model}: expected {expected_basins} "
                f"basins, found {len(df)}."
            )

        if df[id_col].duplicated().any():
            duplicates = (
                df.loc[
                    df[id_col].duplicated(),
                    id_col,
                ]
                .astype(str)
                .tolist()
            )

            raise RuntimeError(
                f"{model}: duplicate basin IDs: "
                f"{duplicates[:10]}"
            )

        results[model] = df

    reference_ids = set(
        results["STL-Q"][id_col]
    )

    for model, df in results.items():
        ids = set(df[id_col])

        if ids != reference_ids:
            missing = sorted(reference_ids - ids)
            extra = sorted(ids - reference_ids)

            raise RuntimeError(
                f"{model}: basin set differs from STL-Q. "
                f"missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )

    return results


def build_performance_summary(
    results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for model, df in results.items():
        row = {
            "model": model,
            "n_basins": len(df),
        }

        for target in [
            "streamflow",
            "evapotranspiration",
        ]:
            for metric in METRICS:
                col = f"{target}_{metric}"

                if col not in df.columns:
                    continue

                x = pd.to_numeric(
                    df[col],
                    errors="coerce",
                ).dropna()

                row[f"{col}_n"] = len(x)
                row[f"{col}_median"] = x.median()
                row[f"{col}_mean"] = x.mean()
                row[f"{col}_q25"] = x.quantile(0.25)
                row[f"{col}_q75"] = x.quantile(0.75)
                row[f"{col}_min"] = x.min()
                row[f"{col}_max"] = x.max()

        rows.append(row)

    return pd.DataFrame(rows)


def build_combined_per_basin(
    results: Dict[str, pd.DataFrame],
    id_col: str,
) -> pd.DataFrame:
    reference = (
        results["STL-Q"][[id_col]]
        .copy()
        .sort_values(id_col)
        .reset_index(drop=True)
    )

    combined = reference

    for model, df in results.items():
        slug = MODEL_SLUG[model]

        metric_cols = []

        for target in [
            "streamflow",
            "evapotranspiration",
        ]:
            for metric in METRICS:
                col = f"{target}_{metric}"

                if col in df.columns:
                    metric_cols.append(col)

        selected = df[
            [id_col] + metric_cols
        ].copy()

        selected = selected.rename(
            columns={
                col: f"{slug}_{col}"
                for col in metric_cols
            }
        )

        combined = combined.merge(
            selected,
            on=id_col,
            how="left",
            validate="one_to_one",
        )

    # Historical Chapter 3 Delta_NSE columns retained for compatibility
    # with audit, sensitivity-analysis, and plotting scripts.
    delta_rules = {
        "Delta_NSE_HardMTL_minus_STLQ": (
            "Hard_MTL_streamflow_nse",
            "STL_Q_streamflow_nse",
        ),
        "Delta_NSE_MMoE_minus_STLQ": (
            "MMoE_streamflow_nse",
            "STL_Q_streamflow_nse",
        ),
        "Delta_NSE_CGC_minus_STLQ": (
            "CGC_streamflow_nse",
            "STL_Q_streamflow_nse",
        ),
        "Delta_NSE_HardMTL_ET_minus_STLET": (
            "Hard_MTL_evapotranspiration_nse",
            "STL_ET_evapotranspiration_nse",
        ),
        "Delta_NSE_MMoE_ET_minus_STLET": (
            "MMoE_evapotranspiration_nse",
            "STL_ET_evapotranspiration_nse",
        ),
        "Delta_NSE_CGC_ET_minus_STLET": (
            "CGC_evapotranspiration_nse",
            "STL_ET_evapotranspiration_nse",
        ),

        # Pairwise CGC-minus-MTL differences used by spatial diagnostics.
        "Delta_NSE_CGC_minus_HardMTL": (
            "CGC_streamflow_nse",
            "Hard_MTL_streamflow_nse",
        ),
        "Delta_NSE_CGC_minus_MMoE": (
            "CGC_streamflow_nse",
            "MMoE_streamflow_nse",
        ),
        "Delta_NSE_CGC_ET_minus_HardMTL": (
            "CGC_evapotranspiration_nse",
            "Hard_MTL_evapotranspiration_nse",
        ),
        "Delta_NSE_CGC_ET_minus_MMoE": (
            "CGC_evapotranspiration_nse",
            "MMoE_evapotranspiration_nse",
        ),
    }

    for delta_col, (model_col, baseline_col) in delta_rules.items():
        if model_col not in combined.columns:
            raise RuntimeError(f"Missing model metric column: {model_col}")
        if baseline_col not in combined.columns:
            raise RuntimeError(f"Missing baseline metric column: {baseline_col}")

        combined[delta_col] = (
            pd.to_numeric(combined[model_col], errors="coerce")
            - pd.to_numeric(combined[baseline_col], errors="coerce")
        )

    return combined


def build_transfer_results(
    results: Dict[str, pd.DataFrame],
    id_col: str,
):
    long_rows = []
    summary_rows = []

    for target, baseline_model in (
        TARGET_BASELINE.items()
    ):
        col = f"{target}_nse"

        baseline = results[
            baseline_model
        ][[id_col, col]].copy()

        baseline = baseline.rename(
            columns={col: "stl_nse"}
        )

        for model in MTL_MODELS:
            if col not in results[model].columns:
                continue

            current = results[
                model
            ][[id_col, col]].copy()

            current = current.rename(
                columns={col: "mtl_nse"}
            )

            paired = baseline.merge(
                current,
                on=id_col,
                how="inner",
                validate="one_to_one",
            )

            paired["delta_nse"] = (
                paired["mtl_nse"]
                - paired["stl_nse"]
            )

            paired["transfer_direction"] = "neutral"

            paired.loc[
                paired["delta_nse"] > 0,
                "transfer_direction",
            ] = "positive"

            paired.loc[
                paired["delta_nse"] < 0,
                "transfer_direction",
            ] = "negative"

            for _, row in paired.iterrows():
                long_rows.append(
                    {
                        id_col: row[id_col],
                        "target": target,
                        "model": model,
                        "baseline_model": (
                            baseline_model
                        ),
                        "stl_nse": row["stl_nse"],
                        "mtl_nse": row["mtl_nse"],
                        "delta_nse": (
                            row["delta_nse"]
                        ),
                        "transfer_direction": (
                            row[
                                "transfer_direction"
                            ]
                        ),
                    }
                )

            delta = paired["delta_nse"]

            summary_rows.append(
                {
                    "target": target,
                    "model": model,
                    "baseline_model": (
                        baseline_model
                    ),
                    "n_basins": len(paired),
                    "median_delta_nse": (
                        delta.median()
                    ),
                    "mean_delta_nse": (
                        delta.mean()
                    ),
                    "q25_delta_nse": (
                        delta.quantile(0.25)
                    ),
                    "q75_delta_nse": (
                        delta.quantile(0.75)
                    ),
                    "positive_rate": (
                        (delta > 0).mean()
                    ),
                    "negative_rate": (
                        (delta < 0).mean()
                    ),
                    "neutral_rate": (
                        (delta == 0).mean()
                    ),
                }
            )

    return (
        pd.DataFrame(long_rows),
        pd.DataFrame(summary_rows),
    )


def print_key_results(
    performance: pd.DataFrame,
    transfer: pd.DataFrame,
) -> None:
    display_cols = [
        "model",
        "n_basins",
        "streamflow_nse_median",
        "streamflow_kge_median",
        "evapotranspiration_nse_median",
        "evapotranspiration_kge_median",
    ]

    display_cols = [
        c for c in display_cols
        if c in performance.columns
    ]

    print("\n" + "=" * 110)
    print(
        "CHAPTER 3 INDEPENDENT TEST "
        "PERFORMANCE SUMMARY"
    )
    print("=" * 110)

    print(
        performance[
            display_cols
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print(
        "CHAPTER 3 TEST-BASED "
        "TRANSFER SUMMARY"
    )
    print("=" * 110)

    transfer_display = transfer[
        [
            "target",
            "model",
            "median_delta_nse",
            "positive_rate",
            "negative_rate",
        ]
    ].copy()

    print(
        transfer_display.to_string(
            index=False
        )
    )


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir

    if not output_dir.is_absolute():
        output_dir = (
            PROJECT_ROOT /
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    id_col = detect_id_column()

    results = load_results(
        id_col=id_col,
        expected_basins=args.expected_basins,
    )

    performance = build_performance_summary(
        results
    )

    combined = build_combined_per_basin(
        results,
        id_col,
    )

    transfer_long, transfer_summary = (
        build_transfer_results(
            results,
            id_col,
        )
    )

    performance_path = (
        output_dir /
        "ch3_test_performance_summary.csv"
    )

    combined_path = (
        output_dir /
        "ch3_per_basin_all_models.csv"
    )

    transfer_long_path = (
        output_dir /
        "ch3_test_transfer_long.csv"
    )

    transfer_summary_path = (
        output_dir /
        "ch3_test_transfer_summary.csv"
    )

    performance.to_csv(
        performance_path,
        index=False,
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    transfer_long.to_csv(
        transfer_long_path,
        index=False,
    )

    transfer_summary.to_csv(
        transfer_summary_path,
        index=False,
    )

    print_key_results(
        performance,
        transfer_summary,
    )

    print("\nSaved formal TEST summaries:")
    print(f"  {performance_path}")
    print(f"  {combined_path}")
    print(f"  {transfer_long_path}")
    print(f"  {transfer_summary_path}")


if __name__ == "__main__":
    main()
