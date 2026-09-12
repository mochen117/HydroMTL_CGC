#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build final comprehensive hydroclimatic tables for Chapter 4.

Experiment 1
------------
Q -> SSM

Models:
    STL-SSM
    Hard-MTL-Scratch
    CGC-Scratch
    Hard-MTL-QPre
    CGC-QPre

Experiment 2
------------
SSM -> Q under PUB conditions

Models:
    STL-Q
    Hard-MTL-Q
    CGC-Q

The final tables integrate:
    - absolute NSE
    - pairwise Delta NSE
    - positive/negative rates
    - bootstrap confidence intervals
    - Wilcoxon p-values
    - Benjamini-Hochberg FDR-adjusted p-values

Outputs
-------
experiments/ch4_summary/

    Table4_hydroclimate_absolute_performance.csv
    Table4_hydroclimate_pairwise_statistics.csv
    Table4_exp1_Q_to_SSM_comprehensive.csv
    Table4_exp2_SSM_to_Q_comprehensive.csv
    Table4_hydroclimate_comprehensive.csv
"""

from pathlib import Path
import logging

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "experiments/ch4_summary"

GROUP_ORDER = ["Dry", "Snow", "Wet"]

EXPERIMENTS = {
    "Experiment_1": {
        "direction": "Q_to_SSM",
        "target": "SSM",
        "absolute_path": (
            PROJECT_ROOT
            / "experiments/ch4_qssm/hydroclimate_groups"
            / "qssm_ssm_absolute_nse_group_summary.csv"
        ),
        "significance_path": (
            PROJECT_ROOT
            / "experiments/ch4_qssm/hydroclimate_groups"
            / "qssm_ssm_significance.csv"
        ),
        "models": {
            "stl": "STL-SSM",
            "hard_scratch": "Hard-MTL-Scratch",
            "cgc_scratch": "CGC-Scratch",
            "hard_qpre": "Hard-MTL-QPre",
            "cgc_qpre": "CGC-QPre",
        },
        "absolute_columns": {
            "median": "median_NSE_SSM",
            "mean": "mean_NSE_SSM",
            "q25": "q25_NSE_SSM",
            "q75": "q75_NSE_SSM",
            "ge_0p50": "NSE_SSM_ge_0p50_rate",
            "ge_0p60": "NSE_SSM_ge_0p60_rate",
        },
    },
    "Experiment_2": {
        "direction": "SSM_to_Q",
        "target": "Q",
        "absolute_path": (
            PROJECT_ROOT
            / "experiments/ch4_qssm_pub/hydroclimate_groups"
            / "pub_q_absolute_nse_group_summary.csv"
        ),
        "significance_path": (
            PROJECT_ROOT
            / "experiments/ch4_qssm_pub/hydroclimate_groups"
            / "pub_q_significance.csv"
        ),
        "models": {
            "stl": "STL-Q",
            "hard": "Hard-MTL-Q",
            "cgc": "CGC-Q",
        },
        "absolute_columns": {
            "median": "median_NSE_Q",
            "mean": "mean_NSE_Q",
            "q25": "q25_NSE_Q",
            "q75": "q75_NSE_Q",
            "ge_0p50": "NSE_Q_ge_0p50_rate",
            "ge_0p60": "NSE_Q_ge_0p60_rate",
        },
    },
}

PAIRWISE_FIELDS = [
    "median_delta_nse",
    "q25_delta_nse",
    "q75_delta_nse",
    "ci95_low",
    "ci95_high",
    "positive_rate",
    "negative_rate",
    "wilcoxon_pvalue",
    "pvalue_fdr_bh",
    "significant_0p05",
    "significant_fdr_0p05",
    "ci_excludes_zero",
    "robust_fdr_0p05",
]

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def require_file(path: Path) -> None:
    """Validate an expected result file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}"
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


def load_absolute_tables() -> pd.DataFrame:
    """Build a standardized long-format absolute NSE table."""
    frames = []

    for experiment, config in EXPERIMENTS.items():
        path = config["absolute_path"]
        frame = pd.read_csv(path)
        cols = config["absolute_columns"]

        required = [
            "hydroclimate_group",
            "model",
            "n_basins",
            *cols.values(),
        ]
        require_columns(frame, required, str(path))

        frame = frame.rename(columns={
            cols["median"]: "median_nse",
            cols["mean"]: "mean_nse",
            cols["q25"]: "q25_nse",
            cols["q75"]: "q75_nse",
            cols["ge_0p50"]: "nse_ge_0p50_rate",
            cols["ge_0p60"]: "nse_ge_0p60_rate",
        })

        model_to_key = {
            label: key
            for key, label in config["models"].items()
        }

        frame["model_key"] = frame["model"].map(model_to_key)

        if frame["model_key"].isna().any():
            unknown = frame.loc[
                frame["model_key"].isna(),
                "model",
            ].unique().tolist()
            raise ValueError(
                f"Unknown models in {path}: {unknown}"
            )

        frame.insert(0, "experiment", experiment)
        frame.insert(1, "direction", config["direction"])
        frame.insert(2, "target", config["target"])

        frames.append(frame)

    output = pd.concat(frames, ignore_index=True)

    output["hydroclimate_group"] = pd.Categorical(
        output["hydroclimate_group"],
        categories=GROUP_ORDER,
        ordered=True,
    )

    return output.sort_values(
        ["experiment", "hydroclimate_group", "model_key"]
    ).reset_index(drop=True)


def load_pairwise_tables() -> pd.DataFrame:
    """Build a standardized long-format pairwise statistics table."""
    frames = []

    required = [
        "hydroclimate_group",
        "comparison_key",
        "comparison",
        "comparison_type",
        "n_basins",
        *PAIRWISE_FIELDS,
    ]

    for experiment, config in EXPERIMENTS.items():
        path = config["significance_path"]
        frame = pd.read_csv(path)

        require_columns(frame, required, str(path))

        frame.insert(0, "experiment", experiment)
        frame.insert(1, "direction", config["direction"])
        frame.insert(2, "target", config["target"])

        frames.append(frame)

    output = pd.concat(frames, ignore_index=True)

    output["hydroclimate_group"] = pd.Categorical(
        output["hydroclimate_group"],
        categories=GROUP_ORDER,
        ordered=True,
    )

    return output.sort_values(
        ["experiment", "hydroclimate_group", "comparison_key"]
    ).reset_index(drop=True)


def get_single_row(
    frame: pd.DataFrame,
    mask: pd.Series,
    description: str,
) -> pd.Series:
    """Return exactly one matching row."""
    subset = frame.loc[mask]

    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one row for {description}, found {len(subset)}."
        )

    return subset.iloc[0]


def build_experiment_wide_table(
    experiment: str,
    absolute: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    """Build one comprehensive row per hydroclimatic group."""
    config = EXPERIMENTS[experiment]

    abs_exp = absolute[
        absolute["experiment"] == experiment
    ]
    pair_exp = pairwise[
        pairwise["experiment"] == experiment
    ]

    comparison_keys = (
        pair_exp["comparison_key"]
        .drop_duplicates()
        .tolist()
    )

    records = []

    for group in GROUP_ORDER:
        group_abs = abs_exp[
            abs_exp["hydroclimate_group"].astype(str) == group
        ]
        group_pair = pair_exp[
            pair_exp["hydroclimate_group"].astype(str) == group
        ]

        if group_abs.empty:
            raise RuntimeError(
                f"No absolute results for {experiment}/{group}."
            )

        record = {
            "experiment": experiment,
            "direction": config["direction"],
            "target": config["target"],
            "hydroclimate_group": group,
            "n_basins": int(group_abs.iloc[0]["n_basins"]),
        }

        for model_key, model_label in config["models"].items():
            row = get_single_row(
                group_abs,
                group_abs["model"] == model_label,
                f"{experiment}/{group}/{model_label}",
            )

            prefix = model_key

            record[f"{prefix}_median_nse"] = row["median_nse"]
            record[f"{prefix}_q25_nse"] = row["q25_nse"]
            record[f"{prefix}_q75_nse"] = row["q75_nse"]
            record[f"{prefix}_nse_ge_0p50_rate"] = (
                row["nse_ge_0p50_rate"]
            )
            record[f"{prefix}_nse_ge_0p60_rate"] = (
                row["nse_ge_0p60_rate"]
            )

        for comparison_key in comparison_keys:
            row = get_single_row(
                group_pair,
                group_pair["comparison_key"] == comparison_key,
                f"{experiment}/{group}/{comparison_key}",
            )

            for field in PAIRWISE_FIELDS:
                record[f"{comparison_key}_{field}"] = row[field]

        records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    setup_logging()

    for config in EXPERIMENTS.values():
        require_file(config["absolute_path"])
        require_file(config["significance_path"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    absolute = load_absolute_tables()
    pairwise = load_pairwise_tables()

    exp1_wide = build_experiment_wide_table(
        "Experiment_1",
        absolute,
        pairwise,
    )
    exp2_wide = build_experiment_wide_table(
        "Experiment_2",
        absolute,
        pairwise,
    )

    comprehensive = pd.concat(
        [exp1_wide, exp2_wide],
        ignore_index=True,
        sort=False,
    )

    absolute_path = (
        OUTPUT_DIR
        / "Table4_hydroclimate_absolute_performance.csv"
    )
    pairwise_path = (
        OUTPUT_DIR
        / "Table4_hydroclimate_pairwise_statistics.csv"
    )
    exp1_path = (
        OUTPUT_DIR
        / "Table4_exp1_Q_to_SSM_comprehensive.csv"
    )
    exp2_path = (
        OUTPUT_DIR
        / "Table4_exp2_SSM_to_Q_comprehensive.csv"
    )
    comprehensive_path = (
        OUTPUT_DIR
        / "Table4_hydroclimate_comprehensive.csv"
    )

    absolute.to_csv(absolute_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    exp1_wide.to_csv(exp1_path, index=False)
    exp2_wide.to_csv(exp2_path, index=False)
    comprehensive.to_csv(comprehensive_path, index=False)

    logger.info("Saved absolute performance table: %s", absolute_path)
    logger.info("Saved pairwise statistics table: %s", pairwise_path)
    logger.info("Saved Experiment 1 table: %s", exp1_path)
    logger.info("Saved Experiment 2 table: %s", exp2_path)
    logger.info(
        "Saved comprehensive Chapter 4 table: %s",
        comprehensive_path,
    )
    logger.info("Chapter 4 hydroclimatic summary completed.")


if __name__ == "__main__":
    main()