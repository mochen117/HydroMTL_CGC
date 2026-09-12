#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise hydroclimatic significance analysis for Chapter 4.

Two comparison sets are supported:

q_to_ssm
    Chapter 4 Experiment 1: Q-assisted SSM prediction, including
    scratch-trained and Q-pretrained multitask models.

pub_q
    Chapter 4 Experiment 2: SSM-assisted streamflow prediction
    under PUB conditions.

For each hydroclimatic group and paired comparison, the script reports
basin-wise Delta NSE statistics, a bootstrap 95% confidence interval of
the median, a two-sided Wilcoxon signed-rank p-value, and a
Benjamini-Hochberg FDR-adjusted p-value.

Each script invocation defines one FDR family. Therefore, Experiment 1
and Experiment 2 should be run separately.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


GROUP_ORDER = ["Dry", "Snow", "Wet"]

COMPARISON_SETS = {
    "q_to_ssm": [
        {
            "key": "hard_scratch_minus_stl",
            "label": "Hard-MTL-Scratch minus STL-SSM",
            "type": "transfer_vs_stl",
            "lhs": "Hard_Scratch_SSM_NSE",
            "rhs": "STL_SSM_NSE",
        },
        {
            "key": "cgc_scratch_minus_stl",
            "label": "CGC-Scratch minus STL-SSM",
            "type": "transfer_vs_stl",
            "lhs": "CGC_Scratch_SSM_NSE",
            "rhs": "STL_SSM_NSE",
        },
        {
            "key": "cgc_scratch_minus_hard_scratch",
            "label": "CGC-Scratch minus Hard-MTL-Scratch",
            "type": "architecture_scratch",
            "lhs": "CGC_Scratch_SSM_NSE",
            "rhs": "Hard_Scratch_SSM_NSE",
        },
        {
            "key": "hard_qpre_minus_stl",
            "label": "Hard-MTL-QPre minus STL-SSM",
            "type": "pretrained_transfer_vs_stl",
            "lhs": "Hard_QPre_SSM_NSE",
            "rhs": "STL_SSM_NSE",
        },
        {
            "key": "cgc_qpre_minus_stl",
            "label": "CGC-QPre minus STL-SSM",
            "type": "pretrained_transfer_vs_stl",
            "lhs": "CGC_QPre_SSM_NSE",
            "rhs": "STL_SSM_NSE",
        },
        {
            "key": "hard_qpre_minus_hard_scratch",
            "label": "Hard-MTL-QPre minus Hard-MTL-Scratch",
            "type": "pretraining_gain",
            "lhs": "Hard_QPre_SSM_NSE",
            "rhs": "Hard_Scratch_SSM_NSE",
        },
        {
            "key": "cgc_qpre_minus_cgc_scratch",
            "label": "CGC-QPre minus CGC-Scratch",
            "type": "pretraining_gain",
            "lhs": "CGC_QPre_SSM_NSE",
            "rhs": "CGC_Scratch_SSM_NSE",
        },
        {
            "key": "cgc_qpre_minus_hard_qpre",
            "label": "CGC-QPre minus Hard-MTL-QPre",
            "type": "architecture_qpre",
            "lhs": "CGC_QPre_SSM_NSE",
            "rhs": "Hard_QPre_SSM_NSE",
        },
    ],
    "pub_q": [
        {
            "key": "hard_minus_stl",
            "label": "Hard-MTL-Q minus STL-Q",
            "type": "transfer_vs_stl",
            "lhs": "Hard_MTL_Q_NSE",
            "rhs": "STL_Q_NSE",
        },
        {
            "key": "cgc_minus_stl",
            "label": "CGC-Q minus STL-Q",
            "type": "transfer_vs_stl",
            "lhs": "CGC_Q_NSE",
            "rhs": "STL_Q_NSE",
        },
        {
            "key": "cgc_minus_hard",
            "label": "CGC-Q minus Hard-MTL-Q",
            "type": "architecture_comparison",
            "lhs": "CGC_Q_NSE",
            "rhs": "Hard_MTL_Q_NSE",
        },
    ],
}

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure compact console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chapter 4 hydroclimatic paired significance analysis."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Basin-level hydroclimatic analysis CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output significance CSV.",
    )
    parser.add_argument(
        "--comparison-set",
        required=True,
        choices=sorted(COMPARISON_SETS),
        help="Predefined pairwise-comparison set.",
    )
    parser.add_argument(
        "--group-column",
        default="hydroclimate_group",
        help="Hydroclimatic group column.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def finite_delta(
    frame: pd.DataFrame,
    lhs: str,
    rhs: str,
) -> np.ndarray:
    """Return finite basin-wise paired differences."""
    paired = frame[[lhs, rhs]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid = (
        np.isfinite(paired[lhs].to_numpy(dtype=float))
        & np.isfinite(paired[rhs].to_numpy(dtype=float))
    )
    paired = paired.loc[valid]

    return (
        paired[lhs].to_numpy(dtype=float)
        - paired[rhs].to_numpy(dtype=float)
    )


def bootstrap_median_ci(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calculate a percentile bootstrap CI for the median."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.nan, np.nan

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    rng = np.random.default_rng(seed)
    medians = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=values.size,
            replace=True,
        )
        medians[index] = np.median(sample)

    return (
        float(np.quantile(medians, alpha / 2.0)),
        float(np.quantile(medians, 1.0 - alpha / 2.0)),
    )


def wilcoxon_pvalue(delta: np.ndarray) -> float:
    """Calculate a two-sided paired Wilcoxon signed-rank p-value."""
    delta = np.asarray(delta, dtype=float)

    if delta.size == 0:
        return np.nan

    if np.allclose(delta, 0.0):
        return 1.0

    try:
        result = wilcoxon(
            delta,
            alternative="two-sided",
            zero_method="wilcox",
        )
    except ValueError:
        return np.nan

    return float(result.pvalue)


def adjust_fdr_bh(pvalues: pd.Series) -> pd.Series:
    """Apply the Benjamini-Hochberg FDR correction."""
    adjusted = pd.Series(
        np.nan,
        index=pvalues.index,
        dtype=float,
    )
    valid = pvalues.dropna()

    if valid.empty:
        return adjusted

    values = valid.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]

    n_tests = len(ranked)
    corrected = (
        ranked
        * n_tests
        / np.arange(1, n_tests + 1)
    )
    corrected = np.minimum.accumulate(
        corrected[::-1]
    )[::-1]
    corrected = np.clip(
        corrected,
        0.0,
        1.0,
    )

    restored = np.empty(
        n_tests,
        dtype=float,
    )
    restored[order] = corrected
    adjusted.loc[valid.index] = restored

    return adjusted


def analyze_comparison(
    subset: pd.DataFrame,
    lhs: str,
    rhs: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Calculate paired statistics for one hydroclimatic group."""
    delta = finite_delta(
        subset,
        lhs,
        rhs,
    )

    if delta.size == 0:
        return {
            "n_basins": 0,
            "median_delta_nse": np.nan,
            "mean_delta_nse": np.nan,
            "q25_delta_nse": np.nan,
            "q75_delta_nse": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "positive_rate": np.nan,
            "negative_rate": np.nan,
            "zero_rate": np.nan,
            "wilcoxon_pvalue": np.nan,
        }

    ci_low, ci_high = bootstrap_median_ci(
        delta,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "n_basins": int(delta.size),
        "median_delta_nse":
            float(np.median(delta)),
        "mean_delta_nse":
            float(np.mean(delta)),
        "q25_delta_nse":
            float(np.quantile(delta, 0.25)),
        "q75_delta_nse":
            float(np.quantile(delta, 0.75)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "positive_rate":
            float(np.mean(delta > 0)),
        "negative_rate":
            float(np.mean(delta < 0)),
        "zero_rate":
            float(np.mean(delta == 0)),
        "wilcoxon_pvalue":
            wilcoxon_pvalue(delta),
    }


def validate_input(
    frame: pd.DataFrame,
    comparison_set: str,
    group_column: str,
) -> None:
    """Validate columns and hydroclimatic groups."""
    comparisons = COMPARISON_SETS[
        comparison_set
    ]

    required = {group_column}
    for spec in comparisons:
        required.update(
            [spec["lhs"], spec["rhs"]]
        )

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing input columns for {comparison_set}: "
            f"{sorted(missing)}"
        )

    groups = set(
        frame[group_column]
        .dropna()
        .astype(str)
    )
    missing_groups = set(
        GROUP_ORDER
    ).difference(groups)

    if missing_groups:
        raise ValueError(
            "Missing required hydroclimatic groups: "
            f"{sorted(missing_groups)}"
        )


def main() -> None:
    """Run hydroclimatic significance analysis."""
    setup_logging()
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    logger.info(
        "Loading input: %s",
        args.input,
    )

    frame = pd.read_csv(args.input)

    validate_input(
        frame,
        args.comparison_set,
        args.group_column,
    )

    comparisons = COMPARISON_SETS[
        args.comparison_set
    ]
    records: list[dict[str, object]] = []

    for group_index, group in enumerate(
        GROUP_ORDER
    ):
        subset = frame[
            frame[args.group_column].astype(str)
            == group
        ]

        logger.info(
            "%s: %d basins.",
            group,
            len(subset),
        )

        for comparison_index, spec in enumerate(
            comparisons
        ):
            stats = analyze_comparison(
                subset=subset,
                lhs=spec["lhs"],
                rhs=spec["rhs"],
                n_bootstrap=args.n_bootstrap,
                seed=(
                    args.seed
                    + group_index * 100
                    + comparison_index
                ),
            )

            records.append({
                "hydroclimate_group": group,
                "comparison_key": spec["key"],
                "comparison": spec["label"],
                "comparison_type": spec["type"],
                **stats,
            })

    output = pd.DataFrame(records)

    output["significant_0p05"] = (
        output["wilcoxon_pvalue"] < 0.05
    )
    output["ci_excludes_zero"] = (
        (output["ci95_low"] > 0)
        | (output["ci95_high"] < 0)
    )

    # One script invocation defines one BH-FDR family.
    output["pvalue_fdr_bh"] = adjust_fdr_bh(
        output["wilcoxon_pvalue"]
    )
    output["significant_fdr_0p05"] = (
        output["pvalue_fdr_bh"] < 0.05
    )
    output["robust_fdr_0p05"] = (
        output["significant_fdr_0p05"]
        & output["ci_excludes_zero"]
    )

    expected_tests = (
        len(GROUP_ORDER)
        * len(comparisons)
    )
    if len(output) != expected_tests:
        raise RuntimeError(
            f"Expected {expected_tests} tests, "
            f"found {len(output)}."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        args.output,
        index=False,
    )

    logger.info(
        "Saved significance analysis: %s",
        args.output,
    )
    logger.info(
        "Completed %d tests in one BH-FDR family "
        "using comparison set '%s'.",
        len(output),
        args.comparison_set,
    )


if __name__ == "__main__":
    main()