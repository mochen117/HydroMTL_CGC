#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze overall paired significance for Chapter 4 primary targets.

Experiment 1
    Q-assisted SSM prediction.
    Primary target: SSM.

Experiment 2
    SSM-assisted streamflow prediction under PUB conditions.
    Primary target: Q.

For each experiment and metric (NSE/KGE), the script reports basin-wise
paired differences, bootstrap 95% confidence intervals of the median,
two-sided Wilcoxon signed-rank tests, and Benjamini-Hochberg FDR correction.

Each experiment-metric combination defines an independent FDR family.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.io_utils import normalize_basin_id  # noqa: E402


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/ch4_summary/teacher_revision"
    / "ch4_overall_primary_significance_nse_kge.csv"
)

EXP1_MODELS = {
    "stl": (
        "STL-SSM",
        "experiments/ch4a_formal_stl_ssm_seed42/test_per_basin_metrics.csv",
    ),
    "hard_scratch": (
        "Hard-MTL-Scratch",
        "experiments/ch4a_formal_hps_qssm_seed42/test_per_basin_metrics.csv",
    ),
    "cgc_scratch": (
        "CGC-Scratch",
        "experiments/ch4a_formal_cgc_qssm_seed42/test_per_basin_metrics.csv",
    ),
    "hard_qpre": (
        "Hard-MTL-QPre",
        "experiments/ch4a_formal_hps_qpre_finetune_qssm_seed42/"
        "test_per_basin_metrics.csv",
    ),
    "cgc_qpre": (
        "CGC-QPre",
        "experiments/ch4a_formal_cgc_qpre_finetune_qssm_seed42/"
        "test_per_basin_metrics.csv",
    ),
}

EXP2_ENSEMBLE = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/summary"
    / "ch4b_pub_ensemble_per_basin_metrics.csv"
)

EXP1_COMPARISONS = [
    (
        "hard_scratch_minus_stl",
        "Hard-MTL-Scratch minus STL-SSM",
        "transfer_vs_stl",
        "hard_scratch",
        "stl",
    ),
    (
        "cgc_scratch_minus_stl",
        "CGC-Scratch minus STL-SSM",
        "transfer_vs_stl",
        "cgc_scratch",
        "stl",
    ),
    (
        "cgc_scratch_minus_hard_scratch",
        "CGC-Scratch minus Hard-MTL-Scratch",
        "architecture_scratch",
        "cgc_scratch",
        "hard_scratch",
    ),
    (
        "hard_qpre_minus_stl",
        "Hard-MTL-QPre minus STL-SSM",
        "pretrained_transfer_vs_stl",
        "hard_qpre",
        "stl",
    ),
    (
        "cgc_qpre_minus_stl",
        "CGC-QPre minus STL-SSM",
        "pretrained_transfer_vs_stl",
        "cgc_qpre",
        "stl",
    ),
    (
        "hard_qpre_minus_hard_scratch",
        "Hard-MTL-QPre minus Hard-MTL-Scratch",
        "pretraining_gain",
        "hard_qpre",
        "hard_scratch",
    ),
    (
        "cgc_qpre_minus_cgc_scratch",
        "CGC-QPre minus CGC-Scratch",
        "pretraining_gain",
        "cgc_qpre",
        "cgc_scratch",
    ),
    (
        "cgc_qpre_minus_hard_qpre",
        "CGC-QPre minus Hard-MTL-QPre",
        "architecture_qpre",
        "cgc_qpre",
        "hard_qpre",
    ),
]

EXP2_COMPARISONS = [
    (
        "hard_minus_stl",
        "Hard-MTL-Q minus STL-Q",
        "transfer_vs_stl",
        "hps_target_ssm",
        "stl_q",
    ),
    (
        "cgc_minus_stl",
        "CGC-Q minus STL-Q",
        "transfer_vs_stl",
        "cgc_target_ssm",
        "stl_q",
    ),
    (
        "cgc_minus_hard",
        "CGC-Q minus Hard-MTL-Q",
        "architecture_comparison",
        "cgc_target_ssm",
        "hps_target_ssm",
    ),
]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze overall Chapter 4 primary-task significance."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure compact console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def resolve_path(path: str | Path) -> Path:
    """Resolve repository-relative paths."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize basin identifiers."""
    frame = frame.copy()

    for candidate in (
        "gauge_id",
        "basin_id",
        "gage_id",
        "Unnamed: 0",
        frame.columns[0],
    ):
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


def load_exp1() -> pd.DataFrame:
    """Load Experiment 1 primary SSM NSE/KGE."""
    output: pd.DataFrame | None = None

    for key, (_, relative_path) in EXP1_MODELS.items():
        path = resolve_path(relative_path)

        if not path.exists():
            raise FileNotFoundError(path)

        frame = normalize_frame(
            pd.read_csv(path)
        )

        required = {"ssm_nse", "ssm_kge"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}"
            )

        part = frame[
            ["gauge_id", "ssm_nse", "ssm_kge"]
        ].rename(
            columns={
                "ssm_nse": f"{key}_NSE",
                "ssm_kge": f"{key}_KGE",
            }
        )

        output = (
            part
            if output is None
            else output.merge(
                part,
                on="gauge_id",
                how="inner",
                validate="one_to_one",
            )
        )

    if output is None or len(output) != 592:
        raise RuntimeError(
            f"Expected 592 Experiment 1 basins, "
            f"found {0 if output is None else len(output)}."
        )

    return output


def load_exp2() -> pd.DataFrame:
    """Load Experiment 2 formal PUB Q NSE/KGE."""
    if not EXP2_ENSEMBLE.exists():
        raise FileNotFoundError(EXP2_ENSEMBLE)

    frame = pd.read_csv(
        EXP2_ENSEMBLE,
        dtype={"gauge_id": str},
    )
    frame["gauge_id"] = frame["gauge_id"].map(
        normalize_basin_id
    )

    required = {
        "gauge_id",
        "scenario",
        "streamflow_nse",
        "streamflow_kge",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing PUB columns: {sorted(missing)}"
        )

    scenarios = {
        "stl_q",
        "hps_target_ssm",
        "cgc_target_ssm",
    }
    available = set(frame["scenario"].unique())
    missing_scenarios = scenarios.difference(available)

    if missing_scenarios:
        raise ValueError(
            f"Missing PUB scenarios: {sorted(missing_scenarios)}"
        )

    pieces = []

    for metric, column in (
        ("NSE", "streamflow_nse"),
        ("KGE", "streamflow_kge"),
    ):
        wide = (
            frame.pivot(
                index="gauge_id",
                columns="scenario",
                values=column,
            )
            .reset_index()
        )

        wide = wide.rename(
            columns={
                scenario: f"{scenario}_{metric}"
                for scenario in scenarios
            }
        )
        pieces.append(wide)

    output = pieces[0].merge(
        pieces[1],
        on="gauge_id",
        how="inner",
        validate="one_to_one",
    )

    if len(output) != 592:
        raise RuntimeError(
            f"Expected 592 Experiment 2 basins, found {len(output)}."
        )

    return output


def finite_delta(
    frame: pd.DataFrame,
    lhs: str,
    rhs: str,
) -> np.ndarray:
    """Return finite paired differences."""
    lhs_values = pd.to_numeric(
        frame[lhs],
        errors="coerce",
    ).to_numpy(dtype=float)

    rhs_values = pd.to_numeric(
        frame[rhs],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(lhs_values)
        & np.isfinite(rhs_values)
    )

    return lhs_values[valid] - rhs_values[valid]


def bootstrap_median_ci(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Calculate percentile bootstrap 95% CI of the median."""
    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    medians = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=values.size,
            replace=True,
        )
        medians[index] = np.median(sample)

    return (
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def wilcoxon_pvalue(delta: np.ndarray) -> float:
    """Calculate a two-sided Wilcoxon signed-rank p-value."""
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
    """Apply Benjamini-Hochberg FDR correction."""
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


def summarize_delta(
    delta: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Summarize one paired effect."""
    ci_low, ci_high = bootstrap_median_ci(
        delta,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "n_basins": int(delta.size),
        "median_delta": float(np.median(delta)),
        "mean_delta": float(np.mean(delta)),
        "q25_delta": float(np.quantile(delta, 0.25)),
        "q75_delta": float(np.quantile(delta, 0.75)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "positive_rate": float(np.mean(delta > 0)),
        "negative_rate": float(np.mean(delta < 0)),
        "zero_rate": float(np.mean(delta == 0)),
        "wilcoxon_pvalue": wilcoxon_pvalue(delta),
    }


def analyze_experiment(
    frame: pd.DataFrame,
    experiment: str,
    direction: str,
    target: str,
    comparisons: list[tuple[str, str, str, str, str]],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Analyze NSE and KGE paired effects for one experiment."""
    records = []

    for metric_index, metric in enumerate(("NSE", "KGE")):
        for comparison_index, (
            key,
            label,
            comparison_type,
            lhs,
            rhs,
        ) in enumerate(comparisons):

            lhs_column = f"{lhs}_{metric}"
            rhs_column = f"{rhs}_{metric}"

            delta = finite_delta(
                frame,
                lhs_column,
                rhs_column,
            )

            stats = summarize_delta(
                delta,
                n_bootstrap=n_bootstrap,
                seed=(
                    seed
                    + metric_index * 1000
                    + comparison_index
                ),
            )

            records.append({
                "experiment": experiment,
                "direction": direction,
                "target": target,
                "metric": metric,
                "comparison_key": key,
                "comparison": label,
                "comparison_type": comparison_type,
                **stats,
            })

    return pd.DataFrame(records)


def main() -> None:
    """Run overall Chapter 4 paired significance analysis."""
    setup_logging()
    args = parse_args()

    exp1 = load_exp1()
    exp2 = load_exp2()

    result1 = analyze_experiment(
        exp1,
        experiment="Experiment_1",
        direction="Q_to_SSM",
        target="SSM",
        comparisons=EXP1_COMPARISONS,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    result2 = analyze_experiment(
        exp2,
        experiment="Experiment_2",
        direction="SSM_to_Q",
        target="Q",
        comparisons=EXP2_COMPARISONS,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 10000,
    )

    output = pd.concat(
        [result1, result2],
        ignore_index=True,
    )

    output["ci_excludes_zero"] = (
        (output["ci95_low"] > 0)
        | (output["ci95_high"] < 0)
    )

    output["pvalue_fdr_bh"] = np.nan

    for (_, _), index in output.groupby(
        ["experiment", "metric"]
    ).groups.items():
        output.loc[index, "pvalue_fdr_bh"] = (
            adjust_fdr_bh(
                output.loc[index, "wilcoxon_pvalue"]
            )
        )

    output["significant_fdr_0p05"] = (
        output["pvalue_fdr_bh"] < 0.05
    )
    output["robust_fdr_0p05"] = (
        output["significant_fdr_0p05"]
        & output["ci_excludes_zero"]
    )

    expected = {
        ("Experiment_1", "NSE"): 8,
        ("Experiment_1", "KGE"): 8,
        ("Experiment_2", "NSE"): 3,
        ("Experiment_2", "KGE"): 3,
    }

    for family, expected_count in expected.items():
        actual = len(
            output[
                (output["experiment"] == family[0])
                & (output["metric"] == family[1])
            ]
        )
        if actual != expected_count:
            raise RuntimeError(
                f"{family}: expected {expected_count} tests, "
                f"found {actual}."
            )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved overall significance table: %s",
        output_path,
    )

    for (experiment, metric), group in output.groupby(
        ["experiment", "metric"]
    ):
        logger.info(
            "%s %s: %d tests, %d robust.",
            experiment,
            metric,
            len(group),
            int(group["robust_fdr_0p05"].sum()),
        )


if __name__ == "__main__":
    main()