#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chapter 4B PUB mechanism analysis.

This script analyzes basin-wise PUB streamflow performance without using
hydroclimatic basin grouping. It is designed for the Chapter 4B experiment
with three scenarios:

    1) STL-Q
    2) Hard-MTL-PUB with target-basin SSM supervision
    3) CGC-PUB with target-basin SSM supervision

Main questions
--------------
1. What are the absolute NSE levels of the three models?
2. Does target-basin SSM produce positive transfer under Hard-MTL?
3. Does CGC reduce negative transfer relative to Hard-MTL?
4. Is the CGC gain broadly distributed or driven by a small number of basins?
5. Does the transfer effect depend on the STL-PUB baseline difficulty?

Notes
-----
- Basin grouping variables (aridity, snow fraction, HUC2, etc.) are intentionally
  excluded from this analysis.
- The primary descriptive statistics are median, IQR, bootstrap confidence
  intervals, and positive/negative transfer rates.
- Mean values are retained only as diagnostics because NSE-related distributions
  may contain strong negative tails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean, wilcoxon


DEFAULT_INPUT = Path(
    "experiments/ch4_qssm_pub/summary/ch4b_pub_effects_with_ch3_metadata.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/ch4_qssm_pub/mechanism"
)

MODEL_COLUMNS: Dict[str, str] = {
    "STL-Q": "stl_q",
    "Hard-MTL-PUB": "hps_target_ssm",
    "CGC-PUB": "cgc_target_ssm",
}

EFFECT_COLUMNS: Dict[str, str] = {
    "Hard-MTL minus STL": "delta_nse_hps_minus_stl",
    "CGC minus STL": "delta_nse_cgc_minus_stl",
    "CGC minus Hard-MTL": "delta_nse_cgc_minus_hps",
}


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a user-supplied path relative to the project root."""
    return path if path.is_absolute() else project_root / path


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the median."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, x.size), replace=True)
    medians = np.median(samples, axis=1)
    return (
        float(np.quantile(medians, alpha / 2)),
        float(np.quantile(medians, 1 - alpha / 2)),
    )


def empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted x values and empirical cumulative probabilities."""
    x = np.asarray(values, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def safe_wilcoxon(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test with graceful handling of edge cases."""
    pair = pd.concat([a, b], axis=1).dropna()
    if pair.empty:
        return np.nan, np.nan

    diff = pair.iloc[:, 0] - pair.iloc[:, 1]
    if np.allclose(diff.to_numpy(), 0.0):
        return 0.0, 1.0

    stat, p_value = wilcoxon(
        pair.iloc[:, 0],
        pair.iloc[:, 1],
        alternative="two-sided",
        zero_method="wilcox",
    )
    return float(stat), float(p_value)


def summarize_absolute_performance(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize absolute NSE for each model."""
    records = []

    for model, column in MODEL_COLUMNS.items():
        x = pd.to_numeric(df[column], errors="coerce").dropna()
        ci_low, ci_high = bootstrap_median_ci(
            x.to_numpy(),
            n_boot=n_boot,
            seed=seed,
        )

        records.append(
            {
                "model": model,
                "n_basins": int(x.size),
                "median_nse": float(x.median()),
                "median_ci95_low": ci_low,
                "median_ci95_high": ci_high,
                "q25_nse": float(x.quantile(0.25)),
                "q75_nse": float(x.quantile(0.75)),
                "mean_nse": float(x.mean()),
                "trimmed_mean_5pct": float(trim_mean(x.to_numpy(), 0.05)),
                "min_nse": float(x.min()),
                "max_nse": float(x.max()),
                "negative_nse_rate": float((x < 0).mean()),
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_effects(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize paired NSE effects and transfer rates."""
    records = []

    for comparison, column in EFFECT_COLUMNS.items():
        x = pd.to_numeric(df[column], errors="coerce").dropna()
        ci_low, ci_high = bootstrap_median_ci(
            x.to_numpy(),
            n_boot=n_boot,
            seed=seed,
        )

        records.append(
            {
                "comparison": comparison,
                "n_basins": int(x.size),
                "median_delta_nse": float(x.median()),
                "median_ci95_low": ci_low,
                "median_ci95_high": ci_high,
                "q25_delta_nse": float(x.quantile(0.25)),
                "q75_delta_nse": float(x.quantile(0.75)),
                "mean_delta_nse": float(x.mean()),
                "trimmed_mean_5pct": float(trim_mean(x.to_numpy(), 0.05)),
                "positive_rate": float((x > 0).mean()),
                "negative_rate": float((x < 0).mean()),
                "zero_rate": float(np.isclose(x, 0.0).mean()),
                "delta_lt_minus_0p1_rate": float((x < -0.1).mean()),
                "delta_lt_minus_0p5_rate": float((x < -0.5).mean()),
                "delta_lt_minus_1_rate": float((x < -1.0).mean()),
                "delta_gt_0p1_rate": float((x > 0.1).mean()),
                "min_delta_nse": float(x.min()),
                "max_delta_nse": float(x.max()),
            }
        )

    return pd.DataFrame.from_records(records)


def build_transfer_transition_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify basins by Hard-MTL and CGC transfer signs.

    The four classes are model-result categories, not hydrological basin groups.
    """
    out = df[
        [
            "gauge_id",
            "fold_id",
            "stl_q",
            "hps_target_ssm",
            "cgc_target_ssm",
            "delta_nse_hps_minus_stl",
            "delta_nse_cgc_minus_stl",
            "delta_nse_cgc_minus_hps",
        ]
    ].copy()

    hard = out["delta_nse_hps_minus_stl"]
    cgc = out["delta_nse_cgc_minus_stl"]

    conditions = [
        (hard > 0) & (cgc > 0),
        (hard < 0) & (cgc > 0),
        (hard > 0) & (cgc < 0),
        (hard < 0) & (cgc < 0),
    ]
    labels = [
        "both_positive",
        "hard_negative_cgc_positive",
        "hard_positive_cgc_negative",
        "both_negative",
    ]

    out["transfer_transition"] = np.select(
        conditions,
        labels,
        default="zero_or_tie",
    )

    counts = (
        out["transfer_transition"]
        .value_counts(dropna=False)
        .rename_axis("transfer_transition")
        .reset_index(name="n_basins")
    )
    counts["rate"] = counts["n_basins"] / len(out)

    hard_negative = hard < 0
    recovered = hard_negative & (cgc > 0)

    diagnostics = pd.DataFrame(
        [
            {
                "metric": "hard_negative_basins",
                "value": int(hard_negative.sum()),
            },
            {
                "metric": "hard_negative_rate",
                "value": float(hard_negative.mean()),
            },
            {
                "metric": "hard_negative_recovered_by_cgc_basins",
                "value": int(recovered.sum()),
            },
            {
                "metric": "recovery_rate_conditional_on_hard_negative",
                "value": (
                    float(recovered.sum() / hard_negative.sum())
                    if hard_negative.sum() > 0
                    else np.nan
                ),
            },
            {
                "metric": "cgc_better_than_hard_rate",
                "value": float((out["delta_nse_cgc_minus_hps"] > 0).mean()),
            },
        ]
    )

    return out, pd.concat([counts, diagnostics], ignore_index=True, sort=False)


def summarize_baseline_dependency(
    df: pd.DataFrame,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify how transfer effects vary with STL-PUB baseline NSE."""
    correlation_records = []
    for comparison, effect_col in {
        "Hard-MTL minus STL": "delta_nse_hps_minus_stl",
        "CGC minus STL": "delta_nse_cgc_minus_stl",
        "CGC minus Hard-MTL": "delta_nse_cgc_minus_hps",
    }.items():
        pair = df[["stl_q", effect_col]].dropna()
        rho, p_value = spearmanr(pair["stl_q"], pair[effect_col])
        correlation_records.append(
            {
                "comparison": comparison,
                "n_basins": int(len(pair)),
                "spearman_rho_with_stl_nse": float(rho),
                "p_value": float(p_value),
            }
        )

    valid = df[["gauge_id", "fold_id", "stl_q"] + list(EFFECT_COLUMNS.values())].dropna(
        subset=["stl_q"]
    ).copy()

    # qcut creates equal-count bins and avoids arbitrary NSE thresholds.
    valid["stl_nse_difficulty_bin"] = pd.qcut(
        valid["stl_q"],
        q=n_bins,
        labels=[f"Q{i}" for i in range(1, n_bins + 1)],
        duplicates="drop",
    )

    rows = []
    for bin_name, group in valid.groupby("stl_nse_difficulty_bin", observed=True):
        for comparison, effect_col in EFFECT_COLUMNS.items():
            x = group[effect_col].dropna()
            rows.append(
                {
                    "stl_nse_difficulty_bin": str(bin_name),
                    "comparison": comparison,
                    "n_basins": int(len(x)),
                    "median_stl_nse": float(group["stl_q"].median()),
                    "median_delta_nse": float(x.median()),
                    "positive_rate": float((x > 0).mean()),
                    "negative_rate": float((x < 0).mean()),
                }
            )

    return (
        pd.DataFrame.from_records(correlation_records),
        pd.DataFrame.from_records(rows),
    )


def summarize_fold_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize model performance and paired effects separately by PUB fold."""
    rows = []

    for fold_id, group in df.groupby("fold_id"):
        for model, column in MODEL_COLUMNS.items():
            x = group[column].dropna()
            rows.append(
                {
                    "fold_id": fold_id,
                    "quantity_type": "absolute_nse",
                    "name": model,
                    "n_basins": int(len(x)),
                    "median": float(x.median()),
                    "q25": float(x.quantile(0.25)),
                    "q75": float(x.quantile(0.75)),
                    "positive_rate": np.nan,
                    "negative_rate": float((x < 0).mean()),
                }
            )

        for comparison, column in EFFECT_COLUMNS.items():
            x = group[column].dropna()
            rows.append(
                {
                    "fold_id": fold_id,
                    "quantity_type": "delta_nse",
                    "name": comparison,
                    "n_basins": int(len(x)),
                    "median": float(x.median()),
                    "q25": float(x.quantile(0.25)),
                    "q75": float(x.quantile(0.75)),
                    "positive_rate": float((x > 0).mean()),
                    "negative_rate": float((x < 0).mean()),
                }
            )

    return pd.DataFrame.from_records(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon tests across basin-wise model NSE values."""
    comparisons = [
        ("Hard-MTL-PUB vs STL-Q", "hps_target_ssm", "stl_q"),
        ("CGC-PUB vs STL-Q", "cgc_target_ssm", "stl_q"),
        ("CGC-PUB vs Hard-MTL-PUB", "cgc_target_ssm", "hps_target_ssm"),
    ]

    rows = []
    for name, a_col, b_col in comparisons:
        stat, p_value = safe_wilcoxon(df[a_col], df[b_col])
        rows.append(
            {
                "comparison": name,
                "wilcoxon_statistic": stat,
                "p_value": p_value,
            }
        )

    return pd.DataFrame.from_records(rows)


def export_tail_diagnostics(
    df: pd.DataFrame,
    output_dir: Path,
    n_worst: int,
) -> pd.DataFrame:
    """Export the strongest negative-transfer basins for manual inspection."""
    rows = []
    for comparison, effect_col in EFFECT_COLUMNS.items():
        selected_columns = [
            "gauge_id",
            "fold_id",
            "stl_q",
            "hps_target_ssm",
            "cgc_target_ssm",
            "delta_nse_hps_minus_stl",
            "delta_nse_cgc_minus_stl",
            "delta_nse_cgc_minus_hps",
        ]
        worst = df[selected_columns].nsmallest(n_worst, effect_col).copy()
        worst.insert(0, "comparison", comparison)

        safe_name = (
            comparison.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        worst.to_csv(
            output_dir / f"worst_{n_worst}_{safe_name}.csv",
            index=False,
        )
        rows.append(worst)

    return pd.concat(rows, ignore_index=True)


def plot_absolute_nse_cdf(df: pd.DataFrame, output_dir: Path) -> None:
    """CDF of absolute basin-wise NSE."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    for model, column in MODEL_COLUMNS.items():
        x, y = empirical_cdf(df[column].to_numpy())
        ax.plot(x, y, linewidth=1.6, label=model)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Basin-wise NSE")
    ax.set_ylabel("Cumulative probability")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_dir / "fig_pub_absolute_nse_cdf.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / "fig_pub_absolute_nse_cdf.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta_nse_cdf(df: pd.DataFrame, output_dir: Path) -> None:
    """CDF of paired NSE changes."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    for comparison, column in EFFECT_COLUMNS.items():
        x, y = empirical_cdf(df[column].to_numpy())
        ax.plot(x, y, linewidth=1.6, label=comparison)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Delta$NSE")
    ax.set_ylabel("Cumulative probability")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_dir / "fig_pub_delta_nse_cdf.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / "fig_pub_delta_nse_cdf.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stl_vs_models(df: pd.DataFrame, output_dir: Path) -> None:
    """1:1 basin-wise comparison against STL-Q."""
    for model, column in {
        "Hard-MTL-PUB": "hps_target_ssm",
        "CGC-PUB": "cgc_target_ssm",
    }.items():
        pair = df[["stl_q", column]].dropna()

        fig, ax = plt.subplots(figsize=(4.7, 4.7))
        ax.scatter(pair["stl_q"], pair[column], s=14, alpha=0.65)

        all_values = np.concatenate([pair["stl_q"].to_numpy(), pair[column].to_numpy()])
        lo = float(np.nanpercentile(all_values, 1))
        hi = float(np.nanpercentile(all_values, 99))
        if np.isclose(lo, hi):
            lo, hi = float(np.nanmin(all_values)), float(np.nanmax(all_values))

        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("STL-Q NSE")
        ax.set_ylabel(f"{model} NSE")
        fig.tight_layout()

        safe_name = model.lower().replace("-", "_")
        fig.savefig(
            output_dir / f"fig_pub_1to1_{safe_name}.png",
            dpi=400,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"fig_pub_1to1_{safe_name}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_recovery_quadrant(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot the key mechanism diagnostic:
    x = Hard-MTL minus STL
    y = CGC minus Hard-MTL
    """
    x = df["delta_nse_hps_minus_stl"]
    y = df["delta_nse_cgc_minus_hps"]

    finite = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x[finite], y[finite], s=14, alpha=0.65)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"Hard-MTL transfer: $\Delta$NSE$_{Hard-STL}$")
    ax.set_ylabel(r"CGC structural effect: $\Delta$NSE$_{CGC-Hard}$")
    fig.tight_layout()

    fig.savefig(output_dir / "fig_pub_hard_to_cgc_recovery_quadrant.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / "fig_pub_hard_to_cgc_recovery_quadrant.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_baseline_dependency(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plots of STL-PUB difficulty versus transfer effect."""
    for comparison, effect_col in {
        "Hard-MTL minus STL": "delta_nse_hps_minus_stl",
        "CGC minus STL": "delta_nse_cgc_minus_stl",
    }.items():
        pair = df[["stl_q", effect_col]].dropna()
        rho, p_value = spearmanr(pair["stl_q"], pair[effect_col])

        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        ax.scatter(pair["stl_q"], pair[effect_col], s=14, alpha=0.65)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("STL-Q PUB NSE")
        ax.set_ylabel(r"$\Delta$NSE")
        ax.set_title(rf"Spearman $\rho$={rho:.3f}, p={p_value:.3g}")
        fig.tight_layout()

        safe_name = comparison.lower().replace(" ", "_").replace("-", "_")
        fig.savefig(
            output_dir / f"fig_pub_baseline_dependency_{safe_name}.png",
            dpi=400,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"fig_pub_baseline_dependency_{safe_name}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def write_run_metadata(
    output_dir: Path,
    input_path: Path,
    n_basins: int,
    n_boot: int,
    seed: int,
) -> None:
    """Write a minimal reproducibility record."""
    metadata = {
        "input_file": str(input_path),
        "n_basins": int(n_basins),
        "bootstrap_replicates": int(n_boot),
        "random_seed": int(seed),
        "basin_grouping_used": False,
        "primary_metric": "NSE",
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Chapter 4B PUB streamflow transfer mechanisms "
            "without hydroclimatic basin grouping."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Per-basin PUB effect table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for mechanism-analysis outputs.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap replicates for median confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by bootstrap resampling.",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=20,
        help="Number of strongest negative-transfer basins to export.",
    )
    parser.add_argument(
        "--difficulty-bins",
        type=int,
        default=5,
        help="Number of equal-count STL-NSE difficulty bins.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_path = resolve_path(project_root, args.input)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required = [
        "gauge_id",
        "fold_id",
        *MODEL_COLUMNS.values(),
        *EFFECT_COLUMNS.values(),
    ]
    require_columns(df, required)

    # Keep one row per PUB target basin.
    if df["gauge_id"].duplicated().any():
        duplicated = df.loc[df["gauge_id"].duplicated(keep=False), "gauge_id"].tolist()
        raise ValueError(
            "Duplicate gauge_id values detected in the PUB per-basin table. "
            f"Examples: {duplicated[:10]}"
        )

    absolute_summary = summarize_absolute_performance(
        df,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    effect_summary = summarize_effects(
        df,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    transition_per_basin, transition_summary = build_transfer_transition_table(df)
    baseline_corr, baseline_bins = summarize_baseline_dependency(
        df,
        n_bins=args.difficulty_bins,
    )
    fold_summary = summarize_fold_robustness(df)
    statistical_tests = paired_tests(df)
    tail_table = export_tail_diagnostics(
        df,
        output_dir=output_dir,
        n_worst=args.worst_n,
    )

    absolute_summary.to_csv(output_dir / "pub_absolute_nse_summary.csv", index=False)
    effect_summary.to_csv(output_dir / "pub_effect_summary_extended.csv", index=False)
    transition_per_basin.to_csv(output_dir / "pub_transfer_transition_per_basin.csv", index=False)
    transition_summary.to_csv(output_dir / "pub_transfer_transition_summary.csv", index=False)
    baseline_corr.to_csv(output_dir / "pub_baseline_dependency_spearman.csv", index=False)
    baseline_bins.to_csv(output_dir / "pub_baseline_difficulty_bins.csv", index=False)
    fold_summary.to_csv(output_dir / "pub_fold_robustness_summary.csv", index=False)
    statistical_tests.to_csv(output_dir / "pub_paired_wilcoxon_tests.csv", index=False)
    tail_table.to_csv(output_dir / "pub_extreme_tail_combined.csv", index=False)

    plot_absolute_nse_cdf(df, output_dir)
    plot_delta_nse_cdf(df, output_dir)
    plot_stl_vs_models(df, output_dir)
    plot_recovery_quadrant(df, output_dir)
    plot_baseline_dependency(df, output_dir)

    write_run_metadata(
        output_dir=output_dir,
        input_path=input_path,
        n_basins=len(df),
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    print("=" * 78)
    print("Chapter 4B PUB mechanism analysis completed")
    print("=" * 78)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Basins: {len(df)}")
    print("\nAbsolute NSE summary:")
    print(absolute_summary.to_string(index=False))
    print("\nTransfer-effect summary:")
    print(effect_summary.to_string(index=False))
    print("\nTransfer-transition summary:")
    print(transition_summary.to_string(index=False))
    print("\nPaired Wilcoxon tests:")
    print(statistical_tests.to_string(index=False))


if __name__ == "__main__":
    main()
