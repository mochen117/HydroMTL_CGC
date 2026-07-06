#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Generate publication-quality Chapter 4 figures for controlled data-condition
#   experiments in hydrological multi-task learning.
#
# Analytical design:
#   Each experiment is evaluated from two complementary perspectives:
#       1. Absolute predictive performance (basin-level NSE distributions).
#       2. Paired CGC gain relative to the task-specific STL baseline:
#              Delta NSE = NSE_CGC - NSE_STL
#
#   This separation prevents absolute predictability and model-transfer benefit
#   from being conflated. Q and ET always use task-specific vertical scales;
#   panels share limits only when they display the same task. All box-and-whisker
#   panels use standard Tukey whiskers (1.5 IQR) and derive their limits
#   independently for Q and ET from the displayed whiskers.
#
# Statistical conventions:
#   - Boxes: 25th-75th percentiles.
#   - Center lines: medians.
#   - Whiskers: Tukey rule (1.5 IQR); outliers are not shown.
#   - Paired statistics use the intersection of basin IDs available for CGC and
#     the corresponding STL model under the same experimental condition.
#   - Bootstrap confidence intervals are calculated for paired median gains.
#   - Wilcoxon signed-rank p-values are adjusted within each experiment using
#     the Holm procedure.
#   - Raw NSE values are used for every statistic. Display limits never modify
#     the underlying data.
#
# Main outputs:
#   - fig4_6_climate_consistency_nse.png/pdf
#   - fig4_7_training_length_nse.png/pdf
#   - fig4_8_basin_diversity_nse.png/pdf
#   - fig4_9_condition_effect_summary.png/pdf
#   - ch4_controlled_nse_per_basin.csv
#   - ch4_controlled_paired_gain_per_basin.csv
#   - ch4_controlled_condition_statistics.csv
# ==============================================================================

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - p-values remain unavailable.
    wilcoxon = None


# ==============================================================================
# Global configuration
# ==============================================================================

JOURNAL_DPI = 600
BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 42

# Journal-style dimensions.
# 90 mm single-column width and 180 mm double-column width.
SINGLE_COLUMN_WIDTH = 3.54
DOUBLE_COLUMN_WIDTH = 7.09
EXPERIMENT_FIGSIZE = (DOUBLE_COLUMN_WIDTH, 6.10)
SUMMARY_FIGSIZE = (DOUBLE_COLUMN_WIDTH, 4.95)

# Display limits are derived from raw Tukey (1.5 IQR) whiskers.
MINIMUM_NSE_LOWER = -1.0
MINIMUM_NSE_UPPER = 1.0
BOX_RANGE_PADDING_RATIO = 0.08
TUKEY_WHISKER_IQR = 1.5
NSE_DISPLAY_LOWER = -1.0
NSE_DISPLAY_UPPER = 1.0
MINIMUM_GAIN_LIMIT_Q = 0.12
MINIMUM_GAIN_LIMIT_ET = 0.06

EDGE_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#222222"
MUTED_TEXT_COLOR = "#666666"

COLORS = {
    "streamflow": "#5B99C5",
    "evapotranspiration": "#FAA256",
    "stl": "#BDBDBD",
    "edge": EDGE_COLOR,
    "grid": GRID_COLOR,
}

TASK_CONFIG: Dict[str, Dict[str, str]] = {
    "streamflow": {
        "baseline_model": "STL-Q",
        "label": "Streamflow (Q)",
        "short_label": "Q",
        "color": COLORS["streamflow"],
    },
    "evapotranspiration": {
        "baseline_model": "STL-ET",
        "label": "Evapotranspiration (ET)",
        "short_label": "ET",
        "color": COLORS["evapotranspiration"],
    },
}

EXPERIMENT_CONFIG: Dict[str, Dict[str, object]] = {
    "climate_consistency": {
        "order": ["Low", "Medium", "High"],
        "title": "Train-test climate consistency",
        "xlabel": "Climate-consistency group",
        "output": "fig4_6_climate_consistency_nse.png",
    },
    "training_length": {
        "order": ["1 yr", "3 yr", "5 yr", "7 yr", "10 yr"],
        "title": "Training data length",
        "xlabel": "Training data length",
        "output": "fig4_7_training_length_nse.png",
    },
    "basin_diversity": {
        # The current grouping varies HUC2 regional coverage and basin count
        # simultaneously; therefore, the scientifically precise figure title is
        # "regional coverage" rather than pure hydrologic diversity.
        "order": ["Low", "Medium", "High"],
        "title": "Training-basin regional coverage",
        "xlabel": "Training-basin regional coverage",
        "output": "fig4_8_basin_diversity_nse.png",
    },
}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project input and output paths."""

    project_root: Path
    experiment_dir: Path
    summary_dir: Path
    figure_dir: Path
    per_basin_path: Path
    climate_group_path: Path
    diversity_group_path: Path
    nse_long_path: Path
    paired_gain_path: Path
    statistics_path: Path


# ==============================================================================
# Argument parsing and paths
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-quality Chapter 4 absolute-NSE and paired-gain figures."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="HydroMTL_CGC project root. Defaults to two levels above this script.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional basin-level experiment CSV path.",
    )
    parser.add_argument(
        "--figure-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional figure output directory. The default is a new task-specific-axis "
            "subdirectory that does not overwrite previous figures."
        ),
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=BOOTSTRAP_REPETITIONS,
        help="Number of bootstrap repetitions for median-gain confidence intervals.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ProjectPaths:
    """Resolve project paths from the command line or script location."""
    script_path = Path(__file__).resolve()
    inferred_root = script_path.parents[2] if len(script_path.parents) >= 3 else Path.cwd()
    project_root = (args.project_root or inferred_root).expanduser().resolve()

    experiment_dir = project_root / "experiments" / "formal_ch4_training_experiments"
    summary_dir = experiment_dir / "summary"
    default_figure_dir = (
        experiment_dir / "figures" / "publication_adaptive_focus_v5"
    )
    figure_dir = (
        args.figure_output_dir.expanduser().resolve()
        if args.figure_output_dir is not None
        else default_figure_dir
    )
    per_basin_path = (
        args.input.expanduser().resolve()
        if args.input is not None
        else summary_dir / "ch4_training_experiment_per_basin.csv"
    )

    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=project_root,
        experiment_dir=experiment_dir,
        summary_dir=summary_dir,
        figure_dir=figure_dir,
        per_basin_path=per_basin_path,
        climate_group_path=summary_dir / "ch4_climate_consistency_groups.csv",
        diversity_group_path=summary_dir / "ch4_basin_diversity_groups.csv",
        nse_long_path=summary_dir / "ch4_controlled_nse_per_basin.csv",
        paired_gain_path=summary_dir / "ch4_controlled_paired_gain_per_basin.csv",
        statistics_path=summary_dir / "ch4_controlled_condition_statistics.csv",
    )


# ==============================================================================
# Plot styling
# ==============================================================================


def choose_serif_font() -> str:
    """Choose an available serif font for publication figures."""
    candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Serif"


def configure_matplotlib() -> None:
    """Configure a thesis- and journal-style Matplotlib theme."""
    font_name = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_name],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "axes.edgecolor": EDGE_COLOR,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "savefig.dpi": JOURNAL_DPI,
            "figure.dpi": 150,
        }
    )
    print(f"[Info] Figure font: {font_name}")


def style_axis(ax: Axes, grid_axis: str = "y") -> None:
    """Apply consistent axis and grid formatting."""
    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.5,
        color=GRID_COLOR,
        alpha=0.65,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save one figure as high-resolution PNG and editable vector PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=JOURNAL_DPI, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[Saved] {path}")
    print(f"[Saved] {path.with_suffix('.pdf')}")


# ==============================================================================
# Data standardization
# ==============================================================================


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def pick_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    """Pick the first case-insensitive column match."""
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise KeyError(
            f"Missing required column. Candidates: {list(candidates)}. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


def normalize_basin_id(value: object) -> str:
    """Normalize basin identifiers as eight-character strings."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def normalize_model_name(value: object) -> str:
    """Normalize model labels."""
    text = str(value).strip()
    lower = text.lower().replace("-", "_")

    if lower in {"stl_q", "stlq"}:
        return "STL-Q"
    if lower in {"stl_et", "stlet"}:
        return "STL-ET"
    if lower == "stl":
        # Ambiguous STL labels should not be silently assigned to both tasks.
        return "STL"
    if lower == "cgc":
        return "CGC"
    return text


def normalize_experiment_type(value: object) -> str:
    """Normalize controlled-experiment names."""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    if "climate" in text or "consistency" in text:
        return "climate_consistency"
    if "length" in text or ("train" in text and ("yr" in text or "year" in text)):
        return "training_length"
    if "diversity" in text or "basin" in text or "coverage" in text:
        return "basin_diversity"
    return text


def normalize_level(value: object) -> str:
    """Normalize condition-level labels."""
    text = str(value).strip()
    lower = text.lower().replace("-", "_").replace(" ", "_")

    mapping = {
        "low": "Low",
        "medium": "Medium",
        "mid": "Medium",
        "high": "High",
        "train_1yr": "1 yr",
        "train_3yr": "3 yr",
        "train_5yr": "5 yr",
        "train_7yr": "7 yr",
        "train_10yr": "10 yr",
        "1yr": "1 yr",
        "3yr": "3 yr",
        "5yr": "5 yr",
        "7yr": "7 yr",
        "10yr": "10 yr",
        "1_year": "1 yr",
        "3_year": "3 yr",
        "5_year": "5 yr",
        "7_year": "7 yr",
        "10_year": "10 yr",
    }
    return mapping.get(lower, text)


def load_per_basin_metrics(paths: ProjectPaths) -> pd.DataFrame:
    """Load and standardize basin-level experiment metrics."""
    require_file(paths.per_basin_path)
    raw = pd.read_csv(paths.per_basin_path)

    experiment_col = pick_column(raw, ["experiment_type", "condition_type", "experiment"])
    level_col = pick_column(raw, ["group_name", "level", "condition_level", "category"])
    model_col = pick_column(raw, ["model_name", "model", "architecture"])
    basin_col = pick_column(raw, ["gauge_id", "gage_id", "basin_id", "station_id", "site_no"])

    q_col = pick_column(
        raw,
        ["streamflow_nse", "streamflow_nse_median", "q_nse", "Val_Q_NSE_Median"],
        required=False,
    )
    et_col = pick_column(
        raw,
        [
            "evapotranspiration_nse",
            "evapotranspiration_nse_median",
            "et_nse",
            "Val_ET_NSE_Median",
        ],
        required=False,
    )
    if q_col is None and et_col is None:
        raise KeyError("No streamflow or evapotranspiration NSE column was found.")

    output = pd.DataFrame(
        {
            "experiment_type": raw[experiment_col].map(normalize_experiment_type),
            "level": raw[level_col].map(normalize_level),
            "model": raw[model_col].map(normalize_model_name),
            "basin_id": raw[basin_col].map(normalize_basin_id),
            "streamflow_nse": (
                pd.to_numeric(raw[q_col], errors="coerce") if q_col is not None else np.nan
            ),
            "evapotranspiration_nse": (
                pd.to_numeric(raw[et_col], errors="coerce") if et_col is not None else np.nan
            ),
        }
    )

    output = output.replace([np.inf, -np.inf], np.nan)
    output = output.dropna(subset=["experiment_type", "level", "model", "basin_id"])
    output = output[output["model"].isin(["CGC", "STL-Q", "STL-ET"])].copy()

    if output.empty:
        raise ValueError(f"No valid Chapter 4 records were found in {paths.per_basin_path}.")

    print(f"[Info] Input table: {paths.per_basin_path}")
    print(f"[Info] Standardized records: {len(output)}")
    return output


def prepare_nse_long(frame: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    """Convert absolute NSE values to one tidy task-wise table."""
    q_frame = frame[frame["model"].isin(["CGC", "STL-Q"])].dropna(
        subset=["streamflow_nse"]
    ).copy()
    q_frame["task"] = "streamflow"
    q_frame["nse"] = q_frame["streamflow_nse"]

    et_frame = frame[frame["model"].isin(["CGC", "STL-ET"])].dropna(
        subset=["evapotranspiration_nse"]
    ).copy()
    et_frame["task"] = "evapotranspiration"
    et_frame["nse"] = et_frame["evapotranspiration_nse"]

    columns = ["experiment_type", "level", "basin_id", "task", "model", "nse"]
    output = pd.concat([q_frame[columns], et_frame[columns]], ignore_index=True)
    output.to_csv(paths.nse_long_path, index=False)
    print(f"[Saved] {paths.nse_long_path}")
    return output


def build_paired_gain_table(nse_long: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    """Build basin-matched CGC-minus-STL differences for every condition."""
    records: List[pd.DataFrame] = []

    for task, task_info in TASK_CONFIG.items():
        baseline_model = task_info["baseline_model"]
        subset = nse_long[
            (nse_long["task"] == task)
            & (nse_long["model"].isin([baseline_model, "CGC"]))
        ].copy()

        pivot = subset.pivot_table(
            index=["experiment_type", "level", "basin_id", "task"],
            columns="model",
            values="nse",
            aggfunc="mean",
        ).reset_index()

        if baseline_model not in pivot.columns or "CGC" not in pivot.columns:
            continue

        paired = pivot.dropna(subset=[baseline_model, "CGC"]).copy()
        paired = paired.rename(
            columns={baseline_model: "stl_nse", "CGC": "cgc_nse"}
        )
        paired["baseline_model"] = baseline_model
        paired["delta_nse"] = paired["cgc_nse"] - paired["stl_nse"]
        records.append(paired)

    if not records:
        raise ValueError("No valid basin-matched CGC-STL pairs were found.")

    output = pd.concat(records, ignore_index=True)
    output.to_csv(paths.paired_gain_path, index=False)
    print(f"[Saved] {paths.paired_gain_path}")
    return output


# ==============================================================================
# Statistical summaries
# ==============================================================================


def bootstrap_median_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Estimate a percentile-bootstrap confidence interval for the median."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    medians = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = rng.choice(array, size=array.size, replace=True)
        medians[index] = np.median(sample)

    alpha = 1.0 - confidence
    return (
        float(np.quantile(medians, alpha / 2.0)),
        float(np.quantile(medians, 1.0 - alpha / 2.0)),
    )


def rank_biserial_effect(values: np.ndarray) -> float:
    """Compute matched-pairs rank-biserial effect size."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array != 0.0)]
    if array.size == 0:
        return 0.0

    ranks = pd.Series(np.abs(array)).rank(method="average").to_numpy(dtype=float)
    positive_sum = float(ranks[array > 0.0].sum())
    negative_sum = float(ranks[array < 0.0].sum())
    denominator = positive_sum + negative_sum
    return 0.0 if denominator == 0.0 else (positive_sum - negative_sum) / denominator


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Apply Holm step-down adjustment while preserving NaN positions."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.where(np.isfinite(values))[0]
    if valid_indices.size == 0:
        return adjusted

    valid_values = values[valid_indices]
    order = np.argsort(valid_values)
    ordered = valid_values[order]
    m = len(ordered)

    ordered_adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, p_value in enumerate(ordered):
        candidate = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, candidate)
        ordered_adjusted[rank] = running_max

    inverse_order = np.empty(m, dtype=int)
    inverse_order[order] = np.arange(m)
    adjusted[valid_indices] = ordered_adjusted[inverse_order]
    return adjusted


def quantiles(values: np.ndarray) -> Dict[str, float]:
    """Return selected distribution quantiles."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: np.nan for key in ["p05", "p25", "p50", "p75", "p95"]}

    q05, q25, q50, q75, q95 = np.quantile(array, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "p05": float(q05),
        "p25": float(q25),
        "p50": float(q50),
        "p75": float(q75),
        "p95": float(q95),
    }


def compute_condition_statistics(
    paired: pd.DataFrame,
    paths: ProjectPaths,
    bootstrap_repetitions: int,
) -> pd.DataFrame:
    """Compute absolute and paired statistics for every experiment-condition-task."""
    records: List[Dict[str, object]] = []

    grouped = paired.groupby(["experiment_type", "level", "task"], sort=False)
    for group_index, ((experiment_type, level, task), group) in enumerate(grouped):
        stl = group["stl_nse"].to_numpy(dtype=float)
        cgc = group["cgc_nse"].to_numpy(dtype=float)
        delta = group["delta_nse"].to_numpy(dtype=float)

        stl_q = quantiles(stl)
        cgc_q = quantiles(cgc)
        delta_q = quantiles(delta)
        ci_low, ci_high = bootstrap_median_ci(
            delta,
            repetitions=bootstrap_repetitions,
            seed=RANDOM_SEED + group_index,
        )

        p_value = np.nan
        if wilcoxon is not None:
            try:
                p_value = float(wilcoxon(delta, zero_method="wilcox").pvalue)
            except ValueError:
                p_value = np.nan

        records.append(
            {
                "experiment_type": experiment_type,
                "level": level,
                "task": task,
                "baseline_model": TASK_CONFIG[task]["baseline_model"],
                "n_pairs": int(len(group)),
                "stl_median_nse": stl_q["p50"],
                "cgc_median_nse": cgc_q["p50"],
                "stl_p25": stl_q["p25"],
                "stl_p75": stl_q["p75"],
                "cgc_p25": cgc_q["p25"],
                "cgc_p75": cgc_q["p75"],
                "paired_median_delta_nse": delta_q["p50"],
                "paired_delta_p05": delta_q["p05"],
                "paired_delta_p25": delta_q["p25"],
                "paired_delta_p75": delta_q["p75"],
                "paired_delta_p95": delta_q["p95"],
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "win_rate": float(np.mean(delta > 0.0) * 100.0),
                "loss_rate": float(np.mean(delta < 0.0) * 100.0),
                "wilcoxon_p": p_value,
                "rank_biserial": rank_biserial_effect(delta),
                "stl_below_minus_one_rate": float(np.mean(stl < -1.0) * 100.0),
                "cgc_below_minus_one_rate": float(np.mean(cgc < -1.0) * 100.0),
            }
        )

    summary = pd.DataFrame(records)
    summary["wilcoxon_p_holm"] = np.nan
    for experiment_type, index in summary.groupby("experiment_type").groups.items():
        summary.loc[index, "wilcoxon_p_holm"] = holm_adjust(
            summary.loc[index, "wilcoxon_p"].to_numpy(dtype=float)
        )

    summary.to_csv(paths.statistics_path, index=False)
    print(f"[Saved] {paths.statistics_path}")
    return summary


# ==============================================================================
# Optional group metadata
# ==============================================================================


def load_group_metadata(paths: ProjectPaths) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Load optional climate and HUC2 group metadata for informative x labels."""
    metadata: Dict[Tuple[str, str], Dict[str, object]] = {}

    if paths.climate_group_path.exists():
        climate = pd.read_csv(paths.climate_group_path, dtype={"gauge_id": str})
        group_col = pick_column(
            climate,
            ["consistency_group", "group", "level"],
            required=False,
        )
        similarity_col = pick_column(
            climate,
            ["climate_similarity", "similarity"],
            required=False,
        )
        if group_col is not None:
            climate["level"] = climate[group_col].map(normalize_level)
            for level, group in climate.groupby("level"):
                item: Dict[str, object] = {"group_basin_count": int(group["gauge_id"].nunique())}
                if similarity_col is not None:
                    item["median_climate_similarity"] = float(
                        pd.to_numeric(group[similarity_col], errors="coerce").median()
                    )
                metadata[("climate_consistency", level)] = item

    if paths.diversity_group_path.exists():
        diversity = pd.read_csv(paths.diversity_group_path, dtype={"gauge_id": str, "huc_02": str})
        group_col = pick_column(
            diversity,
            ["diversity_group", "group", "level"],
            required=False,
        )
        if group_col is not None:
            diversity["level"] = diversity[group_col].map(normalize_level)
            for level, group in diversity.groupby("level"):
                metadata[("basin_diversity", level)] = {
                    "group_basin_count": int(group["gauge_id"].nunique()),
                    "n_huc2_regions": int(group["huc_02"].nunique()),
                }

    return metadata


def build_level_label(
    experiment_type: str,
    level: str,
    n_pairs: int,
    metadata: Mapping[Tuple[str, str], Mapping[str, object]],
) -> str:
    """Build a concise and unambiguous condition label."""
    lines = [level]
    item = metadata.get((experiment_type, level), {})

    if experiment_type == "basin_diversity":
        n_huc2 = item.get("n_huc2_regions")
        if n_huc2 is not None:
            lines.append(f"{int(n_huc2)} HUC2")

    lines.append(f"n={n_pairs}")
    return "\n".join(lines)


# ==============================================================================
# Custom distribution graphics
# ==============================================================================


def tukey_box_statistics(values: np.ndarray) -> Dict[str, float]:
    """Return quartiles and Tukey-whisker endpoints for one numeric array."""
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "lower_whisker": np.nan,
            "upper_whisker": np.nan,
        }

    q1, median, q3 = np.quantile(array, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - TUKEY_WHISKER_IQR * iqr
    upper_fence = q3 + TUKEY_WHISKER_IQR * iqr
    inliers = array[(array >= lower_fence) & (array <= upper_fence)]
    if inliers.size == 0:
        inliers = array

    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "lower_whisker": float(np.min(inliers)),
        "upper_whisker": float(np.max(inliers)),
    }


def draw_quantile_box(
    ax: Axes,
    x_position: float,
    values: np.ndarray,
    color: str,
    width: float,
    *,
    zorder: int = 3,
) -> Dict[str, float]:
    """Draw one Tukey boxplot (P25-P75; whiskers within 1.5 IQR)."""
    stats = tukey_box_statistics(values)
    if not np.isfinite(stats["median"]):
        return {"p25": np.nan, "p50": np.nan, "p75": np.nan}

    cap_width = width * 0.58
    ax.plot(
        [x_position, x_position],
        [stats["lower_whisker"], stats["upper_whisker"]],
        color=EDGE_COLOR,
        linewidth=0.78,
        zorder=zorder,
        clip_on=True,
    )
    for y_value in [stats["lower_whisker"], stats["upper_whisker"]]:
        ax.plot(
            [x_position - cap_width / 2.0, x_position + cap_width / 2.0],
            [y_value, y_value],
            color=EDGE_COLOR,
            linewidth=0.78,
            zorder=zorder,
            clip_on=True,
        )

    ax.add_patch(
        Rectangle(
            (x_position - width / 2.0, stats["q1"]),
            width,
            max(stats["q3"] - stats["q1"], 1e-12),
            facecolor=color,
            edgecolor=EDGE_COLOR,
            linewidth=0.85,
            alpha=0.90,
            zorder=zorder + 1,
            clip_on=True,
        )
    )
    ax.plot(
        [x_position - width / 2.0, x_position + width / 2.0],
        [stats["median"], stats["median"]],
        color="black",
        linewidth=1.30,
        zorder=zorder + 2,
        clip_on=True,
    )
    return {
        "p25": stats["q1"],
        "p50": stats["median"],
        "p75": stats["q3"],
    }


def annotate_median(
    ax: Axes,
    x_position: float,
    median: float,
    color: str = TEXT_COLOR,
    fontweight: str = "normal",
) -> None:
    """Annotate one median without obscuring the median line."""
    if not np.isfinite(median):
        return
    ax.annotate(
        f"{median:.2f}",
        xy=(x_position, median),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=color,
        fontweight=fontweight,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.15},
        zorder=10,
        clip_on=True,
    )


def tukey_whisker_envelope(values: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Return the global Tukey-whisker envelope across arrays."""
    lower_values: List[float] = []
    upper_values: List[float] = []
    for values_i in values:
        stats = tukey_box_statistics(values_i)
        if np.isfinite(stats["lower_whisker"]):
            lower_values.append(stats["lower_whisker"])
            upper_values.append(stats["upper_whisker"])
    if not lower_values:
        return 0.0, 1.0
    return min(lower_values), max(upper_values)


def padded_limits(
    lower: float,
    upper: float,
    *,
    minimum_span: float,
) -> Tuple[float, float]:
    """Add proportional padding while enforcing a minimum axis span."""
    span = max(upper - lower, minimum_span)
    center = 0.5 * (lower + upper)
    if upper - lower < minimum_span:
        lower = center - 0.5 * minimum_span
        upper = center + 0.5 * minimum_span
        span = minimum_span
    padding = span * BOX_RANGE_PADDING_RATIO
    return float(lower - padding), float(upper + padding)


def quartile_envelope(values: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Return the global P25-P75 envelope across numeric arrays."""
    q1_values: List[float] = []
    q3_values: List[float] = []
    for values_i in values:
        stats = tukey_box_statistics(values_i)
        if np.isfinite(stats["median"]):
            q1_values.append(stats["q1"])
            q3_values.append(stats["q3"])
    if not q1_values:
        return 0.0, 1.0
    return min(q1_values), max(q3_values)


def focused_absolute_nse_limits(
    values: Sequence[np.ndarray],
    task: str,
) -> Tuple[float, float]:
    """Return task-specific absolute NSE limits from the Tukey-whisker envelope.

    Chapter 4 figures are now drawn separately by task, so the axis range can be
    derived directly from the task-specific whisker envelope instead of forcing a
    common display window. This preserves the full visible box-whisker geometry
    for each task while avoiding the misleading compression caused by shared axes.
    """
    lower, upper = tukey_whisker_envelope(values)
    lower = min(lower, 0.0)
    upper = max(upper, 0.80 if task == "streamflow" else 0.75)
    lower, upper = padded_limits(
        lower,
        upper,
        minimum_span=0.95 if task == "streamflow" else 1.10,
    )
    upper = min(1.00, upper)
    lower_floor = -3.00 if task == "streamflow" else -5.00
    lower = max(lower_floor, lower)
    return float(lower), float(upper)


def focused_gain_limits(
    values: Sequence[np.ndarray],
    task: str,
) -> Tuple[float, float]:
    """Return task-specific paired-gain limits from the Tukey-whisker envelope."""
    lower, upper = tukey_whisker_envelope(values)
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)
    lower, upper = padded_limits(
        lower,
        upper,
        minimum_span=0.12 if task == "streamflow" else 0.08,
    )
    clip_lower = -0.45 if task == "streamflow" else -0.40
    clip_upper = 0.35 if task == "streamflow" else 0.25
    lower = max(clip_lower, lower)
    upper = min(clip_upper, upper)
    return float(lower), float(upper)


def significance_label(p_value: float) -> str:
    """Return a compact significance label for Holm-adjusted p-values."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# ==============================================================================
# Experiment figures
# ==============================================================================


def _style_native_boxplot(
    box_obj: Mapping[str, object],
    colors: Sequence[str],
) -> None:
    """Style a native Matplotlib boxplot consistently."""
    for patch, color in zip(box_obj["boxes"], colors):  # type: ignore[index]
        patch.set_facecolor(color)
        patch.set_alpha(0.90)
        patch.set_edgecolor(EDGE_COLOR)
        patch.set_linewidth(0.85)
    for median in box_obj["medians"]:  # type: ignore[index]
        median.set_color("black")
        median.set_linewidth(1.30)
    for whisker in box_obj["whiskers"]:  # type: ignore[index]
        whisker.set_color(EDGE_COLOR)
        whisker.set_linewidth(0.78)
        whisker.set_clip_on(True)
    for cap in box_obj["caps"]:  # type: ignore[index]
        cap.set_color(EDGE_COLOR)
        cap.set_linewidth(0.78)
        cap.set_clip_on(True)


def plot_absolute_panel(
    ax: Axes,
    paired: pd.DataFrame,
    experiment_type: str,
    task: str,
    order: Sequence[str],
    metadata: Mapping[Tuple[str, str], Mapping[str, object]],
    panel_label: str,
    y_limits: Tuple[float, float],
) -> None:
    """Plot absolute STL and CGC NSE distributions for one task."""
    task_info = TASK_CONFIG[task]
    labels: List[str] = []
    arrays: List[np.ndarray] = []
    positions: List[float] = []
    colors: List[str] = []
    medians: List[Tuple[float, float, str]] = []

    offset = 0.20
    width = 0.34 if len(order) <= 3 else 0.28

    for index, level in enumerate(order, start=1):
        group = paired[
            (paired["experiment_type"] == experiment_type)
            & (paired["task"] == task)
            & (paired["level"] == level)
        ]
        labels.append(build_level_label(experiment_type, level, len(group), metadata))
        if group.empty:
            continue

        stl_values = group["stl_nse"].to_numpy(dtype=float)
        cgc_values = group["cgc_nse"].to_numpy(dtype=float)
        stl_position = index - offset
        cgc_position = index + offset

        arrays.extend([stl_values, cgc_values])
        positions.extend([stl_position, cgc_position])
        colors.extend([COLORS["stl"], task_info["color"]])
        medians.extend([
            (stl_position, float(np.nanmedian(stl_values)), "#555555"),
            (cgc_position, float(np.nanmedian(cgc_values)), TEXT_COLOR),
        ])

    if arrays:
        box_obj = ax.boxplot(
            arrays,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            whis=TUKEY_WHISKER_IQR,
            manage_ticks=False,
            zorder=3,
        )
        _style_native_boxplot(box_obj, colors)
        for x_position, median, color in medians:
            annotate_median(
                ax,
                x_position,
                median,
                color=color,
                fontweight="bold" if color == TEXT_COLOR else "normal",
            )

    ax.axhline(
        0.0,
        color=EDGE_COLOR,
        linestyle="--",
        linewidth=0.70,
        clip_on=True,
        zorder=1,
    )
    ax.set_xlim(0.45, len(order) + 0.55)
    ax.set_ylim(*y_limits)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("NSE")
    ax.set_title(f"({panel_label}) {task_info['label']}: absolute NSE", loc="left", pad=4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
    style_axis(ax, "y")


def plot_paired_gain_panel(
    ax: Axes,
    paired: pd.DataFrame,
    statistics: pd.DataFrame,
    experiment_type: str,
    task: str,
    order: Sequence[str],
    metadata: Mapping[Tuple[str, str], Mapping[str, object]],
    panel_label: str,
    y_limits: Tuple[float, float],
) -> None:
    """Plot basin-wise paired CGC-minus-STL NSE gains for one task."""
    task_info = TASK_CONFIG[task]
    labels: List[str] = []
    arrays: List[np.ndarray] = []
    positions: List[float] = []
    medians: List[Tuple[float, float]] = []

    for index, level in enumerate(order, start=1):
        group = paired[
            (paired["experiment_type"] == experiment_type)
            & (paired["task"] == task)
            & (paired["level"] == level)
        ]
        labels.append(build_level_label(experiment_type, level, len(group), metadata))
        if group.empty:
            continue

        values = group["delta_nse"].to_numpy(dtype=float)
        arrays.append(values)
        positions.append(float(index))
        medians.append((float(index), float(np.nanmedian(values))))

        stat_row = statistics[
            (statistics["experiment_type"] == experiment_type)
            & (statistics["task"] == task)
            & (statistics["level"] == level)
        ]
        if not stat_row.empty:
            row = stat_row.iloc[0]
            label = (
                f"W={row['win_rate']:.0f}% "
                f"{significance_label(float(row['wilcoxon_p_holm']))}"
            ).strip()
            y_top = y_limits[1] - 0.035 * (y_limits[1] - y_limits[0])
            ax.text(
                index,
                y_top,
                label,
                ha="center",
                va="top",
                fontsize=6.4,
                color=MUTED_TEXT_COLOR,
                clip_on=True,
            )

    if arrays:
        box_obj = ax.boxplot(
            arrays,
            positions=positions,
            widths=0.52 if len(order) <= 3 else 0.44,
            patch_artist=True,
            showfliers=False,
            whis=TUKEY_WHISKER_IQR,
            manage_ticks=False,
            zorder=3,
        )
        _style_native_boxplot(box_obj, [task_info["color"]] * len(arrays))
        for x_position, median in medians:
            annotate_median(ax, x_position, median, fontweight="bold")

    ax.axhline(
        0.0,
        color=EDGE_COLOR,
        linestyle="--",
        linewidth=0.80,
        clip_on=True,
        zorder=1,
    )
    ax.set_xlim(0.45, len(order) + 0.55)
    ax.set_ylim(*y_limits)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Paired $\Delta$NSE (CGC $-$ STL)")
    ax.set_title(f"({panel_label}) {task_info['label']}: paired CGC gain", loc="left", pad=4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
    style_axis(ax, "y")


def plot_experiment_figure(
    paired: pd.DataFrame,
    statistics: pd.DataFrame,
    metadata: Mapping[Tuple[str, str], Mapping[str, object]],
    paths: ProjectPaths,
    experiment_type: str,
    order: Sequence[str],
    title: str,
    output_name: str,
) -> None:
    """Plot one 2x2 publication figure for a controlled data-condition experiment."""
    subset = paired[paired["experiment_type"] == experiment_type].copy()
    if subset.empty:
        print(f"[Skip] No paired data found for {experiment_type}.")
        return

    absolute_limits_by_task: Dict[str, Tuple[float, float]] = {}
    gain_limits_by_task: Dict[str, Tuple[float, float]] = {}
    for task in TASK_CONFIG:
        task_subset = subset[subset["task"] == task]
        absolute_limits_by_task[task] = focused_absolute_nse_limits(
            [
                task_subset["stl_nse"].to_numpy(dtype=float),
                task_subset["cgc_nse"].to_numpy(dtype=float),
            ],
            task,
        )
        gain_limits_by_task[task] = focused_gain_limits(
            [task_subset["delta_nse"].to_numpy(dtype=float)],
            task,
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=EXPERIMENT_FIGSIZE,
        sharex=False,
        sharey=False,
        constrained_layout=False,
    )

    plot_absolute_panel(
        axes[0, 0], paired, experiment_type, "streamflow", order, metadata, "a",
        absolute_limits_by_task["streamflow"],
    )
    plot_absolute_panel(
        axes[0, 1], paired, experiment_type, "evapotranspiration", order, metadata, "b",
        absolute_limits_by_task["evapotranspiration"],
    )
    plot_paired_gain_panel(
        axes[1, 0], paired, statistics, experiment_type, "streamflow", order, metadata, "c",
        gain_limits_by_task["streamflow"],
    )
    plot_paired_gain_panel(
        axes[1, 1], paired, statistics, experiment_type, "evapotranspiration", order, metadata, "d",
        gain_limits_by_task["evapotranspiration"],
    )

    legend_handles = [
        Patch(
            facecolor=COLORS["stl"],
            edgecolor=EDGE_COLOR,
            linewidth=0.75,
            label="Task-specific STL baseline",
        ),
        Patch(
            facecolor=COLORS["streamflow"],
            edgecolor=EDGE_COLOR,
            linewidth=0.75,
            label="CGC / paired gain (Q)",
        ),
        Patch(
            facecolor=COLORS["evapotranspiration"],
            edgecolor=EDGE_COLOR,
            linewidth=0.75,
            label="CGC / paired gain (ET)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.suptitle(title, fontsize=10.5, y=0.995)
    fig.text(
        0.01,
        0.010,
        (
            "Boxes show P25-P75; center lines are medians; whiskers follow the Tukey 1.5-IQR rule. "
            "Whisker segments outside the task-specific display range are clipped; W is the basin-level CGC win rate; "
            "significance uses Holm-adjusted Wilcoxon tests; all statistics use raw values."
        ),
        fontsize=6.0,
        color=MUTED_TEXT_COLOR,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.855,
        bottom=0.135,
        wspace=0.25,
        hspace=0.52,
    )
    save_figure(fig, paths.figure_dir / output_name)

# ==============================================================================
# Cross-experiment paired-gain summary
# ==============================================================================


def task_suffix(task: str) -> str:
    """Return a compact file-name suffix for one task."""
    return "streamflow" if task == "streamflow" else "evapotranspiration"


def experiment_output_path(paths: ProjectPaths, output_name: str, task: str) -> Path:
    """Return the output path for one task-specific Chapter 4 experiment figure."""
    stem = Path(output_name).stem
    return paths.figure_dir / f"{stem}_{task_suffix(task)}.png"


def plot_task_experiment_figure(
    paired: pd.DataFrame,
    statistics: pd.DataFrame,
    metadata: Mapping[Tuple[str, str], Mapping[str, object]],
    paths: ProjectPaths,
    experiment_type: str,
    order: Sequence[str],
    title: str,
    output_name: str,
    task: str,
) -> None:
    """Plot one Chapter 4 experiment as a task-specific two-panel figure."""
    subset = paired[
        (paired["experiment_type"] == experiment_type)
        & (paired["task"] == task)
    ].copy()
    if subset.empty:
        print(f"[Skip] No paired data found for {experiment_type} / {task}.")
        return

    absolute_arrays: List[np.ndarray] = []
    gain_arrays: List[np.ndarray] = []
    for level in order:
        group = subset[subset["level"] == level]
        if group.empty:
            continue
        absolute_arrays.extend([
            group["stl_nse"].to_numpy(dtype=float),
            group["cgc_nse"].to_numpy(dtype=float),
        ])
        gain_arrays.append(group["delta_nse"].to_numpy(dtype=float))

    absolute_limits = focused_absolute_nse_limits(absolute_arrays, task)
    gain_limits = focused_gain_limits(gain_arrays, task)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_COLUMN_WIDTH, 5.60),
        sharex=False,
        sharey=False,
        constrained_layout=False,
    )

    plot_absolute_panel(
        axes[0], paired, experiment_type, task, order, metadata, "a", absolute_limits
    )
    plot_paired_gain_panel(
        axes[1], paired, statistics, experiment_type, task, order, metadata, "b", gain_limits
    )

    task_label = TASK_CONFIG[task]["label"]
    fig.suptitle(f"{title} — {task_label}", fontsize=10.2, y=0.985)
    fig.text(
        0.02,
        0.014,
        (
            "Boxes show P25-P75; center lines are medians; whiskers follow the Tukey 1.5-IQR rule. "
            "Task-specific figures use independent vertical scales; W is the basin-level CGC win rate; "
            "significance uses Holm-adjusted Wilcoxon tests; all statistics use raw values."
        ),
        fontsize=6.0,
        color=MUTED_TEXT_COLOR,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.90,
        bottom=0.11,
        hspace=0.40,
    )
    save_figure(fig, experiment_output_path(paths, output_name, task))


def plot_condition_effect_summary(
    statistics: pd.DataFrame,
    paths: ProjectPaths,
) -> None:
    """Plot task-specific paired median gains and bootstrap intervals separately."""
    experiment_order = [
        "climate_consistency",
        "training_length",
        "basin_diversity",
    ]
    experiment_titles = {
        "climate_consistency": "Climate consistency",
        "training_length": "Training data length",
        "basin_diversity": "Basin regional coverage",
    }
    task_order = ["streamflow", "evapotranspiration"]

    for task in task_order:
        task_rows = statistics[statistics["task"] == task]
        lower_values = task_rows["bootstrap_ci_low"].to_numpy(dtype=float)
        upper_values = task_rows["bootstrap_ci_high"].to_numpy(dtype=float)
        lower_values = lower_values[np.isfinite(lower_values)]
        upper_values = upper_values[np.isfinite(upper_values)]

        if lower_values.size == 0 or upper_values.size == 0:
            task_limits = (-0.05, 0.05)
        else:
            lower = min(float(np.min(lower_values)), 0.0)
            upper = max(float(np.max(upper_values)), 0.0)
            lower, upper = padded_limits(
                lower,
                upper,
                minimum_span=0.05 if task == "streamflow" else 0.03,
            )
            task_limits = (lower, upper)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(DOUBLE_COLUMN_WIDTH, 2.85),
            sharex=False,
            sharey=False,
            constrained_layout=False,
        )

        for panel_index, (ax, experiment_type) in enumerate(zip(axes, experiment_order)):
            config = EXPERIMENT_CONFIG[experiment_type]
            available_levels = set(
                statistics.loc[
                    (statistics["experiment_type"] == experiment_type)
                    & (statistics["task"] == task),
                    "level",
                ]
            )
            order = [level for level in config["order"] if level in available_levels]
            x_positions = np.arange(len(order), dtype=float)

            rows = (
                statistics[
                    (statistics["experiment_type"] == experiment_type)
                    & (statistics["task"] == task)
                    & (statistics["level"].isin(order))
                ]
                .set_index("level")
                .reindex(order)
            )
            medians = rows["paired_median_delta_nse"].to_numpy(dtype=float)
            ci_low = rows["bootstrap_ci_low"].to_numpy(dtype=float)
            ci_high = rows["bootstrap_ci_high"].to_numpy(dtype=float)
            errors = np.vstack([medians - ci_low, ci_high - medians])

            ax.errorbar(
                x_positions,
                medians,
                yerr=errors,
                fmt="o",
                markersize=4.2,
                linewidth=1.10,
                capsize=2.6,
                capthick=0.85,
                color=TASK_CONFIG[task]["color"],
                clip_on=True,
            )
            ax.axhline(
                0.0,
                color=EDGE_COLOR,
                linestyle="--",
                linewidth=0.75,
                clip_on=True,
            )
            ax.set_xticks(x_positions)
            if experiment_type == "basin_diversity":
                fallback_huc = {
                    "Low": "3 HUC2",
                    "Medium": "8 HUC2",
                    "High": "All HUC2",
                }
                ax.set_xticklabels([f"{level}\n{fallback_huc[level]}" for level in order])
            else:
                ax.set_xticklabels(order)

            ax.set_title(
                f"({chr(97 + panel_index)}) {experiment_titles[experiment_type]}",
                loc="left",
                pad=4,
            )
            ax.set_ylim(*task_limits)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            style_axis(ax, "y")

        axes[0].set_ylabel(
            rf"{TASK_CONFIG[task]['short_label']} paired median $\Delta$NSE (CGC $-$ STL)"
        )
        fig.suptitle(
            f"{TASK_CONFIG[task]['label']}: condition-effect summary",
            fontsize=10.0,
            y=0.99,
        )
        fig.text(
            0.01,
            0.012,
            "Points show paired median CGC-STL gains; error bars show bootstrap 95% confidence intervals. "
            "Each task uses its own vertical axis.",
            fontsize=6.2,
            color=MUTED_TEXT_COLOR,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.99,
            top=0.78,
            bottom=0.18,
            wspace=0.28,
        )
        output = paths.figure_dir / f"fig4_9_condition_effect_summary_{task_suffix(task)}.png"
        save_figure(fig, output)

# ==============================================================================
# Diagnostics
# ==============================================================================


def print_design_diagnostics(
    paired: pd.DataFrame,
    metadata: Mapping[Tuple[str, str], Mapping[str, object]],
) -> None:
    """Print condition coverage and warn about changing evaluation populations."""
    print("=" * 100)
    print("Chapter 4 paired-data coverage")

    for experiment_type, config in EXPERIMENT_CONFIG.items():
        print(f"\n[{experiment_type}]")
        for level in config["order"]:
            subset = paired[
                (paired["experiment_type"] == experiment_type)
                & (paired["level"] == level)
            ]
            if subset.empty:
                continue
            counts = subset.groupby("task")["basin_id"].nunique().to_dict()
            meta = metadata.get((experiment_type, level), {})
            print(
                f"  {level:<8} paired basins={counts}; "
                f"group metadata={dict(meta) if meta else 'not available'}"
            )

    diversity = paired[paired["experiment_type"] == "basin_diversity"]
    if not diversity.empty:
        counts = (
            diversity.groupby(["level", "task"])["basin_id"]
            .nunique()
            .unstack("task")
        )
        if len(counts.drop_duplicates()) > 1:
            print(
                "\n[Warning] Basin-diversity levels contain different evaluation-basin "
                "populations. Absolute NSE differences may therefore reflect both training "
                "regional coverage and changing basin composition. For a controlled causal "
                "comparison, retain a fixed evaluation set and vary only the training subset."
            )

    print("=" * 100)


# ==============================================================================
# Main program
# ==============================================================================


def main() -> None:
    """Generate Chapter 4 task-specific controlled-condition figures and statistics."""
    args = parse_arguments()
    paths = resolve_paths(args)
    configure_matplotlib()

    raw = load_per_basin_metrics(paths)
    nse_long = prepare_nse_long(raw, paths)
    paired = build_paired_gain_table(nse_long, paths)
    statistics = compute_condition_statistics(
        paired,
        paths,
        bootstrap_repetitions=max(1_000, int(args.bootstrap_repetitions)),
    )
    metadata = load_group_metadata(paths)

    print_design_diagnostics(paired, metadata)

    for experiment_type, config in EXPERIMENT_CONFIG.items():
        available_levels = set(
            paired.loc[
                paired["experiment_type"] == experiment_type,
                "level",
            ].unique()
        )
        valid_order = [level for level in config["order"] if level in available_levels]
        if not valid_order:
            print(f"[Skip] No valid levels for {experiment_type}.")
            continue

        for task in TASK_CONFIG:
            plot_task_experiment_figure(
                paired=paired,
                statistics=statistics,
                metadata=metadata,
                paths=paths,
                experiment_type=experiment_type,
                order=valid_order,
                title=str(config["title"]),
                output_name=str(config["output"]),
                task=task,
            )

    plot_condition_effect_summary(statistics, paths)

    print("=" * 100)
    print("Chapter 4 task-specific publication figures completed successfully.")
    print(f"Figure directory: {paths.figure_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()