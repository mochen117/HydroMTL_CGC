# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Generate publication-quality Chapter 3 figures for hydrological multi-task
#   learning experiments, with emphasis on negative-transfer diagnosis and the
#   mitigation effect of the CGC architecture relative to hard parameter sharing.
#
# Main analytical questions:
#   1. How large is the overall performance difference among STL, Hard-MTL,
#      MMoE, and CGC?
#   2. How frequently and how severely does negative transfer occur?
#   3. Do streamflow and evapotranspiration improve simultaneously?
#   4. Does CGC outperform Hard-MTL at the basin level?
#   5. Where does CGC mitigate hard-sharing degradation spatially?
#   6. Do the CGC task gates exhibit task-specific routing patterns?
#
# Core outputs:
#   - fig3_1_overall_nse_performance.png/pdf
#   - fig3_2_negative_transfer_diagnosis.png/pdf
#   - fig3_3_joint_task_transfer_quadrants.png/pdf
#   - fig3_4_cgc_vs_hard_pairwise_comparison.png/pdf
#   - fig3_5_cgc_minus_hard_spatial_mitigation.png/pdf
#   - fig3_6_cgc_gate_specialization.png/pdf
#
# Supplementary outputs:
#   - supplementary/figS3_1_streamflow_metrics_boxplot.png/pdf
#   - supplementary/figS3_2_evapotranspiration_metrics_boxplot.png/pdf
#   - supplementary/figS3_3_nse_cdf.png/pdf
#   - supplementary/figS3_4_cgc_vs_stl_pairwise_comparison.png/pdf
# ============================================================================== 

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - spatial figures are skipped gracefully.
    gpd = None  # type: ignore[assignment]

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - p-values are reported as NaN.
    wilcoxon = None


# ==============================================================================
# Project paths
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"
FIG_DIR = CH3_DIR / "figures"
SUPPLEMENTARY_FIG_DIR = FIG_DIR / "supplementary"

PER_BASIN_PATH = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
GATE_SUMMARY_PATH = SUMMARY_DIR / "ch3_gate_utilization_summary.csv"
GATE_LONG_PATH = SUMMARY_DIR / "ch3_gate_utilization_long.csv"

TRANSFER_ANALYSIS_PATH = SUMMARY_DIR / "ch3_transfer_analysis_table.csv"
PAIRWISE_STATS_PATH = SUMMARY_DIR / "ch3_pairwise_statistics.csv"
NEGATIVE_TRANSFER_RATE_PATH = SUMMARY_DIR / "ch3_negative_transfer_rates.csv"
JOINT_TRANSFER_SUMMARY_PATH = SUMMARY_DIR / "ch3_joint_transfer_outcomes.csv"
GATE_SPECIALIZATION_PATH = SUMMARY_DIR / "ch3_gate_specialization_summary.csv"

BASIN_SHP_PATH = Path(
    "/home/mochen/hydro_data/camels/camels_us/"
    "basin_set_full_res/HCDN_nhru_final_671.shp"
)

US_STATE_SHP_PATH = Path(
    "/home/mochen/.local/share/cartopy/shapefiles/natural_earth/cultural/"
    "ne_50m_admin_1_states_provinces_lakes.shp"
)

MAP_CRS = "EPSG:5070"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUPPLEMENTARY_FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Model, task, and figure configuration
# ==============================================================================

MODELS_Q = ["STL_Q", "Hard_MTL", "MMoE", "CGC"]
MODELS_ET = ["STL_ET", "Hard_MTL", "MMoE", "CGC"]
MTL_MODELS = ["Hard_MTL", "MMoE", "CGC"]
TASKS = ["streamflow", "evapotranspiration"]

DISPLAY_LABELS = {
    "STL_Q": "STL-Q",
    "STL_ET": "STL-ET",
    "Hard_MTL": "Hard-MTL",
    "MMoE": "MMoE",
    "CGC": "CGC",
}

TASK_LABELS = {
    "streamflow": "Streamflow (Q)",
    "evapotranspiration": "Evapotranspiration (ET)",
}

PALETTE = {
    "q": "#5B99C5",
    "et": "#FAA256",
    "hard_q": "#B1DDF0",
    "mmoe_q": "#C2BFD7",
    "hard_et": "#F7BFBF",
    "mmoe_et": "#F0AFAF",
    "stl": "#BDBDBD",
    "gain": "#5B99C5",
    "loss": "#E7A1A1",
    "hard": "#7F7F7F",
    "mmoe": "#8D79B5",
    "cgc": "#1F4E79",
}

TASK_COLORS = {
    "streamflow": PALETTE["q"],
    "evapotranspiration": PALETTE["et"],
}

Q_MODEL_COLORS = {
    "STL_Q": PALETTE["stl"],
    "Hard_MTL": PALETTE["hard_q"],
    "MMoE": PALETTE["mmoe_q"],
    "CGC": PALETTE["q"],
}

ET_MODEL_COLORS = {
    "STL_ET": PALETTE["stl"],
    "Hard_MTL": PALETTE["hard_et"],
    "MMoE": PALETTE["mmoe_et"],
    "CGC": PALETTE["et"],
}

TRANSFER_MODEL_COLORS = {
    "Hard_MTL": PALETTE["hard"],
    "MMoE": PALETTE["mmoe"],
    "CGC": PALETTE["cgc"],
}

QUADRANT_COLORS = {
    "both_improved": "#4C956C",
    "q_degraded_et_improved": "#E9A03B",
    "both_degraded": "#C95757",
    "q_improved_et_degraded": "#7B6FB3",
    "near_zero": "#BDBDBD",
}

QUADRANT_LABELS = {
    "both_improved": "Both tasks improved",
    "q_degraded_et_improved": "Q degraded, ET improved",
    "both_degraded": "Both tasks degraded",
    "q_improved_et_degraded": "Q improved, ET degraded",
    "near_zero": "Near-zero change",
}

METRIC_PANELS = [
    ("bias", "Bias"),
    ("rmse", "RMSE"),
    ("corr", "Corr"),
    ("nse", "NSE"),
    ("kge", "KGE"),
]

NSE_DISPLAY_RANGE = (-1.0, 1.0)
NSE_MAP_RANGE = (0.0, 1.0)
NEGATIVE_TRANSFER_THRESHOLDS = (0.0, 0.05, 0.10)
JOINT_TRANSFER_TOLERANCE = 0.0
BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 42

EDGE_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
ANNOTATION_COLOR = "#8B0000"
STATE_LINE_COLOR = "#9A9A9A"
BASEMAP_FACE = "#E2E2E2"


# ==============================================================================
# General utilities
# ==============================================================================


def require_file(path: Path) -> None:
    """Raise a clear error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


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


def set_publication_style() -> None:
    """Set global Matplotlib parameters for journal-quality output."""
    font_name = choose_serif_font()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_name],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.9,
            "axes.edgecolor": EDGE_COLOR,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )
    print(f"[Info] Figure font: {font_name}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save one figure in high-resolution PNG and vector PDF formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[Saved] {path}")
    print(f"[Saved] {path.with_suffix('.pdf')}")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize basin identifiers to eight-character strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to finite numeric values and remove missing values."""
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def metric_column(model: str, task: str, metric: str) -> str:
    """Build the canonical column name for a model-task metric."""
    return f"{model}_{task}_{metric}"


def task_model_colors(task: str) -> Mapping[str, str]:
    """Return task-specific model colors."""
    if task == "streamflow":
        return Q_MODEL_COLORS
    if task == "evapotranspiration":
        return ET_MODEL_COLORS
    raise ValueError(f"Unsupported task: {task}")


def style_axis(ax: Axes, grid_axis: str = "y") -> None:
    """Apply consistent axis and grid styling."""
    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.55,
        color=GRID_COLOR,
        alpha=0.65,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(0.9)


def style_boxplot(box_obj: Mapping[str, object], colors: Sequence[str]) -> None:
    """Apply consistent styling to a Matplotlib boxplot."""
    for patch, color in zip(box_obj["boxes"], colors):  # type: ignore[index]
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
        patch.set_edgecolor(EDGE_COLOR)
        patch.set_linewidth(0.9)

    for median in box_obj["medians"]:  # type: ignore[index]
        median.set_color("black")
        median.set_linewidth(1.35)

    for whisker in box_obj["whiskers"]:  # type: ignore[index]
        whisker.set_color("black")
        whisker.set_linewidth(0.9)

    for cap in box_obj["caps"]:  # type: ignore[index]
        cap.set_color("black")
        cap.set_linewidth(0.9)


def annotate_box_medians(
    ax: Axes,
    values: Sequence[pd.Series],
    positions: Sequence[float],
    decimals: int = 2,
) -> None:
    """Annotate median values immediately above the median lines."""
    medians = [float(np.nanmedian(clean_numeric(value))) for value in values]
    y_min, y_max = ax.get_ylim()
    offset = max(y_max - y_min, 1e-6) * 0.006

    for x_value, median in zip(positions, medians):
        ax.text(
            x_value,
            median + offset,
            f"{median:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=10,
        )


def robust_symmetric_limit(
    values: Sequence[np.ndarray],
    quantile: float = 0.98,
    minimum: float = 0.10,
    maximum: float = 0.50,
) -> float:
    """Estimate a robust symmetric plotting limit across multiple arrays."""
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    combined = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    combined = combined[np.isfinite(combined)]

    if combined.size == 0:
        return minimum

    limit = float(np.quantile(np.abs(combined), quantile))
    return float(np.clip(limit, minimum, maximum))


def load_per_basin_table() -> pd.DataFrame:
    """Load basin-level Chapter 3 metrics for all compared models."""
    require_file(PER_BASIN_PATH)
    table = pd.read_csv(PER_BASIN_PATH, dtype={"gauge_id": str})

    if "gauge_id" not in table.columns:
        raise ValueError(f"Missing 'gauge_id' in {PER_BASIN_PATH}.")

    table["gauge_id"] = normalize_gauge_id(table["gauge_id"])
    print(f"[Info] Basin-level table: {PER_BASIN_PATH}")
    print(f"[Info] Basin records: {len(table)}")
    return table


def collect_metric_series(
    table: pd.DataFrame,
    models: Sequence[str],
    task: str,
    metric: str,
) -> Dict[str, pd.Series]:
    """Collect valid metric series for selected models."""
    output: Dict[str, pd.Series] = {}

    for model in models:
        column = metric_column(model, task, metric)
        if column not in table.columns:
            print(f"[Skip] Missing metric column: {column}")
            continue

        values = clean_numeric(table[column])
        if not values.empty:
            output[model] = values

    return output


# ==============================================================================
# Transfer-analysis table and statistical summaries
# ==============================================================================


def build_transfer_analysis_table(table: pd.DataFrame) -> pd.DataFrame:
    """Build one basin-level table for all transfer and pairwise comparisons."""
    output = table[["gauge_id"]].copy()

    q_stl = metric_column("STL_Q", "streamflow", "nse")
    et_stl = metric_column("STL_ET", "evapotranspiration", "nse")
    q_hard = metric_column("Hard_MTL", "streamflow", "nse")
    et_hard = metric_column("Hard_MTL", "evapotranspiration", "nse")
    q_mmoe = metric_column("MMoE", "streamflow", "nse")
    et_mmoe = metric_column("MMoE", "evapotranspiration", "nse")
    q_cgc = metric_column("CGC", "streamflow", "nse")
    et_cgc = metric_column("CGC", "evapotranspiration", "nse")

    required = [
        q_stl,
        et_stl,
        q_hard,
        et_hard,
        q_mmoe,
        et_mmoe,
        q_cgc,
        et_cgc,
    ]

    missing = [column for column in required if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required NSE columns: {missing}")

    for column in required:
        output[column] = pd.to_numeric(table[column], errors="coerce")

    output["hard_delta_q"] = output[q_hard] - output[q_stl]
    output["hard_delta_et"] = output[et_hard] - output[et_stl]
    output["mmoe_delta_q"] = output[q_mmoe] - output[q_stl]
    output["mmoe_delta_et"] = output[et_mmoe] - output[et_stl]
    output["cgc_delta_q"] = output[q_cgc] - output[q_stl]
    output["cgc_delta_et"] = output[et_cgc] - output[et_stl]

    output["cgc_minus_hard_q"] = output[q_cgc] - output[q_hard]
    output["cgc_minus_hard_et"] = output[et_cgc] - output[et_hard]
    output["cgc_minus_mmoe_q"] = output[q_cgc] - output[q_mmoe]
    output["cgc_minus_mmoe_et"] = output[et_cgc] - output[et_mmoe]

    output = output.replace([np.inf, -np.inf], np.nan)
    output.to_csv(TRANSFER_ANALYSIS_PATH, index=False)
    print(f"[Saved] {TRANSFER_ANALYSIS_PATH}")
    return output


def bootstrap_median_ci(
    values: pd.Series,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    confidence: float = 0.95,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """Estimate a bootstrap confidence interval for a paired median difference."""
    array = clean_numeric(values).to_numpy(dtype=float)
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


def rank_biserial_effect(values: pd.Series) -> float:
    """Compute the paired rank-biserial effect size from signed differences."""
    array = clean_numeric(values).to_numpy(dtype=float)
    array = array[array != 0.0]

    if array.size == 0:
        return 0.0

    ranks = pd.Series(np.abs(array)).rank(method="average").to_numpy(dtype=float)
    positive_rank_sum = float(ranks[array > 0.0].sum())
    negative_rank_sum = float(ranks[array < 0.0].sum())
    denominator = positive_rank_sum + negative_rank_sum

    if denominator == 0.0:
        return 0.0

    return (positive_rank_sum - negative_rank_sum) / denominator


def paired_difference_statistics(values: pd.Series) -> Dict[str, float]:
    """Summarize paired model differences using robust and nonparametric metrics."""
    array = clean_numeric(values)
    if array.empty:
        return {
            "n": 0,
            "median_difference": np.nan,
            "mean_difference": np.nan,
            "win_rate": np.nan,
            "loss_rate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "wilcoxon_p": np.nan,
            "rank_biserial": np.nan,
        }

    ci_low, ci_high = bootstrap_median_ci(array)

    if wilcoxon is None:
        p_value = np.nan
    else:
        try:
            test = wilcoxon(array.to_numpy(dtype=float), zero_method="wilcox")
            p_value = float(test.pvalue)
        except ValueError:
            p_value = np.nan

    return {
        "n": int(len(array)),
        "median_difference": float(array.median()),
        "mean_difference": float(array.mean()),
        "win_rate": float((array > 0.0).mean() * 100.0),
        "loss_rate": float((array < 0.0).mean() * 100.0),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": p_value,
        "rank_biserial": rank_biserial_effect(array),
    }


def export_pairwise_statistics(transfer: pd.DataFrame) -> pd.DataFrame:
    """Export paired CGC comparisons against Hard-MTL, MMoE, and STL."""
    comparisons = {
        ("streamflow", "CGC minus STL"): "cgc_delta_q",
        ("evapotranspiration", "CGC minus STL"): "cgc_delta_et",
        ("streamflow", "CGC minus Hard-MTL"): "cgc_minus_hard_q",
        ("evapotranspiration", "CGC minus Hard-MTL"): "cgc_minus_hard_et",
        ("streamflow", "CGC minus MMoE"): "cgc_minus_mmoe_q",
        ("evapotranspiration", "CGC minus MMoE"): "cgc_minus_mmoe_et",
    }

    records: List[Dict[str, object]] = []
    for (task, comparison), column in comparisons.items():
        statistics = paired_difference_statistics(transfer[column])
        records.append(
            {
                "task": task,
                "comparison": comparison,
                "difference_column": column,
                **statistics,
            }
        )

    summary = pd.DataFrame(records)
    summary.to_csv(PAIRWISE_STATS_PATH, index=False)
    print(f"[Saved] {PAIRWISE_STATS_PATH}")
    return summary


def compute_negative_transfer_rates(transfer: pd.DataFrame) -> pd.DataFrame:
    """Compute negative-transfer rates under multiple severity thresholds."""
    columns = {
        ("streamflow", "Hard_MTL"): "hard_delta_q",
        ("streamflow", "MMoE"): "mmoe_delta_q",
        ("streamflow", "CGC"): "cgc_delta_q",
        ("evapotranspiration", "Hard_MTL"): "hard_delta_et",
        ("evapotranspiration", "MMoE"): "mmoe_delta_et",
        ("evapotranspiration", "CGC"): "cgc_delta_et",
    }

    records: List[Dict[str, object]] = []

    for (task, model), column in columns.items():
        values = clean_numeric(transfer[column])
        if values.empty:
            continue

        for threshold in NEGATIVE_TRANSFER_THRESHOLDS:
            records.append(
                {
                    "task": task,
                    "model": model,
                    "threshold": threshold,
                    "negative_transfer_rate": float(
                        (values < -threshold).mean() * 100.0
                    ),
                    "positive_transfer_rate": float(
                        (values > threshold).mean() * 100.0
                    ),
                    "median_delta_nse": float(values.median()),
                }
            )

    summary = pd.DataFrame(records)
    summary.to_csv(NEGATIVE_TRANSFER_RATE_PATH, index=False)
    print(f"[Saved] {NEGATIVE_TRANSFER_RATE_PATH}")
    return summary


def classify_joint_transfer(
    delta_q: pd.Series,
    delta_et: pd.Series,
    tolerance: float = JOINT_TRANSFER_TOLERANCE,
) -> pd.Series:
    """Classify basin-level Q-ET transfer outcomes into four main quadrants."""
    q_values = pd.to_numeric(delta_q, errors="coerce")
    et_values = pd.to_numeric(delta_et, errors="coerce")

    conditions = [
        (q_values > tolerance) & (et_values > tolerance),
        (q_values < -tolerance) & (et_values > tolerance),
        (q_values < -tolerance) & (et_values < -tolerance),
        (q_values > tolerance) & (et_values < -tolerance),
    ]

    labels = [
        "both_improved",
        "q_degraded_et_improved",
        "both_degraded",
        "q_improved_et_degraded",
    ]

    outcome = np.select(conditions, labels, default="near_zero")
    return pd.Series(outcome, index=delta_q.index, dtype="object")


def build_joint_transfer_summary(transfer: pd.DataFrame) -> pd.DataFrame:
    """Create a long-format summary of joint Q-ET transfer outcomes."""
    model_columns = {
        "Hard_MTL": ("hard_delta_q", "hard_delta_et"),
        "MMoE": ("mmoe_delta_q", "mmoe_delta_et"),
        "CGC": ("cgc_delta_q", "cgc_delta_et"),
    }

    records: List[Dict[str, object]] = []

    for model, (q_column, et_column) in model_columns.items():
        valid = transfer[["gauge_id", q_column, et_column]].dropna().copy()
        valid["outcome"] = classify_joint_transfer(
            valid[q_column],
            valid[et_column],
        )

        counts = valid["outcome"].value_counts(dropna=False)
        total = max(len(valid), 1)

        for outcome in QUADRANT_LABELS:
            count = int(counts.get(outcome, 0))
            records.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "count": count,
                    "percentage": count / total * 100.0,
                    "tolerance": JOINT_TRANSFER_TOLERANCE,
                }
            )

    summary = pd.DataFrame(records)
    summary.to_csv(JOINT_TRANSFER_SUMMARY_PATH, index=False)
    print(f"[Saved] {JOINT_TRANSFER_SUMMARY_PATH}")
    return summary


# ==============================================================================
# Core Figure 3-1: overall performance
# ==============================================================================


def plot_overall_nse_performance(
    table: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
) -> None:
    """Plot overall NSE distributions and direct CGC-Hard paired summaries."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)

    configurations = [
        (axes[0], "streamflow", MODELS_Q, "(a) Streamflow"),
        (axes[1], "evapotranspiration", MODELS_ET, "(b) Evapotranspiration"),
    ]

    for ax, task, models, title in configurations:
        data = collect_metric_series(table, models, task, "nse")
        available = [model for model in models if model in data]
        values = [data[model] for model in available]
        positions = np.arange(1, len(values) + 1)
        colors = [task_model_colors(task)[model] for model in available]

        box_obj = ax.boxplot(
            [value.to_numpy(dtype=float) for value in values],
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
            whis=(5, 95),
            tick_labels=[DISPLAY_LABELS[model] for model in available],
        )
        style_boxplot(box_obj, colors)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylim(*NSE_DISPLAY_RANGE)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Model")
        style_axis(ax, "y")
        annotate_box_medians(ax, values, positions)

        task_stats = pairwise_stats[
            (pairwise_stats["task"] == task)
            & (pairwise_stats["comparison"] == "CGC minus Hard-MTL")
        ]

        if not task_stats.empty:
            row = task_stats.iloc[0]
            p_value = row["wilcoxon_p"]
            p_text = "NA" if not np.isfinite(p_value) else f"{p_value:.2e}"
            text = (
                f"CGC − Hard-MTL\n"
                f"Median ΔNSE: {row['median_difference']:+.3f}\n"
                f"95% CI: [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]\n"
                f"CGC win rate: {row['win_rate']:.1f}%\n"
                f"Wilcoxon p: {p_text}"
            )
            ax.text(
                0.03,
                0.04,
                text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.2,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "linewidth": 0.7,
                    "alpha": 0.94,
                    "pad": 3.5,
                },
            )

    axes[0].set_ylabel("NSE")
    fig.suptitle(
        "Overall performance and paired CGC–Hard-MTL differences",
        fontsize=13.0,
        y=0.995,
    )
    fig.text(
        0.01,
        0.01,
        "Boxes show the 25th–75th percentiles; whiskers show the 5th–95th "
        "percentiles; median values are annotated.",
        fontsize=8.1,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.18, wspace=0.16)
    save_figure(fig, FIG_DIR / "fig3_1_overall_nse_performance.png")


# ==============================================================================
# Core Figure 3-2: negative-transfer diagnosis
# ==============================================================================


def plot_negative_transfer_diagnosis(
    transfer: pd.DataFrame,
    rate_summary: pd.DataFrame,
) -> None:
    """Plot transfer-effect distributions and threshold-dependent risk rates."""
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))

    task_columns = {
        "streamflow": {
            "Hard_MTL": "hard_delta_q",
            "MMoE": "mmoe_delta_q",
            "CGC": "cgc_delta_q",
        },
        "evapotranspiration": {
            "Hard_MTL": "hard_delta_et",
            "MMoE": "mmoe_delta_et",
            "CGC": "cgc_delta_et",
        },
    }

    for column_index, task in enumerate(TASKS):
        ax_box = axes[0, column_index]
        columns = task_columns[task]
        values = [clean_numeric(transfer[column]) for column in columns.values()]
        models = list(columns.keys())
        positions = np.arange(1, len(models) + 1)

        box_obj = ax_box.boxplot(
            [value.to_numpy(dtype=float) for value in values],
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
            whis=(5, 95),
            tick_labels=[DISPLAY_LABELS[model] for model in models],
        )
        style_boxplot(
            box_obj,
            [task_model_colors(task)[model] for model in models],
        )
        ax_box.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

        limit = robust_symmetric_limit(
            [value.to_numpy(dtype=float) for value in values],
            quantile=0.98,
            minimum=0.08,
            maximum=0.35,
        )
        ax_box.set_ylim(-limit, limit)
        ax_box.set_title(
            f"({'a' if column_index == 0 else 'b'}) {TASK_LABELS[task]} transfer effect",
            loc="left",
        )
        ax_box.set_ylabel(r"$\Delta$NSE relative to STL")
        style_axis(ax_box, "y")
        annotate_box_medians(ax_box, values, positions)

        ax_rate = axes[1, column_index]
        subset = rate_summary[rate_summary["task"] == task].copy()
        threshold_labels = ["Any\n(< 0)", "Moderate\n(< −0.05)", "Severe\n(< −0.10)"]

        for model in MTL_MODELS:
            model_data = subset[subset["model"] == model].sort_values("threshold")
            if model_data.empty:
                continue

            line_width = 2.0 if model == "CGC" else 1.4
            marker_size = 6.0 if model == "CGC" else 5.0
            ax_rate.plot(
                np.arange(len(model_data)),
                model_data["negative_transfer_rate"],
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=TRANSFER_MODEL_COLORS[model],
                label=DISPLAY_LABELS[model],
            )

        ax_rate.set_xticks(np.arange(len(NEGATIVE_TRANSFER_THRESHOLDS)))
        ax_rate.set_xticklabels(threshold_labels)
        ax_rate.set_ylim(0.0, 100.0)
        ax_rate.set_ylabel("Negative-transfer rate (%)")
        ax_rate.set_title(
            f"({'c' if column_index == 0 else 'd'}) Severity-dependent risk",
            loc="left",
        )
        style_axis(ax_rate, "y")

    axes[1, 1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Negative-transfer distributions and degradation risk",
        fontsize=13.0,
        y=0.995,
    )
    fig.text(
        0.01,
        0.01,
        "Negative transfer is defined from the basin-level paired difference "
        "between an MTL model and its task-specific STL baseline.",
        fontsize=8.1,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.10, wspace=0.20, hspace=0.35)
    save_figure(fig, FIG_DIR / "fig3_2_negative_transfer_diagnosis.png")


# ==============================================================================
# Core Figure 3-3: joint Q-ET transfer quadrants
# ==============================================================================


def plot_joint_task_transfer_quadrants(
    transfer: pd.DataFrame,
    joint_summary: pd.DataFrame,
) -> None:
    """Plot basin-level joint transfer outcomes for Hard-MTL, MMoE, and CGC."""
    model_columns = {
        "Hard_MTL": ("hard_delta_q", "hard_delta_et"),
        "MMoE": ("mmoe_delta_q", "mmoe_delta_et"),
        "CGC": ("cgc_delta_q", "cgc_delta_et"),
    }

    all_values: List[np.ndarray] = []
    for q_column, et_column in model_columns.values():
        all_values.extend(
            [
                clean_numeric(transfer[q_column]).to_numpy(dtype=float),
                clean_numeric(transfer[et_column]).to_numpy(dtype=float),
            ]
        )

    limit = robust_symmetric_limit(
        all_values,
        quantile=0.98,
        minimum=0.10,
        maximum=0.45,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharex=True, sharey=True)

    for ax, (model, (q_column, et_column)) in zip(axes, model_columns.items()):
        data = transfer[[q_column, et_column]].copy()
        data.columns = ["delta_q", "delta_et"]
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data["outcome"] = classify_joint_transfer(
            data["delta_q"],
            data["delta_et"],
        )

        for outcome in QUADRANT_LABELS:
            subset = data[data["outcome"] == outcome]
            if subset.empty:
                continue

            ax.scatter(
                subset["delta_q"].clip(-limit, limit),
                subset["delta_et"].clip(-limit, limit),
                s=16,
                alpha=0.48,
                color=QUADRANT_COLORS[outcome],
                edgecolor="none",
                rasterized=True,
                zorder=3,
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(DISPLAY_LABELS[model])
        ax.set_xlabel(r"$\Delta$NSE$_Q$ relative to STL-Q")
        style_axis(ax, "both")

        summary = joint_summary[joint_summary["model"] == model].set_index("outcome")
        both_gain = float(summary.loc["both_improved", "percentage"])
        q_negative = float(summary.loc["q_degraded_et_improved", "percentage"])
        both_loss = float(summary.loc["both_degraded", "percentage"])
        et_negative = float(summary.loc["q_improved_et_degraded", "percentage"])

        text = (
            f"Both improved: {both_gain:.1f}%\n"
            f"Q degraded only: {q_negative:.1f}%\n"
            f"ET degraded only: {et_negative:.1f}%\n"
            f"Both degraded: {both_loss:.1f}%"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            bbox={
                "facecolor": "white",
                "edgecolor": "#C7C7C7",
                "linewidth": 0.6,
                "alpha": 0.92,
                "pad": 3.0,
            },
        )

    axes[0].set_ylabel(r"$\Delta$NSE$_{ET}$ relative to STL-ET")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=QUADRANT_COLORS[outcome],
            markeredgecolor="none",
            label=QUADRANT_LABELS[outcome],
        )
        for outcome in [
            "both_improved",
            "q_degraded_et_improved",
            "both_degraded",
            "q_improved_et_degraded",
        ]
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Joint streamflow–evapotranspiration transfer outcomes",
        fontsize=13.0,
        y=0.995,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.20, wspace=0.18)
    save_figure(fig, FIG_DIR / "fig3_3_joint_task_transfer_quadrants.png")


# ==============================================================================
# Core Figure 3-4: direct CGC versus Hard-MTL comparison
# ==============================================================================


def prepare_pairwise_model_data(
    table: pd.DataFrame,
    reference_model: str,
    candidate_model: str,
    task: str,
) -> Tuple[pd.DataFrame, str, str]:
    """Prepare paired basin-level NSE data for two models."""
    reference_column = metric_column(reference_model, task, "nse")
    candidate_column = metric_column(candidate_model, task, "nse")

    missing = [
        column
        for column in [reference_column, candidate_column]
        if column not in table.columns
    ]
    if missing:
        raise KeyError(f"Missing paired comparison columns: {missing}")

    output = table[["gauge_id", reference_column, candidate_column]].copy()
    output[reference_column] = pd.to_numeric(output[reference_column], errors="coerce")
    output[candidate_column] = pd.to_numeric(output[candidate_column], errors="coerce")
    output = output.replace([np.inf, -np.inf], np.nan).dropna()
    output["difference"] = output[candidate_column] - output[reference_column]
    return output, reference_column, candidate_column


def format_p_value(value: float) -> str:
    """Format a statistical p-value for compact figure annotation."""
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def plot_pairwise_scatter_panel(
    ax: Axes,
    data: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    task: str,
    title: str,
    xlabel: str,
    ylabel: str,
    statistics: Mapping[str, float],
) -> None:
    """Plot one paired model-comparison panel with robust summary statistics."""
    low, high = NSE_DISPLAY_RANGE
    visible = (
        data[reference_column].between(low, high)
        & data[candidate_column].between(low, high)
    )
    plot_data = data.loc[visible].copy()

    improved = plot_data[plot_data["difference"] > 0.0]
    degraded = plot_data[plot_data["difference"] < 0.0]
    unchanged = plot_data[plot_data["difference"] == 0.0]

    ax.scatter(
        improved[reference_column],
        improved[candidate_column],
        s=16,
        alpha=0.48,
        color=PALETTE["gain"],
        edgecolor="none",
        rasterized=True,
        label="CGC better",
    )
    ax.scatter(
        degraded[reference_column],
        degraded[candidate_column],
        s=16,
        alpha=0.48,
        color=PALETTE["loss"],
        edgecolor="none",
        rasterized=True,
        label="CGC worse",
    )
    if not unchanged.empty:
        ax.scatter(
            unchanged[reference_column],
            unchanged[candidate_column],
            s=16,
            alpha=0.45,
            color="#9E9E9E",
            edgecolor="none",
            rasterized=True,
            label="No change",
        )

    ax.plot([low, high], [low, high], "k--", linewidth=1.0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax, "both")

    outside_count = int((~visible).sum())
    text = (
        f"CGC win rate: {statistics['win_rate']:.1f}%\n"
        f"Median difference: {statistics['median_difference']:+.3f}\n"
        f"95% CI: [{statistics['ci_low']:+.3f}, {statistics['ci_high']:+.3f}]\n"
        f"Rank-biserial: {statistics['rank_biserial']:+.3f}\n"
        f"Wilcoxon p: {format_p_value(statistics['wilcoxon_p'])}\n"
        f"Outside display range: {outside_count}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "linewidth": 0.7,
            "alpha": 0.94,
            "pad": 3.5,
        },
    )


def plot_cgc_vs_hard_pairwise_comparison(
    table: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
) -> None:
    """Plot direct basin-level CGC-versus-Hard-MTL NSE comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))

    configurations = [
        (
            axes[0],
            "streamflow",
            "(a) Streamflow",
            "Hard-MTL streamflow NSE",
            "CGC streamflow NSE",
        ),
        (
            axes[1],
            "evapotranspiration",
            "(b) Evapotranspiration",
            "Hard-MTL evapotranspiration NSE",
            "CGC evapotranspiration NSE",
        ),
    ]

    for ax, task, title, xlabel, ylabel in configurations:
        data, reference_column, candidate_column = prepare_pairwise_model_data(
            table,
            reference_model="Hard_MTL",
            candidate_model="CGC",
            task=task,
        )

        stats_row = pairwise_stats[
            (pairwise_stats["task"] == task)
            & (pairwise_stats["comparison"] == "CGC minus Hard-MTL")
        ]
        if stats_row.empty:
            raise ValueError(f"Missing CGC-Hard statistics for task: {task}")

        statistics = stats_row.iloc[0].to_dict()
        plot_pairwise_scatter_panel(
            ax=ax,
            data=data,
            reference_column=reference_column,
            candidate_column=candidate_column,
            task=task,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            statistics=statistics,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALETTE["gain"],
            markeredgecolor="none",
            markersize=6,
            label="CGC better than Hard-MTL",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALETTE["loss"],
            markeredgecolor="none",
            markersize=6,
            label="CGC worse than Hard-MTL",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Direct basin-level comparison between CGC and Hard-MTL",
        fontsize=13.0,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.17, wspace=0.22)
    save_figure(fig, FIG_DIR / "fig3_4_cgc_vs_hard_pairwise_comparison.png")


# ==============================================================================
# Core Figure 3-5: spatial mitigation relative to Hard-MTL
# ==============================================================================


def infer_shapefile_gauge_column(frame: "gpd.GeoDataFrame") -> str:
    """Infer the basin identifier column in a CAMELS shapefile."""
    candidates = [
        "gauge_id",
        "GAGE_ID",
        "hru_id",
        "HRU_ID",
        "basin_id",
        "BASIN_ID",
    ]
    lower_map = {str(column).lower(): str(column) for column in frame.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"No gauge ID column found. Available columns: {list(frame.columns)}"
    )


def load_basin_geometries() -> Optional["gpd.GeoDataFrame"]:
    """Load and project CAMELS basin polygons."""
    if gpd is None:
        print("[Skip] GeoPandas is not installed; spatial figures are unavailable.")
        return None

    if not BASIN_SHP_PATH.exists():
        print(f"[Skip] CAMELS basin shapefile not found: {BASIN_SHP_PATH}")
        return None

    basins = gpd.read_file(BASIN_SHP_PATH)
    gauge_column = infer_shapefile_gauge_column(basins)
    basins = basins.rename(columns={gauge_column: "gauge_id"})
    basins["gauge_id"] = normalize_gauge_id(basins["gauge_id"])

    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326", allow_override=True)

    basins = basins.to_crs(MAP_CRS)
    basins = basins[basins.geometry.notna()].copy()
    basins = basins[~basins.geometry.is_empty].copy()
    return basins[["gauge_id", "geometry"]]


def load_state_boundaries() -> Optional["gpd.GeoDataFrame"]:
    """Load and project contiguous U.S. state boundaries."""
    if gpd is None or not US_STATE_SHP_PATH.exists():
        return None

    states = gpd.read_file(US_STATE_SHP_PATH)

    if states.crs is None:
        states = states.set_crs("EPSG:4326", allow_override=True)

    if "admin" in states.columns:
        states = states[states["admin"].astype(str) == "United States of America"]
    elif "adm0_a3" in states.columns:
        states = states[states["adm0_a3"].astype(str) == "USA"]
    elif "iso_a2" in states.columns:
        states = states[states["iso_a2"].astype(str) == "US"]

    states = states.to_crs(MAP_CRS)
    states = states[states.geometry.notna()].copy()
    states = states[~states.geometry.is_empty].copy()
    return states


def projected_conus_extent() -> Tuple[float, float, float, float]:
    """Return a projected extent for the contiguous United States."""
    if gpd is None:
        raise ImportError("GeoPandas is required for projected map extents.")

    minimum_longitude, maximum_longitude = -125.5, -66.5
    minimum_latitude, maximum_latitude = 24.0, 49.5
    n_points = 200

    bottom_longitude = np.linspace(minimum_longitude, maximum_longitude, n_points)
    top_longitude = np.linspace(minimum_longitude, maximum_longitude, n_points)
    left_latitude = np.linspace(minimum_latitude, maximum_latitude, n_points)
    right_latitude = np.linspace(minimum_latitude, maximum_latitude, n_points)

    longitude = np.concatenate(
        [
            bottom_longitude,
            np.full(n_points, maximum_longitude),
            top_longitude[::-1],
            np.full(n_points, minimum_longitude),
        ]
    )
    latitude = np.concatenate(
        [
            np.full(n_points, minimum_latitude),
            right_latitude,
            np.full(n_points, maximum_latitude),
            left_latitude[::-1],
        ]
    )

    boundary = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(longitude, latitude),
        crs="EPSG:4326",
    ).to_crs(MAP_CRS)

    minimum_x, minimum_y, maximum_x, maximum_y = boundary.total_bounds
    padding_x = (maximum_x - minimum_x) * 0.005
    padding_y = (maximum_y - minimum_y) * 0.005

    return (
        float(minimum_x - padding_x),
        float(maximum_x + padding_x),
        float(minimum_y - padding_y),
        float(maximum_y + padding_y),
    )


def prepare_pairwise_spatial_data(
    table: pd.DataFrame,
    reference_model: str,
    candidate_model: str,
    task: str,
) -> pd.DataFrame:
    """Build basin-level pairwise NSE differences for spatial plotting."""
    reference_column = metric_column(reference_model, task, "nse")
    candidate_column = metric_column(candidate_model, task, "nse")

    output = table[["gauge_id", reference_column, candidate_column]].copy()
    output[reference_column] = pd.to_numeric(output[reference_column], errors="coerce")
    output[candidate_column] = pd.to_numeric(output[candidate_column], errors="coerce")
    output["delta_nse"] = output[candidate_column] - output[reference_column]
    return output.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_nse"])


def plot_map_background(
    ax: Axes,
    basins: "gpd.GeoDataFrame",
    states: Optional["gpd.GeoDataFrame"],
) -> None:
    """Draw a consistent basin and state background for spatial figures."""
    basins.plot(
        ax=ax,
        facecolor=BASEMAP_FACE,
        edgecolor="none",
        alpha=0.55,
        zorder=1,
    )

    if states is not None and not states.empty:
        states.boundary.plot(
            ax=ax,
            color=STATE_LINE_COLOR,
            linewidth=0.42,
            zorder=2,
        )


def plot_cgc_minus_hard_spatial_mitigation(table: pd.DataFrame) -> None:
    """Map the spatial difference between CGC and Hard-MTL for both tasks."""
    basins = load_basin_geometries()
    if basins is None:
        return

    states = load_state_boundaries()
    q_difference = prepare_pairwise_spatial_data(
        table,
        reference_model="Hard_MTL",
        candidate_model="CGC",
        task="streamflow",
    )
    et_difference = prepare_pairwise_spatial_data(
        table,
        reference_model="Hard_MTL",
        candidate_model="CGC",
        task="evapotranspiration",
    )

    limit = robust_symmetric_limit(
        [
            q_difference["delta_nse"].to_numpy(dtype=float),
            et_difference["delta_nse"].to_numpy(dtype=float),
        ],
        quantile=0.95,
        minimum=0.05,
        maximum=0.20,
    )
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    figure_height = 4.8
    fig, axes = plt.subplots(1, 2, figsize=(13.0, figure_height))
    extent = projected_conus_extent()
    minimum_x, maximum_x, minimum_y, maximum_y = extent

    configurations = [
        (axes[0], q_difference, "(a) Streamflow"),
        (axes[1], et_difference, "(b) Evapotranspiration"),
    ]

    scatter = None
    for ax, difference, title in configurations:
        merged = basins.merge(difference[["gauge_id", "delta_nse"]], on="gauge_id")
        merged = merged.dropna(subset=["geometry", "delta_nse"]).copy()
        points = merged.copy()
        points["geometry"] = points.geometry.centroid

        plot_map_background(ax, basins, states)
        scatter = ax.scatter(
            points.geometry.x,
            points.geometry.y,
            c=points["delta_nse"].clip(-limit, limit),
            cmap="RdBu_r",
            norm=normalization,
            s=23,
            edgecolors="none",
            alpha=0.92,
            rasterized=True,
            zorder=4,
        )

        positive_rate = float((points["delta_nse"] > 0.0).mean() * 100.0)
        median_difference = float(points["delta_nse"].median())
        ax.set_title(
            f"{title}\nCGC better in {positive_rate:.1f}% of basins; "
            f"median ΔNSE = {median_difference:+.3f}",
            loc="left",
            fontsize=11.0,
        )
        ax.set_xlim(minimum_x, maximum_x)
        ax.set_ylim(minimum_y, maximum_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    if scatter is not None:
        colorbar = fig.colorbar(
            scatter,
            ax=axes,
            orientation="horizontal",
            fraction=0.055,
            pad=0.035,
            aspect=35,
        )
        colorbar.set_label(r"$\Delta$NSE (CGC minus Hard-MTL)")
        colorbar.outline.set_linewidth(0.8)

    fig.suptitle(
        "Spatial distribution of CGC mitigation relative to Hard-MTL",
        fontsize=13.0,
        y=0.995,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12, wspace=0.03)
    save_figure(fig, FIG_DIR / "fig3_5_cgc_minus_hard_spatial_mitigation.png")


# ==============================================================================
# Core Figure 3-6: gate specialization
# ==============================================================================


def expert_sort_key(value: object) -> int:
    """Sort expert identifiers numerically."""
    text = str(value).strip().replace("E", "").replace("e", "")
    try:
        return int(text)
    except ValueError:
        return 10_000


def load_gate_pivot() -> Optional[pd.DataFrame]:
    """Load and normalize mean CGC task-gate utilization by expert."""
    path = GATE_SUMMARY_PATH if GATE_SUMMARY_PATH.exists() else GATE_LONG_PATH
    if not path.exists():
        print("[Skip] Gate utilization table not found.")
        return None

    table = pd.read_csv(path)
    required = {"gate_name", "expert_id", "mean_utilization"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing gate columns in {path}: {sorted(missing)}")

    if "model" in table.columns:
        cgc_only = table[table["model"].astype(str).str.upper() == "CGC"].copy()
        if not cgc_only.empty:
            table = cgc_only

    gate_map = {
        "task_0_gate": "Streamflow gate",
        "task_1_gate": "Evapotranspiration gate",
        "streamflow_gate": "Streamflow gate",
        "evapotranspiration_gate": "Evapotranspiration gate",
        "q_gate": "Streamflow gate",
        "et_gate": "Evapotranspiration gate",
    }

    table["gate_label"] = table["gate_name"].astype(str).map(
        lambda value: gate_map.get(value, value)
    )
    table["expert_id"] = table["expert_id"].astype(str)
    table["mean_utilization"] = pd.to_numeric(
        table["mean_utilization"],
        errors="coerce",
    )
    table = table.dropna(subset=["gate_label", "expert_id", "mean_utilization"])

    pivot = table.pivot_table(
        index="gate_label",
        columns="expert_id",
        values="mean_utilization",
        aggfunc="mean",
    ).fillna(0.0)

    required_gates = ["Streamflow gate", "Evapotranspiration gate"]
    missing_gates = [gate for gate in required_gates if gate not in pivot.index]
    if missing_gates:
        raise ValueError(f"Missing mapped gate rows: {missing_gates}")

    ordered_experts = sorted(pivot.columns, key=expert_sort_key)
    pivot = pivot.loc[required_gates, ordered_experts]

    row_sums = pivot.sum(axis=1).replace(0.0, np.nan)
    pivot = pivot.div(row_sums, axis=0).fillna(0.0)
    return pivot


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Compute normalized Shannon entropy on the interval [0, 1]."""
    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]

    if values.size <= 1:
        return 0.0

    values = values / values.sum()
    entropy = -float(np.sum(values * np.log(values)))
    return entropy / np.log(len(probabilities))


def jensen_shannon_divergence(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Compute base-2 Jensen-Shannon divergence on the interval [0, 1]."""
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    mixture = 0.5 * (p + q)

    def kullback_leibler(left: np.ndarray, right: np.ndarray) -> float:
        mask = (left > 0.0) & (right > 0.0)
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kullback_leibler(p, mixture) + 0.5 * kullback_leibler(q, mixture)


def plot_cgc_gate_specialization() -> None:
    """Plot normalized gate utilization and task-specific routing differences."""
    pivot = load_gate_pivot()
    if pivot is None:
        return

    q_values = pivot.loc["Streamflow gate"].to_numpy(dtype=float)
    et_values = pivot.loc["Evapotranspiration gate"].to_numpy(dtype=float)
    experts = [
        f"E{expert_sort_key(expert)}" if not str(expert).upper().startswith("E") else str(expert)
        for expert in pivot.columns
    ]

    q_entropy = normalized_entropy(q_values)
    et_entropy = normalized_entropy(et_values)
    js_divergence = jensen_shannon_divergence(q_values, et_values)

    summary = pd.DataFrame(
        {
            "metric": [
                "streamflow_gate_entropy",
                "evapotranspiration_gate_entropy",
                "q_et_gate_js_divergence",
            ],
            "value": [q_entropy, et_entropy, js_divergence],
        }
    )
    summary.to_csv(GATE_SPECIALIZATION_PATH, index=False)
    print(f"[Saved] {GATE_SPECIALIZATION_PATH}")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    heatmap_values = np.vstack([q_values, et_values])
    image = axes[0].imshow(
        heatmap_values,
        cmap="Blues",
        vmin=0.0,
        vmax=max(float(heatmap_values.max()), 0.25),
        aspect="auto",
    )
    axes[0].set_xticks(np.arange(len(experts)))
    axes[0].set_xticklabels(experts)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Streamflow gate", "Evapotranspiration gate"])
    axes[0].set_title("(a) Normalized expert utilization", loc="left")

    for row_index in range(heatmap_values.shape[0]):
        for column_index in range(heatmap_values.shape[1]):
            value = heatmap_values[row_index, column_index]
            text_color = "white" if value > heatmap_values.max() * 0.55 else "black"
            axes[0].text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized gate utilization")
    colorbar.outline.set_linewidth(0.8)

    differences = q_values - et_values
    bar_colors = [PALETTE["q"] if value >= 0.0 else PALETTE["et"] for value in differences]
    axes[1].bar(
        np.arange(len(experts)),
        differences,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.70,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_xticks(np.arange(len(experts)))
    axes[1].set_xticklabels(experts)
    axes[1].set_ylabel("Gate difference (Q minus ET)")
    axes[1].set_title("(b) Task-specific routing contrast", loc="left")
    style_axis(axes[1], "y")

    text = (
        f"Normalized entropy\n"
        f"Q gate: {q_entropy:.3f}\n"
        f"ET gate: {et_entropy:.3f}\n\n"
        f"Q–ET JS divergence: {js_divergence:.3f}"
    )
    axes[1].text(
        0.98,
        0.97,
        text,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        bbox={
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "linewidth": 0.7,
            "alpha": 0.94,
            "pad": 3.5,
        },
    )

    legend_handles = [
        Patch(facecolor=PALETTE["q"], edgecolor="black", label="Higher Q utilization"),
        Patch(facecolor=PALETTE["et"], edgecolor="black", label="Higher ET utilization"),
    ]
    axes[1].legend(handles=legend_handles, frameon=False, loc="lower right")

    fig.suptitle(
        "Task-specific expert routing in the CGC architecture",
        fontsize=13.0,
        y=0.995,
    )
    fig.text(
        0.01,
        0.01,
        "Different gate distributions indicate task-specific routing, but do not "
        "by themselves identify the hydrological process represented by an expert.",
        fontsize=8.1,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.16, wspace=0.28)
    save_figure(fig, FIG_DIR / "fig3_6_cgc_gate_specialization.png")


# ==============================================================================
# Supplementary figures
# ==============================================================================


def plot_task_metric_boxplots(
    table: pd.DataFrame,
    task: str,
    models: Sequence[str],
    output_name: str,
    row_title: str,
) -> None:
    """Plot five task-specific evaluation metrics as supplementary boxplots."""
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2))
    axes_flat = axes.ravel()
    colors_by_model = task_model_colors(task)

    for index, (metric, label) in enumerate(METRIC_PANELS):
        ax = axes_flat[index]
        data = collect_metric_series(table, models, task, metric)
        available = [model for model in models if model in data]

        if not available:
            ax.axis("off")
            continue

        values = [data[model] for model in available]
        positions = np.arange(1, len(values) + 1)
        colors = [colors_by_model[model] for model in available]

        box_obj = ax.boxplot(
            [clean_numeric(value).to_numpy(dtype=float) for value in values],
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.52,
            whis=(5, 95),
            tick_labels=[DISPLAY_LABELS[model] for model in available],
        )
        style_boxplot(box_obj, colors)

        ax.set_xlabel(label, fontsize=10.8)
        if label.lower() == "bias":
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.85)

        style_axis(ax, "y")
        annotate_box_medians(ax, values, positions)

    axes_flat[-1].axis("off")
    axes_flat[-1].legend(
        handles=[
            Patch(
                facecolor=colors_by_model[models[0]],
                edgecolor="black",
                label="STL",
            ),
            Patch(
                facecolor=colors_by_model["Hard_MTL"],
                edgecolor="black",
                label="Hard-MTL",
            ),
            Patch(
                facecolor=colors_by_model["MMoE"],
                edgecolor="black",
                label="MMoE",
            ),
            Patch(
                facecolor=colors_by_model["CGC"],
                edgecolor="black",
                label="CGC",
            ),
        ],
        frameon=False,
        loc="center",
    )

    fig.suptitle(row_title, fontsize=13.0, fontstyle="italic", y=0.98)
    fig.text(
        0.01,
        0.01,
        "Outliers are not shown; whiskers represent the 5th and 95th percentiles.",
        fontsize=8.1,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.11,
        wspace=0.38,
        hspace=0.55,
    )
    save_figure(fig, SUPPLEMENTARY_FIG_DIR / output_name)


def plot_supplementary_metric_boxplots(table: pd.DataFrame) -> None:
    """Generate supplementary task-specific metric boxplots."""
    plot_task_metric_boxplots(
        table,
        task="streamflow",
        models=MODELS_Q,
        output_name="figS3_1_streamflow_metrics_boxplot.png",
        row_title="Streamflow performance metrics",
    )
    plot_task_metric_boxplots(
        table,
        task="evapotranspiration",
        models=MODELS_ET,
        output_name="figS3_2_evapotranspiration_metrics_boxplot.png",
        row_title="Evapotranspiration performance metrics",
    )


def empirical_cdf(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted values and empirical cumulative probabilities."""
    array = clean_numeric(values).clip(*NSE_DISPLAY_RANGE).sort_values().to_numpy()
    cumulative = np.arange(1, len(array) + 1, dtype=float) / max(len(array), 1)
    return array, cumulative


def plot_supplementary_nse_cdf(table: pd.DataFrame) -> None:
    """Plot streamflow and evapotranspiration NSE empirical CDFs."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)

    configurations = [
        (axes[0], "streamflow", MODELS_Q, "(a) Streamflow"),
        (axes[1], "evapotranspiration", MODELS_ET, "(b) Evapotranspiration"),
    ]

    for ax, task, models, title in configurations:
        colors = task_model_colors(task)

        for threshold in [0.0, 0.5, 0.75]:
            ax.axvline(
                threshold,
                color="#C8D8E4" if task == "streamflow" else "#F0D1A8",
                linestyle="--",
                linewidth=0.75,
                alpha=0.85,
            )

        data = collect_metric_series(table, models, task, "nse")
        for model in models:
            if model not in data:
                continue

            x_values, y_values = empirical_cdf(data[model])
            high_performance_rate = float((data[model] > 0.75).mean() * 100.0)
            line_width = 1.8 if model == "CGC" else 1.2
            line_style = "-" if model == "CGC" else "--"

            ax.plot(
                x_values,
                y_values,
                color=colors[model],
                linewidth=line_width,
                linestyle=line_style,
                label=f"{DISPLAY_LABELS[model]} (>0.75: {high_performance_rate:.1f}%)",
            )

        ax.set_xlim(*NSE_DISPLAY_RANGE)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("NSE")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="upper left", fontsize=8.2)
        style_axis(ax, "both")

    axes[0].set_ylabel("Cumulative fraction")
    fig.suptitle("Empirical NSE distributions", fontsize=13.0, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.14, wspace=0.16)
    save_figure(fig, SUPPLEMENTARY_FIG_DIR / "figS3_3_nse_cdf.png")


def plot_supplementary_cgc_vs_stl(
    table: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
) -> None:
    """Plot supplementary CGC-versus-STL basin-level comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))

    configurations = [
        (
            axes[0],
            "streamflow",
            "STL_Q",
            "(a) Streamflow",
            "STL-Q NSE",
            "CGC streamflow NSE",
        ),
        (
            axes[1],
            "evapotranspiration",
            "STL_ET",
            "(b) Evapotranspiration",
            "STL-ET NSE",
            "CGC evapotranspiration NSE",
        ),
    ]

    for ax, task, reference_model, title, xlabel, ylabel in configurations:
        data, reference_column, candidate_column = prepare_pairwise_model_data(
            table,
            reference_model=reference_model,
            candidate_model="CGC",
            task=task,
        )

        stats_row = pairwise_stats[
            (pairwise_stats["task"] == task)
            & (pairwise_stats["comparison"] == "CGC minus STL")
        ]
        statistics = stats_row.iloc[0].to_dict()

        plot_pairwise_scatter_panel(
            ax=ax,
            data=data,
            reference_column=reference_column,
            candidate_column=candidate_column,
            task=task,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            statistics=statistics,
        )

    fig.suptitle(
        "Supplementary basin-level comparison between CGC and STL",
        fontsize=13.0,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.22)
    save_figure(fig, SUPPLEMENTARY_FIG_DIR / "figS3_4_cgc_vs_stl_pairwise_comparison.png")


# ==============================================================================
# Main execution
# ==============================================================================


def main() -> None:
    """Generate hypothesis-driven Chapter 3 figures and analysis tables."""
    print("=" * 100)
    print("Chapter 3 Negative-Transfer Figure Generator")
    print("=" * 100)

    set_publication_style()
    per_basin = load_per_basin_table()
    transfer = build_transfer_analysis_table(per_basin)
    pairwise_stats = export_pairwise_statistics(transfer)
    negative_transfer_rates = compute_negative_transfer_rates(transfer)
    joint_transfer_summary = build_joint_transfer_summary(transfer)

    # Core figures for the thesis and manuscript.
    plot_overall_nse_performance(per_basin, pairwise_stats)
    plot_negative_transfer_diagnosis(transfer, negative_transfer_rates)
    plot_joint_task_transfer_quadrants(transfer, joint_transfer_summary)
    plot_cgc_vs_hard_pairwise_comparison(per_basin, pairwise_stats)
    plot_cgc_minus_hard_spatial_mitigation(per_basin)
    plot_cgc_gate_specialization()

    # Supplementary or appendix figures.
    plot_supplementary_metric_boxplots(per_basin)
    plot_supplementary_nse_cdf(per_basin)
    plot_supplementary_cgc_vs_stl(per_basin, pairwise_stats)

    print("=" * 100)
    print("Chapter 3 figure generation completed.")
    print(f"Core figure directory: {FIG_DIR}")
    print(f"Supplementary figure directory: {SUPPLEMENTARY_FIG_DIR}")
    print(f"Summary directory: {SUMMARY_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()