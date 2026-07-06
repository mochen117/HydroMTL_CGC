# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Generate publication-quality Chapter 3 figures for hydrological multi-task
#   learning experiments. The figure set is organized around task asymmetry,
#   negative-transfer diagnosis, joint Q-ET outcomes, basin-level paired model
#   comparisons, spatial heterogeneity, and CGC gate specialization.
#
# Publication-layout principles:
#   1. Use compact double-column dimensions, embedded vector fonts, and 600 dpi PNG.
#   2. Keep panel titles concise; detailed interpretation belongs in the caption.
#   3. Use standard Tukey whiskers (1.5 IQR) with outliers hidden; derive each
#      task-specific axis from the whiskers rather than from extreme tails.
#   4. Use task-specific limits for Q and ET. Limits are shared only across panels
#      that display the same task and therefore require direct comparison.
#   5. Report paired effect size, uncertainty, and significance without overstating
#      small numerical improvements.
#   6. Compare only genuinely shared experts across tasks. Task-specific experts
#      are displayed separately because their network parameters are not identical.
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
from matplotlib.ticker import MaxNLocator, MultipleLocator

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
FIGURE_VARIANT = "publication_adaptive_focus_v5"
BASE_FIG_DIR = CH3_DIR / "figures"
FIG_DIR = BASE_FIG_DIR / FIGURE_VARIANT
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

# Journal-style figure dimensions (inches; approximately 180 mm wide).
DOUBLE_COLUMN_WIDTH = 7.09
FIGSIZE_TWO_PANEL = (DOUBLE_COLUMN_WIDTH, 3.65)
FIGSIZE_FOUR_PANEL = (DOUBLE_COLUMN_WIDTH, 5.45)
FIGSIZE_THREE_PANEL = (DOUBLE_COLUMN_WIDTH, 2.90)
FIGSIZE_MAP = (DOUBLE_COLUMN_WIDTH, 3.10)
FIGSIZE_GATE = (DOUBLE_COLUMN_WIDTH, 3.30)

# Display configuration. Statistical calculations always use raw values.
BOX_RANGE_PADDING_RATIO = 0.08
TUKEY_WHISKER_IQR = 1.5
NSE_DISPLAY_LOWER = -1.0
NSE_DISPLAY_UPPER = 1.0
MINIMUM_TRANSFER_LIMIT_Q = 0.20
MINIMUM_TRANSFER_LIMIT_ET = 0.08
MINIMUM_JOINT_LIMIT = 0.45
PAIRWISE_NSE_RANGE = (-1.0, 1.0)
SPATIAL_DELTA_RANGE = (-0.20, 0.20)

# CGC expert configuration used by the current experiments.
N_SHARED_EXPERTS = 4
N_TASK_SPECIFIC_EXPERTS = 4


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
    """Set global Matplotlib parameters for thesis and journal figures."""
    font_name = choose_serif_font()
    plt.rcParams.update(
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
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    print(f"[Info] Figure font: {font_name}")

def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save one figure as a high-resolution PNG and an editable vector PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(
            path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
    finally:
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
    """Apply a restrained publication-style axis treatment."""
    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.45,
        color=GRID_COLOR,
        alpha=0.55,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

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
    """Annotate medians with a small offset and a non-obtrusive background."""
    for x_value, series in zip(positions, values):
        array = clean_numeric(series).to_numpy(dtype=float)
        if array.size == 0:
            continue
        median = float(np.median(array))
        ax.annotate(
            f"{median:.{decimals}f}",
            xy=(x_value, median),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="semibold",
            color="black",
            clip_on=True,
            zorder=10,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.15,
            },
        )


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


def tukey_whisker_envelope(values: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Return the global Tukey-whisker envelope across numeric arrays."""
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
    padding_ratio: float = BOX_RANGE_PADDING_RATIO,
    minimum_span: float = 0.10,
) -> Tuple[float, float]:
    """Add proportional padding while enforcing a minimum readable span."""
    span = max(upper - lower, minimum_span)
    center = 0.5 * (lower + upper)
    if upper - lower < minimum_span:
        lower = center - 0.5 * minimum_span
        upper = center + 0.5 * minimum_span
        span = minimum_span
    padding = span * padding_ratio
    return float(lower - padding), float(upper + padding)


def nse_boxplot_limits(values: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Return task-specific NSE limits from Tukey whiskers, bounded to [-1, 1]."""
    lower, upper = tukey_whisker_envelope(values)
    lower, upper = padded_limits(lower, upper, minimum_span=0.50)
    lower = max(NSE_DISPLAY_LOWER, lower)
    upper = min(NSE_DISPLAY_UPPER, upper)
    if upper - lower < 0.45:
        center = np.clip(0.5 * (lower + upper), -0.75, 0.75)
        lower = max(NSE_DISPLAY_LOWER, center - 0.225)
        upper = min(NSE_DISPLAY_UPPER, center + 0.225)
    return float(lower), float(upper)


def focused_nse_limits(
    values: Sequence[np.ndarray],
    task: str,
) -> Tuple[float, float]:
    """Return a readable task-specific NSE range while keeping Tukey whiskers visible.

    The lower bound is now anchored to the observed Tukey-whisker minimum instead
    of an aggressively clipped task-specific floor. This change preserves the full
    visible lower whiskers for MMoE and CGC in Figure 3-1 while still avoiding
    excessive white space through mild proportional padding. Statistics are always
    computed from the raw values.
    """
    statistics = [tukey_box_statistics(values_i) for values_i in values]
    statistics = [item for item in statistics if np.isfinite(item["median"])]
    if not statistics:
        return NSE_DISPLAY_RANGE

    q1_min = min(item["q1"] for item in statistics)
    q3_max = max(item["q3"] for item in statistics)
    whisker_min = min(item["lower_whisker"] for item in statistics)
    whisker_max = max(item["upper_whisker"] for item in statistics)

    central_span = max(q3_max - q1_min, 0.20)
    lower = min(whisker_min, q1_min - 0.18 * central_span)
    upper = max(whisker_max, q3_max + 0.14 * central_span)

    if task == "streamflow":
        lower = max(-1.20, lower - 0.04 * central_span)
        upper = min(1.00, upper + 0.03 * central_span)
    elif task == "evapotranspiration":
        lower = max(-1.20, lower - 0.04 * central_span)
        upper = min(1.00, upper + 0.03 * central_span)
    else:
        raise ValueError(f"Unsupported task: {task}")

    if upper - lower < 0.60:
        center = 0.5 * (upper + lower)
        lower = max(-1.20, center - 0.30)
        upper = min(1.00, center + 0.30)

    return float(lower), float(upper)


def metric_boxplot_limits(
    values: Sequence[np.ndarray],
    metric: str,
) -> Tuple[float, float]:
    """Return metric-aware limits using the same Tukey rule as the boxplot."""
    lower, upper = tukey_whisker_envelope(values)

    if metric == "bias":
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    elif metric == "rmse":
        lower = max(0.0, lower)
    elif metric == "corr":
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    elif metric in {"nse", "kge"}:
        lower = max(NSE_DISPLAY_LOWER, lower)
        upper = min(NSE_DISPLAY_UPPER, upper)

    lower, upper = padded_limits(lower, upper, minimum_span=0.20)
    if metric == "rmse":
        lower = max(0.0, lower)
    elif metric == "corr":
        lower, upper = max(0.0, lower), min(1.0, upper)
    elif metric in {"nse", "kge"}:
        lower, upper = max(-1.0, lower), min(1.0, upper)
    return float(lower), float(upper)


def symmetric_tukey_limit(
    values: Sequence[np.ndarray],
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    """Return a symmetric zero-centered limit from task-specific Tukey whiskers."""
    lower, upper = tukey_whisker_envelope(values)
    limit = max(abs(lower), abs(upper), minimum) * (1.0 + BOX_RANGE_PADDING_RATIO)
    if maximum is not None:
        limit = min(limit, maximum)
    return float(limit)


def robust_symmetric_limit(
    values: Sequence[np.ndarray],
    quantile: float = 0.99,
    minimum: float = 0.10,
    maximum: float = 0.75,
) -> float:
    """Estimate a robust symmetric limit for scatter and map visualizations."""
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    combined = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    combined = combined[np.isfinite(combined)]

    if combined.size == 0:
        return minimum

    limit = float(np.quantile(np.abs(combined), quantile)) * 1.05
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
    """Plot task-specific NSE distributions and CGC gains relative to STL.

    Q and ET use independent vertical scales. Paired statistical summaries are
    placed below the plotting area so that no annotation obscures a boxplot.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE_TWO_PANEL,
        sharey=False,
        constrained_layout=False,
    )

    configurations = [
        (axes[0], "streamflow", MODELS_Q, "(a) Streamflow (Q)"),
        (axes[1], "evapotranspiration", MODELS_ET, "(b) Evapotranspiration (ET)"),
    ]

    summary_lines: List[str] = []

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
            widths=0.62,
            whis=TUKEY_WHISKER_IQR,
            tick_labels=[DISPLAY_LABELS[model] for model in available],
            manage_ticks=False,
        )
        style_boxplot(box_obj, colors)

        y_limits = focused_nse_limits(
            [value.to_numpy(dtype=float) for value in values],
            task,
        )
        ax.set_ylim(*y_limits)
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.75,
            clip_on=True,
            zorder=1,
        )
        ax.set_xlim(0.55, len(values) + 0.45)
        ax.set_xticks(positions)
        ax.set_xticklabels([DISPLAY_LABELS[model] for model in available])
        ax.set_title(title, loc="left", pad=4)
        ax.set_ylabel("NSE")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
        style_axis(ax, "y")
        annotate_box_medians(ax, values, positions)

        task_stats = pairwise_stats[
            (pairwise_stats["task"] == task)
            & (pairwise_stats["comparison"] == "CGC minus STL")
        ]
        if task_stats.empty:
            raise ValueError(f"Missing CGC-STL paired statistics for task: {task}")

        row = task_stats.iloc[0]
        summary_lines.append(
            "CGC − task-specific STL: "
            f"median ΔNSE {row['median_difference']:+.3f} "
            f"(95% CI {row['ci_low']:+.3f} to {row['ci_high']:+.3f})\n"
            f"Improved basins {row['win_rate']:.1f}%; "
            f"Wilcoxon p {format_p_value(row['wilcoxon_p'])}"
        )

    # Reserve a dedicated statistics band below the axes.
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.95,
        bottom=0.255,
        wspace=0.26,
    )

    for ax, text_value in zip(axes, summary_lines):
        position = ax.get_position()
        fig.text(
            position.x0,
            0.045,
            text_value,
            ha="left",
            va="bottom",
            fontsize=6.7,
            linespacing=1.25,
            color=EDGE_COLOR,
            bbox={
                "facecolor": "white",
                "edgecolor": "#C4C4C4",
                "linewidth": 0.50,
                "alpha": 1.0,
                "pad": 2.2,
            },
        )

    fig.text(
        0.075,
        0.012,
        "Boxes show P25-P75; center lines are medians; whiskers follow the Tukey 1.5-IQR rule. "
        "Whisker segments outside the displayed task-specific NSE range are clipped; all statistics use raw values.",
        fontsize=6.1,
        color="#666666",
    )
    save_figure(fig, FIG_DIR / "fig3_1_overall_nse_performance.png")

# ==============================================================================
# Core Figure 3-2: negative-transfer diagnosis
# ==============================================================================


def plot_negative_transfer_diagnosis(
    transfer: pd.DataFrame,
    rate_summary: pd.DataFrame,
) -> None:
    """Plot transfer-effect distributions and severity-dependent degradation risk."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 5.65),
        constrained_layout=False,
    )

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
    panel_titles = {
        "streamflow": ("(a) Streamflow transfer effect", "(c) Streamflow degradation risk"),
        "evapotranspiration": (
            "(b) Evapotranspiration transfer effect",
            "(d) Evapotranspiration degradation risk",
        ),
    }

    transfer_limits: Dict[str, float] = {}
    rate_uppers: Dict[str, float] = {}
    for task in TASKS:
        task_values = [
            clean_numeric(transfer[column]).to_numpy(dtype=float)
            for column in task_columns[task].values()
        ]
        transfer_limits[task] = symmetric_tukey_limit(
            task_values,
            minimum=(MINIMUM_TRANSFER_LIMIT_ET if task == "evapotranspiration" else MINIMUM_TRANSFER_LIMIT_Q),
            maximum=0.50,
        )

        task_rates = rate_summary.loc[
            rate_summary["task"] == task, "negative_transfer_rate"
        ]
        maximum_rate = float(task_rates.max()) if not task_rates.empty else 0.0
        rate_uppers[task] = min(
            100.0,
            max(20.0, np.ceil((maximum_rate + 5.0) / 10.0) * 10.0),
        )

    for column_index, task in enumerate(TASKS):
        ax_box = axes[0, column_index]
        columns = task_columns[task]
        models = list(columns.keys())
        values = [clean_numeric(transfer[columns[model]]) for model in models]
        positions = np.arange(1, len(models) + 1)

        box_obj = ax_box.boxplot(
            [value.to_numpy(dtype=float) for value in values],
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.64,
            whis=TUKEY_WHISKER_IQR,
            tick_labels=[DISPLAY_LABELS[model] for model in models],
        )
        style_boxplot(
            box_obj,
            [task_model_colors(task)[model] for model in models],
        )
        ax_box.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        task_transfer_limit = transfer_limits[task]
        ax_box.set_ylim(-task_transfer_limit, task_transfer_limit)
        ax_box.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_box.set_title(panel_titles[task][0], loc="left", pad=4)
        ax_box.set_ylabel(r"$\Delta$NSE relative to STL")
        style_axis(ax_box, "y")
        annotate_box_medians(ax_box, values, positions, decimals=2)

        ax_rate = axes[1, column_index]
        subset = rate_summary[rate_summary["task"] == task].copy()
        threshold_labels = ["Any\n(< 0)", "Moderate\n(< −0.05)", "Severe\n(< −0.10)"]

        for model in MTL_MODELS:
            model_data = subset[subset["model"] == model].sort_values("threshold")
            if model_data.empty:
                continue
            ax_rate.plot(
                np.arange(len(model_data)),
                model_data["negative_transfer_rate"],
                marker="o",
                markersize=4.2 if model == "CGC" else 3.7,
                linewidth=1.65 if model == "CGC" else 1.10,
                color=TRANSFER_MODEL_COLORS[model],
                label=DISPLAY_LABELS[model],
            )

        ax_rate.set_xticks(np.arange(len(NEGATIVE_TRANSFER_THRESHOLDS)))
        ax_rate.set_xticklabels(threshold_labels)
        ax_rate.set_ylim(0.0, rate_uppers[task])
        ax_rate.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax_rate.set_ylabel("Negative-transfer rate (%)")
        ax_rate.set_title(panel_titles[task][1], loc="left", pad=4)
        style_axis(ax_rate, "y")

    axes[1, 1].legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        handlelength=2.0,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.09, wspace=0.25, hspace=0.34)
    save_figure(fig, FIG_DIR / "fig3_2_negative_transfer_diagnosis.png")

# ==============================================================================
# Core Figure 3-3: joint Q-ET transfer quadrants
# ==============================================================================


def plot_joint_task_transfer_quadrants(
    transfer: pd.DataFrame,
    joint_summary: pd.DataFrame,
) -> None:
    """Plot joint Q-ET transfer quadrants with a dedicated lower caption band."""
    model_columns = {
        "Hard_MTL": ("hard_delta_q", "hard_delta_et"),
        "MMoE": ("mmoe_delta_q", "mmoe_delta_et"),
        "CGC": ("cgc_delta_q", "cgc_delta_et"),
    }

    q_values = pd.concat(
        [transfer[q_column] for q_column, _ in model_columns.values()],
        ignore_index=True,
    ).dropna()
    et_values = pd.concat(
        [transfer[et_column] for _, et_column in model_columns.values()],
        ignore_index=True,
    ).dropna()
    q_limit = robust_symmetric_limit(
        q_values, quantile=0.99, minimum=MINIMUM_JOINT_LIMIT, maximum=0.75
    )
    et_limit = robust_symmetric_limit(
        et_values, quantile=0.99, minimum=0.20, maximum=0.75
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGSIZE_THREE_PANEL,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for panel_index, (ax, (model, (q_column, et_column))) in enumerate(
        zip(axes, model_columns.items())
    ):
        data = transfer[[q_column, et_column]].copy()
        data.columns = ["delta_q", "delta_et"]
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data["outcome"] = classify_joint_transfer(data["delta_q"], data["delta_et"])

        visible = (
            data["delta_q"].between(-q_limit, q_limit)
            & data["delta_et"].between(-et_limit, et_limit)
        )
        plot_data = data.loc[visible]

        for outcome in [
            "both_improved",
            "q_degraded_et_improved",
            "both_degraded",
            "q_improved_et_degraded",
            "near_zero",
        ]:
            subset = plot_data[plot_data["outcome"] == outcome]
            if subset.empty:
                continue
            ax.scatter(
                subset["delta_q"],
                subset["delta_et"],
                s=10.0,
                alpha=0.46,
                color=QUADRANT_COLORS[outcome],
                edgecolor="none",
                rasterized=True,
                zorder=3,
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax.set_xlim(-q_limit, q_limit)
        ax.set_ylim(-et_limit, et_limit)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(f"({chr(97 + panel_index)}) {DISPLAY_LABELS[model]}", loc="left", pad=4)
        style_axis(ax, "both")

        summary = joint_summary[joint_summary["model"] == model].set_index("outcome")
        both_gain = float(summary.loc["both_improved", "percentage"])
        both_loss = float(summary.loc["both_degraded", "percentage"])
        outside_count = int((~visible).sum())
        text = f"Both improved: {both_gain:.1f}%\nBoth degraded: {both_loss:.1f}%"
        if outside_count > 0:
            text += f"\nOutside range: {outside_count}"
        ax.text(
            0.04,
            0.96,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            bbox={
                "facecolor": "white",
                "edgecolor": "#BFBFBF",
                "linewidth": 0.5,
                "alpha": 0.94,
                "pad": 2.3,
            },
        )

    axes[0].set_ylabel(r"$\Delta$NSE$_{ET}$ relative to STL-ET")
    fig.supxlabel(r"$\Delta$NSE$_Q$ relative to STL-Q", y=0.105)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.8,
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
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=0.9,
        handletextpad=0.35,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.24, wspace=0.18)
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


def paired_scatter_display_range(
    data: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    *,
    lower_cap: float = -2.0,
    upper_cap: float = 1.05,
) -> Tuple[float, float]:
    """Return a task-specific square display range for one paired scatter."""
    combined = np.concatenate(
        [
            clean_numeric(data[reference_column]).to_numpy(dtype=float),
            clean_numeric(data[candidate_column]).to_numpy(dtype=float),
        ]
    )
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return PAIRWISE_NSE_RANGE

    lower = min(-1.0, float(np.quantile(combined, 0.01)) - 0.03)
    upper = max(1.0, float(np.quantile(combined, 0.99)) + 0.03)
    return max(lower, lower_cap), min(upper, upper_cap)


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
    display_range: Tuple[float, float],
) -> None:
    """Plot one paired model comparison with concise robust statistics."""
    low, high = display_range
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
        s=11.0,
        alpha=0.48,
        color=PALETTE["gain"],
        edgecolor="none",
        rasterized=True,
        label="CGC better",
    )
    ax.scatter(
        degraded[reference_column],
        degraded[candidate_column],
        s=11.0,
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
            s=11.0,
            alpha=0.45,
            color="#9E9E9E",
            edgecolor="none",
            rasterized=True,
            label="No change",
        )

    ax.plot([low, high], [low, high], "k--", linewidth=0.85)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax, "both")

    outside_count = int((~visible).sum())
    text = (
        f"Improved basins = {statistics['win_rate']:.1f}%\n"
        f"Median ΔNSE = {statistics['median_difference']:+.3f}\n"
        f"95% CI [{statistics['ci_low']:+.3f}, {statistics['ci_high']:+.3f}]\n"
        f"Wilcoxon p {format_p_value(statistics['wilcoxon_p'])}"
    )
    if outside_count > 0:
        text += f"\nOutside range = {outside_count}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        bbox={
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "linewidth": 0.55,
            "alpha": 0.94,
            "pad": 2.5,
        },
    )

def plot_cgc_vs_hard_pairwise_comparison(
    table: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
) -> None:
    """Plot direct basin-level CGC-versus-Hard-MTL NSE comparisons."""
    prepared: Dict[str, Tuple[pd.DataFrame, str, str, Tuple[float, float]]] = {}
    for task in TASKS:
        data, reference_column, candidate_column = prepare_pairwise_model_data(
            table,
            reference_model="Hard_MTL",
            candidate_model="CGC",
            task=task,
        )
        display_range = paired_scatter_display_range(
            data, reference_column, candidate_column
        )
        prepared[task] = (data, reference_column, candidate_column, display_range)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 3.55),
        constrained_layout=False,
    )
    configurations = [
        (
            axes[0],
            "streamflow",
            "(a) Streamflow (Q)",
            "Hard-MTL NSE",
            "CGC NSE",
        ),
        (
            axes[1],
            "evapotranspiration",
            "(b) Evapotranspiration (ET)",
            "Hard-MTL NSE",
            "CGC NSE",
        ),
    ]

    for ax, task, title, xlabel, ylabel in configurations:
        data, reference_column, candidate_column, display_range = prepared[task]
        stats_row = pairwise_stats[
            (pairwise_stats["task"] == task)
            & (pairwise_stats["comparison"] == "CGC minus Hard-MTL")
        ]
        if stats_row.empty:
            raise ValueError(f"Missing CGC-Hard statistics for task: {task}")

        plot_pairwise_scatter_panel(
            ax=ax,
            data=data,
            reference_column=reference_column,
            candidate_column=candidate_column,
            task=task,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            statistics=stats_row.iloc[0].to_dict(),
            display_range=display_range,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALETTE["gain"],
            markeredgecolor="none",
            markersize=5,
            label="CGC better than Hard-MTL",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALETTE["loss"],
            markeredgecolor="none",
            markersize=5,
            label="CGC worse than Hard-MTL",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
    )
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
    """Map basin-level CGC-minus-Hard-MTL NSE differences for both tasks."""
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

    normalization = TwoSlopeNorm(
        vmin=SPATIAL_DELTA_RANGE[0],
        vcenter=0.0,
        vmax=SPATIAL_DELTA_RANGE[1],
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE_MAP,
        constrained_layout=True,
    )
    minimum_x, maximum_x, minimum_y, maximum_y = projected_conus_extent()

    configurations = [
        (axes[0], q_difference, "(a) Streamflow (Q)"),
        (axes[1], et_difference, "(b) Evapotranspiration (ET)"),
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
            c=points["delta_nse"].clip(*SPATIAL_DELTA_RANGE),
            cmap="RdBu_r",
            norm=normalization,
            s=10.0,
            edgecolors="none",
            alpha=0.93,
            rasterized=True,
            zorder=4,
        )

        positive_rate = float((points["delta_nse"] > 0.0).mean() * 100.0)
        median_difference = float(points["delta_nse"].median())
        ax.set_title(
            f"{title}\nImproved basins = {positive_rate:.1f}%; median ΔNSE = {median_difference:+.3f}",
            loc="left",
            fontsize=8.4,
            pad=3,
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
            pad=0.025,
            aspect=36,
        )
        colorbar.set_label(r"$\Delta$NSE (CGC − Hard-MTL)")
        colorbar.set_ticks(np.linspace(-0.20, 0.20, 5))
        colorbar.outline.set_linewidth(0.7)

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
    """Plot shared and task-specific CGC gate utilization with explicit margins."""
    pivot = load_gate_pivot()
    if pivot is None:
        return

    if pivot.shape[1] < N_SHARED_EXPERTS + N_TASK_SPECIFIC_EXPERTS:
        raise ValueError(
            "The gate table contains fewer experts than expected for the current CGC configuration: "
            f"found {pivot.shape[1]}, expected at least "
            f"{N_SHARED_EXPERTS + N_TASK_SPECIFIC_EXPERTS}."
        )

    q_values = pivot.loc["Streamflow gate"].to_numpy(dtype=float)
    et_values = pivot.loc["Evapotranspiration gate"].to_numpy(dtype=float)

    q_shared = q_values[:N_SHARED_EXPERTS]
    et_shared = et_values[:N_SHARED_EXPERTS]
    q_specific = q_values[N_SHARED_EXPERTS:N_SHARED_EXPERTS + N_TASK_SPECIFIC_EXPERTS]
    et_specific = et_values[N_SHARED_EXPERTS:N_SHARED_EXPERTS + N_TASK_SPECIFIC_EXPERTS]

    q_entropy = normalized_entropy(q_values)
    et_entropy = normalized_entropy(et_values)
    q_shared_mass = float(q_shared.sum())
    et_shared_mass = float(et_shared.sum())

    q_shared_conditional = q_shared / max(q_shared_mass, 1e-12)
    et_shared_conditional = et_shared / max(et_shared_mass, 1e-12)
    shared_js = jensen_shannon_divergence(q_shared_conditional, et_shared_conditional)

    summary = pd.DataFrame(
        {
            "metric": [
                "streamflow_gate_entropy",
                "evapotranspiration_gate_entropy",
                "streamflow_shared_gate_mass",
                "evapotranspiration_shared_gate_mass",
                "shared_expert_conditional_js_divergence",
            ],
            "value": [
                q_entropy,
                et_entropy,
                q_shared_mass,
                et_shared_mass,
                shared_js,
            ],
        }
    )
    summary.to_csv(GATE_SPECIALIZATION_PATH, index=False)
    print(f"[Saved] {GATE_SPECIALIZATION_PATH}")

    fig = plt.figure(figsize=FIGSIZE_GATE, constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.11,
        width_ratios=[1.22, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.38,
        hspace=0.42,
    )
    shared_ax = fig.add_subplot(grid[:, 0])
    q_ax = fig.add_subplot(grid[0, 1])
    et_ax = fig.add_subplot(grid[1, 1])

    shared_matrix = np.vstack([q_shared, et_shared])
    image = shared_ax.imshow(
        shared_matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=max(float(shared_matrix.max()), 0.10),
        aspect="auto",
    )
    shared_ax.set_xticks(np.arange(N_SHARED_EXPERTS))
    shared_ax.set_xticklabels([f"S{i + 1}" for i in range(N_SHARED_EXPERTS)])
    shared_ax.set_yticks([0, 1])
    shared_ax.set_yticklabels(["Q gate", "ET gate"])
    shared_ax.set_title("(a) Shared-expert utilization", loc="left", pad=4)

    for row_index in range(shared_matrix.shape[0]):
        for column_index in range(shared_matrix.shape[1]):
            value = shared_matrix[row_index, column_index]
            shared_ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if value > shared_matrix.max() * 0.55 else "black",
            )

    colorbar = fig.colorbar(image, ax=shared_ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Gate weight")
    colorbar.outline.set_linewidth(0.7)

    q_positions = np.arange(N_TASK_SPECIFIC_EXPERTS)
    q_ax.bar(
        q_positions,
        q_specific,
        color=PALETTE["q"],
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        width=0.68,
    )
    q_ax.set_xticks(q_positions)
    q_ax.set_xticklabels([f"Q{i + 1}" for i in range(N_TASK_SPECIFIC_EXPERTS)])
    q_ax.set_ylabel("Gate weight")
    q_ax.set_title("(b) Q-specific experts", loc="left", pad=4)
    q_ax.set_ylim(0.0, max(0.80, float(q_specific.max()) * 1.15))
    q_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_axis(q_ax, "y")

    et_positions = np.arange(N_TASK_SPECIFIC_EXPERTS)
    et_ax.bar(
        et_positions,
        et_specific,
        color=PALETTE["et"],
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        width=0.68,
    )
    et_ax.set_xticks(et_positions)
    et_ax.set_xticklabels([f"ET{i + 1}" for i in range(N_TASK_SPECIFIC_EXPERTS)])
    et_ax.set_ylabel("Gate weight")
    et_ax.set_title("(c) ET-specific experts", loc="left", pad=4)
    et_ax.set_ylim(0.0, max(0.80, float(et_specific.max()) * 1.15))
    et_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_axis(et_ax, "y")

    shared_ax.text(
        0.98,
        0.98,
        (
            f"Q entropy = {q_entropy:.3f}\n"
            f"ET entropy = {et_entropy:.3f}\n"
            f"Q shared mass = {q_shared_mass:.3f}\n"
            f"ET shared mass = {et_shared_mass:.3f}\n"
            f"Shared-expert JS = {shared_js:.3f}"
        ),
        transform=shared_ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        bbox={
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "linewidth": 0.5,
            "alpha": 0.95,
            "pad": 2.4,
        },
    )

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
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, 4.75),
        constrained_layout=True,
    )
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
        numeric_values = [clean_numeric(value).to_numpy(dtype=float) for value in values]
        positions = np.arange(1, len(values) + 1)
        colors = [colors_by_model[model] for model in available]

        box_obj = ax.boxplot(
            numeric_values,
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            whis=TUKEY_WHISKER_IQR,
            tick_labels=[DISPLAY_LABELS[model] for model in available],
        )
        style_boxplot(box_obj, colors)

        ax.set_ylim(*metric_boxplot_limits(numeric_values, metric))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))

        ax.set_title(f"({chr(97 + index)}) {label}", loc="left", pad=3)
        if metric == "bias":
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.75)
        style_axis(ax, "y")
        annotate_box_medians(ax, values, positions)

    axes_flat[-1].axis("off")
    axes_flat[-1].legend(
        handles=[
            Patch(facecolor=colors_by_model[models[0]], edgecolor=EDGE_COLOR, label="STL"),
            Patch(facecolor=colors_by_model["Hard_MTL"], edgecolor=EDGE_COLOR, label="Hard-MTL"),
            Patch(facecolor=colors_by_model["MMoE"], edgecolor=EDGE_COLOR, label="MMoE"),
            Patch(facecolor=colors_by_model["CGC"], edgecolor=EDGE_COLOR, label="CGC"),
        ],
        frameon=False,
        loc="center",
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
    """Return an empirical CDF calculated from the unmodified raw values."""
    array = clean_numeric(values).sort_values().to_numpy(dtype=float)
    if array.size == 0:
        return array, array
    probabilities = np.arange(1, array.size + 1, dtype=float) / array.size
    return array, probabilities

def plot_supplementary_nse_cdf(table: pd.DataFrame) -> None:
    """Plot streamflow and evapotranspiration NSE empirical CDFs."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE_TWO_PANEL,
        sharey=True,
        constrained_layout=True,
    )

    configurations = [
        (axes[0], "streamflow", MODELS_Q, "(a) Streamflow (Q)"),
        (axes[1], "evapotranspiration", MODELS_ET, "(b) Evapotranspiration (ET)"),
    ]

    for ax, task, models, title in configurations:
        colors = task_model_colors(task)
        for threshold in [0.0, 0.5, 0.75]:
            ax.axvline(
                threshold,
                color="#C8D8E4" if task == "streamflow" else "#F0D1A8",
                linestyle="--",
                linewidth=0.65,
                alpha=0.85,
            )

        data = collect_metric_series(table, models, task, "nse")
        for model in models:
            if model not in data:
                continue
            x_values, y_values = empirical_cdf(data[model])
            high_rate = float((data[model] > 0.75).mean() * 100.0)
            ax.plot(
                x_values,
                y_values,
                color=colors[model],
                linewidth=1.55 if model == "CGC" else 0.95,
                linestyle="-" if model == "CGC" else "--",
                label=f"{DISPLAY_LABELS[model]} (>0.75: {high_rate:.1f}%)",
            )

        task_values = [
            clean_numeric(series).to_numpy(dtype=float) for series in data.values()
        ]
        combined = np.concatenate(task_values) if task_values else np.asarray([], dtype=float)
        combined = combined[np.isfinite(combined)]
        lower = -1.0 if combined.size == 0 else max(-2.0, float(np.quantile(combined, 0.01)) - 0.05)
        ax.set_xlim(lower, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("NSE")
        ax.set_title(title, loc="left", pad=4)
        ax.legend(frameon=False, loc="upper left", fontsize=6.9)
        style_axis(ax, "both")

    axes[0].set_ylabel("Cumulative fraction")
    save_figure(fig, SUPPLEMENTARY_FIG_DIR / "figS3_3_nse_cdf.png")

def plot_supplementary_cgc_vs_stl(
    table: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
) -> None:
    """Plot supplementary basin-level CGC-versus-STL comparisons."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE_TWO_PANEL,
        constrained_layout=False,
    )

    configurations = [
        (axes[0], "streamflow", "STL_Q", "(a) Streamflow (Q)", "STL-Q NSE", "CGC NSE"),
        (
            axes[1],
            "evapotranspiration",
            "STL_ET",
            "(b) Evapotranspiration (ET)",
            "STL-ET NSE",
            "CGC NSE",
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
        plot_pairwise_scatter_panel(
            ax=ax,
            data=data,
            reference_column=reference_column,
            candidate_column=candidate_column,
            task=task,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            statistics=stats_row.iloc[0].to_dict(),
            display_range=paired_scatter_display_range(
                data, reference_column, candidate_column
            ),
        )

    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.14, wspace=0.24)
    save_figure(fig, SUPPLEMENTARY_FIG_DIR / "figS3_4_cgc_vs_stl_pairwise_comparison.png")

# ==============================================================================
# Main execution
# ==============================================================================


def main() -> None:
    """Generate hypothesis-driven Chapter 3 figures and analysis tables."""
    print("=" * 100)
    print("Chapter 3 Publication Figure Generator: task-specific axes")
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