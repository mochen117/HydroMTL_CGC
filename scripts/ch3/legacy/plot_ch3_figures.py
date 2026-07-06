# ==============================================================================
# Description:
#   Generate Chapter 3 non-spatial figures for thesis-ready model comparison.
#
# Purpose:
#   Produce publication-ready model performance figures for streamflow and
#   evapotranspiration simulations. The main figure compares basin-scale
#   distributions of Bias, RMSE, Corr, NSE, and KGE across STL and MTL models.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_all_models.csv
#   - experiments/formal_ch3_modeling/06_summary/ch3_transfer_long.csv
#   - experiments/formal_ch3_modeling/06_summary/ch3_gate_utilization_summary.csv
#
# Outputs:
#   - fig3_1_median_nse_performance.png
#   - fig3_2_multi_metric_performance_boxplot.png
#   - fig3_3_cgc_vs_stlq_streamflow_nse_1to1.png
#   - fig3_4_cgc_vs_stlet_evapotranspiration_nse_1to1.png
#   - fig3_5_delta_nse_by_task_boxplot.png
#   - fig3_6_transfer_rate_by_task.png
#   - fig3_7_cgc_gate_utilization.png
# ==============================================================================

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"
FIG_DIR = CH3_DIR / "figures"

PER_BASIN_PATH = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
TRANSFER_LONG_PATH = SUMMARY_DIR / "ch3_transfer_long.csv"
GATE_SUMMARY_PATH = SUMMARY_DIR / "ch3_gate_utilization_summary.csv"
GATE_LONG_PATH = SUMMARY_DIR / "ch3_gate_utilization_long.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS_Q = ["STL_Q", "Hard_MTL", "MMoE", "CGC"]
MODELS_ET = ["STL_ET", "Hard_MTL", "MMoE", "CGC"]

DISPLAY_LABELS = {
    "STL_Q": "STL-Q",
    "STL_ET": "STL-ET",
    "Hard_MTL": "Hard-MTL",
    "MMoE": "MMoE",
    "CGC": "CGC",
}

TASK_LABELS = {
    "streamflow": "Streamflow -- Q",
    "evapotranspiration": "Evapotranspiration -- ET",
}

# Okabe-Ito color palette: colorblind-friendly and journal-suitable.
MODEL_COLORS = {
    "STL_Q": "#D55E00",
    "STL_ET": "#D55E00",
    "Hard_MTL": "#0072B2",
    "MMoE": "#009E73",
    "CGC": "#CC79A7",
}

METRIC_PANELS: List[Tuple[str, str, Optional[Tuple[float, float]]]] = [
    ("bias", "Bias", None),
    ("rmse", "RMSE", None),
    ("corr", "Corr", None),
    ("nse", "NSE", (-1.0, 1.0)),
    ("kge", "KGE", (-1.0, 1.0)),
]


def resolve_serif_font() -> str:
    """Return the first available publication-suitable serif font."""
    preferred_fonts = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            return font_name

    return "DejaVu Serif"


def set_publication_style() -> None:
    """Set a portable publication style without unavailable-font warnings."""
    serif_font = resolve_serif_font()

    plt.rcParams.update(
        {
            "font.family": serif_font,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )

    print(f"Using figure font: {serif_font}")


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize CAMELS gauge ids as 8-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to finite numeric values."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def save_figure(path: Path) -> None:
    """Save each figure as high-resolution PNG and vector PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")

    figure = plt.gcf()
    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def load_per_basin_table() -> pd.DataFrame:
    """Load basin-level Chapter 3 model metrics."""
    require_file(PER_BASIN_PATH)
    df = pd.read_csv(PER_BASIN_PATH, dtype={"gauge_id": str})

    if "gauge_id" in df.columns:
        df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    return df


def metric_column(model: str, task: str, metric: str) -> str:
    """Build a standardized model-task-metric column name."""
    return f"{model}_{task}_{metric}"


def collect_metric_series(
    df: pd.DataFrame,
    models: List[str],
    task: str,
    metric: str,
) -> Dict[str, pd.Series]:
    """Collect valid metric series for selected models."""
    result: Dict[str, pd.Series] = {}

    for model in models:
        col = metric_column(model, task, metric)
        if col not in df.columns:
            print(f"[Skip] Missing column: {col}")
            continue

        values = clean_numeric(df[col])
        if not values.empty:
            result[model] = values

    return result


def style_boxplot(box: Dict[str, object], colors: List[str]) -> None:
    """Apply consistent publication-style boxplot formatting."""
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    for whisker in box["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(0.8)

    for cap in box["caps"]:
        cap.set_color("black")
        cap.set_linewidth(0.8)


def add_boxplot_median_labels(
    ax: plt.Axes,
    values: List[pd.Series],
    decimals: int = 2,
) -> None:
    """Place compact median labels directly above the median lines."""
    y_min, y_max = ax.get_ylim()
    offset = max((y_max - y_min) * 0.008, np.finfo(float).eps)

    for position, series in enumerate(values, start=1):
        median = float(np.nanmedian(series))
        ax.text(
            position,
            median + offset,
            f"{median:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="semibold",
            color="black",
            clip_on=True,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 0.10,
            },
            zorder=5,
        )


def boxplot_whisker_limits(values: List[pd.Series]) -> Tuple[float, float]:
    """Return the combined Tukey-whisker range used by standard boxplots."""
    lower_values: List[float] = []
    upper_values: List[float] = []

    for series in values:
        array = np.asarray(series, dtype=float)
        array = array[np.isfinite(array)]
        if array.size == 0:
            continue

        q1, q3 = np.quantile(array, [0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        inliers = array[(array >= lower_fence) & (array <= upper_fence)]

        if inliers.size == 0:
            inliers = array

        lower_values.append(float(np.min(inliers)))
        upper_values.append(float(np.max(inliers)))

    if not lower_values or not upper_values:
        return 0.0, 1.0

    return min(lower_values), max(upper_values)


def apply_metric_ylim(ax: plt.Axes, metric: str, values: List[pd.Series]) -> None:
    """Set readable limits from the same Tukey-whisker rule as the boxplot."""
    lower, upper = boxplot_whisker_limits(values)

    if metric == "corr":
        lower = max(0.0, lower)
        upper = min(1.0, upper)

    if metric in {"nse", "kge"}:
        lower = max(-1.0, lower)
        upper = min(1.0, upper)

    if metric == "bias":
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)

    span = upper - lower
    if not np.isfinite(span) or span <= 0.0:
        span = max(abs(lower), abs(upper), 1.0) * 0.10

    margin = span * 0.10
    y_lower = lower - margin
    y_upper = upper + margin

    if metric == "corr":
        y_lower = max(0.0, y_lower)
        y_upper = min(1.0, y_upper)
    elif metric in {"nse", "kge"}:
        y_lower = max(-1.0, y_lower)
        y_upper = min(1.0, y_upper)

    ax.set_ylim(y_lower, y_upper)


def plot_median_nse_performance(df: pd.DataFrame) -> None:
    """Plot median NSE comparison for streamflow and evapotranspiration."""
    q_data = collect_metric_series(df, MODELS_Q, "streamflow", "nse")
    et_data = collect_metric_series(df, MODELS_ET, "evapotranspiration", "nse")

    x = np.arange(len(MODELS_Q))
    width = 0.36

    q_values = [q_data.get(model, pd.Series(dtype=float)).median() for model in MODELS_Q]

    et_values = []
    for model in MODELS_Q:
        et_model = "STL_ET" if model == "STL_Q" else model
        et_values.append(et_data.get(et_model, pd.Series(dtype=float)).median())

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    bars_q = ax.bar(
        x - width / 2,
        q_values,
        width=width,
        label="Streamflow",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.8,
    )
    bars_et = ax.bar(
        x + width / 2,
        et_values,
        width=width,
        label="Evapotranspiration",
        color="#E69F00",
        edgecolor="black",
        linewidth=0.8,
    )

    for bars in [bars_q, bars_et]:
        for bar in bars:
            value = bar.get_height()
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[m] for m in MODELS_Q])
    ax.set_ylabel("Median NSE")
    ax.set_ylim(0.0, max(0.90, np.nanmax(q_values + et_values) + 0.07))
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.legend(frameon=False, loc="upper right")

    save_figure(FIG_DIR / "fig3_1_median_nse_performance.png")


def plot_multi_metric_performance_boxplot(df: pd.DataFrame) -> None:
    """Plot robust basin-scale metric distributions for both tasks."""
    tasks = [
        ("streamflow", MODELS_Q, "(a) Streamflow, Q"),
        ("evapotranspiration", MODELS_ET, "(b) Evapotranspiration, ET"),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(METRIC_PANELS),
        figsize=(15.8, 7.4),
        constrained_layout=False,
    )

    for row_idx, (task, models, panel_label) in enumerate(tasks):
        for col_idx, (metric, metric_label, _) in enumerate(METRIC_PANELS):
            ax = axes[row_idx, col_idx]
            data = collect_metric_series(df, models, task, metric)
            available_models = [model for model in models if model in data]

            if not available_models:
                ax.axis("off")
                continue

            values = [data[model] for model in available_models]
            colors = [MODEL_COLORS[model] for model in available_models]

            box = ax.boxplot(
                [series.to_numpy() for series in values],
                patch_artist=True,
                showfliers=False,
                widths=0.62,
                whis=1.5,
            )
            style_boxplot(box, colors)
            apply_metric_ylim(ax, metric, values)
            add_boxplot_median_labels(ax, values, decimals=2)

            ax.set_xticks(np.arange(1, len(available_models) + 1))
            ax.set_xticklabels(
                [DISPLAY_LABELS[model] for model in available_models],
                rotation=0,
                ha="center",
            )
            ax.set_xlabel(metric_label, labelpad=5)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
            ax.set_axisbelow(True)

            if metric == "bias":
                ax.axhline(
                    0.0,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    zorder=1,
                )

            if col_idx == 0:
                ax.text(
                    -0.24,
                    0.5,
                    panel_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="semibold",
                )

    legend_items = [
        Patch(
            facecolor=MODEL_COLORS["STL_Q"],
            edgecolor="black",
            linewidth=0.8,
            label="STL",
        ),
        Patch(
            facecolor=MODEL_COLORS["Hard_MTL"],
            edgecolor="black",
            linewidth=0.8,
            label="Hard-MTL",
        ),
        Patch(
            facecolor=MODEL_COLORS["MMoE"],
            edgecolor="black",
            linewidth=0.8,
            label="MMoE",
        ),
        Patch(
            facecolor=MODEL_COLORS["CGC"],
            edgecolor="black",
            linewidth=0.8,
            label="CGC",
        ),
    ]

    fig.legend(
        handles=legend_items,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.56, 0.995),
        handlelength=1.7,
        columnspacing=1.6,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.925,
        bottom=0.09,
        wspace=0.34,
        hspace=0.38,
    )

    save_figure(FIG_DIR / "fig3_2_multi_metric_performance_boxplot.png")


def plot_cgc_vs_baseline_1to1(
    df: pd.DataFrame,
    baseline_model: str,
    cgc_model: str,
    task: str,
    output_name: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot an unclipped basin-level 1:1 comparison of CGC and STL NSE."""
    del title  # Titles belong in the manuscript caption, not inside the panel.

    baseline_col = metric_column(baseline_model, task, "nse")
    cgc_col = metric_column(cgc_model, task, "nse")

    if baseline_col not in df.columns or cgc_col not in df.columns:
        print(f"[Skip] Missing columns for 1:1 plot: {baseline_col}, {cgc_col}")
        return

    data = df[[baseline_col, cgc_col]].copy()
    data.columns = ["baseline", "cgc"]
    data["baseline"] = pd.to_numeric(data["baseline"], errors="coerce")
    data["cgc"] = pd.to_numeric(data["cgc"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        print(f"[Skip] No valid records for {output_name}.")
        return

    x = data["baseline"].to_numpy(dtype=float)
    y = data["cgc"].to_numpy(dtype=float)
    delta = y - x

    positive_rate = float(np.mean(delta > 0.0) * 100.0)
    median_gain = float(np.median(delta))

    data_min = float(min(np.min(x), np.min(y)))
    data_max = float(max(np.max(x), np.max(y)))
    span = max(data_max - data_min, 0.10)
    axis_min = data_min - 0.04 * span
    axis_max = data_max + 0.04 * span

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    ax.scatter(
        x,
        y,
        s=12,
        alpha=0.42,
        color="#0072B2",
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="black",
        linestyle="--",
        linewidth=0.9,
        zorder=3,
    )

    ax.text(
        0.04,
        0.96,
        f"Improved basins: {positive_rate:.1f}%\nMedian ΔNSE: {median_gain:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.55",
            "linewidth": 0.5,
            "alpha": 0.92,
            "pad": 2.5,
        },
        zorder=4,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.15, top=0.98)
    save_figure(FIG_DIR / output_name)


def plot_delta_nse_by_task(df: pd.DataFrame) -> None:
    """Plot NSE gains relative to single-task baselines for both tasks."""
    delta_config = {
        "streamflow": {
            "Hard_MTL": "Delta_NSE_HardMTL_minus_STLQ",
            "MMoE": "Delta_NSE_MMoE_minus_STLQ",
            "CGC": "Delta_NSE_CGC_minus_STLQ",
        },
        "evapotranspiration": {
            "Hard_MTL": "Delta_NSE_HardMTL_ET_minus_STLET",
            "MMoE": "Delta_NSE_MMoE_ET_minus_STLET",
            "CGC": "Delta_NSE_CGC_ET_minus_STLET",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)

    for ax, (task, model_cols) in zip(axes, delta_config.items()):
        values: List[pd.Series] = []
        labels: List[str] = []
        colors: List[str] = []

        for model, col in model_cols.items():
            if col not in df.columns:
                print(f"[Skip] Missing column: {col}")
                continue

            series = clean_numeric(df[col])
            if not series.empty:
                values.append(series)
                labels.append(DISPLAY_LABELS[model])
                colors.append(MODEL_COLORS[model])

        if not values:
            ax.axis("off")
            continue

        box = ax.boxplot(
            [series.values for series in values],
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
        )
        style_boxplot(box, colors)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("(a) Streamflow, Q" if task == "streamflow" else "(b) Evapotranspiration, ET")
        ax.set_ylabel("ΔNSE relative to STL baseline" if task == "streamflow" else "")
        ax.set_ylim(-0.5, 0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.30)
        add_boxplot_median_labels(ax, values, decimals=2)

    save_figure(FIG_DIR / "fig3_5_delta_nse_by_task_boxplot.png")


def plot_transfer_rate_by_task() -> None:
    """Plot positive and negative transfer rates for both tasks."""
    require_file(TRANSFER_LONG_PATH)

    df = pd.read_csv(TRANSFER_LONG_PATH)

    required = ["task", "model", "delta_nse"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {TRANSFER_LONG_PATH}: {missing}")

    records = []

    for (task, model), group in df.groupby(["task", "model"]):
        delta = clean_numeric(group["delta_nse"])
        if delta.empty:
            continue

        records.append(
            {
                "task": task,
                "model": model,
                "positive_rate": float((delta > 0.0).mean() * 100.0),
                "negative_rate": float((delta < 0.0).mean() * 100.0),
            }
        )

    summary = pd.DataFrame(records)

    if summary.empty:
        print("[Skip] No transfer-rate records.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)

    for ax, task in zip(axes, ["streamflow", "evapotranspiration"]):
        plot_df = summary[summary["task"] == task].copy()
        order = [model for model in ["Hard_MTL", "MMoE", "CGC"] if model in plot_df["model"].values]

        if not order:
            ax.axis("off")
            continue

        plot_df = plot_df.set_index("model").loc[order].reset_index()

        x = np.arange(len(plot_df))
        width = 0.36

        bars_pos = ax.bar(
            x - width / 2,
            plot_df["positive_rate"],
            width=width,
            label="Positive",
            color="#0072B2",
            edgecolor="black",
            linewidth=0.8,
        )
        bars_neg = ax.bar(
            x + width / 2,
            plot_df["negative_rate"],
            width=width,
            label="Negative",
            color="#D55E00",
            edgecolor="black",
            linewidth=0.8,
        )

        for bars in [bars_pos, bars_neg]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{bar.get_height():.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_title("(a) Streamflow, Q" if task == "streamflow" else "(b) Evapotranspiration, ET")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_LABELS[m] for m in plot_df["model"]])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.30)

    axes[0].set_ylabel("Basin proportion (%)")
    fig.legend(
        handles=[bars_pos[0], bars_neg[0]],
        labels=["Positive transfer", "Negative transfer"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.84, wspace=0.16)

    save_figure(FIG_DIR / "fig3_6_transfer_rate_by_task.png")


def expert_sort_key(value: object) -> int:
    """Sort expert ids such as 0, 1, E0, and E1 numerically."""
    text = str(value).strip().replace("E", "").replace("e", "")
    try:
        return int(text)
    except ValueError:
        return 10_000


def plot_gate_utilization() -> None:
    """Plot CGC gate utilization for streamflow and evapotranspiration."""
    path = GATE_SUMMARY_PATH if GATE_SUMMARY_PATH.exists() else GATE_LONG_PATH

    if not path.exists():
        print("[Skip] Gate utilization table not found.")
        return

    df = pd.read_csv(path)

    required = ["gate_name", "expert_id", "mean_utilization"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required gate columns in {path}: {missing}")

    table = df.copy()

    if "model" in table.columns:
        cgc_table = table[table["model"].astype(str).str.upper() == "CGC"].copy()
        if not cgc_table.empty:
            table = cgc_table

    table["gate_name"] = table["gate_name"].astype(str)
    table["expert_id"] = table["expert_id"].astype(str)
    table["mean_utilization"] = pd.to_numeric(table["mean_utilization"], errors="coerce")
    table = table.dropna(subset=["gate_name", "expert_id", "mean_utilization"])

    gate_name_map = {
        "task_0_gate": "Streamflow gate",
        "task_1_gate": "Evapotranspiration gate",
        "streamflow_gate": "Streamflow gate",
        "evapotranspiration_gate": "Evapotranspiration gate",
        "q_gate": "Streamflow gate",
        "et_gate": "Evapotranspiration gate",
    }

    table["gate_label"] = table["gate_name"].map(lambda value: gate_name_map.get(value, value))

    pivot = table.pivot_table(
        index="expert_id",
        columns="gate_label",
        values="mean_utilization",
        aggfunc="mean",
    ).fillna(0.0)

    required_gates = ["Streamflow gate", "Evapotranspiration gate"]
    missing_gates = [gate for gate in required_gates if gate not in pivot.columns]
    if missing_gates:
        raise ValueError(
            f"Missing mapped gate columns: {missing_gates}. "
            f"Available gates: {list(pivot.columns)}"
        )

    pivot = pivot[required_gates]
    pivot = pivot.loc[sorted(pivot.index, key=expert_sort_key)]

    experts = [
        f"E{expert_sort_key(idx)}" if not str(idx).startswith("E") else str(idx)
        for idx in pivot.index
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.2))

    x = np.arange(len(pivot))
    width = 0.36

    bars_q = ax.bar(
        x - width / 2,
        pivot["Streamflow gate"].values,
        width=width,
        label="Streamflow gate",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.8,
    )
    bars_et = ax.bar(
        x + width / 2,
        pivot["Evapotranspiration gate"].values,
        width=width,
        label="Evapotranspiration gate",
        color="#E69F00",
        edgecolor="black",
        linewidth=0.8,
    )

    for bars in [bars_q, bars_et]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="semibold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(experts)
    ax.set_ylabel("Mean gate utilization")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.legend(frameon=False)

    save_figure(FIG_DIR / "fig3_7_cgc_gate_utilization.png")


def main() -> None:
    """Generate all Chapter 3 non-spatial figures."""
    set_publication_style()

    per_basin = load_per_basin_table()

    plot_median_nse_performance(per_basin)
    plot_multi_metric_performance_boxplot(per_basin)

    plot_cgc_vs_baseline_1to1(
        per_basin,
        baseline_model="STL_Q",
        cgc_model="CGC",
        task="streamflow",
        output_name="fig3_3_cgc_vs_stlq_streamflow_nse_1to1.png",
        xlabel="STL-Q streamflow NSE",
        ylabel="CGC streamflow NSE",
        title="",
    )

    plot_cgc_vs_baseline_1to1(
        per_basin,
        baseline_model="STL_ET",
        cgc_model="CGC",
        task="evapotranspiration",
        output_name="fig3_4_cgc_vs_stlet_evapotranspiration_nse_1to1.png",
        xlabel="STL-ET evapotranspiration NSE",
        ylabel="CGC evapotranspiration NSE",
        title="",
    )

    plot_delta_nse_by_task(per_basin)
    plot_transfer_rate_by_task()
    plot_gate_utilization()

    print(f"All available Chapter 3 figures were saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()