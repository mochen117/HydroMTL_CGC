# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Generate publication-quality Chapter 3 figures for hydrological multi-task
#   learning model comparison and shared-benefit analysis.
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"
FIG_DIR = CH3_DIR / "figures"

PER_BASIN_PATH = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
TRANSFER_LONG_PATH = SUMMARY_DIR / "ch3_transfer_long.csv"
GATE_SUMMARY_PATH = SUMMARY_DIR / "ch3_gate_utilization_summary.csv"
GATE_LONG_PATH = SUMMARY_DIR / "ch3_gate_utilization_long.csv"
SPATIAL_GPKG_PATH = SUMMARY_DIR / "ch3_spatial_basin_metrics.gpkg"

BASIN_SHP_PATH = Path(
    "/home/mochen/hydro_data/camels/camels_us/"
    "basin_set_full_res/HCDN_nhru_final_671.shp"
)

US_STATE_SHP_PATH = Path(
    "/home/mochen/.local/share/cartopy/shapefiles/natural_earth/cultural/"
    "ne_50m_admin_1_states_provinces_lakes.shp"
)

MAP_CRS = "EPSG:5070"
CONUS_EXTENT_LONLAT = (-128.5, -64.0, 23.0, 51.8)

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

MODELS_Q = ["STL_Q", "Hard_MTL", "MMoE", "CGC"]
MODELS_ET = ["STL_ET", "Hard_MTL", "MMoE", "CGC"]
MTL_MODELS = ["Hard_MTL", "MMoE", "CGC"]

DISPLAY_LABELS = {
    "STL_Q": "STL-Q",
    "STL_ET": "STL-ET",
    "Hard_MTL": "Hard-MTL",
    "MMoE": "MMoE",
    "CGC": "CGC",
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
    "loss": "#F0AFAF",
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

CDF_LINESTYLES = {
    "STL_Q": (0, (5, 3)),
    "STL_ET": (0, (5, 3)),
    "Hard_MTL": (0, (5, 2, 1.2, 2)),
    "MMoE": (0, (1.2, 2.0)),
    "CGC": "solid",
}

CDF_LINEWIDTHS = {
    "STL_Q": 1.05,
    "STL_ET": 1.05,
    "Hard_MTL": 1.10,
    "MMoE": 1.10,
    "CGC": 1.35,
}

Q_CDF_COLORS = {
    "STL_Q": "#6F6F6F",
    "Hard_MTL": "#5B99C5",
    "MMoE": "#C2BFD7",
    "CGC": "#1F4E79",
}

ET_CDF_COLORS = {
    "STL_ET": "#6F6F6F",
    "Hard_MTL": "#FAA256",
    "MMoE": "#F0AFAF",
    "CGC": "#C56E1A",
}

TRANSFER_COLORS = {
    "positive": PALETTE["gain"],
    "negative": PALETTE["loss"],
}

ANNOTATION_COLOR = "#8B0000"
EDGE_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
BASEMAP_FACE = "#D8D8D8"
BASEMAP_EDGE = "#F7F7F7"
STATE_LINE_COLOR = "#9A9A9A"

METRIC_PANELS = [
    ("bias", "Bias"),
    ("rmse", "RMSE"),
    ("corr", "Corr"),
    ("nse", "NSE"),
    ("kge", "KGE"),
]

NSE_RANGE = (-1.0, 1.0)
TOP_GAIN_COUNT = 2

NSE_DISPLAY_RANGE = (0.0, 1.0)
TRANSFER_GAIN_DISPLAY_RANGE = (-0.20, 0.20)

NSE_CMAP = "turbo"
TRANSFER_GAIN_CMAP = "coolwarm"

MAP_FIGSIZE = (12.8, 7.2)
MAP_ADJUST = {
    "left": 0.02,
    "right": 0.91,
    "top": 0.93,
    "bottom": 0.04,
}

def require_file(path: Path) -> None:
    """Raise an error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def choose_serif_font() -> str:
    """Choose an available serif font for thesis and journal figures."""
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
    """Set global matplotlib style for publication-quality figures."""
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
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.3,
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


def save_figure(path: Path) -> None:
    """Save current figure as PNG and PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"[Saved] {path}")
    print(f"[Saved] {path.with_suffix('.pdf')}")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize basin IDs to 8-character strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to finite numeric values."""
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def load_per_basin_table() -> pd.DataFrame:
    """Load Chapter 3 basin-level model metrics."""
    require_file(PER_BASIN_PATH)
    df = pd.read_csv(PER_BASIN_PATH, dtype={"gauge_id": str})
    if "gauge_id" not in df.columns:
        raise ValueError(f"Missing 'gauge_id' in {PER_BASIN_PATH}.")
    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])
    print(f"[Info] Basin-level table: {PER_BASIN_PATH}")
    print(f"[Info] Basin records: {len(df)}")
    return df


def metric_column(model: str, task: str, metric: str) -> str:
    """Build a model-task-metric column name."""
    return f"{model}_{task}_{metric}"


def task_model_colors(task: str) -> Dict[str, str]:
    """Return model colors for a specific task."""
    if task == "streamflow":
        return Q_MODEL_COLORS
    if task == "evapotranspiration":
        return ET_MODEL_COLORS
    raise ValueError(f"Unsupported task: {task}")


def task_cdf_colors(task: str) -> Dict[str, str]:
    """Return CDF line colors for a specific task."""
    if task == "streamflow":
        return Q_CDF_COLORS
    if task == "evapotranspiration":
        return ET_CDF_COLORS
    raise ValueError(f"Unsupported task: {task}")


def style_axis(ax: Axes, grid_axis: str = "y") -> None:
    """Apply consistent axis styling."""
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


def collect_metric_series(
    df: pd.DataFrame,
    models: Sequence[str],
    task: str,
    metric: str,
) -> Dict[str, pd.Series]:
    """Collect valid metric values for selected models."""
    output: Dict[str, pd.Series] = {}
    for model in models:
        col = metric_column(model, task, metric)
        if col not in df.columns:
            print(f"[Skip] Missing metric column: {col}")
            continue
        values = clean_numeric(df[col])
        if not values.empty:
            output[model] = values
    return output


def add_bar_labels(
    ax: Axes,
    bars: Iterable,
    decimals: int = 2,
    offset: float = 0.012,
) -> None:
    """Annotate bar heights."""
    for bar in bars:
        value = float(bar.get_height())
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
            )


def style_boxplot(box_obj: Dict[str, object], colors: Sequence[str]) -> None:
    """Apply consistent styling to a matplotlib boxplot."""
    for patch, color in zip(box_obj["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
        patch.set_edgecolor(EDGE_COLOR)
        patch.set_linewidth(0.9)
    for median in box_obj["medians"]:
        median.set_color("black")
        median.set_linewidth(1.35)
    for whisker in box_obj["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(0.9)
    for cap in box_obj["caps"]:
        cap.set_color("black")
        cap.set_linewidth(0.9)


def annotate_box_medians(
    ax: Axes,
    values: Sequence[pd.Series],
    positions: Sequence[float],
    decimals: int = 2,
) -> None:
    """Annotate medians close to the median lines."""
    medians = [float(np.nanmedian(clean_numeric(v))) for v in values]
    y_min, y_max = ax.get_ylim()
    offset = max(y_max - y_min, 1e-6) * 0.002
    for x, median in zip(positions, medians):
        ax.text(
            x,
            median + offset,
            f"{median:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=10,
        )


def plot_median_nse_performance(df: pd.DataFrame) -> None:
    """Plot median NSE comparison between streamflow and evapotranspiration."""
    q_data = collect_metric_series(df, MODELS_Q, "streamflow", "nse")
    et_data = collect_metric_series(df, MODELS_ET, "evapotranspiration", "nse")

    q_values = [
        q_data.get(model, pd.Series(dtype=float)).median()
        for model in MODELS_Q
    ]

    et_values = []
    for model in MODELS_Q:
        et_model = "STL_ET" if model == "STL_Q" else model
        et_values.append(et_data.get(et_model, pd.Series(dtype=float)).median())

    x = np.arange(len(MODELS_Q))
    width = 0.34
    _, ax = plt.subplots(figsize=(8.6, 4.5))

    bars_q = ax.bar(
        x - width / 2,
        q_values,
        width,
        label="Streamflow",
        color=TASK_COLORS["streamflow"],
        edgecolor="black",
        linewidth=0.8,
    )
    bars_et = ax.bar(
        x + width / 2,
        et_values,
        width,
        label="Evapotranspiration",
        color=TASK_COLORS["evapotranspiration"],
        edgecolor="black",
        linewidth=0.8,
    )

    add_bar_labels(ax, bars_q)
    add_bar_labels(ax, bars_et)

    ax.set_xticks(x)
    ax.set_xticklabels(["STL", "Hard-MTL", "MMoE", "CGC"])
    ax.set_ylabel("Median NSE")
    ax.set_ylim(0.0, 0.90)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax, "y")
    save_figure(FIG_DIR / "fig3_1_median_nse_performance.png")


def plot_task_metric_boxplots(
    df: pd.DataFrame,
    task: str,
    models: Sequence[str],
    output_name: str,
    row_title: str,
) -> None:
    """Plot five performance metrics as boxplots for one task."""
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2))
    axes_flat = axes.ravel()
    colors_by_model = task_model_colors(task)

    for index, (metric, label) in enumerate(METRIC_PANELS):
        ax = axes_flat[index]
        data = collect_metric_series(df, models, task, metric)
        available = [model for model in models if model in data]

        if not available:
            ax.axis("off")
            continue

        values = [data[model] for model in available]
        positions = np.arange(1, len(values) + 1)
        colors = [colors_by_model[model] for model in available]

        box_obj = ax.boxplot(
            [clean_numeric(v).values for v in values],
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.52,
            tick_labels=[DISPLAY_LABELS[m] for m in available],
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

    fig.suptitle(row_title, fontsize=13.5, fontstyle="italic", y=0.98)
    fig.text(
        0.01,
        0.01,
        "Outliers are not shown; medians are annotated adjacent to the median lines.",
        fontsize=8.2,
    )
    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.11,
        wspace=0.38,
        hspace=0.55,
    )
    save_figure(FIG_DIR / output_name)


def plot_all_metric_boxplots(df: pd.DataFrame) -> None:
    """Plot task-specific metric boxplots."""
    plot_task_metric_boxplots(
        df,
        "streamflow",
        MODELS_Q,
        "fig3_2a_streamflow_metrics_boxplot.png",
        "(a) Streamflow -- Q",
    )
    plot_task_metric_boxplots(
        df,
        "evapotranspiration",
        MODELS_ET,
        "fig3_2b_evapotranspiration_metrics_boxplot.png",
        "(b) Evapotranspiration -- ET",
    )


def plot_nse_cdf(
    df: pd.DataFrame,
    task: str,
    models: Sequence[str],
    output_name: str,
    xlabel: str,
    title: str,
) -> None:
    """Plot empirical cumulative distribution of NSE."""
    _, ax = plt.subplots(figsize=(7.6, 4.6))
    threshold_color = "#9EC3DD" if task == "streamflow" else "#EFC58F"

    for threshold in [0.00, 0.50, 0.75]:
        ax.axvline(
            threshold,
            color=threshold_color,
            linestyle="--",
            linewidth=0.70,
            alpha=0.80,
            zorder=1,
        )

    colors = task_cdf_colors(task)

    for model in models:
        col = metric_column(model, task, "nse")
        if col not in df.columns:
            print(f"[Skip] Missing CDF column: {col}")
            continue

        values = clean_numeric(df[col])
        if values.empty:
            continue

        x = values.clip(*NSE_RANGE).sort_values().values
        y = np.arange(1, len(x) + 1) / len(x)
        rate = float((values > 0.75).mean() * 100.0)

        ax.plot(
            x,
            y,
            color=colors[model],
            linestyle=CDF_LINESTYLES[model],
            linewidth=CDF_LINEWIDTHS[model],
            label=f"{DISPLAY_LABELS[model]} (NSE > 0.75: {rate:.1f}%)",
            zorder=4 if model == "CGC" else 3,
        )

    ax.set_xlim(*NSE_RANGE)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=3.2)
    style_axis(ax, "both")
    save_figure(FIG_DIR / output_name)


def plot_all_nse_cdfs(df: pd.DataFrame) -> None:
    """Plot streamflow and evapotranspiration NSE CDFs."""
    plot_nse_cdf(
        df,
        "streamflow",
        MODELS_Q,
        "fig3_3a_streamflow_nse_cdf.png",
        "Streamflow NSE",
        "(a) Streamflow NSE distribution",
    )
    plot_nse_cdf(
        df,
        "evapotranspiration",
        MODELS_ET,
        "fig3_3b_evapotranspiration_nse_cdf.png",
        "Evapotranspiration NSE",
        "(b) Evapotranspiration NSE distribution",
    )


def prepare_1to1_data(
    df: pd.DataFrame,
    baseline_model: str,
    task: str,
) -> Tuple[pd.DataFrame, str, str]:
    """Prepare CGC-versus-baseline NSE comparison data."""
    baseline_col = metric_column(baseline_model, task, "nse")
    cgc_col = metric_column("CGC", task, "nse")
    missing = [col for col in [baseline_col, cgc_col] if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required 1:1 columns: {missing}")

    data = df[["gauge_id", baseline_col, cgc_col]].copy()
    data[baseline_col] = pd.to_numeric(data[baseline_col], errors="coerce")
    data[cgc_col] = pd.to_numeric(data[cgc_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["delta_nse"] = data[cgc_col] - data[baseline_col]
    return data, baseline_col, cgc_col


def select_top_gains(
    data: pd.DataFrame,
    baseline_col: str,
    cgc_col: str,
) -> pd.DataFrame:
    """Select the largest visible positive transfer-gain basins."""
    low, high = NSE_RANGE
    visible = data[
        data[baseline_col].between(low, high)
        & data[cgc_col].between(low, high)
        & (data["delta_nse"] > 0.0)
    ].copy()
    return visible.nlargest(TOP_GAIN_COUNT, "delta_nse")


def plot_cgc_vs_baseline_1to1(
    df: pd.DataFrame,
    baseline_model: str,
    task: str,
    output_name: str,
    xlabel: str,
    ylabel: str,
) -> pd.DataFrame:
    """Plot CGC NSE against the single-task baseline NSE."""
    data, baseline_col, cgc_col = prepare_1to1_data(df, baseline_model, task)

    low, high = NSE_RANGE
    visible = data[baseline_col].between(low, high) & data[cgc_col].between(low, high)
    plot_data = data.loc[visible].copy()
    top_gain = select_top_gains(data, baseline_col, cgc_col)

    gain_rate = float((data["delta_nse"] > 0.0).mean() * 100.0)
    median_gain = float(data["delta_nse"].median())
    outside_count = int((~visible).sum())

    _, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.scatter(
        plot_data[baseline_col],
        plot_data[cgc_col],
        s=14,
        alpha=0.42,
        color=TASK_COLORS[task],
        edgecolor="none",
        rasterized=True,
        zorder=3,
    )
    ax.plot([low, high], [low, high], "k--", linewidth=1.0, zorder=2)

    labels = ["Max shared modeling gain", "2nd max shared modeling gain"]
    offsets = [(14, 12), (14, -26)]

    for rank, (_, row) in enumerate(top_gain.iterrows()):
        x = float(row[baseline_col])
        y = float(row[cgc_col])
        ax.scatter(
            [x],
            [y],
            s=70,
            facecolors="none",
            edgecolors=ANNOTATION_COLOR,
            linewidths=1.25,
            zorder=8,
        )
        ax.annotate(
            f"{labels[rank]}\n{row['gauge_id']}",
            xy=(x, y),
            xytext=offsets[rank],
            textcoords="offset points",
            ha="left",
            va="bottom" if offsets[rank][1] >= 0 else "top",
            fontsize=8.5,
            color=ANNOTATION_COLOR,
            arrowprops={
                "arrowstyle": "->",
                "color": ANNOTATION_COLOR,
                "linewidth": 0.9,
            },
            zorder=9,
        )

    ax.text(
        0.04,
        0.96,
        (
            f"Improved basins: {gain_rate:.1f}%\n"
            f"Median ΔNSE: {median_gain:+.2f}\n"
            f"Outside display window: {outside_count}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "linewidth": 0.7,
            "alpha": 0.94,
            "pad": 4.0,
        },
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax, "both")
    save_figure(FIG_DIR / output_name)

    output = data[["gauge_id", baseline_col, cgc_col, "delta_nse"]].copy()
    output["task"] = task
    output["baseline_col"] = baseline_col
    output["cgc_col"] = cgc_col
    return output


def infer_shapefile_gauge_column(gdf: "gpd.GeoDataFrame") -> str:
    """Infer the basin ID column in a CAMELS shapefile."""
    candidates = [
        "gauge_id",
        "GAGE_ID",
        "gauge_id",
        "hru_id",
        "HRU_ID",
        "basin_id",
        "BASIN_ID",
    ]
    lower_map = {str(col).lower(): str(col) for col in gdf.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(f"No gauge ID column found. Available columns: {list(gdf.columns)}")


def load_basin_geometries() -> Optional["gpd.GeoDataFrame"]:
    """Load CAMELS basin polygons."""
    if gpd is None:
        print("[Skip] GeoPandas is not installed.")
        return None

    if not BASIN_SHP_PATH.exists():
        print(f"[Skip] CAMELS basin shapefile not found: {BASIN_SHP_PATH}")
        return None

    basins = gpd.read_file(BASIN_SHP_PATH)
    gauge_col = infer_shapefile_gauge_column(basins)
    basins = basins.rename(columns={gauge_col: "gauge_id"})
    basins["gauge_id"] = normalize_gauge_id(basins["gauge_id"])

    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326", allow_override=True)

    basins = basins.to_crs(MAP_CRS)
    basins = basins[basins.geometry.notna()].copy()
    basins = basins[~basins.geometry.is_empty].copy()

    return basins[["gauge_id", "geometry"]]


def load_state_boundaries() -> Optional["gpd.GeoDataFrame"]:
    """Load US state boundaries."""
    if gpd is None:
        return None

    if not US_STATE_SHP_PATH.exists():
        print(f"[Info] US state shapefile not found: {US_STATE_SHP_PATH}")
        return None

    states = gpd.read_file(US_STATE_SHP_PATH)

    if states.crs is None:
        states = states.set_crs("EPSG:4326", allow_override=True)

    if "admin" in states.columns:
        states = states[states["admin"].astype(str) == "United States of America"].copy()
    elif "adm0_a3" in states.columns:
        states = states[states["adm0_a3"].astype(str) == "USA"].copy()
    elif "iso_a2" in states.columns:
        states = states[states["iso_a2"].astype(str) == "US"].copy()

    states = states.to_crs(MAP_CRS)
    states = states[states.geometry.notna()].copy()
    states = states[~states.geometry.is_empty].copy()
    return states


def projected_conus_extent(basins: Optional["gpd.GeoDataFrame"] = None) -> Tuple[float, float, float, float]:
    """Return a robust projected CONUS extent using tightly tuned lon-lat boundaries."""
    if gpd is None:
        raise ImportError("GeoPandas is required for projected map extent.")

    # Tightened CONUS boundaries to eliminate excessive white spaces while preserving all US lands
    min_lon, max_lon, min_lat, max_lat = (-125.5, -66.5, 24.0, 49.5)

    n = 200
    bottom_lon = np.linspace(min_lon, max_lon, n)
    top_lon = np.linspace(min_lon, max_lon, n)
    left_lat = np.linspace(min_lat, max_lat, n)
    right_lat = np.linspace(min_lat, max_lat, n)

    lon = np.concatenate(
        [
            bottom_lon,
            np.full(n, max_lon),
            top_lon[::-1],
            np.full(n, min_lon),
        ]
    )
    lat = np.concatenate(
        [
            np.full(n, min_lat),
            right_lat,
            np.full(n, max_lat),
            left_lat[::-1],
        ]
    )

    boundary = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    ).to_crs(MAP_CRS)

    minx, miny, maxx, maxy = boundary.total_bounds

    # Minimal padding to avoid clipping outermost land shapes
    pad_x = (maxx - minx) * 0.005
    pad_y = (maxy - miny) * 0.005

    return (
        float(minx - pad_x),
        float(maxx + pad_x),
        float(miny - pad_y),
        float(maxy + pad_y),
    )


def plot_single_transfer_gain_map(
    basins: "gpd.GeoDataFrame",
    states: Optional["gpd.GeoDataFrame"],
    transfer_data: pd.DataFrame,
    task_label: str,
    marker: str,
    output_name: str,
    title: str,
) -> None:
    """Plot task-specific spatial transfer effects for all basins."""
    if transfer_data.empty:
        print(f"[Skip] No transfer records for {task_label}.")
        return

    data = basins.merge(transfer_data, on="gauge_id", how="inner")
    data = data.dropna(subset=["delta_nse"]).copy()

    if data.empty:
        print(f"[Skip] No valid ΔNSE basins for {task_label}.")
        return

    points = data.copy()
    points["geometry"] = points.geometry.centroid

    values = points["delta_nse"].to_numpy(dtype=float)
    abs_limit = 0.20
    points["plot_delta_nse"] = points["delta_nse"].clip(-abs_limit, abs_limit)

    top_gain = points.nlargest(2, "delta_nse").copy()

    bounds = projected_conus_extent(basins)
    minx, maxx, miny, maxy = bounds
    aspect = (maxx - minx) / (maxy - miny)

    height_inches = 5.2
    width_inches = height_inches * aspect + 1.1

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    basins.plot(
        ax=ax,
        facecolor="#E2E2E2",
        edgecolor="none",
        alpha=0.55,
        zorder=1,
    )

    if states is not None and not states.empty:
        states.boundary.plot(
            ax=ax,
            color="#A8A8A8",
            linewidth=0.45,
            zorder=2,
        )

    plot_points = points.sort_values("delta_nse")

    scatter = ax.scatter(
        plot_points.geometry.x,
        plot_points.geometry.y,
        c=plot_points["plot_delta_nse"],
        cmap="RdBu_r",
        vmin=-abs_limit,
        vmax=abs_limit,
        s=24,
        edgecolors="none",
        alpha=0.92,
        rasterized=True,
        zorder=4,
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.030,
        pad=0.018,
    )
    cbar.set_label(r"$\Delta$NSE (CGC minus STL)", rotation=90, labelpad=8)
    cbar.outline.set_linewidth(0.8)

    annotation_records = [
        ("Max-diff basin", top_gain.iloc[0], (14, 14)),
        ("2nd-max-diff basin", top_gain.iloc[1], (14, -24)),
    ]

    for label, row, offset in annotation_records:
        ax.scatter(
            row.geometry.x,
            row.geometry.y,
            marker=marker,
            s=95,
            facecolors="none",
            edgecolors=ANNOTATION_COLOR,
            linewidths=1.35,
            zorder=7,
        )

        ax.annotate(
            f"{label}\n{row['gauge_id']}",
            xy=(row.geometry.x, row.geometry.y),
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=8.2,
            color=ANNOTATION_COLOR,
            arrowprops={
                "arrowstyle": "->",
                "color": ANNOTATION_COLOR,
                "linewidth": 0.85,
            },
            zorder=8,
        )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=6, fontsize=12.5)

    fig.subplots_adjust(
        left=0.01,
        right=0.92,
        top=0.90,
        bottom=0.01,
    )

    save_figure(FIG_DIR / output_name)

def plot_single_nse_spatial_map(
    basins: "gpd.GeoDataFrame",
    states: Optional["gpd.GeoDataFrame"],
    per_basin: pd.DataFrame,
    value_col: str,
    output_name: str,
    title: str,
    colorbar_label: str,
) -> None:
    """Plot the spatial distribution of basin-level CGC NSE."""
    if value_col not in per_basin.columns:
        print(f"[Skip] Missing spatial NSE column: {value_col}")
        return

    data = per_basin[["gauge_id", value_col]].copy()
    data["gauge_id"] = normalize_gauge_id(data["gauge_id"])
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")

    merged = basins.merge(data, on="gauge_id", how="inner")
    merged = merged.dropna(subset=[value_col, "geometry"]).copy()

    if merged.empty:
        print(f"[Skip] No valid basin geometries for {value_col}.")
        return

    points = merged.copy()
    points["geometry"] = points.geometry.centroid
    points["plot_value"] = points[value_col].clip(*NSE_DISPLAY_RANGE)

    bounds = projected_conus_extent(basins)
    minx, maxx, miny, maxy = bounds
    aspect = (maxx - minx) / (maxy - miny)

    height_inches = 5.2
    width_inches = height_inches * aspect + 1.1

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    basins.plot(
        ax=ax,
        facecolor="#E2E2E2",
        edgecolor="none",
        alpha=0.45,
        zorder=1,
    )

    if states is not None and not states.empty:
        states.boundary.plot(
            ax=ax,
            color="#A8A8A8",
            linewidth=0.45,
            zorder=2,
        )

    scatter = ax.scatter(
        points.geometry.x,
        points.geometry.y,
        c=points["plot_value"],
        cmap=NSE_CMAP,
        vmin=NSE_DISPLAY_RANGE[0],
        vmax=NSE_DISPLAY_RANGE[1],
        s=24,
        edgecolors="none",
        alpha=0.92,
        rasterized=True,
        zorder=4,
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.030,
        pad=0.018,
    )
    cbar.set_label(colorbar_label, rotation=90, labelpad=8)
    cbar.outline.set_linewidth(0.8)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=6, fontsize=12.5)

    fig.subplots_adjust(
        left=0.01,
        right=0.92,
        top=0.90,
        bottom=0.01,
    )
    save_figure(FIG_DIR / output_name)

def plot_spatial_maps(
    per_basin: pd.DataFrame,
    q_transfer: pd.DataFrame,
    et_transfer: pd.DataFrame,
) -> None:
    """Plot CGC NSE maps and shared modeling gain maps for Chapter 3."""
    basins = load_basin_geometries()

    if basins is None:
        print("[Skip] Chapter 3 spatial maps were not generated.")
        return

    states = load_state_boundaries()

    plot_single_nse_spatial_map(
        basins=basins,
        states=states,
        per_basin=per_basin,
        value_col=metric_column("CGC", "streamflow", "nse"),
        output_name="fig3_5a_cgc_streamflow_nse_spatial_distribution.png",
        title="(a) Spatial distribution of CGC streamflow NSE",
        colorbar_label="CGC-Q NSE",
    )

    plot_single_nse_spatial_map(
        basins=basins,
        states=states,
        per_basin=per_basin,
        value_col=metric_column("CGC", "evapotranspiration", "nse"),
        output_name="fig3_5b_cgc_evapotranspiration_nse_spatial_distribution.png",
        title="(b) Spatial distribution of CGC evapotranspiration NSE",
        colorbar_label="CGC-ET NSE",
    )

    plot_single_transfer_gain_map(
        basins=basins,
        states=states,
        transfer_data=q_transfer,
        task_label="Streamflow",
        marker="o",
        output_name="fig3_6a_streamflow_shared_modeling_gain_spatial_distribution.png",
        title="(a) Spatial distribution of streamflow shared modeling gain",
    )

    plot_single_transfer_gain_map(
        basins=basins,
        states=states,
        transfer_data=et_transfer,
        task_label="Evapotranspiration",
        marker="o",
        output_name="fig3_6b_evapotranspiration_shared_modeling_gain_spatial_distribution.png",
        title="(b) Spatial distribution of evapotranspiration shared modeling gain",
    )


def boxplot_whisker_limits(values: Sequence[pd.Series]) -> Tuple[float, float]:
    """Estimate visible whisker limits for boxplots without outliers."""
    lows, highs = [], []

    for series in values:
        arr = clean_numeric(series)
        if arr.empty:
            continue

        q1 = float(arr.quantile(0.25))
        q3 = float(arr.quantile(0.75))
        iqr = q3 - q1
        inside = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]

        lows.append(float(inside.min() if not inside.empty else arr.min()))
        highs.append(float(inside.max() if not inside.empty else arr.max()))

    return (min(lows), max(highs)) if lows and highs else (-1.0, 1.0)


def plot_delta_nse_by_task(df: pd.DataFrame) -> None:
    """Plot shared benefit distributions relative to STL."""
    config = [
        (
            "streamflow",
            "(a) Streamflow",
            {
                "Hard_MTL": "Delta_NSE_HardMTL_minus_STLQ",
                "MMoE": "Delta_NSE_MMoE_minus_STLQ",
                "CGC": "Delta_NSE_CGC_minus_STLQ",
            },
        ),
        (
            "evapotranspiration",
            "(b) Evapotranspiration",
            {
                "Hard_MTL": "Delta_NSE_HardMTL_ET_minus_STLET",
                "MMoE": "Delta_NSE_MMoE_ET_minus_STLET",
                "CGC": "Delta_NSE_CGC_ET_minus_STLET",
            },
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    for ax, (task, title, columns) in zip(axes, config):
        values, labels, models = [], [], []

        for model, col in columns.items():
            if col not in df.columns:
                print(f"[Skip] Missing transfer column: {col}")
                continue

            series = clean_numeric(df[col])
            if not series.empty:
                values.append(series)
                labels.append(DISPLAY_LABELS[model])
                models.append(model)

        if not values:
            ax.axis("off")
            continue

        positions = np.arange(1, len(values) + 1)
        palette = task_model_colors(task)

        box_obj = ax.boxplot(
            [v.values for v in values],
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.56,
            tick_labels=labels,
        )
        style_boxplot(box_obj, [palette[m] for m in models])

        low, high = boxplot_whisker_limits(values)
        span = max(high - low, 1e-6)
        margin = max(span * 0.20, 0.025)
        ax.set_ylim(low - margin, high + margin)

        annotate_box_medians(ax, values, positions)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylabel("ΔNSE relative to STL")
        ax.set_title(title, loc="left")
        style_axis(ax, "y")

    fig.text(0.01, 0.01, "Outliers are not shown.", fontsize=8.2)
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.18,
        wspace=0.28,
    )
    save_figure(FIG_DIR / "fig3_7_delta_nse_by_task_boxplot.png")


def plot_transfer_rate_by_task() -> None:
    """Plot performance improvement and performance reduction proportions."""
    require_file(TRANSFER_LONG_PATH)
    transfer = pd.read_csv(TRANSFER_LONG_PATH)

    required = {"task", "model", "delta_nse"}
    missing = required.difference(transfer.columns)
    if missing:
        raise ValueError(f"Missing columns in {TRANSFER_LONG_PATH}: {sorted(missing)}")

    records = []
    for (task, model), group in transfer.groupby(["task", "model"]):
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

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)

    for ax, task in zip(axes, ["streamflow", "evapotranspiration"]):
        subset = summary[summary["task"] == task].copy()
        order = [model for model in MTL_MODELS if model in set(subset["model"])]

        if not order:
            ax.axis("off")
            continue

        subset = subset.set_index("model").loc[order].reset_index()
        x = np.arange(len(subset))
        width = 0.34

        positive = ax.bar(
            x - width / 2,
            subset["positive_rate"],
            width,
            label="Performance Improvement Ratio",
            color=TRANSFER_COLORS["positive"],
            edgecolor="black",
            linewidth=0.8,
        )
        negative = ax.bar(
            x + width / 2,
            subset["negative_rate"],
            width,
            label="Performance Reduction Ratio",
            color=TRANSFER_COLORS["negative"],
            edgecolor="black",
            linewidth=0.8,
        )

        for bars in [positive, negative]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{bar.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                )

        ax.set_title(
            "(a) Streamflow" if task == "streamflow" else "(b) Evapotranspiration",
            loc="left",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_LABELS[m] for m in subset["model"]])
        ax.set_ylim(0.0, 104.0)
        style_axis(ax, "y")

    axes[0].set_ylabel("Basin proportion (%)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.subplots_adjust(
        left=0.08,
        right=0.86,
        top=0.90,
        bottom=0.15,
        wspace=0.24,
    )
    save_figure(FIG_DIR / "fig3_8_performance_change_ratio_by_task.png")


def expert_sort_key(value: object) -> int:
    """Sort expert IDs numerically."""
    text = str(value).strip().replace("E", "").replace("e", "")
    try:
        return int(text)
    except ValueError:
        return 10_000


def plot_expert_gate_utilization() -> None:
    """Plot CGC gate utilization by expert."""
    path = GATE_SUMMARY_PATH if GATE_SUMMARY_PATH.exists() else GATE_LONG_PATH

    if not path.exists():
        print("[Skip] Gate utilization table not found.")
        return

    table = pd.read_csv(path)

    required = {"gate_name", "expert_id", "mean_utilization"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing gate columns in {path}: {sorted(missing)}")

    if "model" in table.columns:
        cgc = table[table["model"].astype(str).str.upper() == "CGC"].copy()
        if not cgc.empty:
            table = cgc

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
        index="expert_id",
        columns="gate_label",
        values="mean_utilization",
        aggfunc="mean",
    ).fillna(0.0)

    required_gates = ["Streamflow gate", "Evapotranspiration gate"]
    missing_gates = [gate for gate in required_gates if gate not in pivot.columns]
    if missing_gates:
        raise ValueError(f"Missing mapped gate columns: {missing_gates}")

    pivot = pivot[required_gates]
    pivot = pivot.loc[sorted(pivot.index, key=expert_sort_key)]

    experts = [
        f"E{expert_sort_key(i)}" if not str(i).startswith("E") else str(i)
        for i in pivot.index
    ]

    _, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(pivot))
    width = 0.34

    q_bars = ax.bar(
        x - width / 2,
        pivot["Streamflow gate"],
        width,
        label="Streamflow gate",
        color=TASK_COLORS["streamflow"],
        edgecolor="black",
        linewidth=0.8,
    )
    et_bars = ax.bar(
        x + width / 2,
        pivot["Evapotranspiration gate"],
        width,
        label="Evapotranspiration gate",
        color=TASK_COLORS["evapotranspiration"],
        edgecolor="black",
        linewidth=0.8,
    )

    add_bar_labels(ax, q_bars)
    add_bar_labels(ax, et_bars)

    ax.set_xticks(x)
    ax.set_xticklabels(experts)
    ax.set_ylabel("Mean gate utilization")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax, "y")
    save_figure(FIG_DIR / "fig3_9_cgc_gate_utilization.png")


def main() -> None:
    """Run all Chapter 3 figure generation routines."""
    print("=" * 100)
    print("Chapter 3 Figure Generator")
    print("=" * 100)

    set_publication_style()
    per_basin = load_per_basin_table()

    plot_median_nse_performance(per_basin)
    plot_all_metric_boxplots(per_basin)
    plot_all_nse_cdfs(per_basin)

    q_top_gain = plot_cgc_vs_baseline_1to1(
        per_basin,
        baseline_model="STL_Q",
        task="streamflow",
        output_name="fig3_4a_cgc_vs_stlq_streamflow_nse_1to1.png",
        xlabel="STL-Q NSE",
        ylabel="CGC-Q NSE",
    )

    et_top_gain = plot_cgc_vs_baseline_1to1(
        per_basin,
        baseline_model="STL_ET",
        task="evapotranspiration",
        output_name="fig3_4b_cgc_vs_stlet_evapotranspiration_nse_1to1.png",
        xlabel="STL-ET NSE",
        ylabel="CGC-ET NSE",
    )

    plot_spatial_maps(per_basin, q_top_gain, et_top_gain)
    plot_delta_nse_by_task(per_basin)
    plot_transfer_rate_by_task()
    plot_expert_gate_utilization()

    print("=" * 100)
    print("Chapter 3 figure generation completed.")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()