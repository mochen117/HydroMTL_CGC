# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Plot Chapter 4 basin heterogeneity diagnostics.
#
# Purpose:
#   Generate thesis- and journal-style figures for Chapter 4 basin heterogeneity
#   analysis using existing outputs from run_ch4_analysis.py.
#
# Inputs:
#   - experiments/formal_ch4_basin_groups/summary/ch4_basin_metrics_with_geometry.gpkg
#   - experiments/formal_ch4_basin_groups/summary/ch4_huc_group_summary.csv
#   - experiments/formal_ch4_basin_groups/summary/ch4_aridity_group_summary.csv
#   - experiments/formal_ch4_basin_groups/summary/ch4_feature_importance.csv
#
# Outputs:
#   - fig4_1_huc2_region_map.{png,pdf}
#   - fig4_2_aridity_class_map.{png,pdf}
#   - fig4_3_huc_group_delta_nse_boxplot.{png,pdf}
#   - fig4_4_huc_group_positive_transfer_rate.{png,pdf}
#   - fig4_5_basin_attribute_importance.{png,pdf}
# ==============================================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASIN_GROUP_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_basin_groups"
BASIN_GROUP_SUMMARY_DIR = BASIN_GROUP_DIR / "summary"

TRAINING_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
FIGURE_DIR = TRAINING_EXPERIMENT_DIR / "figures"

BASIN_GPKG_PATH = BASIN_GROUP_SUMMARY_DIR / "ch4_basin_metrics_with_geometry.gpkg"
HUC_SUMMARY_PATH = BASIN_GROUP_SUMMARY_DIR / "ch4_huc_group_summary.csv"
ARIDITY_SUMMARY_PATH = BASIN_GROUP_SUMMARY_DIR / "ch4_aridity_group_summary.csv"
FEATURE_IMPORTANCE_PATH = BASIN_GROUP_SUMMARY_DIR / "ch4_feature_importance.csv"

MAP_CRS = "EPSG:5070"
JOURNAL_DPI = 600
FONT_FAMILY = "Times New Roman"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "stl": "#BDBDBD",
    "cgc": "#4C8BBE",
    "blue": "#6BAED6",
    "light_blue": "#9ECAE1",
    "orange": "#F1B76A",
    "green": "#A8D5A2",
    "pink": "#D8A0AE",
    "grey": "#E6E6E6",
    "edge": "#222222",
    "grid": "#D9D9D9",
    "state": "#B8B8B8",
}

ARIDITY_COLORS = {
    "Humid": "#6BAED6",
    "Sub-humid": "#A8D5A2",
    "Semi-arid": "#F1B76A",
    "Arid": "#D8A0AE",
}

HUC2_COLORS = [
    "#6BAED6", "#9ECAE1", "#A8D5A2", "#C7E9C0", "#F1B76A", "#FDD49E",
    "#D8A0AE", "#E7C6D7", "#BDBDBD", "#D9D9D9", "#8DA0CB", "#B3CDE3",
    "#BCAAA4", "#D7CCC8", "#80CBC4", "#B2DFDB", "#FDBF6F", "#CAB2D6",
]


def configure_matplotlib() -> None:
    """Set a consistent journal-style plotting theme."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "DejaVu Serif", "Times"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.9,
            "axes.edgecolor": COLORS["edge"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 12,
            "savefig.dpi": JOURNAL_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_file(path: Path) -> None:
    """Raise a clear error if a required input is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_basin_id(value: object) -> str:
    """Return an eight-digit CAMELS basin identifier."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def normalize_huc2(value: object) -> str:
    """Return a two-digit HUC2 identifier."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    extracted = "".join(ch for ch in text if ch.isdigit())
    if not extracted:
        return text
    return extracted.zfill(2)


def pick_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    """Pick the first matching column by case-insensitive name."""
    lower_map = {column.lower(): column for column in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise KeyError(f"Missing required column. Candidates: {list(candidates)}")
    return None


def standardize_basin_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize basin-level GeoDataFrame column names."""
    out = gdf.copy()

    basin_col = pick_column(
        out,
        ["basin_id", "gauge_id", "gage_id", "hru_id", "site_no"],
        required=False,
    )
    if basin_col is not None:
        out["basin_id"] = out[basin_col].map(normalize_basin_id)

    huc_col = pick_column(
        out,
        ["huc2", "huc_02", "HUC2", "region"],
        required=False,
    )
    if huc_col is not None:
        out["huc2"] = out[huc_col].map(normalize_huc2)

    delta_col = pick_column(
        out,
        [
            "delta_nse_q",
            "Delta_NSE_CGC_minus_STLQ",
            "delta_nse_cgc_minus_stlq",
            "delta_streamflow_nse_cgc_minus_stlq",
            "cgc_minus_stlq_nse",
            "delta_nse",
        ],
        required=False,
    )
    if delta_col is not None:
        out["delta_nse_q"] = pd.to_numeric(out[delta_col], errors="coerce")

    aridity_col = pick_column(
        out,
        ["aridity_class", "aridity_group", "ai_class", "climate_class"],
        required=False,
    )
    if aridity_col is not None:
        out["aridity_class"] = out[aridity_col].astype(str)
    else:
        ai_col = pick_column(
            out,
            ["aridity_index", "aridity", "ai", "p_pet_ratio"],
            required=False,
        )
        if ai_col is not None:
            ai = pd.to_numeric(out[ai_col], errors="coerce")
            out["aridity_class"] = pd.cut(
                ai,
                bins=[-np.inf, 0.65, 1.0, 2.0, np.inf],
                labels=["Arid", "Semi-arid", "Sub-humid", "Humid"],
            ).astype(str)

    if out.crs is None:
        out = out.set_crs("EPSG:4326", allow_override=True)

    return out.to_crs(MAP_CRS)


def load_basin_geodata() -> gpd.GeoDataFrame:
    """Load basin-level metrics with geometry from Chapter 4 analysis output."""
    require_file(BASIN_GPKG_PATH)

    gdf = gpd.read_file(BASIN_GPKG_PATH)
    gdf = standardize_basin_gdf(gdf)

    required = ["huc2", "delta_nse_q"]
    missing = [column for column in required if column not in gdf.columns]
    if missing:
        raise ValueError(
            f"Missing required standardized columns in {BASIN_GPKG_PATH}: {missing}. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    print(f"Matched basin geometries: {len(gdf)}")
    return gdf


def load_huc_summary() -> pd.DataFrame:
    """Load HUC2 summary table generated by analyze_huc_groups.py."""
    require_file(HUC_SUMMARY_PATH)

    df = pd.read_csv(HUC_SUMMARY_PATH, dtype={"huc_02": str})
    required = [
        "huc_02",
        "n_basins",
        "median_delta_nse",
        "q25_delta_nse",
        "q75_delta_nse",
        "positive_transfer_rate_pct",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {HUC_SUMMARY_PATH}: {missing}")

    df = df.copy()
    df["huc2"] = df["huc_02"].map(normalize_huc2)
    for column in required[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["huc2", "median_delta_nse"]).copy()


def load_feature_importance() -> pd.DataFrame:
    """Load feature-importance output from Chapter 4 analysis."""
    require_file(FEATURE_IMPORTANCE_PATH)
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    name_col = pick_column(
        df,
        ["feature", "attribute", "feature_name", "attribute_label", "variable"],
    )
    value_col = pick_column(
        df,
        [
            "importance_mean",
            "permutation_importance_mean",
            "importance",
            "mean_importance",
            "score",
        ],
    )
    std_col = pick_column(
        df,
        ["importance_std", "permutation_importance_std", "std", "std_importance"],
        required=False,
    )

    out = pd.DataFrame()
    out["feature"] = df[name_col].astype(str)
    out["importance"] = pd.to_numeric(df[value_col], errors="coerce")

    if std_col is not None:
        out["importance_std"] = pd.to_numeric(df[std_col], errors="coerce")
    else:
        out["importance_std"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["feature", "importance"])
    out = out.sort_values("importance", ascending=False)
    return out


def load_state_boundaries() -> Optional[gpd.GeoDataFrame]:
    """Load optional US state boundaries if available."""
    candidates = [
        PROJECT_ROOT / "data" / "shapefiles" / "cb_2018_us_state_500k.shp",
        PROJECT_ROOT / "data" / "us_states" / "cb_2018_us_state_500k.shp",
        PROJECT_ROOT / "data" / "natural_earth" / "ne_110m_admin_1_states_provinces.shp",
        Path(
            "/home/mochen/.local/share/cartopy/shapefiles/natural_earth/cultural/"
            "ne_50m_admin_1_states_provinces_lakes.shp"
        ),
        Path(
            "/home/mochen/.local/share/cartopy/shapefiles/natural_earth/cultural/"
            "ne_110m_admin_1_states_provinces_lakes.shp"
        ),
    ]

    state_path = next((path for path in candidates if path.exists()), None)
    if state_path is None:
        return None

    states = gpd.read_file(state_path)
    if states.crs is None:
        states = states.set_crs("EPSG:4326", allow_override=True)

    if "admin" in states.columns:
        states = states[states["admin"].astype(str) == "United States of America"].copy()
    elif "adm0_a3" in states.columns:
        states = states[states["adm0_a3"].astype(str) == "USA"].copy()
    elif "iso_a2" in states.columns:
        states = states[states["iso_a2"].astype(str) == "US"].copy()

    if states.empty:
        return None

    return states.to_crs(MAP_CRS)


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply consistent axis styling."""
    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.6,
        color=COLORS["grid"],
        alpha=0.7,
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["edge"])


def draw_base_map(
    ax: plt.Axes,
    states: Optional[gpd.GeoDataFrame],
    bounds: Tuple[float, float, float, float],
) -> None:
    """Draw state boundaries and set equal-area map extent."""
    if states is not None and not states.empty:
        states.boundary.plot(
            ax=ax,
            color=COLORS["state"],
            linewidth=0.45,
            alpha=0.90,
            zorder=1,
        )

    minx, miny, maxx, maxy = bounds
    pad_x = (maxx - minx) * 0.06
    pad_y = (maxy - miny) * 0.07

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_axis_off()


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure as PNG and PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=JOURNAL_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_huc2_map(gdf: gpd.GeoDataFrame) -> None:
    """Plot basins grouped by HUC2 region."""
    data = gdf.dropna(subset=["huc2"]).copy()
    data = data[data["huc2"].astype(str).str.len() > 0].copy()

    hucs = sorted(data["huc2"].dropna().unique())
    color_map = {huc: HUC2_COLORS[i % len(HUC2_COLORS)] for i, huc in enumerate(hucs)}

    states = load_state_boundaries()
    bounds = data.total_bounds

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    draw_base_map(ax, states, bounds)

    for huc in hucs:
        subset = data[data["huc2"] == huc]
        subset.plot(
            ax=ax,
            facecolor=color_map[huc],
            edgecolor=COLORS["edge"],
            linewidth=0.20,
            alpha=0.92,
            zorder=3,
        )

    handles = [
        Patch(
            facecolor=color_map[huc],
            edgecolor=COLORS["edge"],
            linewidth=0.35,
            label=huc,
        )
        for huc in hucs
    ]

    legend = ax.legend(
        handles=handles,
        title="HUC2",
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        frameon=True,
        ncol=1,
        borderpad=0.6,
        handlelength=1.5,
        labelspacing=0.35,
    )
    legend.get_frame().set_edgecolor("#CCCCCC")
    legend.get_frame().set_linewidth(0.8)

    ax.set_title("(a) CAMELS-US basins grouped by HUC2 region", loc="left", pad=8)
    save_figure(fig, FIGURE_DIR / "fig4_1_huc2_region_map.png")


def plot_aridity_map(gdf: gpd.GeoDataFrame) -> None:
    """Plot basins grouped by aridity class."""
    if "aridity_class" not in gdf.columns:
        print("[Skip] aridity_class not found in basin GeoPackage.")
        return

    data = gdf.dropna(subset=["aridity_class"]).copy()
    data = data[~data["aridity_class"].astype(str).isin(["Unknown", "nan", "None"])].copy()

    order = ["Humid", "Sub-humid", "Semi-arid", "Arid"]
    available = [item for item in order if item in set(data["aridity_class"].astype(str))]
    extra = sorted(set(data["aridity_class"].astype(str)) - set(available))
    classes = available + extra

    states = load_state_boundaries()
    bounds = data.total_bounds

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    draw_base_map(ax, states, bounds)

    for cls in classes:
        subset = data[data["aridity_class"].astype(str) == cls]
        subset.plot(
            ax=ax,
            facecolor=ARIDITY_COLORS.get(cls, COLORS["grey"]),
            edgecolor=COLORS["edge"],
            linewidth=0.20,
            alpha=0.92,
            zorder=3,
        )

    handles = [
        Patch(
            facecolor=ARIDITY_COLORS.get(cls, COLORS["grey"]),
            edgecolor=COLORS["edge"],
            linewidth=0.35,
            label=cls,
        )
        for cls in classes
    ]

    legend = ax.legend(
        handles=handles,
        title="Aridity class",
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        frameon=True,
        borderpad=0.7,
        handlelength=1.5,
    )
    legend.get_frame().set_edgecolor("#CCCCCC")
    legend.get_frame().set_linewidth(0.8)

    ax.set_title("(b) CAMELS-US basins grouped by aridity class", loc="left", pad=8)
    save_figure(fig, FIGURE_DIR / "fig4_2_aridity_class_map.png")


def plot_huc_delta_boxplot(gdf: gpd.GeoDataFrame) -> None:
    """Plot HUC2-level streamflow transfer gain distributions."""
    huc_summary = load_huc_summary()
    huc_summary = huc_summary.sort_values("median_delta_nse", ascending=False)

    ordered_hucs = huc_summary["huc2"].tolist()
    box_values = []
    labels = []
    medians = []

    for row in huc_summary.itertuples(index=False):
        values = (
            gdf.loc[gdf["huc2"] == row.huc2, "delta_nse_q"]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) == 0:
            continue

        box_values.append(values)
        labels.append(f"{row.huc2} (n={int(row.n_basins)})")
        medians.append(float(row.median_delta_nse))

    if not box_values:
        print("[Skip] No valid HUC2 boxplot values.")
        return

    fig_height = max(5.2, 0.30 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    box = ax.boxplot(
        box_values,
        vert=False,
        tick_labels=labels,
        widths=0.58,
        showfliers=False,
        whis=(5, 95),
        patch_artist=True,
        medianprops={"color": COLORS["edge"], "linewidth": 1.2},
        boxprops={
            "facecolor": "#F2F2F2",
            "edgecolor": COLORS["edge"],
            "linewidth": 0.9,
        },
        whiskerprops={"color": COLORS["edge"], "linewidth": 0.9},
        capprops={"color": COLORS["edge"], "linewidth": 0.9},
    )

    for patch in box["boxes"]:
        patch.set_alpha(0.98)

    ax.axvline(0.0, color=COLORS["cgc"], linestyle="--", linewidth=1.0)
    style_axis(ax, grid_axis="x")

    all_values = np.concatenate([values for values in box_values if len(values) > 0])
    lo = max(-0.25, np.nanpercentile(all_values, 1) - 0.03)
    hi = min(0.25, np.nanpercentile(all_values, 99) + 0.03)

    if hi - lo < 0.30:
        center = 0.5 * (hi + lo)
        lo, hi = center - 0.16, center + 0.16

    ax.set_xlim(lo, hi)

    for i, median in enumerate(medians, start=1):
        offset = 0.012 if median >= 0 else -0.012
        ha = "left" if median >= 0 else "right"

        ax.text(
            median + offset,
            i,
            f"{median:.3f}",
            ha=ha,
            va="center",
            fontsize=8.5,
            color=COLORS["edge"],
        )

    ax.set_xlabel(r"$\Delta$NSE (CGC-Q minus STL-Q)")
    ax.set_ylabel("HUC2 region")
    ax.set_title("(c) Streamflow transfer gain by HUC2 region", loc="left", pad=10)
    ax.invert_yaxis()

    save_figure(fig, FIGURE_DIR / "fig4_3_huc_group_delta_nse_boxplot.png")


def plot_huc_positive_rate() -> None:
    """Plot HUC2-level positive transfer rate."""
    summary = load_huc_summary()
    summary = summary.sort_values("positive_transfer_rate_pct", ascending=True)

    y = np.arange(len(summary))
    labels = [
        f"{row.huc2} (n={int(row.n_basins)})"
        for row in summary.itertuples(index=False)
    ]

    fig_height = max(5.2, 0.30 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    bars = ax.barh(
        y,
        summary["positive_transfer_rate_pct"],
        color=COLORS["light_blue"],
        edgecolor=COLORS["edge"],
        linewidth=0.8,
        height=0.68,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Positive transfer rate (%)")
    ax.set_ylabel("HUC2 region")
    ax.set_title("(d) Positive transfer rate by HUC2 region", loc="left", pad=10)
    style_axis(ax, grid_axis="x")

    for bar, value in zip(bars, summary["positive_transfer_rate_pct"]):
        ax.text(
            min(float(value) + 1.2, 98.0),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    save_figure(fig, FIGURE_DIR / "fig4_4_huc_group_positive_transfer_rate.png")


def plot_attribute_importance() -> None:
    """Plot basin attribute importance from Chapter 4 analysis output."""
    importance = load_feature_importance()

    if importance.empty:
        print("[Skip] Empty feature-importance table.")
        return

    plot_df = importance.head(10).copy()
    plot_df = plot_df.sort_values("importance", ascending=True)

    y = np.arange(len(plot_df))
    values = plot_df["importance"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.3, 4.9))

    ax.barh(
        y,
        values,
        color=COLORS["light_blue"],
        edgecolor=COLORS["edge"],
        linewidth=0.8,
        height=0.66,
    )

    if plot_df["importance_std"].notna().any():
        std = plot_df["importance_std"].fillna(0.0).to_numpy(dtype=float)
        ax.errorbar(
            values,
            y,
            xerr=std,
            fmt="none",
            ecolor=COLORS["edge"],
            elinewidth=0.9,
            capsize=2.5,
            zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("Basin attribute")
    ax.set_title("(e) Basin attributes associated with transfer gain", loc="left", pad=10)
    style_axis(ax, grid_axis="x")

    x_max = max(0.01, float(np.nanmax(values)) * 1.18)
    ax.set_xlim(0.0, x_max)

    for yi, value in zip(y, values):
        ax.text(
            value + x_max * 0.015,
            yi,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    save_figure(fig, FIGURE_DIR / "fig4_5_basin_attribute_importance.png")


def main() -> None:
    """Generate Chapter 4 basin heterogeneity diagnostic figures."""
    warnings.filterwarnings("ignore", category=UserWarning)
    configure_matplotlib()

    gdf = load_basin_geodata()

    plot_huc2_map(gdf)
    plot_aridity_map(gdf)
    plot_huc_delta_boxplot(gdf)
    plot_huc_positive_rate()
    plot_attribute_importance()


if __name__ == "__main__":
    main()