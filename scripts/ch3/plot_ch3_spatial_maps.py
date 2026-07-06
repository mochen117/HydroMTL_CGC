# ==============================================================================
# Description:
#   Plot journal-style CAMELS-US spatial maps for Chapter 3.
#
# Purpose:
#   Generate spatial maps for both streamflow and evapotranspiration performance
#   and transfer effects using CAMELS basin polygons and US state boundaries.
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"
FIG_DIR = CH3_DIR / "figures"

INPUT_PATH = SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"
OUTPUT_GPKG = SUMMARY_DIR / "ch3_spatial_basin_metrics.gpkg"

BASIN_SHP_PATH = Path(
    "/home/mochen/hydro_data/camels/camels_us/"
    "basin_set_full_res/HCDN_nhru_final_671.shp"
)

US_STATE_SHP_PATH = Path(
    "/home/mochen/.local/share/cartopy/shapefiles/natural_earth/cultural/"
    "ne_50m_admin_1_states_provinces_lakes.shp"
)

CONUS_EXTENT: Tuple[float, float, float, float] = (-126.0, -66.0, 24.0, 50.0)

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def choose_serif_font() -> str:
    """Choose an available serif font."""
    candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return "DejaVu Serif"


def set_publication_style() -> None:
    """Set stable Matplotlib parameters for spatial maps."""
    font_name = choose_serif_font()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_name],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )
    print(f"Using figure font: {font_name}")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize CAMELS gauge ids as 8-digit strings."""
    return series.astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(8)


def normalize_huc2(series: pd.Series) -> pd.Series:
    """Normalize HUC2 codes as 2-digit strings."""
    return series.astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(2)


def infer_shapefile_gauge_column(gdf: gpd.GeoDataFrame) -> str:
    """Infer gauge id column from the CAMELS basin shapefile."""
    candidates = [
        "gauge_id",
        "GAGE_ID",
        "gage_id",
        "hru_id",
        "HRU_ID",
        "basin_id",
        "BASIN_ID",
    ]
    lower_map = {col.lower(): col for col in gdf.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"No gauge id column found in shapefile. Available columns: {list(gdf.columns)}")


def load_metrics() -> pd.DataFrame:
    """Load Chapter 3 basin-level metrics and metadata."""
    require_file(INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str, "huc_02": str})

    if "gauge_id" not in df.columns:
        raise ValueError("Input table must contain 'gauge_id'.")

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    if "huc_02" in df.columns:
        df["huc_02"] = normalize_huc2(df["huc_02"])

    return df


def load_basin_boundaries() -> gpd.GeoDataFrame:
    """Load CAMELS basin boundary polygons."""
    require_file(BASIN_SHP_PATH)

    basins = gpd.read_file(BASIN_SHP_PATH)
    gauge_col = infer_shapefile_gauge_column(basins)

    basins = basins.rename(columns={gauge_col: "gauge_id"})
    basins["gauge_id"] = normalize_gauge_id(basins["gauge_id"])

    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326", allow_override=True)

    basins = basins.to_crs("EPSG:4326")
    basins = basins[basins.geometry.notna()].copy()
    basins = basins[~basins.geometry.is_empty].copy()

    return basins


def load_us_state_boundaries() -> Optional[gpd.GeoDataFrame]:
    """Load Natural Earth US state/province boundaries."""
    if not US_STATE_SHP_PATH.exists():
        print(f"[Warning] US state boundary file not found: {US_STATE_SHP_PATH}")
        return None

    states = gpd.read_file(US_STATE_SHP_PATH)

    if states.crs is None:
        states = states.set_crs("EPSG:4326", allow_override=True)

    states = states.to_crs("EPSG:4326")

    if "admin" in states.columns:
        states = states[states["admin"].astype(str) == "United States of America"].copy()
    elif "adm0_a3" in states.columns:
        states = states[states["adm0_a3"].astype(str) == "USA"].copy()
    elif "iso_a2" in states.columns:
        states = states[states["iso_a2"].astype(str) == "US"].copy()

    if states.empty:
        print("[Warning] US state filtering returned no records.")
        return None

    minx, maxx = CONUS_EXTENT[0], CONUS_EXTENT[1]
    miny, maxy = CONUS_EXTENT[2], CONUS_EXTENT[3]

    states = states.cx[minx:maxx, miny:maxy].copy()
    states = states[states.geometry.notna()].copy()
    states = states[~states.geometry.is_empty].copy()

    return states


def join_metrics_to_boundaries(
    basins: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Join model metrics to CAMELS basin polygons."""
    merged = basins.merge(metrics, on="gauge_id", how="inner")

    missing = len(metrics) - len(merged)
    if missing > 0:
        print(f"[Warning] {missing} metric basins were not matched to polygons.")

    if merged.empty:
        raise ValueError("No basins were matched between metrics and shapefile.")

    print(f"Matched basin geometries: {len(merged)}")
    return merged


def save_figure(path: Path) -> None:
    """Save and close the current figure as PNG and PDF."""
    plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close()

    print(f"Saved: {path}")
    print(f"Saved: {path.with_suffix('.pdf')}")


def setup_axis(title: str) -> Tuple[plt.Figure, plt.Axes]:
    """Create a CONUS longitude-latitude map axis."""
    fig, ax = plt.subplots(figsize=(10.4, 6.0))

    ax.set_xlim(CONUS_EXTENT[0], CONUS_EXTENT[1])
    ax.set_ylim(CONUS_EXTENT[2], CONUS_EXTENT[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, loc="left")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.22)

    return fig, ax


def plot_us_background(ax: plt.Axes, states: Optional[gpd.GeoDataFrame]) -> None:
    """Plot US state boundaries as a light background."""
    if states is None or states.empty:
        return

    states.boundary.plot(
        ax=ax,
        color="0.65",
        linewidth=0.40,
        alpha=0.85,
        zorder=1,
    )


def plot_basin_metric_map(
    gdf: gpd.GeoDataFrame,
    states: Optional[gpd.GeoDataFrame],
    value_col: str,
    output_name: str,
    title: str,
    legend_label: str,
    value_range: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    """Plot one CAMELS basin polygon map colored by a basin-level metric."""
    if value_col not in gdf.columns:
        print(f"[Skip] Missing value column: {value_col}")
        return

    plot_gdf = gdf.copy()
    plot_gdf[value_col] = pd.to_numeric(plot_gdf[value_col], errors="coerce")
    plot_gdf = plot_gdf.dropna(subset=[value_col])

    if plot_gdf.empty:
        print(f"[Skip] No valid records for {value_col}")
        return

    plot_col = value_col
    vmin, vmax = None, None
    norm = None
    clipped_count = 0

    if value_range is not None:
        low, high = value_range
        plot_col = f"{value_col}_display"
        clipped_count = int(((plot_gdf[value_col] < low) | (plot_gdf[value_col] > high)).sum())
        plot_gdf[plot_col] = plot_gdf[value_col].clip(lower=low, upper=high)

        if center_zero:
            norm = TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
        else:
            vmin, vmax = low, high

    _, ax = setup_axis(title)
    plot_us_background(ax, states)

    plot_gdf.plot(
        column=plot_col,
        ax=ax,
        cmap=cmap,
        linewidth=0.06,
        edgecolor="0.25",
        legend=True,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        zorder=3,
        legend_kwds={
            "label": legend_label,
            "shrink": 0.74,
        },
    )

    if value_range is not None and clipped_count > 0:
        ax.text(
            0.01,
            0.02,
            f"Display range: [{value_range[0]:.2f}, {value_range[1]:.2f}]; clipped basins: {clipped_count}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 2.0,
            },
        )

    save_figure(FIG_DIR / output_name)


def main() -> None:
    """Run Chapter 3 spatial mapping routines."""
    set_publication_style()

    metrics = load_metrics()
    basins = load_basin_boundaries()
    states = load_us_state_boundaries()

    basin_metrics = join_metrics_to_boundaries(basins, metrics)
    basin_metrics.to_file(OUTPUT_GPKG, driver="GPKG")
    print(f"Saved: {OUTPUT_GPKG}")

    map_specs = [
        (
            "CGC_streamflow_nse",
            "fig3_10_spatial_cgc_streamflow_nse.png",
            "(a) Spatial distribution of CGC-Q NSE",
            "CGC-Q NSE",
            (-1.0, 1.0),
            "viridis",
            False,
        ),
        (
            "CGC_evapotranspiration_nse",
            "fig3_11_spatial_cgc_evapotranspiration_nse.png",
            "(b) Spatial distribution of CGC-ET NSE",
            "CGC-ET NSE",
            (-1.0, 1.0),
            "viridis",
            False,
        ),
        (
            "Delta_NSE_CGC_minus_STLQ",
            "fig3_12_spatial_delta_nse_cgc_q_minus_stlq.png",
            "(c) Streamflow NSE gain: CGC-Q minus STL-Q",
            "ΔNSE",
            (-0.20, 0.20),
            "coolwarm",
            True,
        ),
        (
            "Delta_NSE_CGC_ET_minus_STLET",
            "fig3_13_spatial_delta_nse_cgc_et_minus_stlet.png",
            "(d) Evapotranspiration NSE gain: CGC-ET minus STL-ET",
            "ΔNSE",
            (-0.20, 0.20),
            "coolwarm",
            True,
        ),
        (
            "Delta_NSE_CGC_minus_HardMTL",
            "fig3_14_spatial_delta_nse_cgc_q_minus_hard_mtl.png",
            "(e) Streamflow NSE gain: CGC-Q minus Hard-MTL",
            "ΔNSE",
            (-0.20, 0.20),
            "coolwarm",
            True,
        ),
        (
            "Delta_NSE_CGC_minus_MMoE",
            "fig3_15_spatial_delta_nse_cgc_q_minus_mmoe.png",
            "(f) Streamflow NSE gain: CGC-Q minus MMoE",
            "ΔNSE",
            (-0.20, 0.20),
            "coolwarm",
            True,
        ),
        (
            "Delta_NSE_CGC_ET_minus_HardMTL",
            "fig3_16_spatial_delta_nse_cgc_et_minus_hard_mtl.png",
            "(g) Evapotranspiration NSE gain: CGC-ET minus Hard-MTL",
            "ΔNSE",
            (-0.20, 0.20),
            "coolwarm",
            True,
        ),
        (
            "Delta_NSE_CGC_ET_minus_MMoE",
            "fig3_17_spatial_delta_nse_cgc_et_minus_mmoe.png",
            "(h) Evapotranspiration NSE gain: CGC-ET minus MMoE",
            "ΔNSE",
            (-0.20, 0.20),
            "coolwarm",
            True,
        ),
    ]

    for value_col, output_name, title, legend_label, value_range, cmap, center_zero in map_specs:
        plot_basin_metric_map(
            gdf=basin_metrics,
            states=states,
            value_col=value_col,
            output_name=output_name,
            title=title,
            legend_label=legend_label,
            value_range=value_range,
            cmap=cmap,
            center_zero=center_zero,
        )


if __name__ == "__main__":
    main()