# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Plot Chapter 4 hydroclimatic nonstationarity diagnostics.
#
# Purpose:
#   Generate supplementary figures supporting the motivation of Chapter 4:
#   hydroclimatic variables may show significant temporal changes, which can
#   lead to differences between training and testing hydroclimatic conditions.
#
# Inputs:
#   - experiments/formal_ch4_training_experiments/summary/
#     ch4_hydroclimate_nonstationarity_summary.csv
#   - experiments/formal_ch4_training_experiments/summary/
#     ch4_hydroclimate_nonstationarity_per_basin.csv
#   - experiments/formal_ch4_training_experiments/summary/
#     ch4_hydroclimate_representative_basins.csv
#
# Outputs:
#   - figS3_hydroclimate_nonstationarity_bar.png
#   - figS4_representative_nonstationary_basins.png
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"

CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
SUMMARY_DIR = CH4_DIR / "summary"
FIGURE_DIR = CH4_DIR / "figures"

SUMMARY_PATH = SUMMARY_DIR / "ch4_hydroclimate_nonstationarity_summary.csv"
PER_BASIN_PATH = SUMMARY_DIR / "ch4_hydroclimate_nonstationarity_per_basin.csv"
REPRESENTATIVE_PATH = SUMMARY_DIR / "ch4_hydroclimate_representative_basins.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

JOURNAL_DPI = 600

VARIABLE_LABELS = {
    "precipitation": "Precipitation",
    "temperature": "Temperature",
    "evapotranspiration": "Evapotranspiration",
    "streamflow": "Streamflow",
}

VARIABLE_ORDER = [
    "precipitation",
    "temperature",
    "evapotranspiration",
    "streamflow",
]

VARIABLE_UNITS = {
    "precipitation": "Annual mean precipitation",
    "temperature": "Annual mean temperature",
    "evapotranspiration": "Annual mean evapotranspiration",
    "streamflow": "Annual mean streamflow",
}

VARIABLE_CANDIDATES: Dict[str, List[str]] = {
    "precipitation": [
        "total_precipitation",
        "precipitation",
        "precip",
        "prcp",
        "P",
    ],
    "temperature": [
        "temperature",
        "temp",
        "t_mean",
        "tas",
        "T",
    ],
    "evapotranspiration": [
        "evapotranspiration",
        "actual_evapotranspiration",
        "ET",
        "aet",
    ],
    "streamflow": [
        "streamflow",
        "discharge",
        "runoff",
        "Q",
    ],
}

COLORS = {
    "increase": "#5B99C5",
    "decrease": "#F0AFAF",
    "non_significant": "#CDE0CC",
    "line": "#222222",
    "trend": "#B22222",
    "edge": "#222222",
    "grid": "#D9D9D9",
}


def require_file(path: Path) -> None:
    """Validate that a required file exists."""
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

    for name in candidates:
        if name in available:
            return name

    return "DejaVu Serif"


def configure_matplotlib() -> None:
    """Configure publication-style matplotlib parameters."""
    font_name = choose_serif_font()

    mpl.rcParams.update(
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
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.9,
            "axes.edgecolor": COLORS["edge"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "savefig.dpi": JOURNAL_DPI,
            "figure.dpi": 150,
        }
    )

    print(f"[Info] Figure font: {font_name}")


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save one figure as PNG and PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=JOURNAL_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(fig)

    print(f"[Saved] {output_path}")
    print(f"[Saved] {output_path.with_suffix('.pdf')}")


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply consistent axis styling."""
    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.55,
        color=COLORS["grid"],
        alpha=0.70,
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(COLORS["edge"])
        spine.set_linewidth(0.9)


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    require_file(path)
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_gauge_id(value: object) -> str:
    """Normalize basin ID as an 8-digit string."""
    return str(value).strip().replace(".0", "").zfill(8)


def infer_variable(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    """Infer variable name from candidate names."""
    lower_map = {name.lower(): name for name in ds.data_vars}

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]

    return None


def infer_time_dim(da: xr.DataArray) -> str:
    """Infer time dimension name."""
    for dim in da.dims:
        if dim.lower() in {"time", "date", "datetime"}:
            return dim

    return da.dims[0]


def annual_series(
    da: xr.DataArray,
    period: Sequence[str],
) -> pd.Series:
    """Aggregate source data to annual mean values."""
    time_dim = infer_time_dim(da)
    sub = da.sel({time_dim: slice(period[0], period[1])})

    if sub.size == 0:
        return pd.Series(dtype=float)

    frame = sub.to_dataframe(name="value").reset_index()
    frame[time_dim] = pd.to_datetime(frame[time_dim], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=[time_dim, "value"])

    if frame.empty:
        return pd.Series(dtype=float)

    frame["year"] = frame[time_dim].dt.year

    annual = frame.groupby("year")["value"].mean()
    annual = annual.replace([np.inf, -np.inf], np.nan).dropna()

    return annual


def load_representative_series(
    gauge_id: str,
    variable: str,
) -> pd.Series:
    """Load annual series for one representative basin and variable."""
    cfg = load_yaml(BASE_CONFIG)
    data_root = Path(cfg["data"]["data_root"])
    train_period = cfg["data"]["train_period"]
    test_period = cfg["data"]["test_period"]
    analysis_period = [train_period[0], test_period[1]]

    nc_path = data_root / f"gage_{normalize_gauge_id(gauge_id)}.nc"
    require_file(nc_path)

    with xr.open_dataset(nc_path) as ds:
        var_name = infer_variable(ds, VARIABLE_CANDIDATES[variable])
        if var_name is None:
            raise ValueError(
                f"Could not find variable '{variable}' in {nc_path}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        series = annual_series(ds[var_name], analysis_period)

    return series


def read_nonstationarity_summary() -> pd.DataFrame:
    """Read nonstationarity summary table."""
    require_file(SUMMARY_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    required = {
        "variable",
        "n_basins",
        "significant_increase_count",
        "significant_decrease_count",
        "non_significant_count",
        "significant_rate_pct",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in {SUMMARY_PATH}: {sorted(missing)}")

    summary["variable"] = summary["variable"].astype(str)
    summary = summary[summary["variable"].isin(VARIABLE_ORDER)].copy()
    summary["variable_order"] = summary["variable"].map(
        {name: i for i, name in enumerate(VARIABLE_ORDER)}
    )
    summary = summary.sort_values("variable_order")

    return summary


def read_representative_basins() -> pd.DataFrame:
    """Read representative nonstationary basin table."""
    require_file(REPRESENTATIVE_PATH)
    table = pd.read_csv(REPRESENTATIVE_PATH, dtype={"gauge_id": str})

    required = {
        "variable",
        "representative_type",
        "gauge_id",
        "mk_z",
        "mk_p_value",
        "sen_slope",
        "trend_direction",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {REPRESENTATIVE_PATH}: {sorted(missing)}")

    table["gauge_id"] = table["gauge_id"].map(normalize_gauge_id)

    return table


def annotate_stacked_bars(
    ax: plt.Axes,
    x: np.ndarray,
    bottoms: np.ndarray,
    values: np.ndarray,
) -> None:
    """Annotate stacked bars when segments are large enough."""
    for xi, bottom, value in zip(x, bottoms, values):
        if not np.isfinite(value) or value < 5.0:
            continue

        ax.text(
            xi,
            bottom + value / 2.0,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=8.6,
            color="black",
        )


def plot_nonstationarity_bar(summary: pd.DataFrame) -> None:
    """Plot proportions of significant increasing/decreasing trends."""
    plot_df = summary.copy()

    n = pd.to_numeric(plot_df["n_basins"], errors="coerce").to_numpy(dtype=float)
    inc = pd.to_numeric(
        plot_df["significant_increase_count"],
        errors="coerce",
    ).to_numpy(dtype=float)
    dec = pd.to_numeric(
        plot_df["significant_decrease_count"],
        errors="coerce",
    ).to_numpy(dtype=float)
    ns = pd.to_numeric(
        plot_df["non_significant_count"],
        errors="coerce",
    ).to_numpy(dtype=float)

    inc_pct = inc / n * 100.0
    dec_pct = dec / n * 100.0
    ns_pct = ns / n * 100.0

    labels = [VARIABLE_LABELS[v] for v in plot_df["variable"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    bars_dec = ax.bar(
        x,
        dec_pct,
        color=COLORS["decrease"],
        edgecolor=COLORS["edge"],
        linewidth=0.8,
        label="Significant decrease",
        zorder=3,
    )
    bars_ns = ax.bar(
        x,
        ns_pct,
        bottom=dec_pct,
        color=COLORS["non_significant"],
        edgecolor=COLORS["edge"],
        linewidth=0.8,
        label="Not significant",
        zorder=3,
    )
    bars_inc = ax.bar(
        x,
        inc_pct,
        bottom=dec_pct + ns_pct,
        color=COLORS["increase"],
        edgecolor=COLORS["edge"],
        linewidth=0.8,
        label="Significant increase",
        zorder=3,
    )

    annotate_stacked_bars(ax, x, np.zeros_like(dec_pct), dec_pct)
    annotate_stacked_bars(ax, x, dec_pct, ns_pct)
    annotate_stacked_bars(ax, x, dec_pct + ns_pct, inc_pct)

    for xi, total_n, sig_pct in zip(
        x,
        plot_df["n_basins"],
        plot_df["significant_rate_pct"],
    ):
        ax.text(
            xi,
            102.0,
            f"n={int(total_n)}\nSig.={float(sig_pct):.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.3,
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Basin proportion (%)")
    ax.set_ylim(0.0, 112.0)
    ax.set_title(
        "Hydroclimatic nonstationarity detected by Mann-Kendall tests",
        loc="left",
        pad=10,
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        columnspacing=1.1,
        handlelength=1.3,
    )

    style_axis(ax, grid_axis="y")

    fig.text(
        0.01,
        0.01,
        "Significance level: two-sided α = 0.10.",
        ha="left",
        va="bottom",
        fontsize=8.4,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.78,
        bottom=0.17,
    )

    save_figure(
        fig,
        FIGURE_DIR / "figS3_hydroclimate_nonstationarity_bar.png",
    )


def fit_trend_line(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a simple linear trend line for visualization."""
    y = pd.to_numeric(series, errors="coerce").dropna()
    x = y.index.to_numpy(dtype=float)
    values = y.to_numpy(dtype=float)

    if len(values) < 2:
        return x, np.full_like(x, np.nan, dtype=float)

    slope, intercept = np.polyfit(x, values, deg=1)
    fitted = slope * x + intercept

    return x, fitted


def format_representative_title(row: pd.Series) -> str:
    """Format representative basin panel title."""
    variable = VARIABLE_LABELS.get(row["variable"], row["variable"])
    direction = str(row["trend_direction"]).replace("_", " ")
    basin = normalize_gauge_id(row["gauge_id"])

    return (
        f"{variable}: {direction}\n"
        f"Basin {basin}, Z={float(row['mk_z']):.2f}"
    )


def choose_representative_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Choose a compact representative set for plotting."""
    preferred_variables = ["precipitation", "temperature", "streamflow"]

    selected = table[
        table["variable"].isin(preferred_variables)
        & table["representative_type"].isin(
            ["strongest_increase", "strongest_decrease"]
        )
    ].copy()

    if selected.empty:
        selected = table.head(6).copy()

    selected["variable_order"] = selected["variable"].map(
        {name: i for i, name in enumerate(preferred_variables)}
    ).fillna(99)

    selected["type_order"] = selected["representative_type"].map(
        {"strongest_increase": 0, "strongest_decrease": 1}
    ).fillna(99)

    selected = selected.sort_values(["variable_order", "type_order"])
    selected = selected.head(6)

    return selected


def plot_representative_basins(representative: pd.DataFrame) -> None:
    """Plot annual series for representative nonstationary basins."""
    selected = choose_representative_rows(representative)

    if selected.empty:
        print("[Skip] No representative nonstationary basins available.")
        return

    n_panels = len(selected)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(9.6, max(3.0 * n_rows, 4.0)),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (_, row) in zip(axes_flat, selected.iterrows()):
        variable = str(row["variable"])
        gauge_id = normalize_gauge_id(row["gauge_id"])

        try:
            series = load_representative_series(gauge_id, variable)
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Failed to load basin {gauge_id}\n{exc}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
            )
            ax.set_axis_off()
            continue

        if series.empty:
            ax.text(
                0.5,
                0.5,
                f"No annual data: {gauge_id}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
            )
            ax.set_axis_off()
            continue

        ax.plot(
            series.index,
            series.values,
            marker="o",
            markersize=3.4,
            linewidth=1.25,
            color=COLORS["line"],
            markerfacecolor="white",
            markeredgecolor=COLORS["line"],
            markeredgewidth=0.7,
            zorder=3,
        )

        years, fitted = fit_trend_line(series)
        if len(years) > 0 and np.isfinite(fitted).any():
            ax.plot(
                years,
                fitted,
                color=COLORS["trend"],
                linewidth=1.35,
                linestyle="--",
                zorder=4,
            )

        ax.set_title(format_representative_title(row), loc="left", pad=6)
        ax.set_xlabel("Year")
        ax.set_ylabel(VARIABLE_UNITS.get(variable, variable))
        style_axis(ax, grid_axis="both")

    for ax in axes_flat[n_panels:]:
        ax.set_axis_off()

    fig.suptitle(
        "Representative basins with significant hydroclimatic trends",
        fontsize=12.6,
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.08,
        wspace=0.28,
        hspace=0.48,
    )

    save_figure(
        fig,
        FIGURE_DIR / "figS4_representative_nonstationary_basins.png",
    )


def main() -> None:
    """Generate Chapter 4 nonstationarity diagnostic figures."""
    print("=" * 100)
    print("Chapter 4 nonstationarity diagnostic plotting")
    print("=" * 100)

    configure_matplotlib()

    summary = read_nonstationarity_summary()
    representative = read_representative_basins()

    plot_nonstationarity_bar(summary)
    plot_representative_basins(representative)

    print("=" * 100)
    print("Chapter 4 nonstationarity diagnostic figures completed.")
    print(f"Output directory: {FIGURE_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()