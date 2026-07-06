# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Plot Chapter 4 controlled data-condition absolute NSE performance boxplots.
#
# Purpose:
#   Generate publication-quality figures for three controlled Chapter 4 experiments.
#   Each figure shows basin-level absolute NSE performance comparison of CGC 
#   relative to the corresponding single-task models (STL-Q / STL-ET).
#       Streamflow:        CGC  vs STL-Q (using absolute NSE)
#       Evapotranspiration: CGC vs STL-ET (using absolute NSE)
#
# Inputs:
#   - experiments/formal_ch4_training_experiments/summary/
#     ch4_training_experiment_per_basin.csv
#
# Outputs:
#   - fig4_6_climate_consistency_nse.{png,pdf}
#   - fig4_7_training_length_nse.{png,pdf}
#   - fig4_8_basin_diversity_nse.{png,pdf}
#   - ch4_controlled_nse_per_basin.csv
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
SUMMARY_DIR = EXPERIMENT_DIR / "summary"
FIGURE_DIR = EXPERIMENT_DIR / "figures"

PER_BASIN_PATH = SUMMARY_DIR / "ch4_training_experiment_per_basin.csv"
NSE_PER_BASIN_PATH = SUMMARY_DIR / "ch4_controlled_nse_per_basin.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

JOURNAL_DPI = 600
EDGE_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"

# Core palette consistent with academic styling
COLORS = {
    "streamflow": "#5B99C5",
    "evapotranspiration": "#FAA256",
    "stl": "#BDBDBD",
    "edge": EDGE_COLOR,
    "grid": GRID_COLOR,
}

EXPERIMENT_CONFIG: Dict[str, Dict[str, object]] = {
    "climate_consistency": {
        "order": ["Low", "Medium", "High"],
        "xlabel": "Train-test climate consistency",
        "title": "Climate consistency",
        "output": "fig4_6_climate_consistency_nse.png",
    },
    "training_length": {
        "order": ["1 yr", "3 yr", "5 yr", "7 yr", "10 yr"],
        "xlabel": "Training data length",
        "title": "Training data length",
        "output": "fig4_7_training_length_nse.png",
    },
    "basin_diversity": {
        "order": ["Low", "Medium", "High"],
        "xlabel": "Training basin diversity",
        "title": "Training basin diversity",
        "output": "fig4_8_basin_diversity_nse.png",
    },
}


def cleanup_obsolete_files() -> None:
    """Automatically remove obsolete duplicate transfer_gain files to keep work space clean."""
    print("Checking for obsolete duplicate files...")
    # Clean up obsolete duplicate figures
    for config in EXPERIMENT_CONFIG.values():
        for suffix in [".png", ".pdf"]:
            base_name = str(config["output"]).replace("_nse.png", "")
            old_fig_name = f"{base_name}_transfer_gain{suffix}"
            old_path = FIGURE_DIR / old_fig_name
            if old_path.exists():
                try:
                    old_path.unlink()
                    print(f"  [Removed obsolete figure]: {old_path.name}")
                except Exception as e:
                    print(f"  [Warning] Could not remove {old_path.name}: {e}")

    # Clean up obsolete duplicate CSV tables
    for old_csv_name in [
        "ch4_controlled_transfer_gain_per_basin.csv",
        "ch4_transfer_gain_summary_by_conditions.csv"
    ]:
        old_csv_path = SUMMARY_DIR / old_csv_name
        if old_csv_path.exists():
            try:
                old_csv_path.unlink()
                print(f"  [Removed obsolete CSV]: {old_csv_path.name}")
            except Exception as e:
                print(f"  [Warning] Could not remove {old_csv_path.name}: {e}")


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def choose_serif_font() -> str:
    """Choose an available serif font for academic-style figures."""
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
    """Set a consistent thesis- and journal-style plotting theme."""
    font_name = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_name],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": COLORS["edge"],
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "savefig.dpi": JOURNAL_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
        raise KeyError(
            f"Missing required column. Candidates: {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_basin_id(value: object) -> str:
    """Normalize basin identifiers as 8-digit strings."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def normalize_model_name(value: object) -> str:
    """Normalize model names."""
    text = str(value).strip()
    lower = text.lower().replace("-", "_")

    if lower in {"stl_q", "stlq", "stl"}:
        return "STL-Q"
    if lower in {"stl_et", "stlet"}:
        return "STL-ET"
    if lower == "cgc":
        return "CGC"

    return text


def normalize_experiment_type(value: object) -> str:
    """Normalize controlled experiment names."""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    if "climate" in text or "consistency" in text:
        return "climate_consistency"
    if "length" in text or ("train" in text and ("yr" in text or "year" in text)):
        return "training_length"
    if "diversity" in text or "basin" in text:
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


def load_per_basin_metrics() -> pd.DataFrame:
    """Load and standardize basin-level experiment metrics."""
    require_file(PER_BASIN_PATH)

    raw = pd.read_csv(PER_BASIN_PATH)

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

    out = pd.DataFrame()
    out["experiment_type"] = raw[experiment_col].map(normalize_experiment_type)
    out["level"] = raw[level_col].map(normalize_level)
    out["model"] = raw[model_col].map(normalize_model_name)
    out["basin_id"] = raw[basin_col].map(normalize_basin_id)

    out["streamflow_nse"] = (
        pd.to_numeric(raw[q_col], errors="coerce") if q_col is not None else np.nan
    )
    out["evapotranspiration_nse"] = (
        pd.to_numeric(raw[et_col], errors="coerce") if et_col is not None else np.nan
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["experiment_type", "level", "model", "basin_id"])
    out = out[out["model"].isin(["CGC", "STL-Q", "STL-ET"])].copy()

    if out.empty:
        raise ValueError(f"No valid Chapter 4 basin-level records were found in {PER_BASIN_PATH}.")

    return out


def prepare_nse_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy and structure absolute NSE values for plotting."""
    # Streamflow records
    q_df = df[df["model"].isin(["CGC", "STL-Q"])].dropna(subset=["streamflow_nse"]).copy()
    q_df["task"] = "streamflow"
    q_df["nse"] = q_df["streamflow_nse"]
    q_df = q_df[["experiment_type", "level", "basin_id", "task", "model", "nse"]]

    # Evapotranspiration records
    et_df = df[df["model"].isin(["CGC", "STL-ET"])].dropna(subset=["evapotranspiration_nse"]).copy()
    et_df["task"] = "evapotranspiration"
    et_df["nse"] = et_df["evapotranspiration_nse"]
    et_df = et_df[["experiment_type", "level", "basin_id", "task", "model", "nse"]]

    nse_df = pd.concat([q_df, et_df], ignore_index=True)
    nse_df.to_csv(NSE_PER_BASIN_PATH, index=False)
    return nse_df


def style_axis(ax: Axes) -> None:
    """Apply consistent axis styling."""
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.55,
        color=COLORS["grid"],
        alpha=0.65,
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["edge"])


def style_boxplot(box_obj: Dict[str, object], colors: Sequence[str]) -> None:
    """Apply standard academic formatting to boxplot components."""
    for patch, color in zip(box_obj["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(COLORS["edge"])
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


def get_y_limits_for_panel(values_stl: Sequence[np.ndarray], values_cgc: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Calculate optimal Y-axis limits ensuring 5th and 95th percentile whisker caps are fully visible with safe margins."""
    all_arrays = values_stl + values_cgc
    # Whiskers represent 5th and 95th percentiles due to whis=(5, 95)
    lows = [float(np.nanpercentile(arr, 5)) for arr in all_arrays if len(arr) > 0]
    highs = [float(np.nanpercentile(arr, 95)) for arr in all_arrays if len(arr) > 0]

    if not lows or not highs:
        return -0.2, 1.02

    min_val = min(lows)
    max_val = max(highs)

    # Since the raw data has been safely clipped to -1.0, min_val is guaranteed >= -1.0.
    y_min = min_val
    y_max = min(max_val, 1.0)

    span = y_max - y_min
    
    # 13% bottom margin and 8% top margin ensures whisker caps have perfect breathing room and do not touch borders
    ylim_low = y_min - span * 0.13
    ylim_high = min(1.05, y_max + span * 0.08)

    return ylim_low, ylim_high


def annotate_medians(
    ax: Axes,
    positions: Sequence[float],
    values: Sequence[np.ndarray],
    color: str = "black",
    weight: str = "normal",
) -> None:
    """Annotate boxplot medians centered on the median lines with a white badge to prevent clashing."""
    for x, val_arr in zip(positions, values):
        if len(val_arr) == 0:
            continue
        median = float(np.nanmedian(val_arr))
        # Center the text badge right on the median line, masking any overlapping lines
        ax.text(
            x,
            median,
            f"{median:.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight=weight,
            color=color,
            bbox=dict(
                facecolor="white",
                alpha=0.90,
                edgecolor="none",
                pad=1.2,
            ),
            clip_on=False,
            zorder=10,
        )


def plot_task_panel(
    ax: Axes,
    data: pd.DataFrame,
    experiment_type: str,
    task: str,
    order: Sequence[str],
    panel_title: str,
) -> None:
    """Plot one task panel with side-by-side boxplots."""
    baseline_model = "STL-Q" if task == "streamflow" else "STL-ET"

    # Define standard hydrological plotting threshold floor for absolute NSE
    DATA_FLOOR = -1.0

    positions_stl = []
    positions_cgc = []
    values_stl = []
    values_cgc = []
    labels = []

    for i, level in enumerate(order):
        pos = i + 1
        sub_lvl = data[
            (data["experiment_type"] == experiment_type)
            & (data["task"] == task)
            & (data["level"] == level)
        ]

        v_stl = sub_lvl[sub_lvl["model"] == baseline_model]["nse"].dropna().to_numpy()
        v_cgc = sub_lvl[sub_lvl["model"] == "CGC"]["nse"].dropna().to_numpy()

        # Clip the raw NSE values to the standard hydrology display floor [1]
        # This prevents extreme outliers (like NSE = -5.0) from extending whiskers off-screen,
        # ensuring both upper and lower caps (5% and 95%) are fully, cleanly drawn within the margins [1].
        v_stl_clipped = np.clip(v_stl, DATA_FLOOR, 1.0)
        v_cgc_clipped = np.clip(v_cgc, DATA_FLOOR, 1.0)

        values_stl.append(v_stl_clipped)
        values_cgc.append(v_cgc_clipped)

        # Displace STL left, CGC right
        positions_stl.append(pos - 0.2)
        positions_cgc.append(pos + 0.2)
        labels.append(f"{level}\n(n={len(v_cgc)})")

    # Calculate optimal Y-limits based on perfectly-clipped data [1]
    ylim_low, ylim_high = get_y_limits_for_panel(values_stl, values_cgc)
    ax.set_ylim(ylim_low, ylim_high)

    # Plot STL baseline boxplots
    box_stl = ax.boxplot(
        values_stl,
        positions=positions_stl,
        patch_artist=True,
        showfliers=False,
        widths=0.3,
        whis=(5, 95),
    )

    # Plot CGC joint boxplots
    box_cgc = ax.boxplot(
        values_cgc,
        positions=positions_cgc,
        patch_artist=True,
        showfliers=False,
        widths=0.3,
        whis=(5, 95),
    )

    # Apply task-specific colors and formatting
    task_color = COLORS["streamflow"] if task == "streamflow" else COLORS["evapotranspiration"]
    style_boxplot(box_stl, [COLORS["stl"]] * len(order))
    style_boxplot(box_cgc, [task_color] * len(order))

    # Reference levels
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=0.9, zorder=0)
    ax.axhline(0.0, color=COLORS["edge"], linestyle="--", linewidth=0.7, zorder=0)

    # Annotations centered on the line inside white background badges
    annotate_medians(ax, positions_stl, values_stl, color="#444444")
    annotate_medians(ax, positions_cgc, values_cgc, color="black", weight="bold")

    # Set independent labels
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(panel_title, loc="left", pad=7)

    style_axis(ax)


def plot_experiment(
    data: pd.DataFrame,
    experiment_type: str,
    order: Sequence[str],
    title: str,
    xlabel: str,
    output_name: str,
) -> None:
    """Plot Q and ET absolute NSE comparison for one controlled experiment."""
    sub = data[data["experiment_type"] == experiment_type].copy()
    if sub.empty:
        print(f"[Skip] No absolute NSE data found for {experiment_type}.")
        return

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=False)

    plot_task_panel(
        ax=axes[0],
        data=sub,
        experiment_type=experiment_type,
        task="streamflow",
        order=order,
        panel_title="(a) Streamflow",
    )
    plot_task_panel(
        ax=axes[1],
        data=sub,
        experiment_type=experiment_type,
        task="evapotranspiration",
        order=order,
        panel_title="(b) Evapotranspiration",
    )

    axes[0].set_ylabel("Streamflow NSE")
    axes[1].set_ylabel("Evapotranspiration NSE")

    # Centered Master Title
    fig.suptitle(f"{title} Model Comparison (STL vs CGC)", fontsize=12.2, y=0.985, fontweight="bold")
    
    # Unified Global Legend at the top (avoiding clashing inside subplots entirely)
    legend_handles = [
        Patch(facecolor=COLORS["stl"], edgecolor=COLORS["edge"], linewidth=0.9, label="STL Baseline"),
        Patch(facecolor=COLORS["streamflow"], edgecolor=COLORS["edge"], linewidth=0.9, label="CGC (Streamflow)"),
        Patch(facecolor=COLORS["evapotranspiration"], edgecolor=COLORS["edge"], linewidth=0.9, label="CGC (Evapotranspiration)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.925),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )

    # Global X-label positioned safely
    fig.text(0.53, 0.05, xlabel, ha="center", va="center", fontsize=10.5)
    
    # Footnote at the bottom
    fig.text(
        0.02,
        0.012,
        "Outliers are not shown; whiskers denote the 5th and 95th percentiles. Bold text represents CGC.",
        fontsize=7.5,
        color="#444444"
    )

    # Exquisitely balanced subplot grid margins
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.81,
        bottom=0.21,
        wspace=0.28,
    )

    # Save output cleanly
    save_figure(fig, FIGURE_DIR / output_name)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure in both high-resolution PNG and vector PDF formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=JOURNAL_DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {path.name} (and .pdf)")


def print_coverage(nse_df: pd.DataFrame) -> None:
    """Print data coverage information."""
    print("=" * 100)
    print("Chapter 4 controlled absolute NSE plotting coverage:")
    for experiment_type in ["climate_consistency", "training_length", "basin_diversity"]:
        sub = nse_df[nse_df["experiment_type"] == experiment_type]
        for task in ["streamflow", "evapotranspiration"]:
            task_sub = sub[sub["task"] == task]
            print(f"  {experiment_type} - {task}: {len(task_sub)} paired basin records")
    print("=" * 100)


def main() -> None:
    """Generate Chapter 4 controlled absolute NSE figures."""
    configure_matplotlib()
    
    # Cleanup any old redundant / duplicate files first to keep workspace clean
    cleanup_obsolete_files()
    
    per_basin = load_per_basin_metrics()
    nse_data = prepare_nse_data(per_basin)
    
    print_coverage(nse_data)

    # Plot the three controlled experiment figures (no fig4_9)
    for experiment_type, config in EXPERIMENT_CONFIG.items():
        sub = nse_data[nse_data["experiment_type"] == experiment_type]
        valid_order = [
            level for level in config["order"]
            if level in set(sub["level"].unique())
        ]

        if not valid_order:
            print(f"[Skip] No valid condition levels for {experiment_type}.")
            continue

        plot_experiment(
            data=nse_data,
            experiment_type=experiment_type,
            order=valid_order,
            title=str(config["title"]),
            xlabel=str(config["xlabel"]),
            output_name=str(config["output"]),
        )

    print("=" * 100)
    print("Chapter 4 controlled absolute NSE plotting completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()