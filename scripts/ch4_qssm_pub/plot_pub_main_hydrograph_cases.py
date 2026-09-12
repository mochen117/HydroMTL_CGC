#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot selected Chapter 4B PUB hydrograph cases with event precipitation.

Each figure contains:
1. the full PUB test-period streamflow hydrograph;
2. a selected high-flow event with precipitation and streamflow.

Formal daily predictions are loaded from the verified PUB prediction
exports. Precipitation is read from the same basin NetCDF forcing files
used by the Chapter 4B experiments.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from rank_pub_flood_events import (
    load_basin_predictions,
    normalize_gauge_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CASES = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydrograph_candidates"
    / "pub_hydrograph_main_cases.csv"
)

DEFAULT_MASTER = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydrograph_candidates"
    / "pub_hydrograph_candidate_master.csv"
)

DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydrograph_candidates"
    / "event_screening"
    / "pub_highflow_event_screening_summary.csv"
)

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "output_592_basins"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydrograph_candidates"
    / "main_case_figures"
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected PUB hydrograph cases with precipitation."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--pre-days",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--post-days",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_inputs(
    cases_path: Path,
    master_path: Path,
    events_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(
        cases_path,
        dtype={"gauge_id": str},
    )
    master = pd.read_csv(
        master_path,
        dtype={"gauge_id": str},
    )
    events = pd.read_csv(
        events_path,
        dtype={"gauge_id": str},
    )

    for frame in (cases, master, events):
        frame["gauge_id"] = frame["gauge_id"].map(
            normalize_gauge_id
        )

    cases["event_peak_date"] = pd.to_datetime(
        cases["event_peak_date"]
    )
    events["event_peak_date"] = pd.to_datetime(
        events["event_peak_date"]
    )

    required = {
        "gauge_id",
        "fold_id",
        "hydroclimate_group",
        "event_peak_date",
        "case_type",
        "case_order",
    }
    missing = required.difference(cases.columns)

    if missing:
        raise ValueError(
            f"Missing case columns: {sorted(missing)}"
        )

    return cases, master, events


def load_precipitation(
    data_root: Path,
    gauge_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read event precipitation from the formal basin forcing file."""
    path = data_root / f"gage_{gauge_id}.nc"

    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        if "total_precipitation" not in ds:
            raise KeyError(
                f"{gauge_id}: total_precipitation not found in {path}."
            )

        data = ds["total_precipitation"].sel(
            time=slice(start, end)
        )

        frame = pd.DataFrame({
            "time": pd.to_datetime(data["time"].values),
            "precipitation": np.asarray(
                data.values,
                dtype=float,
            ),
        })

    expected = pd.date_range(
        start,
        end,
        freq="D",
    )

    actual = pd.DatetimeIndex(frame["time"])

    if not actual.equals(expected):
        raise ValueError(
            f"{gauge_id}: precipitation time axis does not match "
            f"the requested event window."
        )

    if not np.isfinite(frame["precipitation"]).all():
        raise ValueError(
            f"{gauge_id}: non-finite precipitation detected."
        )

    return frame


def get_basin_metrics(
    master: pd.DataFrame,
    gauge_id: str,
) -> pd.Series:
    rows = master.loc[
        master["gauge_id"] == gauge_id
    ]

    if len(rows) != 1:
        raise ValueError(
            f"{gauge_id}: expected one master row, found {len(rows)}."
        )

    return rows.iloc[0]


def get_event_metrics(
    events: pd.DataFrame,
    gauge_id: str,
    peak_date: pd.Timestamp,
) -> pd.Series:
    rows = events.loc[
        (events["gauge_id"] == gauge_id)
        & (events["event_peak_date"] == peak_date)
    ]

    if len(rows) != 1:
        raise ValueError(
            f"{gauge_id} {peak_date.date()}: expected one event row, "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


def plot_case(
    case: pd.Series,
    basin_metrics: pd.Series,
    event_metrics: pd.Series,
    data_root: Path,
    output_dir: Path,
    pre_days: int,
    post_days: int,
    dpi: int,
) -> dict[str, object]:
    gauge_id = normalize_gauge_id(
        case["gauge_id"]
    )
    fold_id = int(case["fold_id"])
    group = str(case["hydroclimate_group"])
    peak_date = pd.Timestamp(
        case["event_peak_date"]
    )

    frame = load_basin_predictions(
        fold_id=fold_id,
        gauge_id=gauge_id,
    )
    frame["time"] = pd.to_datetime(
        frame["time"]
    )

    event_start = peak_date - pd.Timedelta(
        days=pre_days
    )
    event_end = peak_date + pd.Timedelta(
        days=post_days
    )

    event = frame.loc[
        (frame["time"] >= event_start)
        & (frame["time"] <= event_end)
    ].copy()

    if event.empty:
        raise RuntimeError(
            f"{gauge_id}: empty event window."
        )

    precipitation = load_precipitation(
        data_root=data_root,
        gauge_id=gauge_id,
        start=event_start,
        end=event_end,
    )

    if not event["time"].reset_index(drop=True).equals(
        precipitation["time"].reset_index(drop=True)
    ):
        raise ValueError(
            f"{gauge_id}: event streamflow and precipitation dates differ."
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        gridspec_kw={
            "height_ratios": [1.05, 1.0],
        },
    )

    ax_full, ax_event = axes

    # ------------------------------------------------------------------
    # Panel (a): Full PUB test period.
    # ------------------------------------------------------------------
    ax_full.plot(
        frame["time"],
        frame["obs"],
        linewidth=1.10,
        label="Observed Q",
    )
    ax_full.plot(
        frame["time"],
        frame["STL"],
        linewidth=0.90,
        label="STL-Q",
    )
    ax_full.plot(
        frame["time"],
        frame["Hard"],
        linewidth=0.90,
        label="Hard-MTL",
    )
    ax_full.plot(
        frame["time"],
        frame["CGC"],
        linewidth=1.00,
        label="CGC",
    )

    ax_full.axvline(
        peak_date,
        linewidth=0.8,
        linestyle="--",
    )

    ax_full.set_ylabel(
        r"Streamflow ($\mathrm{m^3\,s^{-1}}$)"
    )

    ax_full.set_title(
        f"(a) Full PUB test period | Basin {gauge_id} ({group})\n"
        f"NSE: STL={basin_metrics['PUB_STL_Q_NSE']:.3f}, "
        f"Hard={basin_metrics['PUB_Hard_MTL_Q_NSE']:.3f}, "
        f"CGC={basin_metrics['PUB_CGC_Q_NSE']:.3f}"
    )

    ax_full.legend(
        frameon=False,
        ncol=4,
        loc="upper right",
    )

    # ------------------------------------------------------------------
    # Panel (b): Event hydrograph.
    # ------------------------------------------------------------------
    ax_event.plot(
        event["time"],
        event["obs"],
        linewidth=1.60,
        label="Observed Q",
        zorder=5,
    )
    ax_event.plot(
        event["time"],
        event["STL"],
        linewidth=1.30,
        label="STL-Q",
        zorder=5,
    )
    ax_event.plot(
        event["time"],
        event["Hard"],
        linewidth=1.30,
        label="Hard-MTL",
        zorder=5,
    )
    ax_event.plot(
        event["time"],
        event["CGC"],
        linewidth=1.45,
        label="CGC",
        zorder=5,
    )

    ax_event.axvline(
        peak_date,
        linewidth=0.8,
        linestyle="--",
        zorder=4,
    )

    ax_event.set_xlabel("Date")
    ax_event.set_ylabel(
        r"Streamflow ($\mathrm{m^3\,s^{-1}}$)"
    )

    ax_event.set_title(
        f"(b) Selected high-flow event: {peak_date.date()}\n"
        f"Event NSE: STL={event_metrics['STL_nse']:.3f}, "
        f"Hard={event_metrics['Hard_nse']:.3f}, "
        f"CGC={event_metrics['CGC_nse']:.3f}"
    )

    # ------------------------------------------------------------------
    # Event precipitation on an inverted secondary y-axis.
    # ------------------------------------------------------------------
    ax_precip = ax_event.twinx()

    ax_precip.bar(
        precipitation["time"],
        precipitation["precipitation"],
        width=0.8,
        alpha=0.25,
        label="Precipitation",
        zorder=1,
    )

    precip_max = float(
        precipitation["precipitation"].max()
    )

    ax_precip.set_ylim(
        max(precip_max * 3.2, 1.0),
        0.0,
    )

    ax_precip.set_ylabel("Precipitation")

    ax_precip.grid(False)

    # Keep precipitation behind streamflow curves.
    ax_event.set_zorder(
        ax_precip.get_zorder() + 1
    )
    ax_event.patch.set_visible(False)

    # Event annotations.
    annotation = (
        "Peak error: "
        f"STL={100 * event_metrics['STL_peak_rel_error']:+.1f}%, "
        f"Hard={100 * event_metrics['Hard_peak_rel_error']:+.1f}%, "
        f"CGC={100 * event_metrics['CGC_peak_rel_error']:+.1f}%\n"
        "Peak timing: "
        f"STL={event_metrics['STL_peak_timing_error_days']:+.0f} d, "
        f"Hard={event_metrics['Hard_peak_timing_error_days']:+.0f} d, "
        f"CGC={event_metrics['CGC_peak_timing_error_days']:+.0f} d"
    )

    ax_event.text(
        0.01,
        0.97,
        annotation,
        transform=ax_event.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        zorder=10,
    )

    ax_event.xaxis.set_major_locator(
        mdates.DayLocator(interval=2)
    )
    ax_event.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )

    for label in ax_event.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")

    fig.tight_layout()

    stem = (
        f"case{int(case['case_order']):02d}_"
        f"{gauge_id}_{group.lower()}"
    )

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    LOGGER.info(
        "Saved %s",
        png_path,
    )

    return {
        "case_order":
            int(case["case_order"]),
        "gauge_id":
            gauge_id,
        "fold_id":
            fold_id,
        "hydroclimate_group":
            group,
        "case_type":
            case["case_type"],
        "event_peak_date":
            peak_date.date(),
        "event_precipitation_total":
            float(
                precipitation[
                    "precipitation"
                ].sum()
            ),
        "event_precipitation_max":
            precip_max,
        "png_path":
            str(png_path),
        "pdf_path":
            str(pdf_path),
        "PUB_STL_Q_NSE":
            basin_metrics["PUB_STL_Q_NSE"],
        "PUB_Hard_MTL_Q_NSE":
            basin_metrics["PUB_Hard_MTL_Q_NSE"],
        "PUB_CGC_Q_NSE":
            basin_metrics["PUB_CGC_Q_NSE"],
        "STL_event_NSE":
            event_metrics["STL_nse"],
        "Hard_event_NSE":
            event_metrics["Hard_nse"],
        "CGC_event_NSE":
            event_metrics["CGC_nse"],
    }


def main() -> None:
    setup_logging()
    args = parse_args()

    cases_path = resolve_path(
        args.cases
    )
    master_path = resolve_path(
        args.master
    )
    events_path = resolve_path(
        args.events
    )
    data_root = resolve_path(
        args.data_root
    )
    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases, master, events = load_inputs(
        cases_path,
        master_path,
        events_path,
    )

    cases = cases.sort_values(
        "case_order"
    ).reset_index(drop=True)

    records = []

    for _, case in cases.iterrows():
        gauge_id = normalize_gauge_id(
            case["gauge_id"]
        )

        LOGGER.info(
            "Plotting basin %s.",
            gauge_id,
        )

        basin_metrics = get_basin_metrics(
            master,
            gauge_id,
        )
        event_metrics = get_event_metrics(
            events,
            gauge_id,
            pd.Timestamp(
                case["event_peak_date"]
            ),
        )

        records.append(
            plot_case(
                case=case,
                basin_metrics=basin_metrics,
                event_metrics=event_metrics,
                data_root=data_root,
                output_dir=output_dir,
                pre_days=args.pre_days,
                post_days=args.post_days,
                dpi=args.dpi,
            )
        )

    manifest = pd.DataFrame(records)

    manifest_path = (
        output_dir
        / "main_case_figure_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    LOGGER.info(
        "Saved figure manifest: %s",
        manifest_path,
    )

    LOGGER.info(
        "Completed %d main hydrograph cases.",
        len(manifest),
    )


if __name__ == "__main__":
    main()
