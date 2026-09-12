#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rank representative high-flow events for Chapter 4B PUB hydrograph analysis.

The script evaluates previously selected basin candidates using formal PUB
daily predictions. Candidate high-flow events are identified from observed
streamflow using basin-specific Q95 local peaks.

For each event, the script reports:
- observed and simulated peak magnitude;
- peak timing error;
- event volume error;
- event RMSE and NSE;
- CGC improvement relative to STL and Hard-MTL.

The resulting tables are intended for representative-event screening rather
than for basin-wide flood-frequency analysis.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SHORTLIST = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydrograph_candidates"
    / "pub_hydrograph_event_shortlist.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/hydrograph_candidates"
    / "event_screening"
)

RUN_ROOT = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/runs"
)

SCENARIOS = {
    "STL": "stl_q",
    "Hard": "hps_target_ssm",
    "CGC": "cgc_target_ssm",
}

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank PUB high-flow events for hydrograph screening."
    )
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=DEFAULT_SHORTLIST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=0.95,
        help="Observed-flow quantile used as peak threshold.",
    )
    parser.add_argument(
        "--min-peak-distance",
        type=int,
        default=15,
        help="Minimum separation between candidate peaks in days.",
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
        "--events-per-basin",
        type=int,
        default=5,
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def normalize_gauge_id(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text.zfill(8)


def decode_basin_values(values: np.ndarray) -> list[str]:
    result = []

    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        result.append(
            normalize_gauge_id(value)
        )

    return result


def run_path(
    fold_id: int,
    scenario: str,
) -> Path:
    return (
        RUN_ROOT
        / (
            f"ch4b_pub_formal_f{fold_id:02d}_"
            f"{scenario}_seed42"
        )
        / "test_predictions_and_weights.nc"
    )


def load_one_model(
    fold_id: int,
    scenario: str,
    gauge_id: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    path = run_path(
        fold_id,
        scenario,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        required = {
            "streamflow_obs",
            "streamflow_sim",
        }
        missing = required.difference(ds.data_vars)

        if missing:
            raise ValueError(
                f"{path}: missing variables {sorted(missing)}"
            )

        if "basin" not in ds.coords:
            raise ValueError(
                f"{path}: missing basin coordinate."
            )

        basin_ids = decode_basin_values(
            ds["basin"].values
        )

        if gauge_id not in basin_ids:
            raise ValueError(
                f"{gauge_id} not found in {path}."
            )

        basin_index = basin_ids.index(gauge_id)

        obs = np.asarray(
            ds["streamflow_obs"]
            .isel(basin=basin_index)
            .values,
            dtype=float,
        )

        sim = np.asarray(
            ds["streamflow_sim"]
            .isel(basin=basin_index)
            .values,
            dtype=float,
        )

        time = pd.to_datetime(
            ds["time"].values
        )

    return time, obs, sim


def load_basin_predictions(
    fold_id: int,
    gauge_id: str,
) -> pd.DataFrame:
    frames = {}
    reference_time = None
    reference_obs = None

    for label, scenario in SCENARIOS.items():
        time, obs, sim = load_one_model(
            fold_id=fold_id,
            scenario=scenario,
            gauge_id=gauge_id,
        )

        if reference_time is None:
            reference_time = time
            reference_obs = obs
        else:
            if not np.array_equal(
                reference_time.values,
                time.values,
            ):
                raise ValueError(
                    f"{gauge_id}: inconsistent time axes."
                )

            valid = (
                np.isfinite(reference_obs)
                & np.isfinite(obs)
            )

            if valid.any() and not np.allclose(
                reference_obs[valid],
                obs[valid],
                rtol=1e-7,
                atol=1e-10,
            ):
                raise ValueError(
                    f"{gauge_id}: inconsistent observations "
                    f"across scenarios."
                )

        frames[label] = sim

    if reference_time is None or reference_obs is None:
        raise RuntimeError(
            f"No prediction data loaded for {gauge_id}."
        )

    return pd.DataFrame(
        {
            "time": reference_time,
            "obs": reference_obs,
            "STL": frames["STL"],
            "Hard": frames["Hard"],
            "CGC": frames["CGC"],
        }
    )


def nse(
    obs: np.ndarray,
    sim: np.ndarray,
) -> float:
    valid = (
        np.isfinite(obs)
        & np.isfinite(sim)
    )

    if valid.sum() < 3:
        return np.nan

    obs_valid = obs[valid]
    sim_valid = sim[valid]

    denominator = np.sum(
        (obs_valid - np.mean(obs_valid)) ** 2
    )

    if denominator <= 0:
        return np.nan

    return float(
        1.0
        - np.sum(
            (sim_valid - obs_valid) ** 2
        )
        / denominator
    )


def rmse(
    obs: np.ndarray,
    sim: np.ndarray,
) -> float:
    valid = (
        np.isfinite(obs)
        & np.isfinite(sim)
    )

    if not valid.any():
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                (sim[valid] - obs[valid]) ** 2
            )
        )
    )


def relative_error(
    simulated: float,
    observed: float,
) -> float:
    if (
        not np.isfinite(simulated)
        or not np.isfinite(observed)
        or observed == 0
    ):
        return np.nan

    return float(
        (simulated - observed)
        / observed
    )


def identify_peaks(
    frame: pd.DataFrame,
    quantile: float,
    min_distance: int,
    pre_days: int,
    post_days: int,
) -> tuple[np.ndarray, float]:
    obs = pd.to_numeric(
        frame["obs"],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = np.isfinite(obs)

    if finite.sum() < 20:
        return np.array([], dtype=int), np.nan

    threshold = float(
        np.quantile(
            obs[finite],
            quantile,
        )
    )

    signal = obs.copy()
    signal[~finite] = -np.inf

    peaks, _ = find_peaks(
        signal,
        height=threshold,
        distance=min_distance,
    )

    valid_peaks = [
        index
        for index in peaks
        if (
            index - pre_days >= 0
            and index + post_days < len(frame)
        )
    ]

    return (
        np.asarray(
            valid_peaks,
            dtype=int,
        ),
        threshold,
    )


def summarize_model_event(
    event: pd.DataFrame,
    model: str,
    observed_peak_index: int,
) -> dict[str, float]:
    obs = event["obs"].to_numpy(
        dtype=float,
    )
    sim = event[model].to_numpy(
        dtype=float,
    )

    valid_sim = np.isfinite(sim)

    if not valid_sim.any():
        return {
            f"{model}_peak": np.nan,
            f"{model}_peak_rel_error": np.nan,
            f"{model}_peak_timing_error_days": np.nan,
            f"{model}_volume_rel_error": np.nan,
            f"{model}_rmse": np.nan,
            f"{model}_nse": np.nan,
        }

    observed_peak = float(
        obs[observed_peak_index]
    )

    simulated_peak_index = int(
        np.nanargmax(sim)
    )
    simulated_peak = float(
        sim[simulated_peak_index]
    )

    obs_volume = float(
        np.nansum(obs)
    )
    sim_volume = float(
        np.nansum(sim)
    )

    return {
        f"{model}_peak": simulated_peak,
        f"{model}_peak_rel_error":
            relative_error(
                simulated_peak,
                observed_peak,
            ),
        f"{model}_peak_timing_error_days":
            float(
                simulated_peak_index
                - observed_peak_index
            ),
        f"{model}_volume_rel_error":
            relative_error(
                sim_volume,
                obs_volume,
            ),
        f"{model}_rmse":
            rmse(obs, sim),
        f"{model}_nse":
            nse(obs, sim),
    }


def summarize_event(
    frame: pd.DataFrame,
    peak_index: int,
    threshold: float,
    pre_days: int,
    post_days: int,
) -> dict[str, object]:
    start = peak_index - pre_days
    end = peak_index + post_days

    event = (
        frame.iloc[start:end + 1]
        .reset_index(drop=True)
    )

    local_peak_index = pre_days
    observed_peak = float(
        event.loc[
            local_peak_index,
            "obs",
        ]
    )

    record: dict[str, object] = {
        "event_start": event["time"].iloc[0],
        "event_peak_date":
            event["time"].iloc[local_peak_index],
        "event_end": event["time"].iloc[-1],
        "observed_peak": observed_peak,
        "q95_threshold": threshold,
        "peak_to_q95_ratio":
            observed_peak / threshold
            if threshold > 0
            else np.nan,
        "observed_event_volume":
            float(
                np.nansum(
                    event["obs"].to_numpy(
                        dtype=float,
                    )
                )
            ),
    }

    for model in SCENARIOS:
        record.update(
            summarize_model_event(
                event,
                model=model,
                observed_peak_index=local_peak_index,
            )
        )

    record[
        "rmse_gain_cgc_vs_stl"
    ] = (
        record["STL_rmse"]
        - record["CGC_rmse"]
    )

    record[
        "rmse_gain_cgc_vs_hard"
    ] = (
        record["Hard_rmse"]
        - record["CGC_rmse"]
    )

    record[
        "abs_peak_error_gain_cgc_vs_stl"
    ] = (
        abs(record["STL_peak_rel_error"])
        - abs(record["CGC_peak_rel_error"])
    )

    record[
        "abs_peak_error_gain_cgc_vs_hard"
    ] = (
        abs(record["Hard_peak_rel_error"])
        - abs(record["CGC_peak_rel_error"])
    )

    record[
        "abs_volume_error_gain_cgc_vs_stl"
    ] = (
        abs(record["STL_volume_rel_error"])
        - abs(record["CGC_volume_rel_error"])
    )

    record[
        "abs_volume_error_gain_cgc_vs_hard"
    ] = (
        abs(record["Hard_volume_rel_error"])
        - abs(record["CGC_volume_rel_error"])
    )

    record[
        "hard_event_worse_than_stl"
    ] = (
        record["Hard_rmse"]
        > record["STL_rmse"]
    )

    record[
        "cgc_event_better_than_stl"
    ] = (
        record["CGC_rmse"]
        < record["STL_rmse"]
    )

    record[
        "cgc_event_better_than_hard"
    ] = (
        record["CGC_rmse"]
        < record["Hard_rmse"]
    )

    record[
        "event_recovery_pattern"
    ] = (
        record["hard_event_worse_than_stl"]
        and record["cgc_event_better_than_stl"]
        and record["cgc_event_better_than_hard"]
    )

    record[
        "event_consistent_improvement"
    ] = (
        record["CGC_rmse"]
        < record["Hard_rmse"]
        < record["STL_rmse"]
    )

    return record


def main() -> None:
    setup_logging()
    args = parse_args()

    if not 0 < args.q_threshold < 1:
        raise ValueError(
            "--q-threshold must be between 0 and 1."
        )

    if args.events_per_basin < 1:
        raise ValueError(
            "--events-per-basin must be >= 1."
        )

    shortlist_path = (
        args.shortlist
        if args.shortlist.is_absolute()
        else PROJECT_ROOT / args.shortlist
    )

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shortlist = pd.read_csv(
        shortlist_path,
        dtype={"gauge_id": str},
    )
    shortlist["gauge_id"] = (
        shortlist["gauge_id"]
        .map(normalize_gauge_id)
    )

    required = {
        "gauge_id",
        "fold_id",
        "hydroclimate_group",
        "candidate_type",
        "priority",
    }
    missing = required.difference(
        shortlist.columns
    )

    if missing:
        raise ValueError(
            f"Missing shortlist columns: {sorted(missing)}"
        )

    records = []

    for row in shortlist.itertuples(
        index=False
    ):
        gauge_id = normalize_gauge_id(
            row.gauge_id
        )
        fold_id = int(row.fold_id)

        LOGGER.info(
            "Processing %s (fold %02d).",
            gauge_id,
            fold_id,
        )

        frame = load_basin_predictions(
            fold_id=fold_id,
            gauge_id=gauge_id,
        )

        peaks, threshold = identify_peaks(
            frame=frame,
            quantile=args.q_threshold,
            min_distance=args.min_peak_distance,
            pre_days=args.pre_days,
            post_days=args.post_days,
        )

        if len(peaks) == 0:
            LOGGER.warning(
                "%s: no eligible high-flow peaks.",
                gauge_id,
            )
            continue

        peaks = sorted(
            peaks,
            key=lambda index: frame.loc[
                index,
                "obs",
            ],
            reverse=True,
        )[:args.events_per_basin]

        for rank, peak_index in enumerate(
            peaks,
            start=1,
        ):
            record = summarize_event(
                frame=frame,
                peak_index=peak_index,
                threshold=threshold,
                pre_days=args.pre_days,
                post_days=args.post_days,
            )

            record.update({
                "gauge_id": gauge_id,
                "fold_id": fold_id,
                "hydroclimate_group":
                    row.hydroclimate_group,
                "candidate_type":
                    row.candidate_type,
                "basin_priority":
                    int(row.priority),
                "event_rank_by_peak":
                    rank,
            })

            records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        raise RuntimeError(
            "No flood-event candidates were identified."
        )

    result = result.sort_values(
        [
            "basin_priority",
            "gauge_id",
            "event_rank_by_peak",
        ]
    ).reset_index(drop=True)

    all_path = (
        output_dir
        / "pub_highflow_event_metrics.csv"
    )
    result.to_csv(
        all_path,
        index=False,
    )

    LOGGER.info(
        "Saved event metrics: %s",
        all_path,
    )

    summary_columns = [
        "gauge_id",
        "hydroclimate_group",
        "candidate_type",
        "basin_priority",
        "event_rank_by_peak",
        "event_start",
        "event_peak_date",
        "event_end",
        "observed_peak",
        "peak_to_q95_ratio",
        "STL_rmse",
        "Hard_rmse",
        "CGC_rmse",
        "STL_nse",
        "Hard_nse",
        "CGC_nse",
        "STL_peak_rel_error",
        "Hard_peak_rel_error",
        "CGC_peak_rel_error",
        "STL_peak_timing_error_days",
        "Hard_peak_timing_error_days",
        "CGC_peak_timing_error_days",
        "rmse_gain_cgc_vs_stl",
        "rmse_gain_cgc_vs_hard",
        "event_recovery_pattern",
        "event_consistent_improvement",
    ]

    summary = result[
        summary_columns
    ].copy()

    summary_path = (
        output_dir
        / "pub_highflow_event_screening_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    LOGGER.info(
        "Saved screening summary: %s",
        summary_path,
    )

    LOGGER.info(
        "Completed %d events across %d basins.",
        len(result),
        result["gauge_id"].nunique(),
    )


if __name__ == "__main__":
    main()
