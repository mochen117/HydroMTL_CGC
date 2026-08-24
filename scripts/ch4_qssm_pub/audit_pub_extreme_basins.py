#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chapter 4B PUB extreme-basin audit.

This script validates extreme negative NSE values and strong negative-transfer
cases in the Chapter 4B PUB experiment. It works entirely from existing
summary tables and ensemble prediction NetCDF files; no model retraining is
required.

Scenarios
---------
1. STL-Q
2. Hard-MTL-PUB with target-basin SSM supervision
3. CGC-PUB with target-basin SSM supervision

Main objectives
---------------
1. Recompute basin-wise NSE from daily observations and predictions.
2. Verify identical target observations/time coordinates across scenarios.
3. Diagnose whether extreme negative NSE is associated with small observed
   variance, near-zero flow, or genuinely large prediction errors.
4. Export the strongest negative-transfer basins for manual inspection.
5. Plot hydrographs for the most extreme CGC failure cases.

Notes
-----
- Basin grouping attributes are intentionally not used.
- Gauge IDs are normalized to 8-digit CAMELS-style strings by default.
- Existing experiment outputs are read only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_SUMMARY = Path(
    "experiments/ch4_qssm_pub/summary/ch4b_pub_effects_with_ch3_metadata.csv"
)
DEFAULT_ENSEMBLE_DIR = Path("experiments/ch4_qssm_pub/ensemble")
DEFAULT_OUTPUT_DIR = Path("experiments/ch4_qssm_pub/extreme_audit")

SCENARIO_LABELS: Mapping[str, str] = {
    "stl_q": "STL-Q",
    "hps_target_ssm": "Hard-MTL-PUB",
    "cgc_target_ssm": "CGC-PUB",
}

SUMMARY_NSE_COLUMNS: Mapping[str, str] = {
    "stl_q": "stl_q",
    "hps_target_ssm": "hps_target_ssm",
    "cgc_target_ssm": "cgc_target_ssm",
}

OBS_VAR_CANDIDATES: Sequence[str] = (
    "q_obs", "obs_q", "streamflow_obs", "observed_streamflow",
    "q_observed", "q_true", "streamflow_true", "y_true_q",
    "target_q", "obs", "observation", "y_true",
)

PRED_VAR_CANDIDATES: Sequence[str] = (
    "q_pred", "pred_q", "streamflow_pred", "predicted_streamflow",
    "q_sim", "streamflow_sim", "sim_q", "q_prediction",
    "y_pred_q", "pred", "prediction", "y_pred",
)

BASIN_DIM_CANDIDATES: Sequence[str] = (
    "gauge_id", "gage_id", "basin_id", "basin",
    "catchment_id", "catchment",
)

TIME_DIM_CANDIDATES: Sequence[str] = (
    "time", "date", "datetime", "day",
)


@dataclass
class BasinSeries:
    """Observation and prediction series for one basin/scenario."""

    gauge_id: str
    fold_id: int
    scenario: str
    time: np.ndarray
    obs: np.ndarray
    pred: np.ndarray
    source_nc: str


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a path relative to the project root."""
    return path if path.is_absolute() else project_root / path


def normalize_gauge_id(value: object, width: int = 8) -> str:
    """Normalize CAMELS gauge IDs to fixed-width strings."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            text = str(int(float(text)))
        except ValueError:
            pass
    try:
        return f"{int(text):0{width}d}"
    except ValueError:
        return text


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def find_matching_name(
    names: Iterable[str],
    preferred: Optional[str],
    candidates: Sequence[str],
    kind: str,
) -> str:
    """Find an exact or case-insensitive matching variable name."""
    names = list(names)

    if preferred is not None:
        if preferred not in names:
            raise KeyError(
                f"Requested {kind} '{preferred}' not found.\n"
                f"Available names: {names}"
            )
        return preferred

    lower_map = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    role_tokens = (
        ("obs", "observed", "true")
        if kind == "observation variable"
        else ("pred", "prediction", "sim", "simulated")
    )
    q_tokens = ("q", "streamflow", "discharge", "runoff")
    matches = [
        name
        for name in names
        if any(token in name.lower() for token in role_tokens)
        and any(token in name.lower() for token in q_tokens)
    ]

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Unable to auto-detect {kind}.\n"
        f"Available names: {names}\n"
        "Provide the corresponding command-line option explicitly."
    )


def select_q_from_extra_dims(da: xr.DataArray) -> xr.DataArray:
    """Reduce optional task/output dimensions to streamflow where possible."""
    result = da

    for dim in list(result.dims):
        if result.sizes[dim] == 1:
            result = result.isel({dim: 0}, drop=True)

    for dim in list(result.dims):
        if dim.lower() not in {
            "task", "target", "variable", "output", "feature", "channel",
        }:
            continue
        if dim not in result.coords:
            continue

        labels = [str(value).lower() for value in result.coords[dim].values.tolist()]
        q_indices = [
            index
            for index, label in enumerate(labels)
            if label in {"q", "streamflow", "discharge", "runoff"}
            or "streamflow" in label
            or "discharge" in label
            or "runoff" in label
        ]
        if len(q_indices) == 1:
            result = result.isel({dim: q_indices[0]}, drop=True)

    return result


def infer_dimension(
    da: xr.DataArray,
    preferred: Optional[str],
    candidates: Sequence[str],
    kind: str,
) -> str:
    """Infer basin or time dimension from a DataArray."""
    dims = list(da.dims)

    if preferred is not None:
        if preferred not in dims:
            raise KeyError(
                f"Requested {kind} '{preferred}' not found in dimensions: {dims}"
            )
        return preferred

    lower_map = {dim.lower(): dim for dim in dims}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    tokens = (
        ("basin", "gauge", "gage", "catchment")
        if kind == "basin dimension"
        else ("time", "date", "day")
    )
    matches = [
        dim for dim in dims if any(token in dim.lower() for token in tokens)
    ]
    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Unable to infer {kind} from dimensions {dims}. "
        "Provide the corresponding command-line option explicitly."
    )


def align_obs_pred(
    ds: xr.Dataset,
    obs_var: Optional[str],
    pred_var: Optional[str],
    basin_dim: Optional[str],
    time_dim: Optional[str],
) -> Tuple[xr.DataArray, xr.DataArray, str, str]:
    """Detect, align, and order observed/predicted streamflow arrays."""
    obs_name = find_matching_name(
        ds.data_vars, obs_var, OBS_VAR_CANDIDATES, "observation variable"
    )
    pred_name = find_matching_name(
        ds.data_vars, pred_var, PRED_VAR_CANDIDATES, "prediction variable"
    )

    obs = select_q_from_extra_dims(ds[obs_name])
    pred = select_q_from_extra_dims(ds[pred_name])

    basin_name = infer_dimension(
        obs, basin_dim, BASIN_DIM_CANDIDATES, "basin dimension"
    )
    time_name = infer_dimension(
        obs, time_dim, TIME_DIM_CANDIDATES, "time dimension"
    )

    if basin_name not in pred.dims or time_name not in pred.dims:
        raise ValueError(
            "Observation and prediction arrays do not share the inferred "
            f"basin/time dimensions.\nobs dims={obs.dims}, pred dims={pred.dims}"
        )

    allowed_dims = {basin_name, time_name}
    obs_extra = [dim for dim in obs.dims if dim not in allowed_dims]
    pred_extra = [dim for dim in pred.dims if dim not in allowed_dims]
    if obs_extra or pred_extra:
        raise ValueError(
            "Unresolved extra dimensions remain after streamflow selection.\n"
            f"obs extra dims={obs_extra}, pred extra dims={pred_extra}"
        )

    obs, pred = xr.align(obs, pred, join="inner")
    return (
        obs.transpose(basin_name, time_name),
        pred.transpose(basin_name, time_name),
        basin_name,
        time_name,
    )


def locate_prediction_file(scenario_dir: Path, nc_pattern: str) -> Path:
    """Locate exactly one prediction NetCDF file for one fold/scenario."""
    candidates = sorted(scenario_dir.glob(nc_pattern))
    if not candidates:
        candidates = sorted(scenario_dir.rglob(nc_pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No NetCDF files matching '{nc_pattern}' under {scenario_dir}"
        )

    prediction_like = [
        path
        for path in candidates
        if any(token in path.name.lower() for token in ("pred", "prediction"))
    ]
    if len(prediction_like) == 1:
        return prediction_like[0]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Multiple NetCDF files found under {scenario_dir}:\n"
        + "\n".join(str(path) for path in candidates)
        + "\nUse a more specific --nc-pattern."
    )


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency using pairwise finite samples."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = np.asarray(obs[mask], dtype=float)
    sim = np.asarray(sim[mask], dtype=float)

    if obs.size < 2:
        return np.nan

    denominator = float(np.sum((obs - np.mean(obs)) ** 2))
    if np.isclose(denominator, 0.0):
        return np.nan

    numerator = float(np.sum((obs - sim) ** 2))
    return 1.0 - numerator / denominator


def basic_error_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Return MAE, RMSE, and bias using pairwise finite samples."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = np.asarray(obs[mask], dtype=float)
    sim = np.asarray(sim[mask], dtype=float)

    if obs.size == 0:
        return {"n_pair": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}

    error = sim - obs
    return {
        "n_pair": int(obs.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }


def observed_flow_diagnostics(
    obs: np.ndarray,
    zero_flow_threshold: float,
) -> Dict[str, float]:
    """Summarize observed-flow variability and the NSE denominator."""
    x = np.asarray(obs, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "n_obs": 0, "obs_mean": np.nan, "obs_std": np.nan,
            "obs_var": np.nan, "obs_cv": np.nan, "obs_min": np.nan,
            "obs_p05": np.nan, "obs_median": np.nan, "obs_p95": np.nan,
            "obs_max": np.nan, "zero_flow_rate": np.nan,
            "nse_denominator": np.nan, "nse_denominator_per_step": np.nan,
        }

    mean_value = float(np.mean(x))
    std_value = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    var_value = float(np.var(x, ddof=1)) if x.size > 1 else np.nan
    denominator = float(np.sum((x - mean_value) ** 2))
    cv_value = (
        np.nan
        if np.isclose(mean_value, 0.0)
        else std_value / abs(mean_value)
    )

    return {
        "n_obs": int(x.size),
        "obs_mean": mean_value,
        "obs_std": std_value,
        "obs_var": var_value,
        "obs_cv": float(cv_value) if np.isfinite(cv_value) else np.nan,
        "obs_min": float(np.min(x)),
        "obs_p05": float(np.quantile(x, 0.05)),
        "obs_median": float(np.median(x)),
        "obs_p95": float(np.quantile(x, 0.95)),
        "obs_max": float(np.max(x)),
        "zero_flow_rate": float((np.abs(x) <= zero_flow_threshold).mean()),
        "nse_denominator": denominator,
        "nse_denominator_per_step": denominator / x.size,
    }


def extract_all_series(
    ensemble_dir: Path,
    nc_pattern: str,
    obs_var: Optional[str],
    pred_var: Optional[str],
    basin_dim: Optional[str],
    time_dim: Optional[str],
    gauge_width: int,
) -> Dict[Tuple[str, str], BasinSeries]:
    """Read all fold/scenario ensemble prediction files."""
    records: Dict[Tuple[str, str], BasinSeries] = {}

    fold_dirs = sorted(path for path in ensemble_dir.glob("fold*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {ensemble_dir}")

    for fold_dir in fold_dirs:
        digits = "".join(ch for ch in fold_dir.name if ch.isdigit())
        if not digits:
            raise ValueError(f"Unable to parse fold id from {fold_dir.name}")
        fold_id = int(digits)

        for scenario in SCENARIO_LABELS:
            scenario_dir = fold_dir / scenario
            if not scenario_dir.exists():
                raise FileNotFoundError(f"Missing scenario directory: {scenario_dir}")

            nc_path = locate_prediction_file(scenario_dir, nc_pattern)

            with xr.open_dataset(nc_path) as ds:
                obs, pred, basin_name, time_name = align_obs_pred(
                    ds,
                    obs_var=obs_var,
                    pred_var=pred_var,
                    basin_dim=basin_dim,
                    time_dim=time_dim,
                )

                basin_values = (
                    obs.coords[basin_name].values
                    if basin_name in obs.coords
                    else np.arange(obs.sizes[basin_name])
                )
                time_values = (
                    np.asarray(obs.coords[time_name].values)
                    if time_name in obs.coords
                    else np.arange(obs.sizes[time_name])
                )

                for index in range(obs.sizes[basin_name]):
                    gauge_id = normalize_gauge_id(
                        basin_values[index],
                        width=gauge_width,
                    )
                    key = (gauge_id, scenario)
                    if key in records:
                        raise ValueError(
                            f"Duplicate basin/scenario combination detected: {key}"
                        )

                    records[key] = BasinSeries(
                        gauge_id=gauge_id,
                        fold_id=fold_id,
                        scenario=scenario,
                        time=np.asarray(time_values),
                        obs=np.asarray(
                            obs.isel({basin_name: index}).values,
                            dtype=float,
                        ),
                        pred=np.asarray(
                            pred.isel({basin_name: index}).values,
                            dtype=float,
                        ),
                        source_nc=str(nc_path),
                    )

            print(
                f"Loaded fold={fold_id:02d} "
                f"scenario={scenario:<15s} file={nc_path.name}"
            )

    return records


def compare_observations(
    series_by_scenario: Mapping[str, BasinSeries],
    obs_atol: float,
    obs_rtol: float,
) -> Dict[str, object]:
    """Check whether all scenarios use identical target observations."""
    reference = series_by_scenario["stl_q"]
    row: Dict[str, object] = {
        "gauge_id": reference.gauge_id,
        "fold_id": reference.fold_id,
        "all_time_equal": True,
        "all_obs_equal": True,
        "fold_id_equal": True,
        "max_abs_obs_diff": 0.0,
    }

    for scenario in ("hps_target_ssm", "cgc_target_ssm"):
        current = series_by_scenario[scenario]
        fold_equal = reference.fold_id == current.fold_id
        time_equal = (
            reference.time.shape == current.time.shape
            and np.array_equal(reference.time, current.time)
        )

        if reference.obs.shape == current.obs.shape:
            finite_pair = np.isfinite(reference.obs) & np.isfinite(current.obs)
            max_abs_diff = (
                float(
                    np.max(
                        np.abs(
                            reference.obs[finite_pair]
                            - current.obs[finite_pair]
                        )
                    )
                )
                if finite_pair.any()
                else np.nan
            )
            obs_equal = bool(
                np.allclose(
                    reference.obs,
                    current.obs,
                    rtol=obs_rtol,
                    atol=obs_atol,
                    equal_nan=True,
                )
            )
        else:
            max_abs_diff = np.inf
            obs_equal = False

        row[f"time_equal__{scenario}"] = time_equal
        row[f"obs_equal__{scenario}"] = obs_equal
        row[f"fold_equal__{scenario}"] = fold_equal
        row[f"max_abs_obs_diff__{scenario}"] = max_abs_diff
        row["all_time_equal"] = bool(row["all_time_equal"] and time_equal)
        row["all_obs_equal"] = bool(row["all_obs_equal"] and obs_equal)
        row["fold_id_equal"] = bool(row["fold_id_equal"] and fold_equal)

        if np.isfinite(max_abs_diff):
            row["max_abs_obs_diff"] = max(
                float(row["max_abs_obs_diff"]),
                max_abs_diff,
            )
        else:
            row["max_abs_obs_diff"] = np.inf

    return row


def build_audit_tables(
    summary: pd.DataFrame,
    series: Mapping[Tuple[str, str], BasinSeries],
    gauge_width: int,
    zero_flow_threshold: float,
    nse_tolerance: float,
    obs_atol: float,
    obs_rtol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build full basin audit and consistency tables."""
    summary = summary.copy()
    summary["gauge_id"] = summary["gauge_id"].map(
        lambda value: normalize_gauge_id(value, gauge_width)
    )

    if summary["gauge_id"].duplicated().any():
        duplicates = summary.loc[
            summary["gauge_id"].duplicated(keep=False),
            "gauge_id",
        ].tolist()
        raise ValueError(
            "Duplicate gauge IDs detected in summary table. "
            f"Examples: {duplicates[:10]}"
        )

    audit_rows: List[Dict[str, object]] = []
    nse_rows: List[Dict[str, object]] = []
    obs_rows: List[Dict[str, object]] = []
    summary_lookup = summary.set_index("gauge_id", drop=False)

    for gauge_id in summary_lookup.index:
        scenario_series: Dict[str, BasinSeries] = {}
        for scenario in SCENARIO_LABELS:
            key = (gauge_id, scenario)
            if key not in series:
                raise KeyError(
                    f"Missing prediction series for basin={gauge_id}, "
                    f"scenario={scenario}"
                )
            scenario_series[scenario] = series[key]

        obs_check = compare_observations(
            scenario_series,
            obs_atol=obs_atol,
            obs_rtol=obs_rtol,
        )
        obs_rows.append(obs_check)

        summary_row = summary_lookup.loc[gauge_id]
        audit_row: Dict[str, object] = {
            "gauge_id": gauge_id,
            "fold_id": int(summary_row["fold_id"]),
            **observed_flow_diagnostics(
                scenario_series["stl_q"].obs,
                zero_flow_threshold=zero_flow_threshold,
            ),
            "all_time_equal": obs_check["all_time_equal"],
            "all_obs_equal": obs_check["all_obs_equal"],
            "fold_id_equal": obs_check["fold_id_equal"],
            "max_abs_obs_diff": obs_check["max_abs_obs_diff"],
        }

        for scenario, label in SCENARIO_LABELS.items():
            basin_series = scenario_series[scenario]
            recomputed_nse = nse(basin_series.obs, basin_series.pred)
            stored_nse = float(summary_row[SUMMARY_NSE_COLUMNS[scenario]])
            errors = basic_error_metrics(basin_series.obs, basin_series.pred)
            diff = (
                recomputed_nse - stored_nse
                if np.isfinite(recomputed_nse) and np.isfinite(stored_nse)
                else np.nan
            )

            audit_row[f"nse_{scenario}"] = recomputed_nse
            audit_row[f"nse_stored_{scenario}"] = stored_nse
            audit_row[f"nse_abs_diff_{scenario}"] = (
                abs(diff) if np.isfinite(diff) else np.nan
            )
            audit_row[f"mae_{scenario}"] = errors["mae"]
            audit_row[f"rmse_{scenario}"] = errors["rmse"]
            audit_row[f"bias_{scenario}"] = errors["bias"]
            audit_row[f"n_pair_{scenario}"] = errors["n_pair"]

            nse_rows.append(
                {
                    "gauge_id": gauge_id,
                    "fold_id": basin_series.fold_id,
                    "scenario": scenario,
                    "model": label,
                    "stored_nse": stored_nse,
                    "recomputed_nse": recomputed_nse,
                    "difference": diff,
                    "abs_difference": (
                        abs(diff) if np.isfinite(diff) else np.nan
                    ),
                    "within_tolerance": bool(
                        np.isfinite(diff) and abs(diff) <= nse_tolerance
                    ),
                    "n_pair": errors["n_pair"],
                    "source_nc": basin_series.source_nc,
                }
            )

        audit_row["delta_nse_hard_minus_stl"] = (
            audit_row["nse_hps_target_ssm"] - audit_row["nse_stl_q"]
        )
        audit_row["delta_nse_cgc_minus_stl"] = (
            audit_row["nse_cgc_target_ssm"] - audit_row["nse_stl_q"]
        )
        audit_row["delta_nse_cgc_minus_hard"] = (
            audit_row["nse_cgc_target_ssm"]
            - audit_row["nse_hps_target_ssm"]
        )
        audit_rows.append(audit_row)

    return (
        pd.DataFrame.from_records(audit_rows),
        pd.DataFrame.from_records(nse_rows),
        pd.DataFrame.from_records(obs_rows),
    )


def add_extreme_flags(
    audit: pd.DataFrame,
    negative_nse_threshold: float,
    strong_delta_threshold: float,
    severe_delta_threshold: float,
) -> pd.DataFrame:
    """Add explicit extreme/failure flags."""
    out = audit.copy()

    for scenario in SCENARIO_LABELS:
        column = f"nse_{scenario}"
        out[f"flag_{scenario}_nse_lt_0"] = out[column] < 0
        out[f"flag_{scenario}_nse_lt_threshold"] = (
            out[column] < negative_nse_threshold
        )

    for name, column in {
        "hard_minus_stl": "delta_nse_hard_minus_stl",
        "cgc_minus_stl": "delta_nse_cgc_minus_stl",
        "cgc_minus_hard": "delta_nse_cgc_minus_hard",
    }.items():
        out[f"flag_{name}_lt_strong"] = out[column] < strong_delta_threshold
        out[f"flag_{name}_lt_severe"] = out[column] < severe_delta_threshold

    flag_columns = [
        column for column in out.columns if column.startswith("flag_")
    ]
    out["flag_any_extreme"] = out[flag_columns].any(axis=1)
    return out


def export_ranked_tables(
    audit: pd.DataFrame,
    output_dir: Path,
    worst_n: int,
) -> None:
    """Export ranked worst-basin tables for key diagnostics."""
    ranking_specs = {
        "worst_cgc_nse": "nse_cgc_target_ssm",
        "worst_hard_nse": "nse_hps_target_ssm",
        "worst_stl_nse": "nse_stl_q",
        "worst_cgc_minus_stl": "delta_nse_cgc_minus_stl",
        "worst_hard_minus_stl": "delta_nse_hard_minus_stl",
        "worst_cgc_minus_hard": "delta_nse_cgc_minus_hard",
    }
    for stem, column in ranking_specs.items():
        audit.nsmallest(worst_n, column).to_csv(
            output_dir / f"{stem}_{worst_n}_basins.csv",
            index=False,
        )


def select_hydrograph_basins(
    audit: pd.DataFrame,
    plot_n: int,
) -> List[str]:
    """Select representative extreme CGC basins for hydrograph inspection."""
    selected: List[str] = []
    for column in ("nse_cgc_target_ssm", "delta_nse_cgc_minus_stl"):
        for gauge_id in audit.nsmallest(plot_n, column)["gauge_id"].tolist():
            if gauge_id not in selected:
                selected.append(gauge_id)
    return selected[:plot_n]


def plot_hydrograph(
    gauge_id: str,
    audit_row: pd.Series,
    series: Mapping[Tuple[str, str], BasinSeries],
    output_dir: Path,
    max_points: Optional[int],
) -> None:
    """Plot observed Q and all three model predictions for one basin."""
    reference = series[(gauge_id, "stl_q")]
    time = reference.time
    obs = reference.obs

    start = 0
    if max_points is not None and max_points > 0 and len(time) > max_points:
        start = len(time) - max_points
        time = time[start:]
        obs = obs[start:]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(time, obs, linewidth=1.2, label="Observed Q")

    for scenario, label in SCENARIO_LABELS.items():
        pred = series[(gauge_id, scenario)].pred
        if start > 0:
            pred = pred[start:]
        ax.plot(time, pred, linewidth=1.0, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Streamflow")
    ax.set_title(
        f"Basin {gauge_id} | "
        f"NSE: STL={audit_row['nse_stl_q']:.3f}, "
        f"Hard={audit_row['nse_hps_target_ssm']:.3f}, "
        f"CGC={audit_row['nse_cgc_target_ssm']:.3f}"
    )
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()

    fig.savefig(
        output_dir / f"hydrograph_{gauge_id}.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"hydrograph_{gauge_id}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_audit_summary(
    audit: pd.DataFrame,
    nse_check: pd.DataFrame,
    obs_check: pd.DataFrame,
    output_path: Path,
    negative_nse_threshold: float,
    strong_delta_threshold: float,
    severe_delta_threshold: float,
) -> None:
    """Write a compact human-readable audit report."""
    lines: List[str] = [
        "Chapter 4B PUB extreme-basin audit",
        "=" * 72,
        f"Basins audited: {len(audit)}",
        "",
        "Observation/time consistency",
        "-" * 72,
        (
            "Basins with identical time coordinates across scenarios: "
            f"{int(obs_check['all_time_equal'].sum())}/{len(obs_check)}"
        ),
        (
            "Basins with identical observed Q across scenarios: "
            f"{int(obs_check['all_obs_equal'].sum())}/{len(obs_check)}"
        ),
        (
            "Basins with identical fold IDs across scenarios: "
            f"{int(obs_check['fold_id_equal'].sum())}/{len(obs_check)}"
        ),
        "",
        "NSE recomputation",
        "-" * 72,
        (
            "Scenario-basin rows within NSE tolerance: "
            f"{int(nse_check['within_tolerance'].sum())}/{len(nse_check)}"
        ),
        (
            "Maximum absolute NSE recomputation difference: "
            f"{nse_check['abs_difference'].max():.6e}"
        ),
        "",
        "Extreme absolute NSE counts",
        "-" * 72,
    ]

    for scenario, label in SCENARIO_LABELS.items():
        column = f"nse_{scenario}"
        lines.append(
            f"{label}: NSE < 0 = {int((audit[column] < 0).sum())} "
            f"({(audit[column] < 0).mean():.2%}); "
            f"NSE < {negative_nse_threshold:g} = "
            f"{int((audit[column] < negative_nse_threshold).sum())} "
            f"({(audit[column] < negative_nse_threshold).mean():.2%}); "
            f"min NSE = {audit[column].min():.6f}"
        )

    lines.extend(["", "Negative-transfer counts", "-" * 72])
    for name, column in {
        "Hard-MTL minus STL": "delta_nse_hard_minus_stl",
        "CGC minus STL": "delta_nse_cgc_minus_stl",
        "CGC minus Hard-MTL": "delta_nse_cgc_minus_hard",
    }.items():
        lines.append(
            f"{name}: delta < {strong_delta_threshold:g} = "
            f"{int((audit[column] < strong_delta_threshold).sum())} "
            f"({(audit[column] < strong_delta_threshold).mean():.2%}); "
            f"delta < {severe_delta_threshold:g} = "
            f"{int((audit[column] < severe_delta_threshold).sum())} "
            f"({(audit[column] < severe_delta_threshold).mean():.2%}); "
            f"min delta = {audit[column].min():.6f}"
        )

    lines.extend(
        [
            "",
            "Observed-flow variance diagnostics",
            "-" * 72,
            f"Minimum observed-flow variance: {audit['obs_var'].min():.6e}",
            f"Minimum NSE denominator: {audit['nse_denominator'].min():.6e}",
            f"Median NSE denominator: {audit['nse_denominator'].median():.6e}",
            "",
            "Interpretation note",
            "-" * 72,
            (
                "Extreme negative NSE should be interpreted together with the "
                "observed variance/NSE denominator and absolute prediction "
                "errors. A small denominator can amplify NSE negativity even "
                "when RMSE is not extreme."
            ),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit extreme negative NSE and strong negative-transfer basins in "
            "the Chapter 4B PUB experiment."
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nc-pattern", default="*.nc")
    parser.add_argument("--obs-var", default=None)
    parser.add_argument("--pred-var", default=None)
    parser.add_argument("--basin-dim", default=None)
    parser.add_argument("--time-dim", default=None)
    parser.add_argument("--gauge-width", type=int, default=8)
    parser.add_argument("--negative-nse-threshold", type=float, default=-1.0)
    parser.add_argument("--strong-delta-threshold", type=float, default=-0.5)
    parser.add_argument("--severe-delta-threshold", type=float, default=-1.0)
    parser.add_argument("--zero-flow-threshold", type=float, default=1e-8)
    parser.add_argument("--nse-tolerance", type=float, default=1e-6)
    parser.add_argument("--obs-atol", type=float, default=1e-10)
    parser.add_argument("--obs-rtol", type=float, default=1e-10)
    parser.add_argument("--worst-n", type=int, default=20)
    parser.add_argument("--plot-n", type=int, default=10)
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=1500,
        help="Use 0 to plot the full period.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail after writing outputs if core consistency checks do not pass.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    summary_path = resolve_path(project_root, args.summary)
    ensemble_dir = resolve_path(project_root, args.ensemble_dir)
    output_dir = resolve_path(project_root, args.output_dir)
    hydrograph_dir = output_dir / "hydrographs"

    output_dir.mkdir(parents=True, exist_ok=True)
    hydrograph_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

    summary = pd.read_csv(summary_path)
    require_columns(
        summary,
        ["gauge_id", "fold_id", "stl_q", "hps_target_ssm", "cgc_target_ssm"],
    )

    print("=" * 78)
    print("Loading ensemble PUB prediction series")
    print("=" * 78)

    series = extract_all_series(
        ensemble_dir=ensemble_dir,
        nc_pattern=args.nc_pattern,
        obs_var=args.obs_var,
        pred_var=args.pred_var,
        basin_dim=args.basin_dim,
        time_dim=args.time_dim,
        gauge_width=args.gauge_width,
    )

    print("\n" + "=" * 78)
    print("Building basin-wise audit tables")
    print("=" * 78)

    audit, nse_check, obs_check = build_audit_tables(
        summary=summary,
        series=series,
        gauge_width=args.gauge_width,
        zero_flow_threshold=args.zero_flow_threshold,
        nse_tolerance=args.nse_tolerance,
        obs_atol=args.obs_atol,
        obs_rtol=args.obs_rtol,
    )

    audit = add_extreme_flags(
        audit,
        negative_nse_threshold=args.negative_nse_threshold,
        strong_delta_threshold=args.strong_delta_threshold,
        severe_delta_threshold=args.severe_delta_threshold,
    )

    audit.to_csv(output_dir / "all_basin_audit.csv", index=False)
    nse_check.to_csv(output_dir / "nse_recomputation_check.csv", index=False)
    obs_check.to_csv(
        output_dir / "observation_consistency_check.csv",
        index=False,
    )

    extreme = audit.loc[audit["flag_any_extreme"]].copy()
    extreme = extreme.sort_values(
        ["nse_cgc_target_ssm", "delta_nse_cgc_minus_stl"],
        ascending=[True, True],
    )
    extreme.to_csv(output_dir / "extreme_basin_summary.csv", index=False)

    export_ranked_tables(
        audit,
        output_dir=output_dir,
        worst_n=args.worst_n,
    )

    selected_basins = select_hydrograph_basins(
        audit,
        plot_n=args.plot_n,
    )
    audit_lookup = audit.set_index("gauge_id")

    for gauge_id in selected_basins:
        plot_hydrograph(
            gauge_id=gauge_id,
            audit_row=audit_lookup.loc[gauge_id],
            series=series,
            output_dir=hydrograph_dir,
            max_points=(
                None if args.plot_max_points == 0 else args.plot_max_points
            ),
        )

    write_audit_summary(
        audit=audit,
        nse_check=nse_check,
        obs_check=obs_check,
        output_path=output_dir / "audit_summary.txt",
        negative_nse_threshold=args.negative_nse_threshold,
        strong_delta_threshold=args.strong_delta_threshold,
        severe_delta_threshold=args.severe_delta_threshold,
    )

    metadata = {
        "summary_file": str(summary_path),
        "ensemble_dir": str(ensemble_dir),
        "n_basins": int(len(audit)),
        "n_scenarios": int(len(SCENARIO_LABELS)),
        "gauge_width": int(args.gauge_width),
        "negative_nse_threshold": float(args.negative_nse_threshold),
        "strong_delta_threshold": float(args.strong_delta_threshold),
        "severe_delta_threshold": float(args.severe_delta_threshold),
        "zero_flow_threshold": float(args.zero_flow_threshold),
        "nse_tolerance": float(args.nse_tolerance),
        "obs_atol": float(args.obs_atol),
        "obs_rtol": float(args.obs_rtol),
        "worst_n": int(args.worst_n),
        "plot_n": int(args.plot_n),
        "basin_grouping_used": False,
    }
    with (output_dir / "analysis_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("Chapter 4B PUB extreme-basin audit completed")
    print("=" * 78)
    print(f"Basins audited:              {len(audit)}")
    print(
        "Observation consistency:     "
        f"{int(obs_check['all_obs_equal'].sum())}/{len(obs_check)}"
    )
    print(
        "Time-coordinate consistency: "
        f"{int(obs_check['all_time_equal'].sum())}/{len(obs_check)}"
    )
    print(
        "Fold consistency:            "
        f"{int(obs_check['fold_id_equal'].sum())}/{len(obs_check)}"
    )
    print(
        "NSE recomputation pass:       "
        f"{int(nse_check['within_tolerance'].sum())}/{len(nse_check)}"
    )
    print(f"Extreme basins exported:     {len(extreme)}")
    print(f"Hydrographs exported:        {len(selected_basins)}")
    print(f"Output directory:            {output_dir}")

    print("\nKey minima:")
    print(f"  STL-Q minimum NSE:      {audit['nse_stl_q'].min():.6f}")
    print(
        f"  Hard-MTL minimum NSE:   "
        f"{audit['nse_hps_target_ssm'].min():.6f}"
    )
    print(
        f"  CGC minimum NSE:        "
        f"{audit['nse_cgc_target_ssm'].min():.6f}"
    )
    print(
        f"  CGC-STL minimum delta:  "
        f"{audit['delta_nse_cgc_minus_stl'].min():.6f}"
    )
    print(
        f"  Minimum NSE denominator:"
        f" {audit['nse_denominator'].min():.6e}"
    )

    strict_failures: List[str] = []
    if not obs_check["all_obs_equal"].all():
        strict_failures.append("observed Q differs across scenarios")
    if not obs_check["all_time_equal"].all():
        strict_failures.append("time coordinates differ across scenarios")
    if not obs_check["fold_id_equal"].all():
        strict_failures.append("fold IDs differ across scenarios")
    if not nse_check["within_tolerance"].all():
        strict_failures.append(
            "stored and recomputed NSE differ beyond tolerance"
        )

    if args.strict and strict_failures:
        raise RuntimeError(
            "Strict audit failed: " + "; ".join(strict_failures)
        )


if __name__ == "__main__":
    main()
