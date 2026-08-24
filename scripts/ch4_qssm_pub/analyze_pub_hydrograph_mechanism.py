#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chapter 4B PUB hydrograph-mechanism analysis.

This script diagnoses how the PUB streamflow prediction changes at the
hydrograph level for STL-Q, Hard-MTL-PUB, and CGC-PUB. It uses existing
ensemble prediction NetCDF files and does not require model retraining.

Main questions
--------------
1. Is the CGC gain associated with better temporal correlation?
2. Does CGC improve hydrograph variability or long-term bias?
3. Are the gains concentrated in low-, middle-, or high-flow conditions?

Metrics
-------
- KGE (2009) and its components:
    r     : Pearson correlation
    alpha : std(sim) / std(obs)
    beta  : mean(sim) / mean(obs)
- Flow-regime errors using observed-flow quantiles:
    low flow    : Q <= Q30
    middle flow : Q30 < Q < Q90
    high flow   : Q >= Q90

The script attempts to auto-detect common NetCDF variable names. If automatic
detection fails, use --obs-var, --pred-var, --basin-dim, and --time-dim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_ENSEMBLE_DIR = Path("experiments/ch4_qssm_pub/ensemble")
DEFAULT_OUTPUT_DIR = Path("experiments/ch4_qssm_pub/hydrograph_mechanism")

SCENARIO_LABELS: Dict[str, str] = {
    "stl_q": "STL-Q",
    "hps_target_ssm": "Hard-MTL-PUB",
    "cgc_target_ssm": "CGC-PUB",
}

OBS_VAR_CANDIDATES: Sequence[str] = (
    "q_obs",
    "obs_q",
    "streamflow_obs",
    "observed_streamflow",
    "q_observed",
    "q_true",
    "streamflow_true",
    "y_true_q",
    "target_q",
    "obs",
    "observation",
    "y_true",
)

PRED_VAR_CANDIDATES: Sequence[str] = (
    "q_pred",
    "pred_q",
    "streamflow_pred",
    "predicted_streamflow",
    "q_sim",
    "streamflow_sim",
    "sim_q",
    "q_prediction",
    "y_pred_q",
    "pred",
    "prediction",
    "y_pred",
)

BASIN_DIM_CANDIDATES: Sequence[str] = (
    "gauge_id",
    "gage_id",
    "basin_id",
    "basin",
    "catchment_id",
    "catchment",
)

TIME_DIM_CANDIDATES: Sequence[str] = (
    "time",
    "date",
    "datetime",
    "day",
)


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a path relative to the project root."""
    return path if path.is_absolute() else project_root / path


def normalize_identifier(value: object) -> str:
    """Normalize basin identifiers while preserving leading-zero strings."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def find_matching_name(
    names: Iterable[str],
    preferred: Optional[str],
    candidates: Sequence[str],
    kind: str,
) -> str:
    """Find an exact or case-insensitive matching variable/dimension name."""
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

    # Token-based fallback.
    if kind == "observation variable":
        obs_tokens = ("obs", "observed", "true")
        q_tokens = ("q", "streamflow", "discharge", "runoff")
        matches = [
            name for name in names
            if any(token in name.lower() for token in obs_tokens)
            and any(token in name.lower() for token in q_tokens)
        ]
    elif kind == "prediction variable":
        pred_tokens = ("pred", "prediction", "sim", "simulated")
        q_tokens = ("q", "streamflow", "discharge", "runoff")
        matches = [
            name for name in names
            if any(token in name.lower() for token in pred_tokens)
            and any(token in name.lower() for token in q_tokens)
        ]
    else:
        matches = []

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Unable to auto-detect {kind}.\n"
        f"Available names: {names}\n"
        f"Please provide the corresponding command-line option explicitly."
    )


def select_q_from_extra_dims(da: xr.DataArray) -> xr.DataArray:
    """
    Reduce optional task/output dimensions to streamflow if they are present.

    Dimensions with size 1 are squeezed automatically. For labeled task/output
    dimensions, the function selects labels containing Q/streamflow/discharge/runoff.
    """
    result = da

    for dim in list(result.dims):
        if result.sizes[dim] == 1:
            result = result.isel({dim: 0}, drop=True)

    for dim in list(result.dims):
        dim_lower = dim.lower()
        if dim_lower not in {"task", "target", "variable", "output", "feature", "channel"}:
            continue

        if dim not in result.coords:
            continue

        values = [str(v).lower() for v in result.coords[dim].values.tolist()]
        q_indices = [
            i for i, value in enumerate(values)
            if value in {"q", "streamflow", "discharge", "runoff"}
            or "streamflow" in value
            or "discharge" in value
            or "runoff" in value
        ]

        if len(q_indices) == 1:
            result = result.isel({dim: q_indices[0]}, drop=True)

    return result


def locate_prediction_file(
    scenario_dir: Path,
    nc_pattern: str,
) -> Path:
    """Locate a single prediction NetCDF file in one fold/scenario directory."""
    candidates = sorted(scenario_dir.glob(nc_pattern))

    if not candidates:
        # Common fallback: search recursively if the file is nested.
        candidates = sorted(scenario_dir.rglob(nc_pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No NetCDF files matching '{nc_pattern}' under {scenario_dir}"
        )

    # Prefer filenames clearly related to predictions.
    prediction_like = [
        p for p in candidates
        if any(token in p.name.lower() for token in ("pred", "prediction"))
    ]
    if len(prediction_like) == 1:
        return prediction_like[0]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Multiple NetCDF files found under {scenario_dir}:\n"
        + "\n".join(str(p) for p in candidates)
        + "\nUse a more specific --nc-pattern."
    )



def locate_single_seed_run_prediction(
    runs_dir: Path,
    fold_id: int,
    scenario: str,
    nc_pattern: str,
) -> Path:
    """
    Fallback to a formal single-seed run when ensemble NetCDF files are absent.

    This fallback is safe only when exactly one formal run matches the
    fold/scenario. If multiple seeds are present, the function fails rather
    than silently analyzing a non-ensemble member.
    """
    pattern = f"ch4b_pub_formal_f{fold_id:02d}_{scenario}_seed*"
    run_dirs = sorted(p for p in runs_dir.glob(pattern) if p.is_dir())

    if len(run_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one formal single-seed run for fold={fold_id:02d}, "
            f"scenario={scenario}, but found {len(run_dirs)} under {runs_dir}.\n"
            "If this is a multi-seed experiment, generate ensemble NetCDF outputs "
            "or point the script to a directory that contains them."
        )

    return locate_prediction_file(run_dirs[0], nc_pattern=nc_pattern)


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
                f"Requested {kind} '{preferred}' not present in data dimensions: {dims}"
            )
        return preferred

    lower_map = {dim.lower(): dim for dim in dims}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if kind == "basin dimension":
        token_matches = [
            dim for dim in dims
            if any(token in dim.lower() for token in ("basin", "gauge", "gage", "catchment"))
        ]
    else:
        token_matches = [
            dim for dim in dims
            if any(token in dim.lower() for token in ("time", "date", "day"))
        ]

    if len(token_matches) == 1:
        return token_matches[0]

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
    """Find, reduce, align, and order the streamflow observation/prediction arrays."""
    obs_name = find_matching_name(
        ds.data_vars,
        obs_var,
        OBS_VAR_CANDIDATES,
        "observation variable",
    )
    pred_name = find_matching_name(
        ds.data_vars,
        pred_var,
        PRED_VAR_CANDIDATES,
        "prediction variable",
    )

    obs = select_q_from_extra_dims(ds[obs_name])
    pred = select_q_from_extra_dims(ds[pred_name])

    basin_name = infer_dimension(
        obs,
        basin_dim,
        BASIN_DIM_CANDIDATES,
        "basin dimension",
    )
    time_name = infer_dimension(
        obs,
        time_dim,
        TIME_DIM_CANDIDATES,
        "time dimension",
    )

    if basin_name not in pred.dims or time_name not in pred.dims:
        raise ValueError(
            "Observation and prediction arrays do not share the inferred basin/time dimensions.\n"
            f"obs dims={obs.dims}, pred dims={pred.dims}"
        )

    # Ensure no unresolved dimensions remain.
    allowed_dims = {basin_name, time_name}
    obs_extra = [dim for dim in obs.dims if dim not in allowed_dims]
    pred_extra = [dim for dim in pred.dims if dim not in allowed_dims]
    if obs_extra or pred_extra:
        raise ValueError(
            "Unresolved extra dimensions remain after Q selection.\n"
            f"obs extra dims={obs_extra}, pred extra dims={pred_extra}\n"
            "Inspect the dataset structure and provide more specific variables."
        )

    obs, pred = xr.align(obs, pred, join="inner")
    obs = obs.transpose(basin_name, time_name)
    pred = pred.transpose(basin_name, time_name)

    return obs, pred, basin_name, time_name


def pearson_r(obs: np.ndarray, sim: np.ndarray) -> float:
    """Pearson correlation with explicit handling of constant series."""
    if obs.size < 2 or sim.size < 2:
        return np.nan
    if np.isclose(np.std(obs), 0.0) or np.isclose(np.std(sim), 0.0):
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1])


def kge_2009(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Kling-Gupta Efficiency (2009) and its three components."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask].astype(float)
    sim = sim[mask].astype(float)

    if obs.size < 2:
        return {"kge": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}

    mean_obs = float(np.mean(obs))
    std_obs = float(np.std(obs, ddof=1))
    mean_sim = float(np.mean(sim))
    std_sim = float(np.std(sim, ddof=1))

    r = pearson_r(obs, sim)
    alpha = np.nan if np.isclose(std_obs, 0.0) else std_sim / std_obs
    beta = np.nan if np.isclose(mean_obs, 0.0) else mean_sim / mean_obs

    if not np.all(np.isfinite([r, alpha, beta])):
        kge = np.nan
    else:
        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    return {
        "kge": float(kge) if np.isfinite(kge) else np.nan,
        "r": float(r) if np.isfinite(r) else np.nan,
        "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "beta": float(beta) if np.isfinite(beta) else np.nan,
    }


def error_metrics(
    obs: np.ndarray,
    sim: np.ndarray,
    mean_obs_all: float,
    std_obs_all: float,
) -> Dict[str, float]:
    """Compute raw and basin-normalized error metrics."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask].astype(float)
    sim = sim[mask].astype(float)

    if obs.size == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "nmae_by_basin_mean": np.nan,
            "nrmse_by_basin_std": np.nan,
        }

    error = sim - obs
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))

    nmae = np.nan if np.isclose(mean_obs_all, 0.0) else mae / abs(mean_obs_all)
    nrmse = np.nan if np.isclose(std_obs_all, 0.0) else rmse / std_obs_all

    return {
        "n": int(obs.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "nmae_by_basin_mean": float(nmae) if np.isfinite(nmae) else np.nan,
        "nrmse_by_basin_std": float(nrmse) if np.isfinite(nrmse) else np.nan,
    }


def compute_basin_metrics(
    gauge_id: str,
    obs: np.ndarray,
    sim: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> Dict[str, float]:
    """Compute KGE components and low/mid/high-flow error diagnostics."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask].astype(float)
    sim = sim[mask].astype(float)

    if obs.size < 2:
        return {"gauge_id": gauge_id}

    base = {"gauge_id": gauge_id, "n_time_steps": int(obs.size)}
    base.update(kge_2009(obs, sim))

    mean_obs_all = float(np.mean(obs))
    std_obs_all = float(np.std(obs, ddof=1))

    q_low = float(np.quantile(obs, low_quantile))
    q_high = float(np.quantile(obs, high_quantile))
    base["obs_q_low_threshold"] = q_low
    base["obs_q_high_threshold"] = q_high

    regime_masks = {
        "low": obs <= q_low,
        "mid": (obs > q_low) & (obs < q_high),
        "high": obs >= q_high,
    }

    for regime, regime_mask in regime_masks.items():
        metrics = error_metrics(
            obs[regime_mask],
            sim[regime_mask],
            mean_obs_all=mean_obs_all,
            std_obs_all=std_obs_all,
        )
        for metric_name, value in metrics.items():
            base[f"{regime}_{metric_name}"] = value

    return base


def extract_scenario_metrics(
    nc_path: Path,
    fold_id: int,
    scenario: str,
    obs_var: Optional[str],
    pred_var: Optional[str],
    basin_dim: Optional[str],
    time_dim: Optional[str],
    low_quantile: float,
    high_quantile: float,
) -> pd.DataFrame:
    """Read one fold/scenario NetCDF file and compute basin-wise metrics."""
    with xr.open_dataset(nc_path) as ds:
        obs, pred, basin_name, _ = align_obs_pred(
            ds,
            obs_var=obs_var,
            pred_var=pred_var,
            basin_dim=basin_dim,
            time_dim=time_dim,
        )

        if basin_name in obs.coords:
            basin_values = obs.coords[basin_name].values
        else:
            basin_values = np.arange(obs.sizes[basin_name])

        records: List[Dict[str, float]] = []
        for i in range(obs.sizes[basin_name]):
            gauge_id = normalize_identifier(basin_values[i])
            obs_i = obs.isel({basin_name: i}).values
            pred_i = pred.isel({basin_name: i}).values

            record = compute_basin_metrics(
                gauge_id=gauge_id,
                obs=np.asarray(obs_i),
                sim=np.asarray(pred_i),
                low_quantile=low_quantile,
                high_quantile=high_quantile,
            )
            record["fold_id"] = int(fold_id)
            record["scenario"] = scenario
            record["model"] = SCENARIO_LABELS[scenario]
            record["source_nc"] = str(nc_path)
            records.append(record)

    return pd.DataFrame.from_records(records)


def pivot_model_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long model metrics to one row per basin for paired differences."""
    id_cols = ["gauge_id", "fold_id"]
    metric_cols = [
        col for col in long_df.columns
        if col not in id_cols + ["scenario", "model", "source_nc"]
        and pd.api.types.is_numeric_dtype(long_df[col])
    ]

    pieces = []
    for metric in metric_cols:
        pivot = long_df.pivot_table(
            index=id_cols,
            columns="scenario",
            values=metric,
            aggfunc="first",
        )
        pivot.columns = [f"{metric}__{scenario}" for scenario in pivot.columns]
        pieces.append(pivot)

    if not pieces:
        raise RuntimeError("No numeric hydrograph metrics were available for paired analysis.")

    wide = pd.concat(pieces, axis=1).reset_index()
    return wide


def add_paired_improvements(wide: pd.DataFrame) -> pd.DataFrame:
    """Create positive-is-better paired improvement metrics."""
    out = wide.copy()

    # Higher-is-better metrics.
    higher_better = ["kge", "r"]

    # For alpha and beta, closeness to 1 is better.
    for baseline, comparison, label in [
        ("stl_q", "hps_target_ssm", "hard_minus_stl"),
        ("stl_q", "cgc_target_ssm", "cgc_minus_stl"),
        ("hps_target_ssm", "cgc_target_ssm", "cgc_minus_hard"),
    ]:
        for metric in higher_better:
            a = f"{metric}__{comparison}"
            b = f"{metric}__{baseline}"
            if a in out.columns and b in out.columns:
                out[f"improvement_{metric}__{label}"] = out[a] - out[b]

        for metric in ("alpha", "beta"):
            a = f"{metric}__{comparison}"
            b = f"{metric}__{baseline}"
            if a in out.columns and b in out.columns:
                out[f"improvement_{metric}_closeness__{label}"] = (
                    np.abs(out[b] - 1.0) - np.abs(out[a] - 1.0)
                )

        # Lower error is better, so baseline - comparison is positive improvement.
        for regime in ("low", "mid", "high"):
            for metric in ("nmae_by_basin_mean", "nrmse_by_basin_std", "mae", "rmse"):
                a = f"{regime}_{metric}__{comparison}"
                b = f"{regime}_{metric}__{baseline}"
                if a in out.columns and b in out.columns:
                    out[f"improvement_{regime}_{metric}__{label}"] = out[b] - out[a]

    return out


def summarize_improvements(wide: pd.DataFrame) -> pd.DataFrame:
    """Summarize all paired positive-is-better improvement columns."""
    rows = []
    improvement_cols = [c for c in wide.columns if c.startswith("improvement_")]

    for column in improvement_cols:
        x = pd.to_numeric(wide[column], errors="coerce").dropna()
        rows.append(
            {
                "metric": column,
                "n_basins": int(len(x)),
                "median_improvement": float(x.median()),
                "q25_improvement": float(x.quantile(0.25)),
                "q75_improvement": float(x.quantile(0.75)),
                "mean_improvement": float(x.mean()),
                "positive_rate": float((x > 0).mean()),
                "negative_rate": float((x < 0).mean()),
            }
        )

    return pd.DataFrame.from_records(rows)


def plot_kge_component_improvements(wide: pd.DataFrame, output_dir: Path) -> None:
    """Boxplots of KGE-component improvements for Hard-MTL and CGC versus STL."""
    specifications = [
        ("r", "Temporal correlation improvement"),
        ("alpha_closeness", "Variability-ratio closeness improvement"),
        ("beta_closeness", "Bias-ratio closeness improvement"),
    ]

    for metric_key, ylabel in specifications:
        hard_col = f"improvement_{metric_key}__hard_minus_stl"
        cgc_col = f"improvement_{metric_key}__cgc_minus_stl"

        if hard_col not in wide.columns or cgc_col not in wide.columns:
            continue

        data = [
            wide[hard_col].dropna().to_numpy(),
            wide[cgc_col].dropna().to_numpy(),
        ]

        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        ax.boxplot(data, tick_labels=["Hard-MTL", "CGC"], showfliers=False)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_ylabel(ylabel)
        fig.tight_layout()

        fig.savefig(
            output_dir / f"fig_hydrograph_{metric_key}_improvement.png",
            dpi=400,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"fig_hydrograph_{metric_key}_improvement.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_flow_regime_improvements(wide: pd.DataFrame, output_dir: Path) -> None:
    """Compare normalized MAE improvement across low/mid/high flow regimes."""
    for comparison_label, display_name in [
        ("hard_minus_stl", "Hard-MTL minus STL"),
        ("cgc_minus_stl", "CGC minus STL"),
        ("cgc_minus_hard", "CGC minus Hard-MTL"),
    ]:
        columns = [
            f"improvement_{regime}_nmae_by_basin_mean__{comparison_label}"
            for regime in ("low", "mid", "high")
        ]

        if not all(column in wide.columns for column in columns):
            continue

        data = [wide[column].dropna().to_numpy() for column in columns]

        fig, ax = plt.subplots(figsize=(5.3, 4.2))
        ax.boxplot(
            data,
            tick_labels=["Low", "Middle", "High"],
            showfliers=False,
        )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Observed-flow regime")
        ax.set_ylabel("nMAE improvement (positive = better)")
        ax.set_title(display_name)
        fig.tight_layout()

        safe_name = comparison_label.lower()
        fig.savefig(
            output_dir / f"fig_hydrograph_flow_regime_{safe_name}.png",
            dpi=400,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"fig_hydrograph_flow_regime_{safe_name}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def write_dataset_inventory(
    rows: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    """Write the NetCDF files and detected variables used in the analysis."""
    pd.DataFrame.from_records(rows).to_csv(
        output_dir / "hydrograph_dataset_inventory.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Chapter 4B PUB hydrograph mechanisms from existing "
            "ensemble prediction NetCDF files."
        )
    )
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=DEFAULT_ENSEMBLE_DIR,
        help="Root directory containing foldXX/scenario ensemble outputs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/ch4_qssm_pub/runs"),
        help=(
            "Fallback directory for formal single-seed run NetCDF files when "
            "ensemble NetCDF files are not present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for hydrograph-mechanism outputs.",
    )
    parser.add_argument(
        "--nc-pattern",
        default="*.nc",
        help="NetCDF glob pattern within each fold/scenario directory.",
    )
    parser.add_argument(
        "--obs-var",
        default=None,
        help="Observed streamflow variable name. Auto-detected by default.",
    )
    parser.add_argument(
        "--pred-var",
        default=None,
        help="Predicted streamflow variable name. Auto-detected by default.",
    )
    parser.add_argument(
        "--basin-dim",
        default=None,
        help="Basin dimension name. Auto-detected by default.",
    )
    parser.add_argument(
        "--time-dim",
        default=None,
        help="Time dimension name. Auto-detected by default.",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.30,
        help="Observed-flow quantile defining the low-flow regime.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.90,
        help="Observed-flow quantile defining the high-flow regime.",
    )
    args = parser.parse_args()

    if not (0.0 < args.low_quantile < args.high_quantile < 1.0):
        raise ValueError(
            "--low-quantile and --high-quantile must satisfy "
            "0 < low < high < 1."
        )

    project_root = Path(__file__).resolve().parents[2]
    ensemble_dir = resolve_path(project_root, args.ensemble_dir)
    runs_dir = resolve_path(project_root, args.runs_dir)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

    all_frames: List[pd.DataFrame] = []
    inventory_rows: List[Dict[str, object]] = []

    fold_dirs = sorted(
        p for p in ensemble_dir.glob("fold*")
        if p.is_dir()
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold directories found under {ensemble_dir}"
        )

    for fold_dir in fold_dirs:
        digits = "".join(ch for ch in fold_dir.name if ch.isdigit())
        if not digits:
            raise ValueError(f"Unable to parse fold id from: {fold_dir.name}")
        fold_id = int(digits)

        for scenario in SCENARIO_LABELS:
            scenario_dir = fold_dir / scenario
            if not scenario_dir.exists():
                raise FileNotFoundError(
                    f"Missing scenario directory: {scenario_dir}"
                )

            try:
                nc_path = locate_prediction_file(
                    scenario_dir,
                    nc_pattern=args.nc_pattern,
                )
                nc_source = "ensemble"
            except FileNotFoundError:
                nc_path = locate_single_seed_run_prediction(
                    runs_dir=runs_dir,
                    fold_id=fold_id,
                    scenario=scenario,
                    nc_pattern=args.nc_pattern,
                )
                nc_source = "single_seed_run_fallback"

            # Open briefly to record dataset structure for reproducibility.
            with xr.open_dataset(nc_path) as ds:
                inventory_rows.append(
                    {
                        "fold_id": fold_id,
                        "scenario": scenario,
                        "nc_path": str(nc_path),
                        "nc_source": nc_source,
                        "data_vars": "|".join(ds.data_vars),
                        "dims": "|".join(f"{k}:{v}" for k, v in ds.sizes.items()),
                    }
                )

            frame = extract_scenario_metrics(
                nc_path=nc_path,
                fold_id=fold_id,
                scenario=scenario,
                obs_var=args.obs_var,
                pred_var=args.pred_var,
                basin_dim=args.basin_dim,
                time_dim=args.time_dim,
                low_quantile=args.low_quantile,
                high_quantile=args.high_quantile,
            )
            all_frames.append(frame)

            print(
                f"Processed fold={fold_id:02d} "
                f"scenario={scenario:<15s} "
                f"basins={len(frame)} "
                f"source={nc_source} "
                f"file={nc_path.name}"
            )

    long_df = pd.concat(all_frames, ignore_index=True)

    # Every PUB basin should occur once per scenario.
    duplicate_mask = long_df.duplicated(
        subset=["gauge_id", "scenario"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = long_df.loc[
            duplicate_mask,
            ["gauge_id", "fold_id", "scenario"],
        ].head(20)
        raise ValueError(
            "Duplicate basin/scenario records detected after concatenating folds.\n"
            + examples.to_string(index=False)
        )

    wide_df = pivot_model_metrics(long_df)
    wide_df = add_paired_improvements(wide_df)
    summary_df = summarize_improvements(wide_df)

    long_df.to_csv(
        output_dir / "pub_hydrograph_metrics_long.csv",
        index=False,
    )
    wide_df.to_csv(
        output_dir / "pub_hydrograph_metrics_paired.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "pub_hydrograph_improvement_summary.csv",
        index=False,
    )
    write_dataset_inventory(inventory_rows, output_dir)

    plot_kge_component_improvements(wide_df, output_dir)
    plot_flow_regime_improvements(wide_df, output_dir)

    metadata = {
        "ensemble_dir": str(ensemble_dir),
        "runs_dir_fallback": str(runs_dir),
        "low_flow_quantile": args.low_quantile,
        "high_flow_quantile": args.high_quantile,
        "obs_var_override": args.obs_var,
        "pred_var_override": args.pred_var,
        "basin_dim_override": args.basin_dim,
        "time_dim_override": args.time_dim,
        "basin_grouping_used": False,
        "kge_definition": "KGE 2009",
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("Chapter 4B PUB hydrograph-mechanism analysis completed")
    print("=" * 78)
    print(f"Ensemble root: {ensemble_dir}")
    print(f"Output:        {output_dir}")
    print(f"Long rows:     {len(long_df)}")
    print(f"Unique basins: {long_df['gauge_id'].nunique()}")
    print("\nPaired hydrograph-improvement summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
