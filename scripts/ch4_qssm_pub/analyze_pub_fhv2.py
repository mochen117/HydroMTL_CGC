#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chapter 4B PUB high-flow analysis using FHV2%.

Purpose
-------
Evaluate whether target-basin SSM supervision improves high-flow
streamflow simulation under PUB conditions.

The analysis uses the formal 5-fold PUB prediction exports and computes
FHV for the highest 2% of the flow-duration curve.

FHV = 100 * sum(Qsim_high - Qobs_high) / sum(Qobs_high)

Observed and simulated discharge series are independently sorted in
descending order before extracting the highest 2%, following the
flow-duration-curve definition of FHV.

Outputs
-------
1. Basin-wise signed FHV and absolute FHV.
2. Overall summaries for STL-Q, Hard-MTL, and CGC.
3. Wet/Snow/Dry hydroclimate summaries.
4. Paired statistics for absolute FHV differences.
5. Analysis metadata.

Interpretation
--------------
FHV < 0 : systematic underestimation of high-flow volume.
FHV > 0 : systematic overestimation of high-flow volume.
|FHV|    : high-flow magnitude error; lower is better.

For paired absolute-FHV differences:
delta = |FHV|_model - |FHV|_reference

delta < 0 indicates improvement.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_ROOT = (
    PROJECT_ROOT
    / "experiments/ch4_qssm_pub/runs"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/ch4_summary/teacher_revision"
)

DEFAULT_CLIMATE_FILE = Path(
    "/home/mochen/hydro_data/camels/camels_us/"
    "camels_clim.txt"
)

SCENARIOS = {
    "STL": "stl_q",
    "Hard": "hps_target_ssm",
    "CGC": "cgc_target_ssm",
}

PAIRWISE_COMPARISONS = (
    ("Hard", "STL"),
    ("CGC", "STL"),
    ("CGC", "Hard"),
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute formal PUB FHV2% high-flow metrics "
            "for STL, Hard-MTL, and CGC."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=RUN_ROOT,
    )
    parser.add_argument(
        "--climate-file",
        type=Path,
        default=DEFAULT_CLIMATE_FILE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--high-flow-fraction",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--expected-basins",
        type=int,
        default=592,
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def resolve_path(path: Path) -> Path:
    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def normalize_gauge_id(value: object) -> str:
    """Normalize basin identifiers to eight-character gauge IDs."""
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode()

    text = str(value).strip()

    if text.endswith(".0"):
        prefix = text[:-2]
        if prefix.isdigit():
            text = prefix

    return text.zfill(8)


def run_path(
    run_root: Path,
    fold_id: int,
    scenario: str,
) -> Path:
    return (
        run_root
        / (
            f"ch4b_pub_formal_f{fold_id:02d}_"
            f"{scenario}_seed42"
        )
        / "test_predictions_and_weights.nc"
    )


def decode_basin_values(
    values: np.ndarray,
) -> list[str]:
    return [
        normalize_gauge_id(value)
        for value in values
    ]


def compute_fhv(
    obs: np.ndarray,
    sim: np.ndarray,
    high_flow_fraction: float,
) -> tuple[float, int, int]:
    """
    Compute signed FHV for the highest fraction of the FDC.

    Observed and simulated series are sorted independently in descending
    order, consistent with the flow-duration-curve definition.

    Returns
    -------
    fhv
        Signed FHV in percent.
    n_valid
        Number of finite paired daily values.
    n_high
        Number of values used in the high-flow segment.
    """
    obs = np.asarray(obs, dtype=float).reshape(-1)
    sim = np.asarray(sim, dtype=float).reshape(-1)

    if obs.shape != sim.shape:
        raise ValueError(
            f"Observation/simulation shape mismatch: "
            f"{obs.shape} vs {sim.shape}."
        )

    valid = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid]
    sim = sim[valid]

    n_valid = int(obs.size)

    if n_valid == 0:
        return np.nan, 0, 0

    n_high = max(
        1,
        int(np.round(
            high_flow_fraction * n_valid
        )),
    )

    obs_high = np.sort(obs)[::-1][:n_high]
    sim_high = np.sort(sim)[::-1][:n_high]

    denominator = float(
        np.sum(obs_high)
    )

    if (
        not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return np.nan, n_valid, n_high

    fhv = (
        100.0
        * float(
            np.sum(sim_high - obs_high)
        )
        / denominator
    )

    return float(fhv), n_valid, n_high


def load_fold_data(
    run_root: Path,
    fold_id: int,
    scenario: str,
) -> Dict[str, object]:
    """Load one formal PUB fold/scenario prediction export."""
    path = run_path(
        run_root=run_root,
        fold_id=fold_id,
        scenario=scenario,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        required = {
            "streamflow_obs",
            "streamflow_sim",
        }
        missing = required.difference(
            ds.data_vars
        )

        if missing:
            raise ValueError(
                f"{path}: missing variables "
                f"{sorted(missing)}."
            )

        if "basin" not in ds.coords:
            raise ValueError(
                f"{path}: missing basin coordinate."
            )

        if "time" not in ds.coords:
            raise ValueError(
                f"{path}: missing time coordinate."
            )

        basin_ids = decode_basin_values(
            ds["basin"].values
        )
        time = pd.DatetimeIndex(
            pd.to_datetime(
                ds["time"].values
            )
        )

        obs = np.asarray(
            ds["streamflow_obs"].values,
            dtype=float,
        )
        sim = np.asarray(
            ds["streamflow_sim"].values,
            dtype=float,
        )

    if obs.shape != sim.shape:
        raise ValueError(
            f"{path}: obs/sim shape mismatch "
            f"{obs.shape} vs {sim.shape}."
        )

    if obs.shape != (
        len(basin_ids),
        len(time),
    ):
        raise ValueError(
            f"{path}: unexpected array shape "
            f"{obs.shape}; basin={len(basin_ids)}, "
            f"time={len(time)}."
        )

    return {
        "path": path,
        "basin_ids": basin_ids,
        "time": time,
        "obs": obs,
        "sim": sim,
    }


def check_fold_consistency(
    fold_id: int,
    data: Dict[str, Dict[str, object]],
) -> None:
    """Verify basin, time, and observation consistency across scenarios."""
    reference = data["STL"]

    ref_ids = reference["basin_ids"]
    ref_time = reference["time"]
    ref_obs = reference["obs"]

    if len(set(ref_ids)) != len(ref_ids):
        raise ValueError(
            f"Fold {fold_id}: duplicate STL basin IDs."
        )

    ref_index = {
        gauge_id: i
        for i, gauge_id in enumerate(ref_ids)
    }

    for label in ("Hard", "CGC"):
        current = data[label]

        ids = current["basin_ids"]
        time = current["time"]
        obs = current["obs"]

        if set(ids) != set(ref_ids):
            missing = sorted(
                set(ref_ids).difference(ids)
            )
            extra = sorted(
                set(ids).difference(ref_ids)
            )
            raise ValueError(
                f"Fold {fold_id} {label}: basin set mismatch. "
                f"Missing={missing[:5]}, extra={extra[:5]}."
            )

        if not ref_time.equals(time):
            raise ValueError(
                f"Fold {fold_id} {label}: time axis mismatch."
            )

        current_index = {
            gauge_id: i
            for i, gauge_id in enumerate(ids)
        }

        for gauge_id in ref_ids:
            ref_values = np.asarray(
                ref_obs[
                    ref_index[gauge_id]
                ],
                dtype=float,
            )
            cur_values = np.asarray(
                obs[
                    current_index[gauge_id]
                ],
                dtype=float,
            )

            if not np.array_equal(
                np.isnan(ref_values),
                np.isnan(cur_values),
            ):
                raise ValueError(
                    f"Fold {fold_id} {label} "
                    f"{gauge_id}: observation missingness mismatch."
                )

            valid = (
                np.isfinite(ref_values)
                & np.isfinite(cur_values)
            )

            if valid.any() and not np.allclose(
                ref_values[valid],
                cur_values[valid],
                rtol=1e-7,
                atol=1e-10,
            ):
                raise ValueError(
                    f"Fold {fold_id} {label} "
                    f"{gauge_id}: observations differ."
                )


def build_per_basin_table(
    run_root: Path,
    high_flow_fraction: float,
) -> pd.DataFrame:
    """Compute FHV for all formal PUB target basins."""
    records = []

    for fold_id in range(1, 6):
        LOGGER.info(
            "Loading formal PUB fold %02d.",
            fold_id,
        )

        fold_data = {
            label: load_fold_data(
                run_root=run_root,
                fold_id=fold_id,
                scenario=scenario,
            )
            for label, scenario in SCENARIOS.items()
        }

        check_fold_consistency(
            fold_id=fold_id,
            data=fold_data,
        )

        reference = fold_data["STL"]
        basin_ids = reference["basin_ids"]

        index_lookup = {
            label: {
                gauge_id: i
                for i, gauge_id in enumerate(
                    fold_data[label]["basin_ids"]
                )
            }
            for label in SCENARIOS
        }

        for gauge_id in basin_ids:
            row = {
                "gauge_id": gauge_id,
                "fold_id": fold_id,
            }

            common_n_valid = None
            common_n_high = None

            for label in SCENARIOS:
                idx = index_lookup[label][gauge_id]

                obs = fold_data[label]["obs"][idx]
                sim = fold_data[label]["sim"][idx]

                fhv, n_valid, n_high = compute_fhv(
                    obs=obs,
                    sim=sim,
                    high_flow_fraction=high_flow_fraction,
                )

                if common_n_valid is None:
                    common_n_valid = n_valid
                    common_n_high = n_high
                else:
                    if (
                        n_valid != common_n_valid
                        or n_high != common_n_high
                    ):
                        raise ValueError(
                            f"{gauge_id}: inconsistent valid/high-flow "
                            f"sample counts across scenarios."
                        )

                row[f"{label}_FHV2"] = fhv
                row[f"{label}_abs_FHV2"] = abs(fhv)

            row["n_valid_days"] = common_n_valid
            row["n_highflow_days"] = common_n_high

            records.append(row)

        LOGGER.info(
            "Fold %02d complete: %d target basins.",
            fold_id,
            len(basin_ids),
        )

    out = pd.DataFrame(records)

    if out["gauge_id"].duplicated().any():
        duplicates = (
            out.loc[
                out["gauge_id"].duplicated(
                    keep=False
                ),
                "gauge_id",
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "Basins occur in more than one PUB test fold: "
            f"{duplicates[:10]}"
        )

    return out.sort_values(
        "gauge_id"
    ).reset_index(drop=True)


def load_hydroclimate_groups(
    climate_file: Path,
) -> pd.DataFrame:
    """
    Build Wet/Snow/Dry groups from CAMELS climate attributes.

    Classification:
    Snow: frac_snow > 0.20
    Wet : remaining basins with aridity < 1.0
    Dry : remaining basins with aridity >= 1.0
    """
    if not climate_file.exists():
        raise FileNotFoundError(
            climate_file
        )

    climate = pd.read_csv(
        climate_file,
        sep=";",
    )

    climate.columns = [
        str(column).strip()
        for column in climate.columns
    ]

    required = {
        "gauge_id",
        "frac_snow",
        "aridity",
    }
    missing = required.difference(
        climate.columns
    )

    if missing:
        raise ValueError(
            f"Climate file missing columns: "
            f"{sorted(missing)}."
        )

    climate["gauge_id"] = (
        climate["gauge_id"]
        .map(normalize_gauge_id)
    )

    climate["frac_snow"] = pd.to_numeric(
        climate["frac_snow"],
        errors="coerce",
    )
    climate["aridity"] = pd.to_numeric(
        climate["aridity"],
        errors="coerce",
    )

    if climate[
        ["frac_snow", "aridity"]
    ].isna().any().any():
        raise ValueError(
            "Non-finite frac_snow/aridity values "
            "detected in climate file."
        )

    snow = climate["frac_snow"] > 0.20

    climate["hydroclimate_group"] = np.where(
        snow,
        "Snow",
        np.where(
            climate["aridity"] < 1.0,
            "Wet",
            "Dry",
        ),
    )

    if climate["gauge_id"].duplicated().any():
        raise ValueError(
            "Duplicate gauge IDs in climate file."
        )

    return climate[
        [
            "gauge_id",
            "frac_snow",
            "aridity",
            "hydroclimate_group",
        ]
    ].copy()


def metric_summary(
    frame: pd.DataFrame,
    group_name: str,
    group_value: str,
) -> list[dict[str, object]]:
    """Summarize signed and absolute FHV for each model."""
    rows = []

    for model in SCENARIOS:
        signed = pd.to_numeric(
            frame[f"{model}_FHV2"],
            errors="coerce",
        )
        absolute = pd.to_numeric(
            frame[f"{model}_abs_FHV2"],
            errors="coerce",
        )

        valid = (
            np.isfinite(signed)
            & np.isfinite(absolute)
        )

        signed = signed[valid]
        absolute = absolute[valid]

        rows.append({
            group_name: group_value,
            "model": model,
            "n": int(len(signed)),
            "median_FHV2": float(
                signed.median()
            ),
            "mean_FHV2": float(
                signed.mean()
            ),
            "q25_FHV2": float(
                signed.quantile(0.25)
            ),
            "q75_FHV2": float(
                signed.quantile(0.75)
            ),
            "median_abs_FHV2": float(
                absolute.median()
            ),
            "mean_abs_FHV2": float(
                absolute.mean()
            ),
            "q25_abs_FHV2": float(
                absolute.quantile(0.25)
            ),
            "q75_abs_FHV2": float(
                absolute.quantile(0.75)
            ),
            "underestimation_rate": float(
                (signed < 0.0).mean()
            ),
            "overestimation_rate": float(
                (signed > 0.0).mean()
            ),
        })

    return rows


def bootstrap_median_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the paired median difference."""
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    medians = np.empty(
        repetitions,
        dtype=float,
    )

    n = int(values.size)

    for i in range(repetitions):
        indices = rng.integers(
            0,
            n,
            size=n,
        )
        medians[i] = np.median(
            values[indices]
        )

    lower, upper = np.quantile(
        medians,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def wilcoxon_pvalue(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan

    if np.allclose(
        values,
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        return 1.0

    result = wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return float(result.pvalue)


def bh_fdr(
    pvalues: np.ndarray,
) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    adjusted = np.full(
        pvalues.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(pvalues)
    valid_indices = np.where(valid)[0]

    if valid_indices.size == 0:
        return adjusted

    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]

    m = len(ranked)

    raw_adjusted = (
        ranked
        * m
        / np.arange(1, m + 1)
    )

    monotonic = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]

    monotonic = np.minimum(
        monotonic,
        1.0,
    )

    restored = np.empty(
        m,
        dtype=float,
    )
    restored[order] = monotonic

    adjusted[valid_indices] = restored

    return adjusted


def build_pairwise_statistics(
    frame: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    """
    Compare absolute FHV between models.

    Negative paired delta means the first model has lower |FHV|.
    """
    rows = []

    for comparison_index, (
        model,
        reference,
    ) in enumerate(
        PAIRWISE_COMPARISONS
    ):
        model_values = pd.to_numeric(
            frame[f"{model}_abs_FHV2"],
            errors="coerce",
        ).to_numpy(dtype=float)

        reference_values = pd.to_numeric(
            frame[f"{reference}_abs_FHV2"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(model_values)
            & np.isfinite(reference_values)
        )

        model_values = model_values[valid]
        reference_values = reference_values[
            valid
        ]

        delta = (
            model_values
            - reference_values
        )

        ci_low, ci_high = (
            bootstrap_median_ci(
                values=delta,
                repetitions=bootstrap_reps,
                seed=seed + comparison_index,
            )
        )

        rows.append({
            "comparison":
                f"{model}_minus_{reference}",
            "metric":
                "abs_FHV2",
            "n":
                int(delta.size),
            "median_delta":
                float(np.median(delta)),
            "mean_delta":
                float(np.mean(delta)),
            "q25_delta":
                float(np.quantile(delta, 0.25)),
            "q75_delta":
                float(np.quantile(delta, 0.75)),
            "bootstrap_ci_low":
                ci_low,
            "bootstrap_ci_high":
                ci_high,
            "better_rate":
                float(np.mean(delta < 0.0)),
            "worse_rate":
                float(np.mean(delta > 0.0)),
            "wilcoxon_p_raw":
                wilcoxon_pvalue(delta),
        })

    out = pd.DataFrame(rows)

    out["wilcoxon_p_fdr"] = bh_fdr(
        out["wilcoxon_p_raw"].to_numpy(
            dtype=float
        )
    )

    ci_excludes_zero = (
        (out["bootstrap_ci_low"] > 0.0)
        | (out["bootstrap_ci_high"] < 0.0)
    )

    out["robust_significant"] = (
        (out["wilcoxon_p_fdr"] < 0.05)
        & ci_excludes_zero
    )

    out["direction"] = np.where(
        out["median_delta"] < 0.0,
        "lower_abs_FHV",
        np.where(
            out["median_delta"] > 0.0,
            "higher_abs_FHV",
            "no_median_difference",
        ),
    )

    return out


def main() -> None:
    setup_logging()
    args = parse_args()

    run_root = resolve_path(
        args.run_root
    )
    climate_file = resolve_path(
        args.climate_file
    )
    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (
        0.0
        < args.high_flow_fraction
        < 1.0
    ):
        raise ValueError(
            "--high-flow-fraction must be "
            "between 0 and 1."
        )

    LOGGER.info(
        "Computing PUB FHV using top %.1f%% "
        "of the flow-duration curve.",
        100.0 * args.high_flow_fraction,
    )

    per_basin = build_per_basin_table(
        run_root=run_root,
        high_flow_fraction=(
            args.high_flow_fraction
        ),
    )

    if (
        args.expected_basins > 0
        and len(per_basin)
        != args.expected_basins
    ):
        raise RuntimeError(
            f"Expected {args.expected_basins} "
            f"unique PUB basins, found "
            f"{len(per_basin)}."
        )

    LOGGER.info(
        "Formal PUB basins: %d unique basins.",
        len(per_basin),
    )

    climate = load_hydroclimate_groups(
        climate_file
    )

    per_basin = per_basin.merge(
        climate,
        on="gauge_id",
        how="left",
        validate="one_to_one",
    )

    if per_basin[
        "hydroclimate_group"
    ].isna().any():
        missing = per_basin.loc[
            per_basin[
                "hydroclimate_group"
            ].isna(),
            "gauge_id",
        ].tolist()

        raise RuntimeError(
            "Missing hydroclimate group for "
            f"{len(missing)} basins: "
            f"{missing[:10]}"
        )

    expected_groups = {
        "Wet": 282,
        "Snow": 168,
        "Dry": 142,
    }

    actual_groups = (
        per_basin[
            "hydroclimate_group"
        ]
        .value_counts()
        .to_dict()
    )

    if actual_groups != expected_groups:
        raise RuntimeError(
            "Unexpected hydroclimate counts. "
            f"Expected={expected_groups}, "
            f"actual={actual_groups}."
        )

    per_basin_path = (
        output_dir
        / "ch4b_pub_fhv2_per_basin.csv"
    )
    per_basin.to_csv(
        per_basin_path,
        index=False,
    )

    overall = pd.DataFrame(
        metric_summary(
            frame=per_basin,
            group_name="scope",
            group_value="Overall",
        )
    )

    overall_path = (
        output_dir
        / "ch4b_pub_fhv2_overall_summary.csv"
    )
    overall.to_csv(
        overall_path,
        index=False,
    )

    hydro_rows = []

    for group in (
        "Dry",
        "Snow",
        "Wet",
    ):
        subset = per_basin.loc[
            per_basin[
                "hydroclimate_group"
            ] == group
        ]

        hydro_rows.extend(
            metric_summary(
                frame=subset,
                group_name="hydroclimate_group",
                group_value=group,
            )
        )

    hydro = pd.DataFrame(
        hydro_rows
    )

    hydro_path = (
        output_dir
        / "ch4b_pub_fhv2_hydroclimate_summary.csv"
    )
    hydro.to_csv(
        hydro_path,
        index=False,
    )

    pairwise = build_pairwise_statistics(
        frame=per_basin,
        bootstrap_reps=(
            args.bootstrap_reps
        ),
        seed=args.seed,
    )

    pairwise_path = (
        output_dir
        / "ch4b_pub_fhv2_pairwise_statistics.csv"
    )
    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    n_high_values = sorted(
        per_basin[
            "n_highflow_days"
        ].dropna().unique().tolist()
    )

    metadata = {
        "analysis": "Chapter 4B PUB FHV2%",
        "run_root": str(run_root),
        "climate_file": str(
            climate_file
        ),
        "n_basins": int(
            len(per_basin)
        ),
        "high_flow_fraction": float(
            args.high_flow_fraction
        ),
        "high_flow_percent": float(
            100.0
            * args.high_flow_fraction
        ),
        "high_flow_day_counts":
            n_high_values,
        "fhv_definition": (
            "Observed and simulated discharge "
            "are independently sorted in "
            "descending order. FHV is the "
            "percent volume bias over the "
            "highest 2% of the FDC."
        ),
        "signed_fhv_interpretation": {
            "negative":
                "high-flow underestimation",
            "zero":
                "unbiased high-flow volume",
            "positive":
                "high-flow overestimation",
        },
        "paired_metric":
            "absolute FHV2",
        "paired_delta_definition":
            "|FHV|_model - |FHV|_reference",
        "paired_improvement":
            "negative delta",
        "bootstrap_repetitions":
            int(args.bootstrap_reps),
        "bootstrap_seed":
            int(args.seed),
        "significance":
            (
                "Two-sided paired Wilcoxon "
                "signed-rank test with BH-FDR "
                "across the three overall "
                "|FHV| comparisons. Robust "
                "significance additionally "
                "requires the bootstrap 95% "
                "CI of paired median delta "
                "to exclude zero."
            ),
        "hydroclimate_rules": {
            "Snow": "frac_snow > 0.20",
            "Wet":
                "remaining basins: aridity < 1.0",
            "Dry":
                "remaining basins: aridity >= 1.0",
        },
    }

    metadata_path = (
        output_dir
        / "ch4b_pub_fhv2_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 78)
    print("OVERALL FHV2% SUMMARY")
    print("=" * 78)

    display_columns = [
        "model",
        "n",
        "median_FHV2",
        "median_abs_FHV2",
        "q25_abs_FHV2",
        "q75_abs_FHV2",
        "underestimation_rate",
    ]

    print(
        overall[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 78)
    print("HYDROCLIMATE FHV2% SUMMARY")
    print("=" * 78)

    print(
        hydro[
            [
                "hydroclimate_group",
                "model",
                "n",
                "median_FHV2",
                "median_abs_FHV2",
                "underestimation_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 78)
    print("PAIRED |FHV2%| STATISTICS")
    print("=" * 78)

    print(
        pairwise[
            [
                "comparison",
                "n",
                "median_delta",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "better_rate",
                "wilcoxon_p_fdr",
                "robust_significant",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\nOutputs:")
    for path in (
        per_basin_path,
        overall_path,
        hydro_path,
        pairwise_path,
        metadata_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
