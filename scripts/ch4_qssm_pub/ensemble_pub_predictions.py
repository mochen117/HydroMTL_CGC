#!/usr/bin/env python3
"""Average physical-domain PUB streamflow predictions across seeds.

The formal ensemble is computed at the prediction level and basin metrics are
recalculated afterwards.  This avoids averaging nonlinear skill metrics such as
NSE or KGE across random initializations.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.native_runtime import bootstrap_native_runtime  # noqa: E402

bootstrap_native_runtime(strict=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    load_json,
    load_yaml,
    normalize_basin_id,
    resolve_project_path,
)
from mtl_cgc.protocols.ch4_qssm_pub.paths import ENSEMBLE_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ENSEMBLE_DIR,
    )
    parser.add_argument(
        "--require-seeds",
        type=int,
        default=0,
        help=(
            "Fail unless each fold/scenario has this many seeds. "
            "Use 5 for the formal multi-seed ensemble."
        ),
    )
    return parser.parse_args()


def _find_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of the expected variables exist: {candidates}")


def _basin_labels(ds: xr.Dataset, sim: xr.DataArray) -> list[str]:
    for name in ("basin_id", "gauge_id", "gage_id", "basin"):
        if name in ds.coords and ds.coords[name].size == sim.shape[0]:
            return [normalize_basin_id(v) for v in ds.coords[name].values.tolist()]
    if "basin" in sim.dims:
        return [normalize_basin_id(v) for v in ds["basin"].values.tolist()]
    raise ValueError("Unable to identify basin labels in PUB NetCDF.")


def _time_values(ds: xr.Dataset, sim: xr.DataArray) -> np.ndarray:
    for name in ("time", "date", "target_date"):
        if name in ds.coords and ds.coords[name].size == sim.shape[1]:
            return ds.coords[name].values
    raise ValueError("Unable to identify time coordinate in PUB NetCDF.")


def _metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(sim)
    o = np.asarray(obs[mask], dtype=float)
    p = np.asarray(sim[mask], dtype=float)
    if o.size < 2:
        return {key: np.nan for key in ("nse", "kge", "rmse", "mae", "bias", "corr")}

    residual = p - o
    denominator = np.sum((o - np.mean(o)) ** 2)
    nse = 1.0 - np.sum(residual**2) / denominator if denominator > 1e-12 else np.nan
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    obs_sum = np.sum(o)
    bias = float(np.sum(residual) / (obs_sum + 1e-8)) if abs(obs_sum) > 1e-8 else np.nan

    if np.std(o) < 1e-12 or np.std(p) < 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(o, p)[0, 1])

    mean_o = np.mean(o)
    std_o = np.std(o)
    if not np.isfinite(corr) or abs(mean_o) < 1e-12 or std_o < 1e-12:
        kge = np.nan
    else:
        alpha = np.std(p) / std_o
        beta = np.mean(p) / mean_o
        kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    return {
        "nse": float(nse),
        "kge": kge,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "corr": corr,
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[int, str], list[tuple[int, Path]]] = defaultdict(list)
    for manifest_arg in args.manifests:
        manifest_path = resolve_project_path(manifest_arg, PROJECT_ROOT)
        manifest = load_json(manifest_path)
        for entry in manifest["entries"]:
            if entry.get("group") != "core":
                continue
            cfg_path = resolve_project_path(entry["config"], PROJECT_ROOT)
            cfg = load_yaml(cfg_path)
            save_root = Path(cfg["experiment"].get("save_dir", "experiments"))
            if not save_root.is_absolute():
                save_root = PROJECT_ROOT / save_root
            nc_path = save_root / cfg["experiment"]["name"] / "test_predictions_and_weights.nc"
            grouped[(int(entry["fold_id"]), str(entry["scenario"]))].append(
                (int(entry["seed"]), nc_path)
            )

    index_rows: list[dict[str, object]] = []
    for (fold_id, scenario), records in sorted(grouped.items()):
        records = sorted(records)
        if args.require_seeds and len(records) != args.require_seeds:
            raise RuntimeError(
                f"fold={fold_id}, scenario={scenario}: found {len(records)} seeds, "
                f"required {args.require_seeds}."
            )
        if not records:
            continue

        sims: list[np.ndarray] = []
        obs_ref: np.ndarray | None = None
        basins_ref: list[str] | None = None
        time_ref: np.ndarray | None = None

        for seed, nc_path in records:
            if not nc_path.exists():
                raise FileNotFoundError(nc_path)
            with xr.open_dataset(nc_path) as ds:
                sim_name = _find_var(ds, ("streamflow_sim", "streamflow_pred", "pred_streamflow"))
                obs_name = _find_var(ds, ("streamflow_obs", "streamflow_target", "obs_streamflow"))
                sim_da = ds[sim_name]
                obs_da = ds[obs_name]
                sim = np.asarray(sim_da.values, dtype=float)
                obs = np.asarray(obs_da.values, dtype=float)
                basins = _basin_labels(ds, sim_da)
                time_values = _time_values(ds, sim_da)

            if sim.ndim != 2 or obs.shape != sim.shape:
                raise ValueError(
                    f"Unexpected Q array shape in {nc_path}: "
                    f"sim={sim.shape}, obs={obs.shape}"
                )
            if basins_ref is None:
                basins_ref, time_ref, obs_ref = basins, time_values, obs
            else:
                if basins != basins_ref:
                    raise ValueError(
                        "Basin order differs across seeds for "
                        f"fold={fold_id}, scenario={scenario}."
                    )
                if not np.array_equal(time_values, time_ref):
                    raise ValueError(
                        "Time axis differs across seeds for "
                        f"fold={fold_id}, scenario={scenario}."
                    )
                if not np.allclose(obs, obs_ref, equal_nan=True, rtol=0.0, atol=1e-10):
                    raise ValueError(
                        "Observed Q differs across seeds for "
                        f"fold={fold_id}, scenario={scenario}."
                    )
            sims.append(sim)

        assert obs_ref is not None and basins_ref is not None and time_ref is not None
        ensemble_sim = np.nanmean(np.stack(sims, axis=0), axis=0)

        fold_dir = out_dir / f"fold{fold_id:02d}" / scenario
        fold_dir.mkdir(parents=True, exist_ok=True)
        ds_out = xr.Dataset(
            data_vars={
                "streamflow_sim": (("basin", "time"), ensemble_sim),
                "streamflow_obs": (("basin", "time"), obs_ref),
            },
            coords={"basin": basins_ref, "time": time_ref},
            attrs={
                "ensemble_method": "arithmetic mean of physical-domain predictions",
                "seed_count": len(records),
                "seeds": ",".join(str(seed) for seed, _ in records),
            },
        )
        nc_out = fold_dir / "ensemble_test_predictions.nc"
        ds_out.to_netcdf(nc_out)

        metric_rows = []
        for basin_idx, basin_id in enumerate(basins_ref):
            values = _metrics(obs_ref[basin_idx], ensemble_sim[basin_idx])
            metric_rows.append(
                {
                    "gauge_id": basin_id,
                    **{f"streamflow_{key}": value for key, value in values.items()},
                }
            )
        csv_out = fold_dir / "ensemble_test_per_basin_metrics.csv"
        pd.DataFrame(metric_rows).to_csv(csv_out, index=False)

        index_rows.append(
            {
                "fold_id": fold_id,
                "scenario": scenario,
                "seed_count": len(records),
                "seeds": ",".join(str(seed) for seed, _ in records),
                "metrics_csv": str(csv_out),
                "predictions_nc": str(nc_out),
            }
        )
        print(f"Ensemble exported: fold={fold_id:02d} scenario={scenario} seeds={len(records)}")

    pd.DataFrame(index_rows).to_csv(out_dir / "ensemble_index.csv", index=False)
    print(f"Ensemble outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
