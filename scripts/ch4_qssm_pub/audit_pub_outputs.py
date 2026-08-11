#!/usr/bin/env python3
"""Audit target-only test outputs for Chapter 4B spatial PUB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.native_runtime import bootstrap_native_runtime  # noqa: E402

bootstrap_native_runtime(strict=True)

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    load_json,
    load_yaml,
    normalize_basin_id,
    read_basin_ids,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def metric_basin_ids(csv_path: Path) -> set[str]:
    frame = pd.read_csv(csv_path)
    for candidate in ("gauge_id", "basin_id", "gage_id", "Unnamed: 0", frame.columns[0]):
        if candidate in frame.columns:
            return {normalize_basin_id(v) for v in frame[candidate].tolist()}
    raise ValueError(f"Cannot identify basin ids in {csv_path}")


def netcdf_basin_ids(dataset: xr.Dataset) -> set[str]:
    for candidate in ("basin_id", "gauge_id", "gage_id", "basin"):
        if candidate in dataset.coords:
            return {normalize_basin_id(v) for v in dataset.coords[candidate].values.tolist()}
    raise ValueError("Prediction NetCDF has no recognizable basin coordinate.")


def validate_dates(dataset: xr.Dataset, expected_start: str, expected_end: str) -> None:
    for name in ("date", "time", "target_date"):
        if name in dataset.coords:
            values = pd.to_datetime(dataset.coords[name].values)
            actual = (values.min().strftime("%Y-%m-%d"), values.max().strftime("%Y-%m-%d"))
            expected = (expected_start, expected_end)
            if actual != expected:
                raise ValueError(f"Unexpected test dates: {actual}; expected {expected}.")
            return
    raise ValueError("Prediction NetCDF has no recognizable date coordinate.")


def validate_q_variables(dataset: xr.Dataset) -> None:
    sim_candidates = {"streamflow_sim", "streamflow_pred", "pred_streamflow"}
    obs_candidates = {"streamflow_obs", "streamflow_target", "obs_streamflow"}
    if not (sim_candidates & set(dataset.data_vars)):
        raise ValueError("Missing streamflow simulation variable in NetCDF.")
    if not (obs_candidates & set(dataset.data_vars)):
        raise ValueError("Missing streamflow observation variable in NetCDF.")


def main() -> None:
    args = parse_args()
    manifest = load_json(resolve_project_path(args.manifest, PROJECT_ROOT))
    errors: list[str] = []

    for entry in manifest["entries"]:
        config_path = resolve_project_path(entry["config"], PROJECT_ROOT)
        config = load_yaml(config_path)
        name = str(config["experiment"]["name"])
        save_root = Path(config["experiment"].get("save_dir", "experiments"))
        if not save_root.is_absolute():
            save_root = PROJECT_ROOT / save_root
        experiment_dir = save_root / name
        csv_path = experiment_dir / "test_per_basin_metrics.csv"
        nc_path = experiment_dir / "test_predictions_and_weights.nc"

        try:
            if not csv_path.exists() or not nc_path.exists():
                raise FileNotFoundError(
                    f"Missing test outputs: csv={csv_path.exists()}, nc={nc_path.exists()}"
                )

            target_file = resolve_project_path(config["pub"]["target_basin_file"], PROJECT_ROOT)
            expected_basins = set(read_basin_ids(target_file))
            csv_basins = metric_basin_ids(csv_path)

            with xr.open_dataset(nc_path) as dataset:
                nc_basins = netcdf_basin_ids(dataset)
                validate_dates(
                    dataset,
                    str(config["data"]["test_period"][0]),
                    str(config["data"]["test_period"][1]),
                )
                validate_q_variables(dataset)

            extra_csv = sorted(csv_basins - expected_basins)
            extra_nc = sorted(nc_basins - expected_basins)
            missing_csv = sorted(expected_basins - csv_basins)
            missing_nc = sorted(expected_basins - nc_basins)

            if extra_csv or extra_nc:
                raise ValueError(
                    f"Non-target basins found. CSV={extra_csv[:10]}, NC={extra_nc[:10]}"
                )
            if args.strict and (missing_csv or missing_nc):
                raise ValueError(
                    f"Missing target basins. CSV={missing_csv[:10]}, NC={missing_nc[:10]}"
                )

            frame = pd.read_csv(csv_path)
            if "streamflow_nse" not in frame.columns:
                raise ValueError("test_per_basin_metrics.csv lacks streamflow_nse.")

            print(
                f"PASS fold={int(entry['fold_id']):02d} "
                f"scenario={entry['scenario']:<18s} "
                f"csv={len(csv_basins):3d} nc={len(nc_basins):3d}"
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")

    if errors:
        raise RuntimeError(f"PUB output audit failed for {len(errors)} experiments.")
    print("PUB output audit passed.")


if __name__ == "__main__":
    main()
