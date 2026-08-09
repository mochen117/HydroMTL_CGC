# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: CAMELS NetCDF extraction and DataLoader construction with
# context-aware N-to-1 alignment and leak-safe scaler fitting.
# ==============================================================================

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtl_cgc.data.data_scalers import HydroScaler
from mtl_cgc.data.data_sets import HydroDataset
from mtl_cgc.utils.temporal import (
    expand_period_for_sequence,
    is_missing_period,
    normalize_period,
)


RawDataDict = Dict[str, Any]
LoaderBundle = Tuple[
    Optional[DataLoader],
    Optional[DataLoader],
    Optional[DataLoader],
    HydroScaler,
]


def impute_dynamic_features(
    dyn_array: np.ndarray,
    causal: bool = False,
) -> np.ndarray:
    """
    Fill missing meteorological drivers.

    Set ``data.causal_dynamic_imputation=true`` for strict forecasting studies.
    Non-causal interpolation is retained as an explicit option for simulation
    settings in which the complete forcing record is available beforehand.
    """
    frame = pd.DataFrame(dyn_array)
    if causal:
        frame = frame.interpolate(method="linear", limit_direction="forward")
        frame = frame.ffill()
    else:
        frame = frame.interpolate(method="linear", limit_direction="both")
        frame = frame.ffill().bfill()

    return frame.fillna(0.0).to_numpy(dtype=np.float32)


def _read_static_value(
    dataset: xr.Dataset,
    var_name: str,
    basin_id: str,
) -> float:
    """Read one static numerical feature from a variable or attribute."""
    value = np.nan
    if var_name in dataset:
        value = float(dataset[var_name].values.item())
    elif var_name in dataset.attrs:
        value = float(dataset.attrs[var_name])

    if var_name in {"area_gages2", "p_mean"}:
        if np.isfinite(value) and value <= 0:
            raise ValueError(
                f"Non-physical static feature: basin={basin_id}, "
                f"feature={var_name}, value={value}. Expected a positive value."
            )

    return value


def _read_categorical_value(dataset: xr.Dataset, var_name: str) -> int:
    """Read one static categorical feature from a variable or attribute."""
    if var_name in dataset:
        raw_value = dataset[var_name].values.item()
        return 0 if pd.isna(raw_value) else int(raw_value)

    if var_name in dataset.attrs:
        try:
            return int(dataset.attrs[var_name])
        except (TypeError, ValueError):
            return 0

    return 0


def _validate_daily_time_axis(
    time_values: np.ndarray,
    period: Sequence[str],
    basin_id: str,
    split_name: str,
) -> None:
    """Require a complete, unique, daily time coordinate for one raw slice."""
    actual = pd.DatetimeIndex(pd.to_datetime(time_values))
    expected = pd.date_range(period[0], period[1], freq="D")

    if not actual.is_monotonic_increasing:
        raise ValueError(
            f"Non-monotonic time coordinate: basin={basin_id}, split={split_name}."
        )
    if not actual.is_unique:
        raise ValueError(
            f"Duplicate time coordinates: basin={basin_id}, split={split_name}."
        )
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise ValueError(
            f"Incomplete daily time axis for basin={basin_id}, split={split_name}, "
            f"period={list(period)}. Missing={len(missing)}, extra={len(extra)}, "
            f"missing_examples={missing[:3].strftime('%Y-%m-%d').tolist()}."
        )


def _load_single_basin(
    nc_path: Path,
    basin_id: str,
    data_cfg: Dict[str, Any],
    split_period: Sequence[str],
    split_name: str,
    ungauged_basins: Optional[List[str]],
    mask_target: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Read one basin and slice the requested inclusive raw period."""
    period = normalize_period(split_period, name=f"{split_name}_read_period")
    start_dt = np.datetime64(period[0])
    end_dt = np.datetime64(period[1])
    causal_imputation = bool(
        data_cfg.get("causal_dynamic_imputation", False)
    )

    with xr.open_dataset(nc_path) as dataset:
        static_num = np.asarray(
            [
                _read_static_value(dataset, feature, basin_id)
                for feature in data_cfg["static_features"]
            ],
            dtype=np.float32,
        )
        static_cat = np.asarray(
            [
                _read_categorical_value(dataset, feature)
                for feature in data_cfg.get("categorical_static_features", [])
            ],
            dtype=np.int64,
        )

        sliced = dataset.sel(time=slice(start_dt, end_dt))
        if sliced.sizes.get("time", 0) == 0:
            raise ValueError(
                f"Empty time slice: basin={basin_id}, split={split_name}, "
                f"period={period}."
            )

        _validate_daily_time_axis(
            time_values=sliced["time"].values,
            period=period,
            basin_id=basin_id,
            split_name=split_name,
        )

        dynamic = np.stack(
            [sliced[feature].values for feature in data_cfg["dynamic_features"]],
            axis=-1,
        )
        dynamic = impute_dynamic_features(
            dynamic,
            causal=causal_imputation,
        )

        targets: Dict[str, np.ndarray] = {}
        ungauged_set = set(ungauged_basins or [])
        normalized_mask_target = str(mask_target).lower()

        for target_cfg in data_cfg["targets"]:
            source_name = target_cfg["name"]
            task_name = str(source_name).lower()
            values = sliced[source_name].values.astype(np.float32)

            if (
                split_name == "train"
                and basin_id in ungauged_set
                and task_name == normalized_mask_target
            ):
                values = np.full_like(values, np.nan)

            targets[task_name] = values

    return dynamic, static_num, static_cat, targets


def load_nc_to_dict(
    data_root: Path,
    basin_ids: List[str],
    data_cfg: Dict[str, Any],
    split_period: Sequence[str],
    split_name: str,
    ungauged_basins: Optional[List[str]] = None,
    mask_target: str = "streamflow",
) -> RawDataDict:
    """Load multiple basin NetCDF files into dense aligned arrays."""
    if not basin_ids:
        raise ValueError(
            f"No basin ids were supplied for split='{split_name}'."
        )

    dynamic_list: List[np.ndarray] = []
    static_num_list: List[np.ndarray] = []
    static_cat_list: List[np.ndarray] = []
    target_lists = {
        str(target["name"]).lower(): []
        for target in data_cfg["targets"]
    }

    show_progress = bool(data_cfg.get("show_loading_progress", False))
    progress = tqdm(
        basin_ids,
        desc=f"Load {split_name.upper():>7}",
        leave=False,
        disable=not show_progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )

    expected_time_steps: Optional[int] = None
    for basin_id in progress:
        nc_path = data_root / f"gage_{basin_id}.nc"
        if not nc_path.exists():
            raise FileNotFoundError(f"Missing basin file: {nc_path}")

        dynamic, static_num, static_cat, targets = _load_single_basin(
            nc_path=nc_path,
            basin_id=basin_id,
            data_cfg=data_cfg,
            split_period=split_period,
            split_name=split_name,
            ungauged_basins=ungauged_basins,
            mask_target=mask_target,
        )

        if expected_time_steps is None:
            expected_time_steps = int(dynamic.shape[0])
        elif dynamic.shape[0] != expected_time_steps:
            raise ValueError(
                f"Time-length mismatch in split='{split_name}': basin={basin_id} "
                f"has {dynamic.shape[0]} steps, expected {expected_time_steps}."
            )

        dynamic_list.append(dynamic)
        static_num_list.append(static_num)
        static_cat_list.append(static_cat)
        for task_name, values in targets.items():
            target_lists[task_name].append(values)

    if show_progress:
        print("", flush=True)

    return {
        "dyn": np.stack(dynamic_list),
        "s_num": np.stack(static_num_list),
        "s_cat": (
            np.stack(static_cat_list)
            if static_cat_list and static_cat_list[0].size > 0
            else None
        ),
        "y_dict": {
            task_name: np.stack(values)
            for task_name, values in target_lists.items()
        },
    }


def assert_temporal_splits(data_cfg: Dict[str, Any]) -> None:
    """
    Validate configured target periods.

    Temporal experiments require strictly ordered, non-overlapping target
    periods. PUB experiments may reuse the same dates because train and test
    leakage is controlled by disjoint basin sets instead of temporal ordering.
    """
    train_period = normalize_period(
        data_cfg["train_period"],
        name="train_period",
    )
    test_period = normalize_period(
        data_cfg["test_period"],
        name="test_period",
    )
    val_period = data_cfg.get("val_period")

    train_start, train_end = map(pd.to_datetime, train_period)
    test_start, test_end = map(pd.to_datetime, test_period)

    if bool(data_cfg.get("spatial_split", False)):
        if not is_missing_period(val_period):
            normalize_period(val_period, name="val_period")
        return

    if is_missing_period(val_period):
        if train_end >= test_start:
            raise ValueError(
                "Invalid temporal split. Expected train_end < test_start for "
                f"train={train_period}, test={test_period}."
            )
        return

    normalized_val = normalize_period(val_period, name="val_period")
    val_start, val_end = map(pd.to_datetime, normalized_val)
    if not (train_end < val_start <= val_end < test_start):
        raise ValueError(
            "Invalid temporal split. Expected "
            "train_end < val_start <= val_end < test_start, but got "
            f"train={train_period}, val={normalized_val}, test={test_period}."
        )


def _fit_scaler(raw_data: RawDataDict, data_cfg: Dict[str, Any]) -> HydroScaler:
    """Fit HydroScaler on the supplied basin-time arrays only."""
    scaler = HydroScaler(data_cfg)
    scaler.fit_transform(
        raw_data["dyn"],
        raw_data["s_num"],
        raw_data.get("s_cat"),
        raw_data["y_dict"],
    )
    return scaler


def _target_only_view(
    raw_data: RawDataDict,
    sequence_length: int,
) -> RawDataDict:
    """
    Return the target-period portion of a context-expanded raw dictionary.

    This view is used only for scaler fitting. Historical context remains
    available to the model but does not influence normalization statistics.
    """
    context_steps = int(sequence_length) - 1
    if context_steps < 0:
        raise ValueError(
            f"sequence_length must be positive, got {sequence_length}."
        )

    return {
        "dyn": raw_data["dyn"][:, context_steps:, :],
        "s_num": raw_data["s_num"],
        "s_cat": raw_data.get("s_cat"),
        "y_dict": {
            task_name: values[:, context_steps:]
            for task_name, values in raw_data["y_dict"].items()
        },
    }


def seed_worker(worker_id: int) -> None:
    """Set deterministic NumPy and Python seeds in a DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(
    config: Dict[str, Any],
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator,
) -> Dict[str, Any]:
    """Build stable DataLoader options for NetCDF-backed datasets."""
    data_cfg = config["data"]
    num_workers = int(data_cfg.get("num_workers", 0))

    kwargs: Dict[str, Any] = {
        "batch_size": int(data_cfg.get("batch_size", 64)),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "num_workers": num_workers,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(
            data_cfg.get("persistent_workers", False)
        )

    return kwargs


def get_hydro_dataloaders(
    config: Dict[str, Any],
    basin_ids: List[str],
    mode: str = "train",
    ungauged_basins: Optional[List[str]] = None,
    mask_target: str = "streamflow",
    scaler_basin_ids: Optional[List[str]] = None,
) -> LoaderBundle:
    """
    Construct context-aware N-to-1 DataLoaders with leak-safe scaling.

    Configured periods always denote target dates. Raw dynamic inputs are read
    from ``target_start - sequence_length + 1`` through ``target_end``. Scaler
    statistics are fitted only on the configured training target period and,
    under PUB evaluation, only on training basins.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"train", "test"}:
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected 'train' or 'test'."
        )

    data_cfg = config["data"]
    data_root = Path(data_cfg["data_root"])
    sequence_length = int(data_cfg.get("sequence_length", 180))
    if sequence_length <= 0:
        raise ValueError(
            f"Invalid sequence_length={sequence_length}. It must be positive."
        )

    assert_temporal_splits(data_cfg)
    sorted_basin_ids = sorted(str(basin_id) for basin_id in basin_ids)
    if not sorted_basin_ids:
        raise FileNotFoundError(
            f"No basin ids were provided. Check data_root={data_root.resolve()} "
            "and the spatial split configuration."
        )

    seed = int(config.get("reproducibility", {}).get("seed", 42))
    generator = torch.Generator().manual_seed(seed)

    if normalized_mode == "train":
        train_period = normalize_period(
            data_cfg["train_period"],
            name="train_period",
        )
        train_read_period = expand_period_for_sequence(
            train_period,
            sequence_length,
        )
        train_raw = load_nc_to_dict(
            data_root=data_root,
            basin_ids=sorted_basin_ids,
            data_cfg=data_cfg,
            split_period=train_read_period,
            split_name="train",
            ungauged_basins=ungauged_basins,
            mask_target=mask_target,
        )

        scaler = _fit_scaler(
            _target_only_view(train_raw, sequence_length),
            data_cfg,
        )
        train_dataset = HydroDataset(
            raw_data=train_raw,
            data_params=data_cfg,
            basin_ids=sorted_basin_ids,
            target_period=train_period,
            mode="train",
            scaler=scaler,
        )
        train_dataset.config = config
        train_loader = DataLoader(
            train_dataset,
            **_loader_kwargs(
                config,
                shuffle=True,
                drop_last=True,
                generator=generator,
            ),
        )

        val_loader: Optional[DataLoader] = None
        val_period = data_cfg.get("val_period")
        if not is_missing_period(val_period):
            normalized_val_period = normalize_period(
                val_period,
                name="val_period",
            )
            val_read_period = expand_period_for_sequence(
                normalized_val_period,
                sequence_length,
            )
            valid_raw = load_nc_to_dict(
                data_root=data_root,
                basin_ids=sorted_basin_ids,
                data_cfg=data_cfg,
                split_period=val_read_period,
                split_name="valid",
                ungauged_basins=None,
                mask_target=mask_target,
            )
            val_dataset = HydroDataset(
                raw_data=valid_raw,
                data_params=data_cfg,
                basin_ids=sorted_basin_ids,
                target_period=normalized_val_period,
                mode="valid",
                scaler=scaler,
            )
            val_dataset.config = config
            val_loader = DataLoader(
                val_dataset,
                **_loader_kwargs(
                    config,
                    shuffle=False,
                    drop_last=False,
                    generator=generator,
                ),
            )

        return train_loader, val_loader, None, scaler

    evaluation_basin_ids = sorted_basin_ids
    scaler_ids = sorted(
        str(basin_id)
        for basin_id in (
            scaler_basin_ids
            if scaler_basin_ids is not None
            else evaluation_basin_ids
        )
    )
    if not scaler_ids:
        raise ValueError("scaler_basin_ids is empty in test mode.")

    if scaler_basin_ids is not None and bool(data_cfg.get("spatial_split", False)):
        overlap = set(evaluation_basin_ids).intersection(scaler_ids)
        if overlap:
            raise ValueError(
                "Scaler leakage risk under spatial-split evaluation: test basins "
                "must not be used for scaler fitting. "
                f"Overlap count={len(overlap)}, examples={sorted(overlap)[:5]}."
            )

    train_period = normalize_period(
        data_cfg["train_period"],
        name="train_period",
    )
    scaler_raw = load_nc_to_dict(
        data_root=data_root,
        basin_ids=scaler_ids,
        data_cfg=data_cfg,
        split_period=train_period,
        split_name="scaler",
        ungauged_basins=None,
        mask_target=mask_target,
    )
    scaler = _fit_scaler(scaler_raw, data_cfg)

    test_period = normalize_period(
        data_cfg["test_period"],
        name="test_period",
    )
    test_read_period = expand_period_for_sequence(
        test_period,
        sequence_length,
    )
    test_raw = load_nc_to_dict(
        data_root=data_root,
        basin_ids=evaluation_basin_ids,
        data_cfg=data_cfg,
        split_period=test_read_period,
        split_name="test",
        ungauged_basins=ungauged_basins,
        mask_target=mask_target,
    )
    test_dataset = HydroDataset(
        raw_data=test_raw,
        data_params=data_cfg,
        basin_ids=evaluation_basin_ids,
        target_period=test_period,
        mode="test",
        scaler=scaler,
    )
    test_dataset.config = config
    test_loader = DataLoader(
        test_dataset,
        **_loader_kwargs(
            config,
            shuffle=False,
            drop_last=False,
            generator=generator,
        ),
    )

    return None, None, test_loader, scaler
