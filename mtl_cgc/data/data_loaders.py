# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Data extractor and DataLoader pipeline for CAMELS NetCDF files.
# Provides leak-safe scaler fitting for PUB experiments and clean data-loading
# progress display.
# ==============================================================================

import sys
import torch
import random
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional, Any
from torch.utils.data import DataLoader

from mtl_cgc.data.data_sets import HydroDataset
from mtl_cgc.data.data_scalers import HydroScaler


def impute_dynamic_features(dyn_array: np.ndarray, causal: bool = False) -> np.ndarray:
    """
    Fill missing meteorological drivers.

    Dynamic meteorological forcings are treated as known external drivers within
    the selected period. If strict causal dynamic imputation is required, set
    config.data.causal_dynamic_imputation = true.
    """
    df = pd.DataFrame(dyn_array)
    if causal:
        df = df.interpolate(method="linear", limit_direction="forward")
        df = df.ffill()
    else:
        df = df.interpolate(method="linear", limit_direction="both")
        df = df.ffill().bfill()
    df = df.fillna(0.0)
    return df.values.astype(np.float32)


def _read_static_value(ds: xr.Dataset, var_name: str, basin_id: str) -> float:
    """Read a static numerical feature from variables or attributes."""
    val = np.nan
    if var_name in ds:
        val = float(ds[var_name].values.item())
    elif var_name in ds.attrs:
        val = float(ds.attrs[var_name])

    if var_name in ["area_gages2", "p_mean"] and np.isfinite(val) and val <= 0:
        raise ValueError(
            f"Non-physical static feature detected: basin={basin_id}, "
            f"feature={var_name}, value={val}. Expected a positive value."
        )
    return val


def _read_categorical_value(ds: xr.Dataset, var_name: str) -> int:
    """Read a static categorical feature from variables or attributes."""
    val = 0
    if var_name in ds:
        raw_val = ds[var_name].values.item()
        if not pd.isna(raw_val):
            val = int(raw_val)
    elif var_name in ds.attrs:
        try:
            val = int(ds.attrs[var_name])
        except (ValueError, TypeError):
            pass
    return val


def _load_single_basin(
    nc_path: Path,
    basin_id: str,
    data_cfg: Dict[str, Any],
    split_period: List[str],
    split_name: str,
    ungauged_basins: Optional[List[str]],
    mask_target: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load one basin file and slice it into the requested time period."""
    start_dt, end_dt = np.datetime64(split_period[0]), np.datetime64(split_period[1])
    causal_imputation = bool(data_cfg.get("causal_dynamic_imputation", False))

    with xr.open_dataset(nc_path) as ds:
        s_num = np.array(
            [_read_static_value(ds, v, basin_id) for v in data_cfg["static_features"]],
            dtype=np.float32,
        )

        s_cat_raw = [
            _read_categorical_value(ds, v)
            for v in data_cfg.get("categorical_static_features", [])
        ]
        s_cat = np.array(s_cat_raw, dtype=np.int64)

        ds_split = ds.sel(time=slice(start_dt, end_dt))
        if len(ds_split.time) == 0:
            raise ValueError(
                f"Empty time slice for basin={basin_id}, split={split_name}, "
                f"period={split_period}."
            )

        dyn = np.stack([ds_split[v].values for v in data_cfg["dynamic_features"]], axis=-1)
        dyn = impute_dynamic_features(dyn, causal=causal_imputation)

        y_dict = {}
        for target in data_cfg["targets"]:
            target_original = target["name"]
            target_name = str(target_original).lower()
            vals = ds_split[target_original].values.astype(np.float32)

            if split_name == "train" and ungauged_basins is not None:
                if basin_id in ungauged_basins and target_name == mask_target:
                    vals = np.full_like(vals, np.nan)

            y_dict[target_name] = vals

    return dyn, s_num, s_cat, y_dict


def load_nc_to_dict(
    data_root: Path,
    basin_ids: List[str],
    data_cfg: Dict[str, Any],
    split_period: List[str],
    split_name: str,
    ungauged_basins: Optional[List[str]] = None,
    mask_target: str = "streamflow",
) -> Dict[str, Any]:
    """Load a list of basin NetCDF files into dense arrays for HydroDataset."""
    if not basin_ids:
        raise ValueError(f"No basin ids passed to load_nc_to_dict for split={split_name}.")

    dyn_list, s_num_list, s_cat_list = [], [], []
    y_dict = {str(t["name"]).lower(): [] for t in data_cfg["targets"]}

    show_loading_progress = data_cfg.get("show_loading_progress", False)

    pbar = tqdm(
        basin_ids,
        desc=f"Load {split_name.upper():>5}",
        leave=False,
        disable=not show_loading_progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )

    for basin_id in pbar:
        nc_path = data_root / f"gage_{basin_id}.nc"
        if not nc_path.exists():
            raise FileNotFoundError(f"Missing basin file: {nc_path}")

        dyn, s_num, s_cat, target_values = _load_single_basin(
            nc_path=nc_path,
            basin_id=basin_id,
            data_cfg=data_cfg,
            split_period=split_period,
            split_name=split_name,
            ungauged_basins=ungauged_basins,
            mask_target=mask_target,
        )

        dyn_list.append(dyn)
        s_num_list.append(s_num)
        s_cat_list.append(s_cat)
        for task_name, vals in target_values.items():
            y_dict[task_name].append(vals)

    if show_loading_progress:
        print("", flush=True)

    return {
        "dyn": np.stack(dyn_list),
        "s_num": np.stack(s_num_list),
        "s_cat": np.stack(s_cat_list) if len(s_cat_list[0]) > 0 else None,
        "y_dict": {k: np.stack(v) for k, v in y_dict.items()},
    }

def assert_temporal_splits(data_cfg: Dict[str, Any]) -> None:
    """Validate strict non-overlapping temporal splits."""
    train_start, train_end = map(pd.to_datetime, data_cfg["train_period"])
    val_start, val_end = map(pd.to_datetime, data_cfg["val_period"])
    test_start, test_end = map(pd.to_datetime, data_cfg["test_period"])

    if not (train_start <= train_end < val_start <= val_end < test_start <= test_end):
        raise ValueError(
            "Temporal leakage risk detected. Expected "
            "train_start <= train_end < val_start <= val_end < test_start <= test_end, "
            f"but got train={data_cfg['train_period']}, "
            f"val={data_cfg['val_period']}, test={data_cfg['test_period']}."
        )

def seed_worker(worker_id: int) -> None:
    """Set deterministic worker seeds for PyTorch DataLoader processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(
    config: Dict[str, Any],
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator,
) -> Dict[str, Any]:
    """
    Build DataLoader keyword arguments.

    NetCDF-backed hydrological datasets are often more stable with
    num_workers=0 under spawn multiprocessing, especially on shared servers.
    """
    data_cfg = config["data"]

    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 64)),
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))

    return kwargs


def get_hydro_dataloaders(
    config: Dict[str, Any],
    basin_ids: List[str],
    mode: str = "train",
    ungauged_basins: Optional[List[str]] = None,
    mask_target: str = "streamflow",
    scaler_basin_ids: Optional[List[str]] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], HydroScaler]:
    """
    Construct DataLoaders with leak-safe scaler fitting.

    Parameters
    ----------
    basin_ids:
        Basins used for the requested split. In train mode, these are training
        basins. In test mode, these are evaluation/test basins.
    scaler_basin_ids:
        Basins used only for fitting the scaler in test mode. For PUB tests,
        this must be the training basin set, not the test basin set.
    """
    data_cfg = config["data"]
    data_root = Path(data_cfg["data_root"])
    assert_temporal_splits(data_cfg)

    basin_ids = sorted([str(b) for b in basin_ids])
    if len(basin_ids) == 0:
        raise FileNotFoundError(
            f"No basin ids were provided. Please check data_root={data_root.resolve()} "
            "and the spatial split configuration."
        )

    seed = int(config.get("reproducibility", {}).get("seed", 42))
    generator = torch.Generator()
    generator.manual_seed(seed)

    if mode == "train":
        train_raw = load_nc_to_dict(
            data_root=data_root,
            basin_ids=basin_ids,
            data_cfg=data_cfg,
            split_period=data_cfg["train_period"],
            split_name="train",
            ungauged_basins=ungauged_basins,
            mask_target=mask_target,
        )
        valid_raw = load_nc_to_dict(
            data_root=data_root,
            basin_ids=basin_ids,
            data_cfg=data_cfg,
            split_period=data_cfg["val_period"],
            split_name="valid",
            ungauged_basins=None,
            mask_target=mask_target,
        )

        train_ds = HydroDataset(train_raw, data_cfg, basin_ids, data_cfg["train_period"], mode="train")
        train_ds.config = config

        val_ds = HydroDataset(valid_raw, data_cfg, basin_ids, data_cfg["val_period"], mode="valid", scaler=train_ds.scaler)
        val_ds.config = config

        train_loader = DataLoader(train_ds, **_loader_kwargs(config, shuffle=True, drop_last=True, generator=generator))
        val_loader = DataLoader(val_ds, **_loader_kwargs(config, shuffle=False, drop_last=False, generator=generator))
        return train_loader, val_loader, None, train_ds.scaler

    if mode == "test":
        scaler_ids = sorted([str(b) for b in (scaler_basin_ids if scaler_basin_ids is not None else basin_ids)])
        
        if scaler_basin_ids is not None and bool(data_cfg.get("spatial_split", False)):
            overlap = set(basin_ids).intersection(set(scaler_ids))
            if overlap:
                raise ValueError(
                    "Scaler leakage risk detected under spatial-split evaluation. "
                    "Test/evaluation basins must not be used for scaler fitting. "
                    f"Overlap count={len(overlap)}; examples={sorted(list(overlap))[:5]}."
                ) 
        if len(scaler_ids) == 0:
            raise ValueError("scaler_basin_ids is empty in test mode.")

        # Leak-safe scaler fitting: scaler is fitted on scaler_ids and train_period only.
        scaler_train_raw = load_nc_to_dict(
            data_root=data_root,
            basin_ids=scaler_ids,
            data_cfg=data_cfg,
            split_period=data_cfg["train_period"],
            split_name="scaler",
            ungauged_basins=None,
            mask_target=mask_target,
        )
        scaler_ds = HydroDataset(scaler_train_raw, data_cfg, scaler_ids, data_cfg["train_period"], mode="train")
        scaler_ds.config = config

        test_raw = load_nc_to_dict(
            data_root=data_root,
            basin_ids=basin_ids,
            data_cfg=data_cfg,
            split_period=data_cfg["test_period"],
            split_name="test",
            ungauged_basins=ungauged_basins,
            mask_target=mask_target,
        )
        test_ds = HydroDataset(test_raw, data_cfg, basin_ids, data_cfg["test_period"], mode="test", scaler=scaler_ds.scaler)
        test_ds.config = config

        test_loader = DataLoader(test_ds, **_loader_kwargs(config, shuffle=False, drop_last=False, generator=generator))
        return None, None, test_loader, scaler_ds.scaler

    raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'test'.")
