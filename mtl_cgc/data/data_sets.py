# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Memory-efficient N-to-1 sequence datasets and spatial splitters
# for HydroMTL.
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from mtl_cgc.data.data_scalers import HydroScaler
from mtl_cgc.utils.temporal import build_prediction_dates, count_inclusive_days


class HydroDataset(Dataset):
    """
    Memory-efficient N-to-1 hydrological sequence dataset.

    The raw arrays must already contain ``sequence_length - 1`` historical
    context days before ``target_period[0]``. Each input window
    ``x[t:t+rho]`` maps strictly to the target at ``t + rho - 1``. Only target
    dates inside ``target_period`` are exposed as samples.
    """

    _SUPPORTED_MODES = {"train", "valid", "validation", "test"}

    def __init__(
        self,
        raw_data: Dict[str, Any],
        data_params: Dict[str, Any],
        basin_ids: List[str],
        target_period: List[str],
        mode: str = "train",
        scaler: Optional[HydroScaler] = None,
    ) -> None:
        self.mode = str(mode).strip().lower()
        if self.mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported dataset mode '{mode}'. "
                f"Expected one of {sorted(self._SUPPORTED_MODES)}."
            )

        self.rho = int(data_params.get("sequence_length", 180))
        if self.rho <= 0:
            raise ValueError(
                f"Invalid sequence_length={self.rho}. It must be positive."
            )

        self.task_names = [
            str(target["name"]).lower()
            for target in data_params.get("targets", [])
        ]
        self.target_configs = {
            str(target["name"]).lower(): dict(target)
            for target in data_params.get("targets", [])
        }

        raw_sample_filter = data_params.get("sample_filter")
        if raw_sample_filter is None:
            self.sample_filter_config: Dict[str, Any] = {}
        elif isinstance(raw_sample_filter, dict):
            self.sample_filter_config = dict(raw_sample_filter)
        else:
            raise TypeError(
                "data.sample_filter must be a dictionary or null, "
                f"got {type(raw_sample_filter).__name__}."
            )

        self.basin_ids = [str(basin_id) for basin_id in basin_ids]
        self.s_cat = raw_data.get("s_cat")
        self.period_start_date = pd.to_datetime(target_period[0])
        self.period_end_date = pd.to_datetime(target_period[1])
        if self.period_start_date > self.period_end_date:
            raise ValueError(
                "Invalid target_period: start date must not be after end date."
            )

        self.pred_dates: Optional[pd.DatetimeIndex] = None
        self.scaler = self._initialize_scaler(
            raw_data=raw_data,
            data_params=data_params,
            scaler=scaler,
        )

        self._validate_raw_arrays()
        self._interpolate_missing_targets()
        self._build_index_arrays()

    def _initialize_scaler(
        self,
        raw_data: Dict[str, Any],
        data_params: Dict[str, Any],
        scaler: Optional[HydroScaler],
    ) -> HydroScaler:
        """
        Fit or apply the HydroScaler.

        A pre-fitted scaler may be supplied in training mode. This is required
        when the raw training arrays include pre-period historical context but
        scaling statistics must be fitted only on the configured target period.
        """
        if scaler is None:
            if self.mode != "train":
                raise ValueError(
                    "A fitted scaler must be supplied for validation or test data."
                )

            fitted_scaler = HydroScaler(data_params)
            self.dyn, self.s_num, _, self.y_dict = fitted_scaler.fit_transform(
                raw_data["dyn"],
                raw_data["s_num"],
                raw_data.get("s_cat"),
                raw_data["y_dict"],
            )
            return fitted_scaler

        self.dyn, self.s_num, _, self.y_dict = scaler.transform(
            raw_data["dyn"],
            raw_data["s_num"],
            raw_data.get("s_cat"),
            raw_data["y_dict"],
        )
        return scaler

    def _validate_raw_arrays(self) -> None:
        """Validate basin and temporal array alignment before indexing."""
        if self.dyn.ndim != 3:
            raise ValueError(
                f"Expected dyn with shape [basin, time, feature], got {self.dyn.shape}."
            )
        if self.s_num.ndim != 2:
            raise ValueError(
                f"Expected s_num with shape [basin, feature], got {self.s_num.shape}."
            )
        if self.dyn.shape[0] != len(self.basin_ids):
            raise ValueError(
                f"Basin count mismatch: dyn contains {self.dyn.shape[0]} basins, "
                f"but basin_ids contains {len(self.basin_ids)}."
            )
        if self.dyn.shape[0] != self.s_num.shape[0]:
            raise ValueError(
                f"Basin dimension mismatch: dyn has {self.dyn.shape[0]} basins, "
                f"s_num has {self.s_num.shape[0]} basins."
            )
        if self.s_cat is not None and self.s_cat.shape[0] != self.dyn.shape[0]:
            raise ValueError(
                f"Basin dimension mismatch: s_cat has {self.s_cat.shape[0]} basins, "
                f"dyn has {self.dyn.shape[0]} basins."
            )

        for task_name in self.task_names:
            if task_name not in self.y_dict:
                raise KeyError(
                    f"Target '{task_name}' is configured but missing from y_dict."
                )

            target = self.y_dict[task_name]
            if target.ndim != 2:
                raise ValueError(
                    f"Expected target '{task_name}' with shape [basin, time], "
                    f"got {target.shape}."
                )
            if target.shape[:2] != self.dyn.shape[:2]:
                raise ValueError(
                    f"Target alignment mismatch for '{task_name}': target shape "
                    f"{target.shape} versus dynamic shape {self.dyn.shape}."
                )

    def _build_index_arrays(self) -> None:
        """Build vectorized basin/time sample indices for the target period."""
        num_basins = int(self.dyn.shape[0])
        expected_target_steps = count_inclusive_days(
            [self.period_start_date, self.period_end_date]
        )
        expected_raw_steps = expected_target_steps + self.rho - 1
        actual_raw_steps = int(self.dyn.shape[1])

        if actual_raw_steps != expected_raw_steps:
            raise RuntimeError(
                "N-to-1 context alignment failure: the raw slice must contain "
                "sequence_length - 1 historical context days before the target "
                f"period. Expected {expected_raw_steps} raw steps for "
                f"target_period=[{self.period_start_date.date()}, "
                f"{self.period_end_date.date()}] and sequence_length={self.rho}, "
                f"but received {actual_raw_steps}."
            )

        self.num_basins = num_basins
        self.num_time_steps = expected_target_steps
        self.pred_dates = build_prediction_dates(
            start_date=self.period_start_date,
            sequence_length=self.rho,
            num_time_steps=expected_target_steps,
        )

        expected_last_date = self.period_end_date.normalize()
        if self.pred_dates[-1] != expected_last_date:
            raise RuntimeError(
                "Prediction-date alignment failure: expected the final target "
                f"date {expected_last_date.date()}, got {self.pred_dates[-1].date()}."
            )

        apply_modes = {
            str(mode).strip().lower()
            for mode in self.sample_filter_config.get("apply_to_modes", [])
        }
        use_sample_filter = (
            bool(self.sample_filter_config.get("enabled", False))
            and self.mode in apply_modes
        )

        if not use_sample_filter:
            self.basin_index = np.repeat(
                np.arange(num_basins, dtype=np.int32),
                expected_target_steps,
            )
            self.time_index = np.tile(
                np.arange(expected_target_steps, dtype=np.int32),
                num_basins,
            )
            self.num_samples = int(self.basin_index.size)
            return

        required_targets = [
            str(target).strip().lower()
            for target in self.sample_filter_config.get(
                "required_valid_targets",
                [],
            )
        ]
        if not required_targets:
            raise ValueError(
                "sample_filter.enabled=True but required_valid_targets is empty."
            )

        target_start = self.rho - 1
        target_stop = target_start + expected_target_steps
        valid_samples = np.ones(
            (num_basins, expected_target_steps),
            dtype=bool,
        )

        for task_name in required_targets:
            if task_name not in self.y_dict:
                raise KeyError(
                    f"sample_filter requires target '{task_name}', but only "
                    f"{sorted(self.y_dict)} are available."
                )

            target_values = self.y_dict[task_name][
                :,
                target_start:target_stop,
            ]
            expected_shape = (num_basins, expected_target_steps)
            if target_values.shape != expected_shape:
                raise RuntimeError(
                    f"Target shape mismatch for sample_filter task '{task_name}': "
                    f"expected {expected_shape}, got {target_values.shape}."
                )

            valid_samples &= np.isfinite(target_values)

        basin_index, time_index = np.where(valid_samples)
        self.basin_index = basin_index.astype(np.int32, copy=False)
        self.time_index = time_index.astype(np.int32, copy=False)
        self.num_samples = int(self.basin_index.size)

        if self.num_samples == 0:
            raise RuntimeError(
                "sample_filter removed every sample. "
                f"Required targets={required_targets}, mode={self.mode}."
            )

    def _is_non_negative_target(self, task_name: str) -> bool:
        """Return whether a target is constrained to non-negative values."""
        config = self.target_configs.get(task_name, {})
        if "non_negative" in config:
            return bool(config["non_negative"])
        if "allow_negative" in config:
            return not bool(config["allow_negative"])
        if "constraint" in config:
            return str(config["constraint"]).lower() == "non_negative"
        return True

    def _interpolate_missing_targets(self) -> None:
        """
        Fill short target gaps during training only when explicitly enabled.

        Validation and test targets remain unchanged so that evaluation relies
        strictly on observed values and NaN masking.
        """
        if self.mode != "train":
            return

        for task_name in self.task_names:
            target_config = self.target_configs.get(task_name, {})
            if not bool(target_config.get("interpolate_missing", False)):
                continue

            interpolation_limit = int(
                target_config.get("interpolation_limit", 3)
            )
            if interpolation_limit <= 0:
                raise ValueError(
                    f"interpolation_limit for target '{task_name}' must be "
                    f"positive, got {interpolation_limit}."
                )

            is_non_negative = self._is_non_negative_target(task_name)

            for basin_idx in range(self.y_dict[task_name].shape[0]):
                series = pd.Series(self.y_dict[task_name][basin_idx])
                if is_non_negative:
                    series = series.clip(lower=0.0)

                interpolated = series.interpolate(
                    method="linear",
                    limit=interpolation_limit,
                    limit_direction="forward",
                )
                if is_non_negative:
                    interpolated = interpolated.clip(lower=0.0)

                self.y_dict[task_name][basin_idx] = interpolated.to_numpy(
                    dtype=np.float32
                )

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        basin_idx = int(self.basin_index[idx])
        sequence_start = int(self.time_index[idx])
        target_idx = sequence_start + self.rho - 1

        features = self.dyn[
            basin_idx,
            sequence_start : sequence_start + self.rho,
            :,
        ]
        if features.shape[0] != self.rho:
            raise RuntimeError(
                f"Invalid sequence length at sample {idx}: expected {self.rho}, "
                f"got {features.shape[0]}."
            )

        item: Dict[str, torch.Tensor] = {
            "features": torch.from_numpy(features).float(),
            "static_num": torch.from_numpy(self.s_num[basin_idx]).float(),
            "basin_idx": torch.tensor(basin_idx, dtype=torch.long),
            "time_idx": torch.tensor(sequence_start, dtype=torch.long),
        }

        if self.s_cat is not None:
            item["categorical_features"] = torch.from_numpy(
                self.s_cat[basin_idx]
            ).long()

        for task_name in self.task_names:
            value = np.asarray(
                [self.y_dict[task_name][basin_idx, target_idx]],
                dtype=np.float32,
            )
            item[task_name] = torch.from_numpy(value).float()

        return item


class BasinSpatialSplitter:
    """Spatial split helper for PUB and regional transfer experiments."""

    def __init__(self, basin_ids: List[str], random_seed: int = 42) -> None:
        self.basin_ids = np.asarray(basin_ids)
        self.seed = int(random_seed)

    def random_kfold_split(
        self,
        n_splits: int = 5,
    ) -> List[Tuple[List[str], List[str]]]:
        """Split catchments into disjoint random K-fold train/test sets."""
        kfold = KFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=self.seed,
        )
        return [
            (
                self.basin_ids[train_idx].tolist(),
                self.basin_ids[test_idx].tolist(),
            )
            for train_idx, test_idx in kfold.split(self.basin_ids)
        ]

    def hydrologic_region_split(
        self,
        metadata: pd.DataFrame,
        region_col: str = "hydrologic_region",
    ) -> List[Dict[str, Any]]:
        """Create leave-one-hydrologic-region-out basin splits."""
        required_columns = {"basin_id", region_col}
        missing_columns = required_columns.difference(metadata.columns)
        if missing_columns:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing_columns)}."
            )

        basin_set = set(self.basin_ids.tolist())
        splits: List[Dict[str, Any]] = []

        for region in metadata[region_col].dropna().unique():
            train_basins = metadata.loc[
                metadata[region_col] != region,
                "basin_id",
            ].astype(str)
            test_basins = metadata.loc[
                metadata[region_col] == region,
                "basin_id",
            ].astype(str)

            splits.append(
                {
                    "test_region": str(region),
                    "train_basins": [
                        basin_id for basin_id in train_basins if basin_id in basin_set
                    ],
                    "test_basins": [
                        basin_id for basin_id in test_basins if basin_id in basin_set
                    ],
                }
            )

        return splits
