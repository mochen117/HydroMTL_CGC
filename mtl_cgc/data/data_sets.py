# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Sequence dataset abstractions for HydroMTL.
# Implements strict temporal coordinate mappings and leave-basin-out splitters.
# Avoids per-sample Python object construction for large hydrological datasets.
# Ensures zero-future leakage: x[t:t+rho] maps strictly to target[t+rho-1].
# ==============================================================================

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


class HydroDataset(Dataset):
    """
    Memory-efficient sequence dataset with strict temporal alignment.

    The dataset stores only vectorized basin/time indices instead of creating
    one Python object per sample. This is critical for CAMELS-scale experiments,
    where one split can easily contain more than one million sequence samples.
    """

    def __init__(
        self,
        raw_data: Dict[str, Any],
        data_params: Dict[str, Any],
        basin_ids: List[str],
        target_period: List[str],
        mode: str = "train",
        scaler: Optional[Any] = None,
    ):
        self.mode = mode
        self.rho = int(data_params.get("sequence_length", 180))
        if self.rho <= 0:
            raise ValueError(f"Invalid sequence_length={self.rho}. It must be positive.")

        self.task_names = [str(t["name"]).lower() for t in data_params.get("targets", [])]
        self.target_configs = {
            str(t["name"]).lower(): dict(t)
            for t in data_params.get("targets", [])
        }

        self.basin_ids = [str(b) for b in basin_ids]
        self.s_cat = raw_data.get("s_cat", None)

        start_date = pd.to_datetime(target_period[0])
        end_date = pd.to_datetime(target_period[1])

        self.pred_dates = pd.date_range(
            start=start_date + pd.Timedelta(days=self.rho - 1),
            end=end_date,
            freq="D",
        )

        from mtl_cgc.data.data_scalers import HydroScaler

        if mode == "train":
            self.scaler = HydroScaler(data_params)
            self.dyn, self.s_num, _, self.y_dict = self.scaler.fit_transform(
                raw_data["dyn"],
                raw_data["s_num"],
                raw_data["s_cat"],
                raw_data["y_dict"],
            )
        else:
            if scaler is None:
                raise ValueError("A fitted scaler must be supplied for non-training datasets.")
            self.scaler = scaler
            self.dyn, self.s_num, _, self.y_dict = self.scaler.transform(
                raw_data["dyn"],
                raw_data["s_num"],
                raw_data["s_cat"],
                raw_data["y_dict"],
            )
        self._validate_raw_arrays()
        self._interpolate_missing_targets()
        self._build_index_arrays()

    def _validate_raw_arrays(self) -> None:
        """Validate core arrays before sample-index construction."""
        if self.dyn.ndim != 3:
            raise ValueError(
                f"Expected dyn with shape [basin, time, feature], got {self.dyn.shape}."
            )

        if self.s_num.ndim != 2:
            raise ValueError(
                f"Expected s_num with shape [basin, feature], got {self.s_num.shape}."
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

        for task in self.task_names:
            if task not in self.y_dict:
                continue

            target = self.y_dict[task]

            if target.ndim != 2:
                raise ValueError(
                    f"Expected target '{task}' with shape [basin, time], got {target.shape}."
                )

            if target.shape[0] != self.dyn.shape[0]:
                raise ValueError(
                    f"Basin dimension mismatch for target '{task}': "
                    f"target has {target.shape[0]} basins, dyn has {self.dyn.shape[0]} basins."
                )

            if target.shape[1] != self.dyn.shape[1]:
                raise ValueError(
                    f"Time dimension mismatch for target '{task}': "
                    f"target has {target.shape[1]} steps, dyn has {self.dyn.shape[1]} steps."
                )
    def _build_index_arrays(self) -> None:
        """Build vectorized sample indices without per-sample Python objects."""
        num_basins = int(self.dyn.shape[0])
        num_time_steps = int(self.dyn.shape[1] - self.rho + 1)

        if num_time_steps <= 0:
            raise RuntimeError(
                f"Invalid sequence configuration: available time steps={self.dyn.shape[1]}, "
                f"sequence_length={self.rho}."
            )

        if len(self.pred_dates) != num_time_steps:
            raise RuntimeError(
                f"Spatio-temporal alignment failure: pred_dates length "
                f"({len(self.pred_dates)}) does not match num_time_steps "
                f"({num_time_steps})."
            )

        self.num_basins = num_basins
        self.num_time_steps = num_time_steps
        self.num_samples = num_basins * num_time_steps

        self.basin_index = np.repeat(
            np.arange(num_basins, dtype=np.int32),
            num_time_steps,
        )
        self.time_index = np.tile(
            np.arange(num_time_steps, dtype=np.int32),
            num_basins,
        )

    def _is_non_negative_target(self, task: str) -> bool:
        """
        Return whether a target should be clipped to non-negative values.

        Supported YAML styles:
            - non_negative: true/false
            - allow_negative: true/false
            - constraint: non_negative
        """
        cfg = self.target_configs.get(task, {})

        if "non_negative" in cfg:
            return bool(cfg["non_negative"])

        if "allow_negative" in cfg:
            return not bool(cfg["allow_negative"])

        if "constraint" in cfg:
            return str(cfg.get("constraint", "")).lower() == "non_negative"

        return True

    def _interpolate_missing_targets(self) -> None:
        """
        Fill short target gaps during training only.

        Validation and test targets are not interpolated here, so evaluation
        remains based on observed values and NaN masking.
        """
        if self.mode != "train":
            return

        for task in self.task_names:
            if task not in self.y_dict:
                continue

            is_non_negative = self._is_non_negative_target(task)

            for b in range(self.y_dict[task].shape[0]):
                series = pd.Series(self.y_dict[task][b])

                if is_non_negative:
                    series = series.clip(lower=0.0)

                interpolated = series.interpolate(
                    method="linear",
                    limit=3,
                    limit_direction="forward",
                )

                if is_non_negative:
                    interpolated = interpolated.clip(lower=0.0)

                self.y_dict[task][b] = interpolated.values.astype(np.float32)

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        b = int(self.basin_index[idx])
        t = int(self.time_index[idx])

        features = self.dyn[b, t:t + self.rho, :]
        static_num = self.s_num[b]

        item = {
            "features": torch.from_numpy(features).float(),
            "static_num": torch.from_numpy(static_num).float(),
            "basin_idx": torch.tensor(b, dtype=torch.long),
            "time_idx": torch.tensor(t, dtype=torch.long),
        }

        if self.s_cat is not None:
            item["categorical_features"] = torch.from_numpy(self.s_cat[b]).long()

        target_t = t + self.rho - 1

        if target_t >= self.dyn.shape[1]:
            raise IndexError(
                f"Target index out of range: target_t={target_t}, "
                f"available_time_steps={self.dyn.shape[1]}."
            )

        for task_name in self.task_names:
            if task_name in self.y_dict:
                if target_t >= self.y_dict[task_name].shape[1]:
                    raise IndexError(
                        f"Target index out of range for task='{task_name}': "
                        f"target_t={target_t}, available={self.y_dict[task_name].shape[1]}."
                    )

                value = np.array(
                    [self.y_dict[task_name][b, target_t]],
                    dtype=np.float32,
                )
                item[task_name] = torch.from_numpy(value).float()

        return item


class BasinSpatialSplitter:
    """Spatial split helper for PUB and regional transfer experiments."""

    def __init__(self, basin_ids: List[str], random_seed: int = 42):
        self.basin_ids = np.array(basin_ids)
        self.seed = random_seed

    def random_kfold_split(self, n_splits: int = 5) -> List[Tuple[List[str], List[str]]]:
        """Split catchments into disjoint random K-fold train/test basin sets."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = []

        for train_idx, test_idx in kf.split(self.basin_ids):
            splits.append(
                (
                    self.basin_ids[train_idx].tolist(),
                    self.basin_ids[test_idx].tolist(),
                )
            )

        return splits

    def hydrologic_region_split(
        self,
        metadata: pd.DataFrame,
        region_col: str = "hydrologic_region",
    ) -> List[Dict[str, Any]]:
        """Split catchments by hydrologic regions for leave-region-out tests."""
        if "basin_id" not in metadata.columns:
            raise ValueError("Metadata must contain a 'basin_id' column.")

        if region_col not in metadata.columns:
            raise ValueError(f"Metadata must contain region column: {region_col}")

        unique_regions = metadata[region_col].dropna().unique()
        basin_set = set(self.basin_ids.tolist())
        splits = []

        for region in unique_regions:
            train_basins = metadata.loc[
                metadata[region_col] != region,
                "basin_id",
            ].astype(str).tolist()

            test_basins = metadata.loc[
                metadata[region_col] == region,
                "basin_id",
            ].astype(str).tolist()

            splits.append(
                {
                    "test_region": str(region),
                    "train_basins": [b for b in train_basins if b in basin_set],
                    "test_basins": [b for b in test_basins if b in basin_set],
                }
            )

        return splits