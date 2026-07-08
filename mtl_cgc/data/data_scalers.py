# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Feature scaling module for HydroMTL_CGC.
#   The streamflow scaling follows the DapengScaler convention:
#       Q(cfs) -> Q(mm/day) / p_mean -> log10(sqrt(x) + 0.1) -> standardization.
#   During evaluation, streamflow is restored to m3/s for consistency with
#   Ouyang-style reporting and hydrological benchmark tables.
# ==============================================================================

from typing import Any, Dict, Optional, Tuple

import numpy as np


class HydroScaler:
    """
    Catchment-aware scaling framework.

    This scaler standardizes dynamic forcing, numerical static attributes, and
    target variables using statistics fitted strictly on the training set.

    Streamflow is treated with basin normalization following the DapengScaler
    convention. Other targets are standardized directly in their native units.
    """

    CFS_TO_M3_PER_DAY = 0.0283168 * 86400.0
    KM2_TO_M2 = 1e6
    MM_TO_M = 1e-3
    LOG_EPSILON = 0.1

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.stat_dict: Dict[str, Any] = {}

        data_cfg = config.get("data", config)
        self.static_features = data_cfg.get("static_features", [])

        self.area_idx = self._get_static_feature_index("area_gages2")
        self.prcp_idx = self._get_static_feature_index("p_mean")

        self.targets_cfg = data_cfg.get("targets", [])
        self.task_names = [str(t["name"]).lower() for t in self.targets_cfg]
        self.q_name = next(
            (task for task in self.task_names if "streamflow" in task),
            "streamflow",
        )

        if self.q_name in self.task_names:
            self._validate_streamflow_scaling_features()


    def _validate_streamflow_scaling_features(self) -> None:
        """Ensure static attributes required by streamflow scaling are present."""
        required = {
            "area_gages2": self.area_idx,
            "p_mean": self.prcp_idx,
        }
        missing = [name for name, index in required.items() if index < 0]
        if missing:
            raise ValueError(
                "Streamflow scaling requires static features: "
                + ", ".join(missing)
            )

    def fit_transform(
        self,
        dyn: np.ndarray,
        s_num: np.ndarray,
        s_cat: Optional[np.ndarray],
        y_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Fit training-set statistics and transform all input arrays."""
        self._validate_input_shapes(dyn=dyn, s_num=s_num, y_dict=y_dict)

        s_num_t = self._fit_transform_static(s_num)
        dyn_t = self._fit_transform_dynamic(dyn)
        target_t = self._fit_transform_targets(s_num=s_num, y_dict=y_dict)

        s_cat_t = np.copy(s_cat) if s_cat is not None else None
        return dyn_t, s_num_t, s_cat_t, target_t

    def transform(
        self,
        dyn: np.ndarray,
        s_num: np.ndarray,
        s_cat: Optional[np.ndarray],
        y_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Transform arrays using fitted training-set statistics."""
        if not self.stat_dict:
            raise RuntimeError("Scaler must be fitted before calling transform().")

        self._validate_input_shapes(dyn=dyn, s_num=s_num, y_dict=y_dict)
        self._validate_required_statistics()

        s_num_t = (s_num - self.stat_dict["s_num_mean"]) / self.stat_dict["s_num_std"]
        s_num_t = np.nan_to_num(s_num_t, nan=0.0)

        dyn_t = (dyn - self.stat_dict["dyn_mean"]) / self.stat_dict["dyn_std"]
        dyn_t = np.nan_to_num(dyn_t, nan=0.0)

        target_t = self._transform_targets(s_num=s_num, y_dict=y_dict)
        s_cat_t = np.copy(s_cat) if s_cat is not None else None

        return dyn_t, s_num_t, s_cat_t, target_t

    def inverse_transform_target_safe(
        self,
        task: str,
        latent_arr: np.ndarray,
        stat_num_scaled: np.ndarray,
    ) -> np.ndarray:
        """
        Invert standardized target values back to physical units.

        Streamflow is restored to m3/s for hydrological evaluation and publication-ready reporting.

        The internal normalization follows the DapengScaler convention:
        Q(cfs) -> Q(mm/day)/p_mean -> log10(sqrt(x)+0.1).

        After inverse transformation, streamflow is returned in m3/s.
        """
        if not self.stat_dict:
            return latent_arr

        task = str(task).lower()
        stat_num_raw = self._inverse_static_features(stat_num_scaled)

        task_mean = self.stat_dict.get(f"{task}_mean", 0.0)
        task_std = self.stat_dict.get(f"{task}_std", 1.0)

        if task != self.q_name:
            return latent_arr * task_std + task_mean

        q_ratio = self._inverse_streamflow_to_prcp_ratio(
            latent_arr=latent_arr,
            task_mean=task_mean,
            task_std=task_std,
        )

        area_raw = self._extract_basin_attribute(stat_num_raw, self.area_idx, "area_gages2")
        prcp_raw = self._extract_basin_attribute(stat_num_raw, self.prcp_idx, "p_mean")

        area_raw = self._align_basin_vector(area_raw, latent_arr)
        prcp_raw = self._align_basin_vector(prcp_raw, latent_arr)

        q_cfs = self._physical_basin_norm(
            flow=q_ratio,
            area=area_raw,
            prcp=prcp_raw,
            to_norm=False,
        )

        q_m3s = np.maximum(q_cfs, 0.0) * 0.0283168

        return q_m3s

    def inverse_streamflow_to_mm_day(
        self,
        latent_arr: np.ndarray,
        stat_num_scaled: np.ndarray,
    ) -> np.ndarray:
        """
        Invert standardized streamflow to runoff depth in mm/day.

        This method is intended for physical diagnostics or water-balance
        constraints. It should not be used for benchmark evaluation when the
        hard-sharing baseline evaluates streamflow in cfs.
        """
        if not self.stat_dict:
            return latent_arr

        stat_num_raw = self._inverse_static_features(stat_num_scaled)
        prcp_raw = self._extract_basin_attribute(stat_num_raw, self.prcp_idx, "p_mean")

        task_mean = self.stat_dict.get(f"{self.q_name}_mean", 0.0)
        task_std = self.stat_dict.get(f"{self.q_name}_std", 1.0)

        q_ratio = self._inverse_streamflow_to_prcp_ratio(
            latent_arr=latent_arr,
            task_mean=task_mean,
            task_std=task_std,
        )

        prcp_raw = self._align_basin_vector(prcp_raw, latent_arr)
        q_mm_day = q_ratio * prcp_raw

        return np.maximum(q_mm_day, 0.0)

    def _fit_transform_static(self, s_num: np.ndarray) -> np.ndarray:
        """Fit and transform numerical static attributes."""
        self.stat_dict["s_num_mean"] = np.nanmean(s_num, axis=0)
        self.stat_dict["s_num_std"] = np.nanstd(s_num, axis=0)
        self.stat_dict["s_num_std"][self.stat_dict["s_num_std"] < 1e-6] = 1.0

        s_num_t = (s_num - self.stat_dict["s_num_mean"]) / self.stat_dict["s_num_std"]
        return np.nan_to_num(s_num_t, nan=0.0)

    def _fit_transform_dynamic(self, dyn: np.ndarray) -> np.ndarray:
        """Fit and transform dynamic forcing variables."""
        self.stat_dict["dyn_mean"] = np.nanmean(dyn, axis=(0, 1))
        self.stat_dict["dyn_std"] = np.nanstd(dyn, axis=(0, 1))
        self.stat_dict["dyn_std"][self.stat_dict["dyn_std"] < 1e-6] = 1.0

        dyn_t = (dyn - self.stat_dict["dyn_mean"]) / self.stat_dict["dyn_std"]
        return np.nan_to_num(dyn_t, nan=0.0)

    def _fit_transform_targets(
        self,
        s_num: np.ndarray,
        y_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Fit and transform target variables."""
        target_t: Dict[str, np.ndarray] = {}

        area = None
        prcp = None
        if self.q_name in self.task_names:
            area = self._extract_basin_attribute(s_num, self.area_idx, "area_gages2")
            prcp = self._extract_basin_attribute(s_num, self.prcp_idx, "p_mean")

        for task in self.task_names:
            raw_y = y_dict[task]

            if task == self.q_name:
                transformed = self._transform_streamflow(
                    raw_y=raw_y,
                    area=area,
                    prcp=prcp,
                )
            else:
                transformed = raw_y

            mean_val = float(np.nanmean(transformed))
            std_val = float(np.nanstd(transformed))

            mean_val = 0.0 if np.isnan(mean_val) else mean_val
            std_val = 1.0 if np.isnan(std_val) or std_val < 1e-6 else std_val

            self.stat_dict[f"{task}_mean"] = mean_val
            self.stat_dict[f"{task}_std"] = std_val
            target_t[task] = (transformed - mean_val) / std_val

        return target_t

    def _transform_targets(
        self,
        s_num: np.ndarray,
        y_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Transform target variables using fitted statistics."""
        target_t: Dict[str, np.ndarray] = {}

        area = None
        prcp = None
        if self.q_name in self.task_names:
            area = self._extract_basin_attribute(s_num, self.area_idx, "area_gages2")
            prcp = self._extract_basin_attribute(s_num, self.prcp_idx, "p_mean")

        for task in self.task_names:
            raw_y = y_dict.get(task)
            if raw_y is None:
                continue

            if task == self.q_name:
                transformed = self._transform_streamflow(
                    raw_y=raw_y,
                    area=area,
                    prcp=prcp,
                )
            else:
                transformed = raw_y

            target_t[task] = (
                transformed - self.stat_dict[f"{task}_mean"]
            ) / self.stat_dict[f"{task}_std"]

        return target_t

    def _transform_streamflow(
        self,
        raw_y: np.ndarray,
        area: np.ndarray,
        prcp: np.ndarray,
    ) -> np.ndarray:
        """
        Transform streamflow from cfs to standardized-ready log space.

        The returned value is not yet standardized:
            raw cfs -> Q(mm/day) / p_mean -> log10(sqrt(x) + 0.1)
        """
        raw_y_safe = np.maximum(raw_y, 0.0)

        q_ratio = self._physical_basin_norm(
            flow=raw_y_safe,
            area=area,
            prcp=prcp,
            to_norm=True,
        )

        q_ratio = np.maximum(q_ratio, 0.0)
        return np.log10(np.sqrt(q_ratio) + self.LOG_EPSILON)

    def _inverse_streamflow_to_prcp_ratio(
        self,
        latent_arr: np.ndarray,
        task_mean: float,
        task_std: float,
    ) -> np.ndarray:
        """Invert standardized streamflow to Q(mm/day) / p_mean."""
        q_log = latent_arr * task_std + task_mean
        q_log = np.clip(q_log, -5.0, 10.0)

        sqrt_q_ratio = np.maximum(
            np.power(10.0, q_log) - self.LOG_EPSILON,
            0.0,
        )
        q_ratio = sqrt_q_ratio ** 2
        return np.maximum(q_ratio, 0.0)

    def _physical_basin_norm(
        self,
        flow: np.ndarray,
        area: np.ndarray,
        prcp: np.ndarray,
        to_norm: bool,
    ) -> np.ndarray:
        """
        Convert streamflow using DapengScaler basin normalization.

        If to_norm is True:
            Q(cfs) -> Q(mm/day) / p_mean

        If to_norm is False:
            Q(mm/day) / p_mean -> Q(cfs)
        """
        area_ex = self._align_basin_vector(area, flow)
        prcp_ex = self._align_basin_vector(prcp, flow)

        area_ex = np.maximum(area_ex, 1e-6)
        prcp_ex = np.maximum(prcp_ex, 1e-6)

        if to_norm:
            return (
                flow
                * self.CFS_TO_M3_PER_DAY
                / (area_ex * self.KM2_TO_M2 * prcp_ex * self.MM_TO_M)
            )

        return (
            flow
            * area_ex
            * self.KM2_TO_M2
            * prcp_ex
            * self.MM_TO_M
            / self.CFS_TO_M3_PER_DAY
        )

    def _inverse_static_features(self, stat_num_scaled: np.ndarray) -> np.ndarray:
        """Restore standardized numerical static attributes to raw values."""
        return (
            stat_num_scaled * self.stat_dict.get("s_num_std", 1.0)
            + self.stat_dict.get("s_num_mean", 0.0)
        )

    def _extract_basin_attribute(
        self,
        stat_matrix: np.ndarray,
        index: int,
        feature_name: str,
    ) -> np.ndarray:
        """Extract a strictly positive basin attribute required by scaling."""
        if index < 0 or stat_matrix.shape[1] <= index:
            raise ValueError(f"Missing required basin attribute: {feature_name}")

        values = np.asarray(stat_matrix[:, index], dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values found in required basin attribute: {feature_name}"
            )

        if np.any(values <= 0.0):
            raise ValueError(
                f"Non-positive values found in required basin attribute: {feature_name}"
            )

        return values

    def _align_basin_vector(
        self,
        values: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Align a basin-level vector to the dimensionality of a target array.

        For [basin, time] arrays, a [basin] vector is expanded to [basin, 1].
        Arrays already aligned are returned unchanged.
        """
        if values.ndim == 1 and reference.ndim == 2:
            return np.expand_dims(values, axis=1)

        return values

    def _get_static_feature_index(self, feature_name: str) -> int:
        """Return the static feature index, or -1 if the feature is unavailable."""
        try:
            return self.static_features.index(feature_name)
        except ValueError:
            return -1

    def _validate_input_shapes(
        self,
        dyn: np.ndarray,
        s_num: np.ndarray,
        y_dict: Dict[str, np.ndarray],
    ) -> None:
        """Validate scaler input dimensions before fitting or transforming."""
        if dyn.ndim != 3:
            raise ValueError(
                f"Expected dyn with shape [basin, time, feature], got {dyn.shape}."
            )

        if s_num.ndim != 2:
            raise ValueError(
                f"Expected s_num with shape [basin, feature], got {s_num.shape}."
            )

        if dyn.shape[0] != s_num.shape[0]:
            raise ValueError(
                f"Basin dimension mismatch: dyn has {dyn.shape[0]} basins, "
                f"s_num has {s_num.shape[0]} basins."
            )

        for task in self.task_names:
            if task not in y_dict:
                continue

            y = y_dict[task]

            if y.ndim != 2:
                raise ValueError(
                    f"Expected target '{task}' with shape [basin, time], "
                    f"got {y.shape}."
                )

            if y.shape[0] != dyn.shape[0]:
                raise ValueError(
                    f"Basin dimension mismatch for target '{task}': "
                    f"target has {y.shape[0]} basins, dyn has {dyn.shape[0]} basins."
                )

            if y.shape[1] != dyn.shape[1]:
                raise ValueError(
                    f"Time dimension mismatch for target '{task}': "
                    f"target has {y.shape[1]} steps, dyn has {dyn.shape[1]} steps."
                )

    def _validate_required_statistics(self) -> None:
        """Ensure all statistics required for transformation are available."""
        required = [
            "s_num_mean",
            "s_num_std",
            "dyn_mean",
            "dyn_std",
        ]

        for task in self.task_names:
            required.append(f"{task}_mean")
            required.append(f"{task}_std")

        missing = [key for key in required if key not in self.stat_dict]

        if missing:
            raise RuntimeError(
                "Scaler is missing fitted statistics required for transform: "
                + ", ".join(missing)
            )