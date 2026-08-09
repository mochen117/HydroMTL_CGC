# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Mask-aware multi-task regression criterion for HydroMTL.
# Supports MSE, RMSE, and MAE in standardized target space, applies independent
# finite-value masks per task, and retains the optional legacy water-balance
# regularization branch used by earlier Q-ET experiments.
# ==============================================================================

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _config_get(container: Any, key: str, default: Any = None) -> Any:
    """Read one configuration value from a mapping or attribute-style object."""
    if container is None:
        return default

    if isinstance(container, Mapping):
        return container.get(key, default)

    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass

    return getattr(container, key, default)


class DynamicMultiTaskLoss(nn.Module):
    """
    Compute weighted, mask-aware regression losses for configured tasks.

    The criterion operates in the standardized target space produced by
    ``HydroScaler``. Missing observations are excluded independently for each
    task, which allows daily Q supervision and sparse SSM supervision to coexist
    in the same mini-batch.

    Supported base losses
    ---------------------
    ``mse``
        Mean squared error.
    ``rmse``
        Root mean squared error, computed as ``sqrt(MSE + eps)``.
    ``mae``
        Mean absolute error.

    Notes
    -----
    The optional water-balance branch is retained for backward compatibility
    with earlier Q-ET experiments. It is disabled in Chapter 4 Experiment A.
    """

    _SUPPORTED_BASE_LOSSES = {"mse", "rmse", "mae"}
    _MISSING_SENTINEL = -9999.0

    def __init__(self, config: Any, stat_dict: Dict[str, Any]) -> None:
        super().__init__()

        self.stat = stat_dict or {}

        data_cfg = _config_get(config, "data", {})
        training_cfg = _config_get(config, "training", {})
        loss_cfg = _config_get(training_cfg, "loss", {})

        self.base_loss = str(
            _config_get(loss_cfg, "base_loss", "rmse")
        ).strip().lower()
        if self.base_loss not in self._SUPPORTED_BASE_LOSSES:
            raise ValueError(
                f"Unsupported base_loss={self.base_loss!r}. "
                f"Expected one of {sorted(self._SUPPORTED_BASE_LOSSES)}."
            )

        self.eps = float(_config_get(loss_cfg, "eps", 1.0e-6))
        if self.eps <= 0.0:
            raise ValueError(
                f"training.loss.eps must be positive, got {self.eps}."
            )

        targets_cfg = list(_config_get(data_cfg, "targets", []))
        if not targets_cfg:
            raise ValueError("data.targets must contain at least one target.")

        self.weights: Dict[str, float] = {}
        for target_cfg in targets_cfg:
            task_name = str(_config_get(target_cfg, "name", "")).strip().lower()
            if not task_name:
                raise ValueError("Every data.targets entry must define a name.")

            weight = float(_config_get(target_cfg, "loss_weight", 1.0))
            if weight < 0.0:
                raise ValueError(
                    f"loss_weight for task {task_name!r} must be non-negative, "
                    f"got {weight}."
                )

            self.weights[task_name] = weight

        self.q_name = next(
            (
                task_name
                for task_name in self.weights
                if "streamflow" in task_name
            ),
            "streamflow",
        )
        self.et_name = next(
            (
                task_name
                for task_name in self.weights
                if "evapo" in task_name
            ),
            "evapotranspiration",
        )

        static_features = list(_config_get(data_cfg, "static_features", []))
        self.prcp_idx = (
            static_features.index("p_mean")
            if "p_mean" in static_features
            else -1
        )

        model_cfg = _config_get(config, "model", {})
        physics_cfg = _config_get(
            _config_get(model_cfg, "physics_constraints", {}),
            "water_balance",
            {},
        )
        self.use_physics = bool(_config_get(physics_cfg, "enabled", False))
        self.alpha = float(_config_get(physics_cfg, "alpha", 0.1))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"water_balance.alpha must be in [0, 1], got {self.alpha}."
            )

    @staticmethod
    def prediction_tensor(prediction: Any) -> torch.Tensor:
        """Convert one model prediction object into a dense tensor."""
        if isinstance(prediction, torch.Tensor):
            return prediction

        if (
            isinstance(prediction, Mapping)
            and "means" in prediction
            and "weights" in prediction
        ):
            means = prediction["means"]
            weights = prediction["weights"]
            return torch.sum(
                means.squeeze(-1) * weights,
                dim=1,
            )

        raise TypeError(
            "Unsupported prediction object. Expected a torch.Tensor or a "
            "mapping containing 'means' and 'weights', got "
            f"{type(prediction).__name__}."
        )

    @classmethod
    def valid_mask(
        cls,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return the finite, aligned observation mask for one task."""
        pred = prediction.reshape(-1)
        obs = target.reshape(-1)

        if pred.shape != obs.shape:
            raise ValueError(
                "Shape mismatch in DynamicMultiTaskLoss: "
                f"prediction shape={pred.shape}, target shape={obs.shape}."
            )

        return (
            torch.isfinite(pred)
            & torch.isfinite(obs)
            & (obs != cls._MISSING_SENTINEL)
        )

    def count_valid_observations(
        self,
        prediction: Any,
        target: torch.Tensor,
    ) -> int:
        """Count valid prediction-observation pairs for one task."""
        pred = self.prediction_tensor(prediction)
        mask = self.valid_mask(pred, target)
        return int(mask.sum().item())

    def compute_task_loss(
        self,
        prediction: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one unweighted, mask-aware task loss."""
        pred = self.prediction_tensor(prediction).reshape(-1)
        obs = target.reshape(-1)
        mask = self.valid_mask(pred, obs)

        if not mask.any():
            # Return a graph-connected zero so backward remains well-defined.
            return (
                torch.nan_to_num(
                    pred,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).sum()
                * 0.0
            )

        residual = pred[mask] - obs[mask]

        if self.base_loss == "mae":
            return residual.abs().mean()

        mse = residual.square().mean()
        if self.base_loss == "mse":
            return mse

        return torch.sqrt(mse + self.eps)

    def forward(
        self,
        preds_dict: Dict[str, Any],
        targets_dict: Dict[str, torch.Tensor],
        s_num: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the weighted mathematical loss and optional physics loss."""
        if not preds_dict:
            raise ValueError("preds_dict is empty.")

        first_prediction = self.prediction_tensor(
            next(iter(preds_dict.values()))
        )
        total_math_loss = (
            torch.nan_to_num(
                first_prediction,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).sum()
            * 0.0
        )

        positive_weight_tasks = 0
        for task_name, weight in self.weights.items():
            if weight <= 0.0:
                continue

            if task_name not in preds_dict:
                raise KeyError(
                    f"Positive-weight task {task_name!r} is missing from "
                    "model predictions."
                )
            if task_name not in targets_dict:
                raise KeyError(
                    f"Positive-weight task {task_name!r} is missing from "
                    "batch targets."
                )

            task_loss = self.compute_task_loss(
                preds_dict[task_name],
                targets_dict[task_name],
            )
            total_math_loss = total_math_loss + weight * task_loss
            positive_weight_tasks += 1

        if positive_weight_tasks == 0:
            raise RuntimeError(
                "No positive-weight task is available for loss calculation."
            )

        total_loss = total_math_loss
        if (
            self.use_physics
            and self.stat
            and self.q_name in preds_dict
            and self.et_name in preds_dict
            and s_num is not None
            and self.prcp_idx >= 0
        ):
            total_loss = self._apply_legacy_water_balance_loss(
                preds_dict=preds_dict,
                s_num=s_num,
                total_math_loss=total_math_loss,
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite total loss detected.")

        return total_loss

    def _apply_legacy_water_balance_loss(
        self,
        preds_dict: Dict[str, Any],
        s_num: torch.Tensor,
        total_math_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the legacy Q-ET water-balance regularization branch.

        This branch is intentionally unchanged in mathematical meaning because
        it belongs to earlier Q-ET experiments and is disabled for Q-SSM. A
        dedicated physical-units review is required before enabling it in a new
        experiment.
        """
        q_prediction = self.prediction_tensor(
            preds_dict[self.q_name]
        ).reshape(-1)
        et_prediction = self.prediction_tensor(
            preds_dict[self.et_name]
        ).reshape(-1)

        q_std = self.stat.get(f"{self.q_name}_std", 1.0)
        q_mean = self.stat.get(f"{self.q_name}_mean", 0.0)
        et_std = self.stat.get(f"{self.et_name}_std", 1.0)
        et_mean = self.stat.get(f"{self.et_name}_mean", 0.0)

        q_log = q_prediction * q_std + q_mean
        q_log = torch.clamp(q_log, -5.0, 10.0)
        sqrt_q_phys = torch.clamp(
            torch.pow(10.0, q_log) - 0.1,
            min=0.0,
        )
        q_phys = sqrt_q_phys.square()

        et_phys = et_prediction * et_std + et_mean
        p_phys = s_num[:, self.prcp_idx].reshape(-1)

        if q_phys.shape != p_phys.shape or et_phys.shape != p_phys.shape:
            raise ValueError(
                "Physical shape mismatch: "
                f"Q={q_phys.shape}, ET={et_phys.shape}, P={p_phys.shape}."
            )

        valid_mask = (
            torch.isfinite(p_phys)
            & torch.isfinite(q_phys)
            & torch.isfinite(et_phys)
        )
        if not valid_mask.any():
            return total_math_loss

        physics_loss = F.mse_loss(
            q_phys[valid_mask] + et_phys[valid_mask],
            p_phys[valid_mask],
        )
        return (
            (1.0 - self.alpha) * total_math_loss
            + self.alpha * physics_loss
        )
