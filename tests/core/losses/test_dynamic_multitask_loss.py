# ==============================================================================
# Tests for the mask-aware DynamicMultiTaskLoss criterion.
# ==============================================================================

from __future__ import annotations

import math

import pytest
import torch

from mtl_cgc.core.losses.crits import DynamicMultiTaskLoss


def _config(
    *,
    targets: list[dict],
    base_loss: str = "rmse",
    eps: float = 1.0e-6,
) -> dict:
    return {
        "data": {
            "targets": targets,
            "static_features": [],
        },
        "training": {
            "loss": {
                "base_loss": base_loss,
                "eps": eps,
            }
        },
        "model": {
            "physics_constraints": {
                "water_balance": {
                    "enabled": False,
                    "alpha": 0.1,
                }
            }
        },
    }


def test_rmse_matches_manual_value() -> None:
    criterion = DynamicMultiTaskLoss(
        _config(
            targets=[{"name": "ssm", "loss_weight": 1.0}],
            base_loss="rmse",
        ),
        {},
    )

    prediction = torch.tensor([[1.0], [3.0]], requires_grad=True)
    target = torch.tensor([[1.0], [1.0]])

    loss = criterion(
        {"ssm": prediction},
        {"ssm": target},
    )

    expected = math.sqrt(2.0 + 1.0e-6)
    assert loss.item() == pytest.approx(expected, rel=1.0e-6)

    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_nan_targets_are_ignored() -> None:
    criterion = DynamicMultiTaskLoss(
        _config(targets=[{"name": "ssm", "loss_weight": 1.0}]),
        {},
    )

    prediction = torch.tensor([[1.0], [9.0], [5.0]], requires_grad=True)
    target = torch.tensor([[1.0], [float("nan")], [1.0]])

    loss = criterion(
        {"ssm": prediction},
        {"ssm": target},
    )

    expected = math.sqrt(8.0 + 1.0e-6)
    assert loss.item() == pytest.approx(expected, rel=1.0e-6)
    assert criterion.count_valid_observations(prediction, target) == 2


def test_all_nan_target_returns_graph_connected_zero() -> None:
    criterion = DynamicMultiTaskLoss(
        _config(targets=[{"name": "ssm", "loss_weight": 1.0}]),
        {},
    )

    prediction = torch.tensor([[1.0], [2.0]], requires_grad=True)
    target = torch.full_like(prediction, float("nan"))

    loss = criterion(
        {"ssm": prediction},
        {"ssm": target},
    )

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_weighted_q_ssm_rmse() -> None:
    criterion = DynamicMultiTaskLoss(
        _config(
            targets=[
                {"name": "streamflow", "loss_weight": 0.5},
                {"name": "ssm", "loss_weight": 0.5},
            ]
        ),
        {},
    )

    q_pred = torch.tensor([[0.0], [2.0]], requires_grad=True)
    q_obs = torch.tensor([[0.0], [0.0]])
    ssm_pred = torch.tensor([[1.0], [3.0]], requires_grad=True)
    ssm_obs = torch.tensor([[1.0], [1.0]])

    loss = criterion(
        {"streamflow": q_pred, "ssm": ssm_pred},
        {"streamflow": q_obs, "ssm": ssm_obs},
    )

    task_rmse = math.sqrt(2.0 + 1.0e-6)
    assert loss.item() == pytest.approx(task_rmse, rel=1.0e-6)


def test_zero_weight_task_may_be_absent() -> None:
    criterion = DynamicMultiTaskLoss(
        _config(
            targets=[
                {"name": "streamflow", "loss_weight": 1.0},
                {"name": "ssm", "loss_weight": 0.0},
            ]
        ),
        {},
    )

    q_pred = torch.tensor([[0.0], [1.0]], requires_grad=True)
    q_obs = torch.tensor([[0.0], [0.0]])

    loss = criterion(
        {"streamflow": q_pred},
        {"streamflow": q_obs},
    )

    assert torch.isfinite(loss)


def test_unsupported_base_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported base_loss"):
        DynamicMultiTaskLoss(
            _config(
                targets=[{"name": "ssm", "loss_weight": 1.0}],
                base_loss="nse",
            ),
            {},
        )
