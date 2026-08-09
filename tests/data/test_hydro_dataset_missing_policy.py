# ==============================================================================
# Tests for target interpolation and train-only sample filtering in HydroDataset.
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from mtl_cgc.data.data_sets import HydroDataset


class IdentityScaler:
    """Minimal fitted-scaler stub for isolated HydroDataset tests."""

    def transform(
        self,
        dyn: np.ndarray,
        s_num: np.ndarray,
        s_cat: Optional[np.ndarray],
        y_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        return (
            np.array(dyn, copy=True),
            np.array(s_num, copy=True),
            None if s_cat is None else np.array(s_cat, copy=True),
            {
                task: np.array(values, copy=True)
                for task, values in y_dict.items()
            },
        )


def _raw_data() -> Dict[str, Any]:
    # sequence_length=3 and a four-day target period require six raw days.
    return {
        "dyn": np.ones((1, 6, 2), dtype=np.float32),
        "s_num": np.ones((1, 1), dtype=np.float32),
        "s_cat": None,
        "y_dict": {
            "ssm": np.array(
                [[0.10, np.nan, 0.20, np.nan, 0.30, np.nan]],
                dtype=np.float32,
            )
        },
    }


def _params(*, use_filter: bool) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "sequence_length": 3,
        "targets": [
            {
                "name": "ssm",
                "interpolate_missing": False,
                "constraint": "non_negative",
            }
        ],
    }

    if use_filter:
        params["sample_filter"] = {
            "enabled": True,
            "required_valid_targets": ["ssm"],
            "apply_to_modes": ["train"],
        }

    return params


def test_stl_ssm_preserves_nan_and_filters_training_samples() -> None:
    dataset = HydroDataset(
        raw_data=_raw_data(),
        data_params=_params(use_filter=True),
        basin_ids=["00000001"],
        target_period=["2020-01-03", "2020-01-06"],
        mode="train",
        scaler=IdentityScaler(),
    )

    target_slice = dataset.y_dict["ssm"][:, 2:]
    assert np.isfinite(target_slice).sum() == 2
    assert np.isnan(target_slice).sum() == 2
    assert len(dataset) == 2
    assert dataset.time_index.tolist() == [0, 2]


def test_joint_dataset_preserves_nan_but_keeps_all_dates() -> None:
    dataset = HydroDataset(
        raw_data=_raw_data(),
        data_params=_params(use_filter=False),
        basin_ids=["00000001"],
        target_period=["2020-01-03", "2020-01-06"],
        mode="train",
        scaler=IdentityScaler(),
    )

    target_slice = dataset.y_dict["ssm"][:, 2:]
    assert np.isfinite(target_slice).sum() == 2
    assert np.isnan(target_slice).sum() == 2
    assert len(dataset) == 4
    assert dataset.time_index.tolist() == [0, 1, 2, 3]


def test_test_mode_does_not_apply_train_only_filter() -> None:
    dataset = HydroDataset(
        raw_data=_raw_data(),
        data_params=_params(use_filter=True),
        basin_ids=["00000001"],
        target_period=["2020-01-03", "2020-01-06"],
        mode="test",
        scaler=IdentityScaler(),
    )

    assert len(dataset) == 4
