"""Regression tests for the PUB wrapper's frozen-main integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtl_cgc.protocols.ch4_qssm_pub.constants import PUBScenario
from mtl_cgc.protocols.ch4_qssm_pub.protocol import PUBProtocol
from scripts.ch4_qssm_pub.pub_main import patch_main_for_pub


def _protocol(tmp_path: Path) -> PUBProtocol:
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("00000001\n", encoding="utf-8")
    target.write_text("00000002\n", encoding="utf-8")
    cfg = {
        "data": {
            "train_period": ["2015-04-01", "2021-09-30"],
            "test_period": ["2015-04-01", "2021-09-30"],
            "val_period": None,
            "spatial_split": True,
        },
        "pub": {
            "enabled": True,
            "protocol_version": "ch4b_pub_v3",
            "fold_id": 1,
            "scenario": PUBScenario.HPS_TARGET_SSM.value,
            "source_basin_file": str(source),
            "target_basin_file": str(target),
            "scaler_fit_scope": "source_only",
            "test_basin_scope": "target_only",
            "evaluation_task": "streamflow",
            "same_period_spatial_cv": True,
            "supervision": {
                "target": {"streamflow": False, "ssm": True},
            },
        },
    }
    return PUBProtocol.from_config(cfg, tmp_path)


def _pub_config() -> dict:
    return {
        "data": {
            "train_period": ["2015-04-01", "2021-09-30"],
            "test_period": ["2015-04-01", "2021-09-30"],
            "spatial_split": True,
        },
        "pub": {"enabled": True},
    }


def test_pub_patch_accepts_same_period_spatial_cv(tmp_path: Path) -> None:
    calls = []

    def original_validator(config):
        calls.append(config)
        raise AssertionError("Frozen temporal hold-out validator should be bypassed for PUB.")

    module = SimpleNamespace(
        get_hydro_dataloaders=lambda *args, **kwargs: None,
        validate_temporal_splits=original_validator,
        build_spatial_split=lambda *args, **kwargs: None,
        load_ungauged_list=lambda path: path,
    )
    patch_main_for_pub(module, _protocol(tmp_path), "train")
    module.validate_temporal_splits(_pub_config())
    assert calls == []


def test_pub_patch_rejects_nonidentical_target_dates(tmp_path: Path) -> None:
    module = SimpleNamespace(
        get_hydro_dataloaders=lambda *args, **kwargs: None,
        validate_temporal_splits=lambda config: None,
        build_spatial_split=lambda *args, **kwargs: None,
        load_ungauged_list=lambda path: path,
    )
    patch_main_for_pub(module, _protocol(tmp_path), "train")
    cfg = _pub_config()
    cfg["data"]["test_period"] = ["2018-10-01", "2021-09-30"]
    with pytest.raises(ValueError, match="identical train_period and test_period"):
        module.validate_temporal_splits(cfg)


def test_pub_patch_delegates_non_pub_validation(tmp_path: Path) -> None:
    calls = []

    def original_validator(config):
        calls.append(config)

    module = SimpleNamespace(
        get_hydro_dataloaders=lambda *args, **kwargs: None,
        validate_temporal_splits=original_validator,
        build_spatial_split=lambda *args, **kwargs: None,
        load_ungauged_list=lambda path: path,
    )
    patch_main_for_pub(module, _protocol(tmp_path), "train")
    config = {"pub": {"enabled": False}, "data": {}}
    module.validate_temporal_splits(config)
    assert calls == [config]
