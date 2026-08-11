"""Unit tests for Chapter 4B PUB supervision and leakage constraints."""

from __future__ import annotations

from pathlib import Path

import pytest

from mtl_cgc.protocols.ch4_qssm_pub.constants import PUBScenario
from mtl_cgc.protocols.ch4_qssm_pub.protocol import PUBProtocol


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def make_config(source: Path, target: Path, scenario: PUBScenario) -> dict:
    return {
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
            "scenario": scenario.value,
            "source_basin_file": str(source),
            "target_basin_file": str(target),
            "scaler_fit_scope": "source_only",
            "test_basin_scope": "target_only",
            "evaluation_task": "streamflow",
            "same_period_spatial_cv": True,
            "supervision": {
                "source": {"streamflow": True, "ssm": scenario is not PUBScenario.STL_Q},
                "target": {
                    "streamflow": False,
                    "ssm": scenario.target_ssm_supervision,
                },
            },
        },
    }


def test_assisted_mtl_training_set_is_source_plus_target(tmp_path: Path) -> None:
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    write_ids(source, ["00000001", "00000002"])
    write_ids(target, ["00000003"])
    protocol = PUBProtocol.from_config(
        make_config(source, target, PUBScenario.CGC_TARGET_SSM), tmp_path
    )
    assert protocol.effective_training_basins == ["00000001", "00000002", "00000003"]
    assert protocol.masked_streamflow_basins == ["00000003"]


def test_stl_q_training_set_is_source_only(tmp_path: Path) -> None:
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    write_ids(source, ["00000001", "00000002"])
    write_ids(target, ["00000003"])
    protocol = PUBProtocol.from_config(
        make_config(source, target, PUBScenario.STL_Q), tmp_path
    )
    assert protocol.effective_training_basins == ["00000001", "00000002"]
    assert protocol.masked_streamflow_basins == []


def test_target_streamflow_supervision_is_forbidden(tmp_path: Path) -> None:
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    write_ids(source, ["00000001"])
    write_ids(target, ["00000002"])
    config = make_config(source, target, PUBScenario.HPS_TARGET_SSM)
    config["pub"]["supervision"]["target"]["streamflow"] = True
    with pytest.raises(ValueError, match="streamflow supervision is forbidden"):
        PUBProtocol.from_config(config, tmp_path)


def test_same_period_is_required(tmp_path: Path) -> None:
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    write_ids(source, ["00000001"])
    write_ids(target, ["00000002"])
    config = make_config(source, target, PUBScenario.HPS_TARGET_SSM)
    config["data"]["test_period"] = ["2018-10-01", "2021-09-30"]
    with pytest.raises(ValueError, match="identical train_period and test_period"):
        PUBProtocol.from_config(config, tmp_path)
