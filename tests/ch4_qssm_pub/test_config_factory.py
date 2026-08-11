"""Unit tests for generated Chapter 4B PUB configurations."""

from __future__ import annotations

from pathlib import Path

from mtl_cgc.protocols.ch4_qssm_pub.config_factory import build_pub_config
from mtl_cgc.protocols.ch4_qssm_pub.constants import PUBScenario, ProtocolDefaults


def base_config() -> dict:
    return {
        "experiment": {"name": "ch4a_base", "save_dir": "experiments"},
        "data": {
            "dynamic_features": ["prcp", "tmax"],
            "static_features": ["area_gages2", "p_mean"],
            "targets": [
                {"name": "streamflow", "loss_weight": 0.5, "interpolate_missing": False},
                {
                    "name": "ssm",
                    "loss_weight": 0.5,
                    "interpolate_missing": False,
                    "unit_scale": 0.01,
                },
            ],
            "sequence_length": 365,
            "batch_size": 64,
        },
        "model": {
            "architecture": "cgc",
            "cgc": {"task_experts": [4, 4]},
            "task_towers": [{"hidden_dim": 64}, {"hidden_dim": 64}],
        },
        "training": {"epochs": 100, "scheduler": {"mode": "min"}},
        "reproducibility": {"seed": 42},
    }


def _ids(tmp_path: Path) -> tuple[Path, Path]:
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    source.write_text("00000001\n", encoding="utf-8")
    target.write_text("00000002\n", encoding="utf-8")
    return source, target


def test_cgc_pub_config_uses_same_period_and_target_ssm(tmp_path: Path) -> None:
    source, target = _ids(tmp_path)
    cfg = build_pub_config(
        base_config(), PUBScenario.CGC_TARGET_SSM, 1, 42,
        source, target, tmp_path, ProtocolDefaults()
    )
    assert cfg["data"]["train_period"] == ["2015-04-01", "2021-09-30"]
    assert cfg["data"]["test_period"] == cfg["data"]["train_period"]
    assert cfg["data"]["val_period"] is None
    assert cfg["pub"]["supervision"]["target"] == {"streamflow": False, "ssm": True}
    assert cfg["model"]["architecture"] == "cgc"
    assert cfg["training"]["early_stopping"]["enabled"] is False


def test_stl_q_has_only_streamflow_head(tmp_path: Path) -> None:
    source, target = _ids(tmp_path)
    cfg = build_pub_config(
        base_config(), PUBScenario.STL_Q, 1, 42,
        source, target, tmp_path, ProtocolDefaults()
    )
    assert cfg["model"]["architecture"] == "stl"
    assert [item["name"] for item in cfg["data"]["targets"]] == ["streamflow"]
    assert cfg["data"]["targets"][0]["loss_weight"] == 1.0
    assert cfg["pub"]["supervision"]["target"]["ssm"] is False


def test_pub_outputs_are_scoped_under_chapter4b_results(tmp_path: Path) -> None:
    source, target = _ids(tmp_path)
    formal = build_pub_config(
        base_config(),
        PUBScenario.HPS_TARGET_SSM,
        1,
        42,
        source,
        target,
        tmp_path,
        ProtocolDefaults(),
        run_profile="formal",
    )
    smoke = build_pub_config(
        base_config(),
        PUBScenario.HPS_TARGET_SSM,
        1,
        42,
        source,
        target,
        tmp_path,
        ProtocolDefaults(epochs=1),
        run_profile="smoke",
    )

    assert formal["experiment"]["save_dir"] == "experiments/ch4_qssm_pub/runs"
    assert formal["experiment"]["name"].startswith("ch4b_pub_formal_")
    assert smoke["experiment"]["name"].startswith("ch4b_pub_smoke_")
    assert formal["experiment"]["name"] != smoke["experiment"]["name"]
    assert formal["pub"]["run_profile"] == "formal"
    assert smoke["pub"]["run_profile"] == "smoke"
