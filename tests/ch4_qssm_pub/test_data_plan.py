"""Unit tests for explicit PUB basin-role plans."""

from __future__ import annotations

from pathlib import Path

from mtl_cgc.protocols.ch4_qssm_pub.constants import PUBScenario
from mtl_cgc.protocols.ch4_qssm_pub.data_adapter import build_data_plan
from mtl_cgc.protocols.ch4_qssm_pub.protocol import PUBProtocol


def protocol(tmp_path: Path, scenario: PUBScenario) -> PUBProtocol:
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    source.write_text("00000001\n00000002\n", encoding="utf-8")
    target.write_text("00000003\n", encoding="utf-8")
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
            "scenario": scenario.value,
            "source_basin_file": str(source),
            "target_basin_file": str(target),
            "scaler_fit_scope": "source_only",
            "test_basin_scope": "target_only",
            "evaluation_task": "streamflow",
            "same_period_spatial_cv": True,
            "supervision": {
                "target": {"streamflow": False, "ssm": scenario.target_ssm_supervision}
            },
        },
    }
    return PUBProtocol.from_config(cfg, tmp_path)


def test_assisted_plan() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = protocol(Path(d), PUBScenario.HPS_TARGET_SSM)
        plan = build_data_plan(p)
        assert plan.training_basins == ("00000001", "00000002", "00000003")
        assert plan.scaler_basins == ("00000001", "00000002")
        assert plan.masked_streamflow_basins == ("00000003",)
        assert plan.evaluation_basins == ("00000003",)


def test_role_aware_filter_keeps_source_q_days_and_target_ssm_days(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import numpy as np

    from mtl_cgc.protocols.ch4_qssm_pub.data_adapter import _apply_role_aware_sample_filter

    p = protocol(tmp_path, PUBScenario.HPS_TARGET_SSM)
    # rho=2 => index 0 is historical context; four target dates occupy 1:5.
    dataset = SimpleNamespace(
        basin_ids=["00000001", "00000002", "00000003"],
        rho=2,
        num_time_steps=4,
        num_basins=3,
        y_dict={
            "streamflow": np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=float,
            ),
            "ssm": np.array(
                [
                    [np.nan, 0.1, np.nan, 0.2, np.nan],
                    [np.nan, np.nan, 0.2, np.nan, 0.3],
                    [np.nan, 0.1, np.nan, 0.2, np.nan],
                ],
                dtype=float,
            ),
        },
    )

    counts = _apply_role_aware_sample_filter(dataset, p)
    assert counts == {"source_samples": 8, "target_samples": 2, "total_samples": 10}
    assert dataset.num_samples == 10
