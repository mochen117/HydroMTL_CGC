#!/usr/bin/env python3
"""Runtime data-semantics audit for Chapter 4B spatial PUB.

The audit verifies the scientific supervision policy on real NetCDF data before
formal training:

- source Q is available;
- source SSM is available where observed;
- target Q is completely masked in training;
- target SSM remains available where observed;
- the effective MTL training basin set includes both source and target basins;
- scaler and final evaluation scopes remain source-only and target-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.native_runtime import bootstrap_native_runtime  # noqa: E402

bootstrap_native_runtime(strict=True)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mtl_cgc.protocols.ch4_qssm_pub.data_adapter import build_data_plan  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.protocol import PUBProtocol  # noqa: E402
from mtl_cgc.utils.temporal import expand_period_for_sequence  # noqa: E402
from mtl_cgc.data.data_loaders import load_nc_to_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-source", type=int, default=3)
    parser.add_argument("--sample-target", type=int, default=3)
    return parser.parse_args()


def finite_count(values: np.ndarray) -> int:
    return int(np.isfinite(values).sum())


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = PUBProtocol.from_config(config, PROJECT_ROOT)
    plan = build_data_plan(protocol)

    source = list(plan.scaler_basins)[: max(1, args.sample_source)]
    target = list(plan.evaluation_basins)[: max(1, args.sample_target)]

    if protocol.scenario.include_target_during_training:
        sampled_training = source + target
        masked = target
    else:
        sampled_training = source
        masked = []

    data_cfg = config["data"]
    period = list(data_cfg["train_period"])
    read_period = expand_period_for_sequence(
        period,
        int(data_cfg.get("sequence_length", 365)),
    )

    data_root = Path(data_cfg["data_root"]).expanduser()
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    try:
        raw = load_nc_to_dict(
        data_root=data_root,
        basin_ids=sampled_training,
        data_cfg=data_cfg,
        split_period=read_period,
        split_name="train",
        ungauged_basins=masked,
        mask_target="streamflow",
        )
    except ImportError as exc:
        message = str(exc)
        if "GLIBCXX_" in message:
            raise RuntimeError(
                "NetCDF runtime failed before the PUB semantic audit. This is "
                "an environment/shared-library issue, not a supervision-policy "
                "failure. Run scripts/ch4_qssm_pub/check_runtime_environment.py "
                "and ensure the active conda environment's libstdc++ is loaded "
                "before retrying."
            ) from exc
        raise

    positions = {basin_id: idx for idx, basin_id in enumerate(sampled_training)}
    source_idx = [positions[item] for item in source]
    target_idx = [positions[item] for item in target if item in positions]

    q = raw["y_dict"].get("streamflow")
    ssm = raw["y_dict"].get("ssm")

    if q is None:
        raise RuntimeError("Configured PUB audit requires streamflow target.")

    source_q = finite_count(q[source_idx])
    target_q = finite_count(q[target_idx]) if target_idx else 0
    source_ssm = finite_count(ssm[source_idx]) if ssm is not None else 0
    target_ssm = finite_count(ssm[target_idx]) if (ssm is not None and target_idx) else 0

    print("=" * 88)
    print("Chapter 4B PUB runtime data-semantics audit")
    print("-" * 88)
    print(f"Scenario                     : {protocol.scenario.value}")
    print(f"Fold                         : {protocol.fold_id:02d}")
    print(f"Source basins (full fold)    : {len(protocol.source_basins)}")
    print(f"Target basins (full fold)    : {len(protocol.target_basins)}")
    print(f"Effective training basins    : {plan.training_count}")
    print(f"Scaler basins                : {plan.scaler_count} (source only)")
    print(f"Evaluation basins            : {plan.evaluation_count} (target only)")
    print(f"Sampled source Q finite      : {source_q}")
    print(f"Sampled source SSM finite    : {source_ssm}")
    print(f"Sampled target Q finite      : {target_q}")
    print(f"Sampled target SSM finite    : {target_ssm}")
    print("=" * 88)

    if source_q <= 0:
        raise RuntimeError("FAIL: no finite source-basin Q supervision found.")

    if protocol.scenario.include_target_during_training:
        if target_q != 0:
            raise RuntimeError(
                "FAIL: target-basin Q was not completely masked during training."
            )
        if ssm is None or target_ssm <= 0:
            raise RuntimeError(
                "FAIL: target-basin SSM was not retained as auxiliary supervision."
            )
        if plan.training_count != len(protocol.source_basins) + len(protocol.target_basins):
            raise RuntimeError("FAIL: assisted MTL training set is not source+target.")
    else:
        if plan.training_count != len(protocol.source_basins):
            raise RuntimeError("FAIL: source-only scenario unexpectedly includes target basins.")

    if set(plan.scaler_basins) != set(protocol.source_basins):
        raise RuntimeError("FAIL: scaler scope is not exactly the source basin set.")
    if set(plan.evaluation_basins) != set(protocol.target_basins):
        raise RuntimeError("FAIL: evaluation scope is not exactly the target basin set.")

    print("PUB semantic audit: PASS")


if __name__ == "__main__":
    main()
