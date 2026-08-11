#!/usr/bin/env python3
"""Audit generated Chapter 4B spatial PUB configurations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.constants import ProtocolDefaults, PUBScenario  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    load_json,
    load_yaml,
    resolve_project_path,
)
from mtl_cgc.protocols.ch4_qssm_pub.protocol import PUBProtocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest, PROJECT_ROOT)
    manifest = load_json(manifest_path)
    run_profile = str(manifest.get("run_profile", "formal")).strip().lower()
    if run_profile not in {"formal", "smoke"}:
        raise ValueError(f"Unsupported manifest run_profile: {run_profile}")
    expected_epochs = int(
        manifest.get("epochs", 1 if run_profile == "smoke" else 100)
    )
    defaults = ProtocolDefaults(epochs=expected_epochs)
    errors: list[str] = []

    for entry in manifest.get("entries", []):
        config_path = resolve_project_path(entry["config"], PROJECT_ROOT)
        try:
            cfg = load_yaml(config_path)
            protocol = PUBProtocol.from_config(cfg, PROJECT_ROOT)
            scenario = PUBScenario(str(entry["scenario"]))
            config_profile = str(cfg.get("pub", {}).get("run_profile", "formal"))
            if config_profile != run_profile:
                raise ValueError(
                    f"Config run_profile={config_profile} does not match "
                    f"manifest run_profile={run_profile}."
                )

            if protocol.scenario is not scenario:
                raise ValueError("Manifest scenario does not match YAML scenario.")

            data = cfg["data"]
            expected_period = [defaults.pub_start, defaults.pub_end]
            if list(data["train_period"]) != expected_period:
                raise ValueError(f"Unexpected train_period: {data['train_period']}")
            if list(data["test_period"]) != expected_period:
                raise ValueError(f"Unexpected test_period: {data['test_period']}")
            if data.get("val_period") is not None:
                raise ValueError("val_period must be null for fixed-epoch spatial PUB.")
            if int(data.get("sequence_length", -1)) != defaults.sequence_length:
                raise ValueError("sequence_length mismatch.")
            if int(data.get("batch_size", -1)) != defaults.batch_size:
                raise ValueError("batch_size mismatch.")
            if bool(data.get("sample_filter", {}).get("enabled", False)):
                raise ValueError(
                    "Global sample_filter must be disabled; PUB uses role-aware filtering."
                )

            targets = {str(t["name"]).lower(): t for t in data["targets"]}
            if set(targets) != set(scenario.active_tasks):
                raise ValueError(
                    f"Task mismatch: configured={sorted(targets)}, "
                    f"expected={sorted(scenario.active_tasks)}"
                )
            if "ssm" in targets and bool(targets["ssm"].get("interpolate_missing", True)):
                raise ValueError("SSM interpolation must be disabled.")

            if scenario is PUBScenario.STL_Q:
                if float(targets["streamflow"]["loss_weight"]) != 1.0:
                    raise ValueError("STL-Q must use streamflow loss_weight=1.0.")
            else:
                q_w = float(targets["streamflow"]["loss_weight"])
                ssm_w = float(targets["ssm"]["loss_weight"])
                if abs(q_w - defaults.streamflow_weight) > 1e-12:
                    raise ValueError(f"Unexpected Q loss weight: {q_w}")
                if abs(ssm_w - defaults.ssm_weight) > 1e-12:
                    raise ValueError(f"Unexpected SSM loss weight: {ssm_w}")

            training = cfg["training"]
            if int(training.get("epochs", -1)) != defaults.epochs:
                raise ValueError("Epoch count mismatch.")
            if bool(training.get("early_stopping", {}).get("enabled", True)):
                raise ValueError("Early stopping must be disabled.")
            if str(training.get("monitor", {}).get("name", "")) != "loss":
                raise ValueError("Training monitor must be loss when validation is absent.")

            evaluation = cfg.get("evaluation_protocol", {})
            if evaluation.get("primary_metric") != "streamflow_nse_median":
                raise ValueError("Formal PUB primary metric must be streamflow_nse_median.")

            print(
                f"PASS fold={protocol.fold_id:02d} "
                f"scenario={scenario.value:<18s} "
                f"train_basins={len(protocol.effective_training_basins):3d} "
                f"scaler={len(protocol.source_basins):3d} "
                f"test={len(protocol.target_basins):3d}"
            )
        except Exception as exc:
            errors.append(f"{config_path}: {exc}")
            print(f"FAIL {config_path}: {exc}")

    if errors:
        print("\nConfiguration audit errors:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print(f"\nConfiguration audit passed for {len(manifest.get('entries', []))} files.")


if __name__ == "__main__":
    main()
