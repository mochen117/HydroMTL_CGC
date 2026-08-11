#!/usr/bin/env python3
"""PUB-aware entry point for the frozen HydroMTL_CGC core runner.

This wrapper deliberately does not modify ``main.py`` or the standard data
loader used by completed Chapter 3 / Chapter 4A experiments.  Instead it
injects an explicit source/target fold and a PUB-specific DataLoader adapter.
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import date
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.native_runtime import bootstrap_native_runtime  # noqa: E402

bootstrap_native_runtime(strict=True)

import yaml  # noqa: E402

from mtl_cgc.protocols.ch4_qssm_pub.data_adapter import (  # noqa: E402
    build_data_plan,
    make_pub_loader,
)
from mtl_cgc.protocols.ch4_qssm_pub.io_utils import atomic_write_json  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.protocol import PUBProtocol  # noqa: E402


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_known_args()


def import_project_main(project_root: Path) -> ModuleType:
    """Import the repository's current ``main.py`` without copying its logic."""

    main_path = project_root / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(main_path)

    spec = importlib.util.spec_from_file_location("hydromtl_project_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import project main module: {main_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_pub_temporal_validator(
    original_validator: Any,
    protocol: PUBProtocol,
):
    """Return a validator that permits same-period spatial PUB safely.

    The frozen project validator enforces a conventional temporal hold-out
    (``train_end < test_start``). Chapter 4B instead holds out basins while
    keeping the target-date period identical between training and evaluation.
    This wrapper bypasses only that temporal non-overlap rule for validated PUB
    configurations and delegates all non-PUB configurations to the original
    project validator.
    """

    def validate(config: Any) -> None:
        pub_cfg = config.get("pub", {}) if hasattr(config, "get") else {}
        if not bool(pub_cfg.get("enabled", False)):
            original_validator(config)
            return

        if not protocol.same_period_spatial_cv:
            raise ValueError("PUB temporal override requires same_period_spatial_cv=true.")

        data_cfg = config.get("data", {})
        if not bool(data_cfg.get("spatial_split", False)):
            raise ValueError("PUB temporal override requires data.spatial_split=true.")

        train_period = list(data_cfg.get("train_period") or [])
        test_period = list(data_cfg.get("test_period") or [])
        if len(train_period) != 2 or len(test_period) != 2:
            raise ValueError(
                "PUB train_period and test_period must each contain [start, end]."
            )
        if train_period != test_period:
            raise ValueError(
                "PUB requires identical train_period and test_period because "
                "generalization is spatial, not temporal. "
                f"Got train={train_period}, test={test_period}."
            )

        start = date.fromisoformat(str(train_period[0]))
        end = date.fromisoformat(str(train_period[1]))
        if start > end:
            raise ValueError(
                f"Invalid PUB target-date period: start={start} is after end={end}."
            )

    return validate


def patch_main_for_pub(
    module: ModuleType,
    protocol: PUBProtocol,
    run_mode: str,
) -> None:
    """Inject explicit spatial folds and role-aware PUB DataLoader behavior."""

    source_basins = sorted(protocol.source_basins)
    target_basins = sorted(protocol.target_basins)
    plan = build_data_plan(protocol)

    def explicit_spatial_split(
        config: Any,
        all_basin_ids: list[str],
    ) -> tuple[list[str], list[str], str]:
        del config
        discovered = {str(item).zfill(8) for item in all_basin_ids}
        missing_source = sorted(set(source_basins) - discovered)
        missing_target = sorted(set(target_basins) - discovered)
        if missing_source or missing_target:
            raise ValueError(
                "PUB fold references basins absent from data_root. "
                f"Missing source={missing_source[:10]}, "
                f"missing target={missing_target[:10]}"
            )
        # ``main.py`` still sees the conventional source=train, target=test
        # split.  The injected PUB loader expands the *effective* MTL training
        # set to source+target when target SSM supervision is enabled.
        return source_basins, target_basins, f"pub:fold{protocol.fold_id:02d}"

    original_loader = module.get_hydro_dataloaders
    original_temporal_validator = getattr(module, "validate_temporal_splits", None)
    if original_temporal_validator is None:
        raise AttributeError(
            "Frozen main.py is missing validate_temporal_splits; PUB overlay "
            "cannot safely inject same-period spatial validation."
        )

    module.build_spatial_split = explicit_spatial_split
    module.get_hydro_dataloaders = make_pub_loader(original_loader, protocol)
    module.validate_temporal_splits = make_pub_temporal_validator(
        original_temporal_validator,
        protocol,
    )

    # Prevent any CLI ungauged list from altering the explicit protocol.
    module.load_ungauged_list = lambda path: None

    print("\n" + "=" * 112)
    print("Chapter 4 Experiment B: spatial PUB protocol injection")
    print("-" * 112)
    print(f"Protocol version          : {protocol.protocol_version}")
    print(f"Fold                      : {protocol.fold_id:02d}")
    print(f"Scenario                  : {protocol.scenario.value}")
    print(f"Mode                      : {run_mode}")
    print(f"Source basins             : {len(source_basins)}")
    print(f"Target basins             : {len(target_basins)}")
    print(f"Effective training basins : {plan.training_count}")
    print(f"Scaler basins             : {plan.scaler_count} (source only)")
    print(f"Masked target-Q basins    : {len(plan.masked_streamflow_basins)}")
    print(f"Evaluation basins         : {plan.evaluation_count} (target only)")
    print("Target Q supervision      : disabled")
    print(
        "Target SSM supervision    : "
        f"{'enabled' if protocol.target_ssm_supervision else 'disabled'}"
    )
    print("Train/test target dates   : identical (spatial PUB)")
    print("=" * 112 + "\n")


def write_protocol_snapshot(
    config: dict[str, Any],
    protocol: PUBProtocol,
    project_root: Path,
    mode: str,
) -> None:
    """Export the exact fold, basin roles, and supervision policy used by a run."""

    save_root = Path(config.get("experiment", {}).get("save_dir", "experiments"))
    if not save_root.is_absolute():
        save_root = project_root / save_root
    experiment_name = str(config["experiment"]["name"])
    save_dir = save_root / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = protocol.snapshot()
    payload["mode"] = mode
    payload["experiment_name"] = experiment_name
    payload["train_period"] = list(config["data"]["train_period"])
    payload["test_period"] = list(config["data"]["test_period"])
    atomic_write_json(save_dir / "pub_protocol_snapshot.json", payload)


def main() -> None:
    wrapper_args, remaining = parse_wrapper_args()
    project_root = wrapper_args.project_root.expanduser().resolve()
    config_path = (
        wrapper_args.config.expanduser().resolve()
        if wrapper_args.config.is_absolute()
        else (project_root / wrapper_args.config).resolve()
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping: {config_path}")

    protocol = PUBProtocol.from_config(config, project_root)
    project_main = import_project_main(project_root)
    patch_main_for_pub(project_main, protocol, wrapper_args.mode)
    write_protocol_snapshot(
        config=config,
        protocol=protocol,
        project_root=project_root,
        mode=wrapper_args.mode,
    )

    sys.argv = [
        str(project_root / "main.py"),
        "--config",
        str(config_path),
        "--mode",
        wrapper_args.mode,
        *remaining,
    ]
    project_main.main()


if __name__ == "__main__":
    main()
