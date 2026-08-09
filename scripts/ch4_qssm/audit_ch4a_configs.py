#!/usr/bin/env python3
"""
Audit Chapter 4 Experiment A Q-to-SSM configurations.

The script validates task activation, loss semantics, temporal periods,
sample filtering, evaluation metrics, and stage-specific epoch settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


ConfigDict = Dict[str, Any]


def require(condition: bool, message: str) -> None:
    """Raise a descriptive error when a protocol condition is violated."""
    if not condition:
        raise ValueError(message)


def classify_stage(config_path: Path) -> str:
    """Classify an Experiment A configuration by file name."""
    name = config_path.stem.lower()

    if "qpre_finetune" in name:
        return "finetune"
    if "_qpre_seed" in name:
        return "pretrain"
    return "baseline"


def load_yaml(path: Path) -> ConfigDict:
    """Load one YAML configuration."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    require(
        isinstance(config, dict),
        f"Configuration must be a dictionary: {path}",
    )
    return config


def normalized_sample_filter(
    data_config: ConfigDict,
) -> ConfigDict | None:
    """Return a normalized sample-filter dictionary."""
    sample_filter = data_config.get("sample_filter")

    if sample_filter in (None, {}):
        return None

    require(
        isinstance(sample_filter, dict),
        "data.sample_filter must be a dictionary or null.",
    )

    return {
        "enabled": bool(sample_filter.get("enabled", False)),
        "required_valid_targets": [
            str(task).strip().lower()
            for task in sample_filter.get(
                "required_valid_targets",
                [],
            )
        ],
        "apply_to_modes": [
            str(mode).strip().lower()
            for mode in sample_filter.get(
                "apply_to_modes",
                [],
            )
        ],
    }


def expected_training_filter(task_name: str) -> ConfigDict:
    """Build the canonical train-only target filter."""
    return {
        "enabled": True,
        "required_valid_targets": [task_name],
        "apply_to_modes": ["train"],
    }


def resolve_config_path(
    raw_path: str,
    project_root: Path,
) -> Path:
    """Resolve a manifest configuration path."""
    path = Path(raw_path)

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit Chapter 4 Experiment A "
            "Q-to-SSM configurations."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    """Audit every YAML file registered in the manifest."""
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.resolve()

    require(
        manifest_path.exists(),
        f"Manifest not found: {manifest_path}",
    )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    require(
        manifest.get("protocol") == "q_to_ssm",
        "Manifest protocol must be 'q_to_ssm'.",
    )

    raw_config_paths = manifest.get("configs")
    require(
        isinstance(raw_config_paths, list)
        and len(raw_config_paths) > 0,
        "Manifest contains no configuration paths.",
    )

    pretrain_period = list(
        manifest["pretrain_period"]
    )
    finetune_period = list(
        manifest["finetune_train_period"]
    )
    final_test_period = list(
        manifest["test_period"]
    )

    expected_joint_epochs = int(
        manifest["joint_epochs"]
    )
    expected_pretrain_epochs = int(
        manifest["pretrain_epochs"]
    )
    expected_batch_size = int(
        manifest["batch_size"]
    )

    config_paths = [
        resolve_config_path(
            raw_path,
            project_root,
        )
        for raw_path in raw_config_paths
    ]

    require(
        len(config_paths) == 8,
        f"Expected 8 configurations, got {len(config_paths)}.",
    )

    experiment_names: Set[str] = set()

    for config_path in sorted(config_paths):
        require(
            config_path.exists(),
            f"Configuration not found: {config_path}",
        )

        config = load_yaml(config_path)
        stage = classify_stage(config_path)

        experiment = config.get("experiment", {})
        data = config.get("data", {})
        model = config.get("model", {})
        training = config.get("training", {})
        evaluation = config.get(
            "evaluation_protocol",
            {},
        )

        require(
            isinstance(experiment, dict),
            f"Missing experiment block: {config_path}",
        )
        require(
            isinstance(data, dict),
            f"Missing data block: {config_path}",
        )
        require(
            isinstance(model, dict),
            f"Missing model block: {config_path}",
        )
        require(
            isinstance(training, dict),
            f"Missing training block: {config_path}",
        )
        require(
            isinstance(evaluation, dict),
            f"Missing evaluation_protocol block: {config_path}",
        )

        experiment_name = str(
            experiment.get("name", "")
        ).strip()

        require(
            experiment_name,
            f"Missing experiment.name: {config_path}",
        )
        require(
            experiment_name not in experiment_names,
            f"Duplicate experiment name: {experiment_name}",
        )
        experiment_names.add(experiment_name)

        targets = data.get("targets", [])
        require(
            isinstance(targets, list) and targets,
            f"data.targets is empty: {config_path}",
        )

        weights = {
            str(target.get("name", ""))
            .strip()
            .lower(): float(
                target.get("loss_weight", 1.0)
            )
            for target in targets
        }

        active_tasks = {
            task_name
            for task_name, weight in weights.items()
            if weight > 0.0
        }

        sample_filter = normalized_sample_filter(data)

        loss_config = training.get("loss", {})
        require(
            isinstance(loss_config, dict),
            f"training.loss must be a dictionary: {config_path}",
        )

        require(
            loss_config.get("base_loss") == "rmse",
            f"base_loss must be rmse: {config_path}",
        )

        require(
            int(data.get("batch_size", -1))
            == expected_batch_size,
            f"Unexpected batch_size: {config_path}",
        )

        for validation_key in (
            "val_period",
            "valid_period",
            "validation_period",
        ):
            require(
                data.get(validation_key) is None,
                f"{validation_key} must be null: {config_path}",
            )

        if stage == "pretrain":
            expected_train_period = pretrain_period

            # Q-pretraining is evaluated over the subsequent
            # Q-SSM fine-tuning period, not the final test period.
            expected_test_period = finetune_period
            expected_epochs = expected_pretrain_epochs
        else:
            expected_train_period = finetune_period
            expected_test_period = final_test_period
            expected_epochs = expected_joint_epochs

        require(
            list(data.get("train_period", []))
            == expected_train_period,
            (
                f"Unexpected train_period in {config_path}: "
                f"{data.get('train_period')}"
            ),
        )

        require(
            list(data.get("test_period", []))
            == expected_test_period,
            (
                f"Unexpected test_period in {config_path}: "
                f"{data.get('test_period')}; "
                f"expected {expected_test_period}"
            ),
        )

        require(
            int(training.get("epochs", -1))
            == expected_epochs,
            (
                f"Unexpected epochs in {config_path}: "
                f"{training.get('epochs')}"
            ),
        )

        early_stopping = training.get(
            "early_stopping",
            {},
        )
        if isinstance(early_stopping, dict):
            require(
                not bool(
                    early_stopping.get(
                        "enabled",
                        False,
                    )
                ),
                f"Early stopping must be disabled: {config_path}",
            )

        physics_config = (
            model.get("physics_constraints", {})
            .get("water_balance", {})
        )
        if isinstance(physics_config, dict):
            require(
                not bool(
                    physics_config.get(
                        "enabled",
                        False,
                    )
                ),
                (
                    "Water-balance loss must be disabled "
                    f"for Q-to-SSM: {config_path}"
                ),
            )

        ssm_target = next(
            (
                target
                for target in targets
                if str(
                    target.get("name", "")
                ).strip().lower() == "ssm"
            ),
            None,
        )

        if ssm_target is not None:
            require(
                ssm_target.get(
                    "interpolate_missing"
                )
                is False,
                (
                    "SSM interpolation must be disabled: "
                    f"{config_path}"
                ),
            )

        primary_metric = evaluation.get(
            "primary_metric"
        )

        if active_tasks == {"ssm"}:
            require(
                primary_metric == "ssm_nse_median",
                f"Invalid SSM primary metric: {config_path}",
            )
            require(
                sample_filter
                == expected_training_filter("ssm"),
                f"Invalid STL-SSM sample filter: {config_path}",
            )

        elif active_tasks == {"streamflow"}:
            require(
                primary_metric
                == "streamflow_nse_median",
                f"Invalid Q primary metric: {config_path}",
            )
            require(
                sample_filter
                == expected_training_filter(
                    "streamflow"
                ),
                f"Invalid Q sample filter: {config_path}",
            )

        elif active_tasks == {
            "streamflow",
            "ssm",
        }:
            require(
                primary_metric == "ssm_nse_median",
                (
                    "Joint Q-SSM primary metric must "
                    f"be ssm_nse_median: {config_path}"
                ),
            )
            require(
                sample_filter is None,
                (
                    "Joint Q-SSM models must retain "
                    f"all dates: {config_path}"
                ),
            )
            require(
                abs(sum(weights.values()) - 1.0)
                < 1.0e-9,
                (
                    "Joint Q-SSM task weights must "
                    f"sum to 1: {config_path}"
                ),
            )

        else:
            raise ValueError(
                "Unexpected active-task combination "
                f"{sorted(active_tasks)} in {config_path}."
            )

        print("=" * 100)
        print("file          :", config_path.name)
        print("stage         :", stage)
        print("active tasks  :", sorted(active_tasks))
        print("train period  :", data.get("train_period"))
        print("test period   :", data.get("test_period"))
        print("epochs        :", training.get("epochs"))
        print("primary metric:", primary_metric)
        print("status        : PASS")

    print(
        f"\nConfiguration audit passed "
        f"for {len(config_paths)} YAML files."
    )


if __name__ == "__main__":
    main()
