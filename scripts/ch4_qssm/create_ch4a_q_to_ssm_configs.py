#!/usr/bin/env python3
"""
Create YAML configurations for Chapter 4 Experiment A: Q-to-SSM transfer.

The reviewed protocol contains four baseline configurations and four
pretraining/fine-tuning configurations:

1. STL-Q
2. STL-SSM
3. HPS-QSSM from scratch
4. CGC-QSSM from scratch
5. HPS Q pretraining
6. HPS Q-pretrained Q+SSM fine-tuning
7. CGC Q pretraining
8. CGC Q-pretrained Q+SSM fine-tuning

Configured periods denote target dates. Historical input context required by
the N-to-1 model is expanded automatically by the DataLoader.

The PUB experiment is intentionally excluded from this generator until its
spatial protocol has been reviewed independently.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ch4_common import (
    apply_common_config,
    dump_yaml,
    load_yaml,
    task_list,
    task_weights,
    write_json,
)


ConfigDict = Dict[str, Any]
Period = Tuple[str, str]

DEFAULT_BATCH_SIZE = 64
DEFAULT_PRETRAIN_PERIOD: Period = (
    "2005-10-01",
    "2015-09-30",
)
DEFAULT_FINETUNE_PERIOD: Period = (
    "2015-10-01",
    "2018-09-30",
)
DEFAULT_TEST_PERIOD: Period = (
    "2018-10-01",
    "2021-09-30",
)


MULTI_TASK_ARCHITECTURES = ("hps", "cgc")


Q_TARGET: ConfigDict = {
    "name": "streamflow",
    "type": "regression",
    "loss_weight": 1.0,
    "constraint": "non_negative",
    "interpolate_missing": True,
    "output_unit": "m3 s-1",
    "long_name": "streamflow",
}


SSM_TARGET: ConfigDict = {
    "name": "ssm",
    "type": "regression",
    "loss_weight": 1.0,
    "constraint": "non_negative",
    "interpolate_missing": False,
    "unit_scale": 0.01,
    "source_unit": "%",
    "output_unit": "m3 m-3",
    "long_name": "surface soil moisture",
}


def normalize_period(
    values: Sequence[str],
    *,
    name: str,
) -> Period:
    """
    Validate and normalize an inclusive daily period.

    Parameters
    ----------
    values:
        Two ISO-formatted dates: start and end.
    name:
        Human-readable period name used in error messages.

    Returns
    -------
    tuple[str, str]
        Validated inclusive start and end dates.
    """
    if len(values) != 2:
        raise ValueError(
            f"{name} must contain exactly two dates, got {values}."
        )

    start_text = str(values[0])
    end_text = str(values[1])

    start_date = date.fromisoformat(start_text)
    end_date = date.fromisoformat(end_text)

    if start_date > end_date:
        raise ValueError(
            f"{name} start date must not exceed its end date: "
            f"{start_text} > {end_text}."
        )

    return start_text, end_text


def validate_protocol_periods(
    pretrain_period: Period,
    finetune_period: Period,
    test_period: Period,
) -> None:
    """
    Validate continuity of the reviewed Experiment A target periods.
    """
    pretrain_end = date.fromisoformat(pretrain_period[1])
    finetune_start = date.fromisoformat(finetune_period[0])
    finetune_end = date.fromisoformat(finetune_period[1])
    test_start = date.fromisoformat(test_period[0])

    if pretrain_end + timedelta(days=1) != finetune_start:
        raise ValueError(
            "Q pretraining and Q+SSM fine-tuning periods must be "
            "consecutive. "
            f"Got pretrain_end={pretrain_period[1]} and "
            f"finetune_start={finetune_period[0]}."
        )

    if finetune_end + timedelta(days=1) != test_start:
        raise ValueError(
            "Q+SSM fine-tuning and test periods must be consecutive. "
            f"Got finetune_end={finetune_period[1]} and "
            f"test_start={test_period[0]}."
        )


VALIDATION_PERIOD_KEYS = (
    "val_period",
    "valid_period",
    "validation_period",
)


def clear_validation_period_aliases(
    config: ConfigDict,
) -> None:
    """
    Remove all validation-period aliases for Experiment A.

    Experiment A uses a train-test-only temporal protocol. Clearing every
    supported alias prevents stale validation periods inherited from the
    template from being interpreted as active validation intervals.
    """
    data_config = config.setdefault("data", {})

    if not isinstance(data_config, dict):
        raise TypeError(
            "The configuration data block must be a dictionary."
        )

    for key in VALIDATION_PERIOD_KEYS:
        data_config[key] = None

    metadata = config.setdefault("ch4_qssm", {})

    if not isinstance(metadata, dict):
        raise TypeError(
            "The ch4_qssm metadata block must be a dictionary."
        )

    for key in VALIDATION_PERIOD_KEYS:
        metadata[key] = None


def save_config(
    config: ConfigDict,
    output_path: Path,
    generated_paths: List[Path],
) -> None:
    """Write one validated YAML configuration and register its path."""
    clear_validation_period_aliases(config)
    dump_yaml(config, output_path)
    generated_paths.append(output_path)
    print(f"Created: {output_path}")


def set_training_sample_filter(
    config: ConfigDict,
    required_targets: Sequence[str],
) -> None:
    """
    Restrict training samples to dates with valid observations for the
    specified targets.

    Evaluation remains mask-based so the complete configured test period is
    preserved in exported time series.
    """
    data_config = config.setdefault("data", {})
    data_config["sample_filter"] = {
        "enabled": True,
        "required_valid_targets": [
            str(target).lower()
            for target in required_targets
        ],
        "apply_to_modes": ["train"],
    }


def build_common_kwargs(
    args: argparse.Namespace,
) -> ConfigDict:
    """Build arguments shared by all Experiment A configurations."""
    basin_file = (
        str(args.basin_file)
        if args.basin_file is not None
        else None
    )

    return {
        "data_root": str(args.data_root),
        "basin_file": basin_file,
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
    }



def normalize_run_tag(raw_value: str) -> str:
    """Normalize an optional experiment tag used to isolate formal outputs."""
    value = str(raw_value or "").strip().lower()

    if not value:
        return ""

    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in value
    ).strip("_-")

    if not normalized:
        raise ValueError(
            f"run_tag contains no usable characters: {raw_value!r}."
        )

    return normalized


def build_experiment_name(
    *,
    run_tag: str,
    suffix: str,
    seed: int,
) -> str:
    """Build a collision-safe experiment name."""
    components = ["ch4a"]

    if run_tag:
        components.append(run_tag)

    components.extend([suffix, f"seed{int(seed)}"])
    return "_".join(components)


def resolve_epoch_plan(
    args: argparse.Namespace,
) -> Tuple[int, int]:
    """Resolve joint/fine-tuning and Q-pretraining epoch counts."""
    if args.epochs is not None:
        joint_epochs = int(args.epochs)
        pretrain_epochs = int(args.epochs)
    else:
        joint_epochs = int(args.joint_epochs)
        pretrain_epochs = int(args.pretrain_epochs)

    if joint_epochs <= 0:
        raise ValueError(
            f"joint_epochs must be positive, got {joint_epochs}."
        )

    if pretrain_epochs <= 0:
        raise ValueError(
            f"pretrain_epochs must be positive, got {pretrain_epochs}."
        )

    return joint_epochs, pretrain_epochs


def set_training_epochs(
    config: ConfigDict,
    epochs: int,
) -> None:
    """Assign a validated fixed epoch count to one configuration."""
    if int(epochs) <= 0:
        raise ValueError(
            f"epochs must be positive, got {epochs}."
        )

    training_config = config.setdefault("training", {})

    if not isinstance(training_config, dict):
        raise TypeError(
            "The training block must be a dictionary."
        )

    training_config["epochs"] = int(epochs)


def build_target_configs(
    task_names: Sequence[str],
    loss_weights: Dict[str, float],
) -> List[ConfigDict]:
    """Build complete target dictionaries for one experiment."""
    catalog = {
        "streamflow": Q_TARGET,
        "ssm": SSM_TARGET,
    }

    targets: List[ConfigDict] = []

    for raw_name in task_names:
        name = str(raw_name).strip().lower()

        if name not in catalog:
            raise ValueError(
                f"Unsupported Chapter 4 target: {name!r}."
            )

        target = deepcopy(catalog[name])
        target["loss_weight"] = float(
            loss_weights[name]
        )
        targets.append(target)

    return targets


def finalize_experiment_a_config(
    config: ConfigDict,
    *,
    task_names: Sequence[str],
    loss_weights: Dict[str, float],
    basin_file: str | None,
    batch_size: int,
    epochs: int,
    require_ssm_observation: bool = False,
) -> None:
    """
    Finalize one Experiment A YAML configuration.

    The generic configuration helper may write target names as strings.
    HydroMTL models require complete target dictionaries, so this function
    overwrites data.targets with validated Q/SSM specifications.
    """
    data_config = config.setdefault("data", {})

    if not isinstance(data_config, dict):
        raise TypeError(
            "The generated data block must be a dictionary."
        )

    normalized_tasks = [
        str(name).strip().lower()
        for name in task_names
    ]

    if not normalized_tasks:
        raise ValueError("Experiment A must contain at least one task.")

    experiment_config = config.setdefault("experiment", {})
    if not isinstance(experiment_config, dict):
        raise TypeError("The experiment block must be a dictionary.")
    experiment_config["task"] = "q_to_ssm_prediction"

    data_config["targets"] = build_target_configs(
        normalized_tasks,
        loss_weights,
    )

    data_config["batch_size"] = int(batch_size)
    data_config["spatial_split"] = False
    data_config["spatial_split_type"] = "none"
    data_config["basin_list_path"] = (
        str(basin_file)
        if basin_file is not None
        else None
    )

    # Experiment A uses fixed train and test periods without validation.
    for key in (
        "val_period",
        "valid_period",
        "validation_period",
    ):
        data_config[key] = None

    # Do not inherit an unrelated sample filter from default.yaml.
    data_config.pop("sample_filter", None)

    # STL-SSM requires a valid SSM observation for each retained sample.
    # Joint Q-SSM models retain all samples and use finite-value loss masks.
    if require_ssm_observation:
        data_config["sample_filter"] = {
            "enabled": True,
            "required_valid_targets": ["ssm"],
            "apply_to_modes": ["train"],
        }

    metadata = config.setdefault("ch4_qssm", {})

    if not isinstance(metadata, dict):
        raise TypeError(
            "The ch4_qssm block must be a dictionary."
        )

    metadata["protocol"] = "q_to_ssm"
    metadata["tasks"] = normalized_tasks
    metadata["loss_weights"] = {
        name: float(loss_weights[name])
        for name in normalized_tasks
    }
    metadata["batch_size"] = int(batch_size)
    metadata["epochs"] = int(epochs)

    # Experiment A has no validation loader. Training loss is therefore the
    # only finite epoch-level quantity available for scheduler and checkpoint
    # control. Fixed-epoch training must not use validation early stopping.
    training_config = config.setdefault("training", {})

    if not isinstance(training_config, dict):
        raise TypeError(
            "The training block must be a dictionary."
        )

    # Keep all redundant weight records synchronized. The runtime criterion
    # reads data.targets[*].loss_weight, while these fields are retained for
    # experiment auditing and backward-compatible reporting.
    training_config["loss_weights"] = {
        name: float(loss_weights[name])
        for name in normalized_tasks
    }

    loss_config = training_config.setdefault("loss", {})
    if not isinstance(loss_config, dict):
        raise TypeError("training.loss must be a dictionary.")
    loss_config["base_loss"] = "rmse"
    loss_config["multi_task_balancing"] = "none"
    loss_config["water_balance_weight"] = 0.0
    loss_config["eps"] = 1.0e-6

    evaluation_config = config.setdefault("evaluation_protocol", {})
    if not isinstance(evaluation_config, dict):
        raise TypeError("evaluation_protocol must be a dictionary.")

    ssm_is_active = (
        "ssm" in normalized_tasks
        and float(loss_weights.get("ssm", 0.0)) > 0.0
    )
    evaluation_config["primary_metric"] = (
        "ssm_nse_median"
        if ssm_is_active
        else "streamflow_nse_median"
    )

    set_training_epochs(config, epochs)

    training_config["monitor"] = {
        "name": "loss",
        "mode": "min",
        "min_delta": 1.0e-4,
    }

    early_stopping_config = training_config.setdefault(
        "early_stopping",
        {},
    )

    if not isinstance(early_stopping_config, dict):
        raise TypeError(
            "training.early_stopping must be a dictionary."
        )

    early_stopping_config["enabled"] = False

    scheduler_config = training_config.get("scheduler")

    if isinstance(scheduler_config, dict):
        scheduler_config["mode"] = "min"

    for key in (
        "val_period",
        "valid_period",
        "validation_period",
    ):
        metadata[key] = None


def create_q_to_ssm_configs(
    args: argparse.Namespace,
) -> List[Path]:
    """Create all reviewed Chapter 4 Experiment A configs."""
    template = load_yaml(args.template)
    output_paths: List[Path] = []
    run_tag = normalize_run_tag(args.run_tag)
    joint_epochs, pretrain_epochs = resolve_epoch_plan(args)

    pretrain_period = normalize_period(
        args.pretrain_period,
        name="pretrain_period",
    )
    finetune_period = normalize_period(
        args.finetune_train_period,
        name="finetune_train_period",
    )
    test_period = normalize_period(
        args.test_period,
        name="test_period",
    )

    validate_protocol_periods(
        pretrain_period,
        finetune_period,
        test_period,
    )

    if args.q_weight <= 0.0:
        raise ValueError(
            f"q_weight must be positive, got {args.q_weight}."
        )

    if args.ssm_weight <= 0.0:
        raise ValueError(
            f"ssm_weight must be positive, got {args.ssm_weight}."
        )

    common_kwargs = build_common_kwargs(args)
    basin_file = common_kwargs["basin_file"]
    output_dir = args.out_dir / "q_to_ssm"

    # ------------------------------------------------------------------
    # STL-SSM baseline.
    # ------------------------------------------------------------------
    config = deepcopy(template)
    experiment_name = build_experiment_name(
        run_tag=run_tag,
        suffix="stl_ssm",
        seed=args.seed,
    )

    apply_common_config(
        config,
        experiment_name=experiment_name,
        architecture="stl",
        tasks=task_list("ssm"),
        loss_weights={"ssm": 1.0},
        train_period=finetune_period,
        valid_period=None,
        test_period=test_period,
        **common_kwargs,
    )


    finalize_experiment_a_config(
        config,
        task_names=("ssm",),
        loss_weights={"ssm": 1.0},
        basin_file=basin_file,
        batch_size=args.batch_size,
        epochs=joint_epochs,
        require_ssm_observation=True,
    )

    save_config(
        config,
        output_dir / f"{experiment_name}.yaml",
        output_paths,
    )


    # ------------------------------------------------------------------
    # STL-Q baseline over the same short joint-training period.
    # ------------------------------------------------------------------
    config = deepcopy(template)
    experiment_name = build_experiment_name(
        run_tag=run_tag,
        suffix="stl_q",
        seed=args.seed,
    )

    apply_common_config(
        config,
        experiment_name=experiment_name,
        architecture="stl",
        tasks=task_list("streamflow"),
        loss_weights={"streamflow": 1.0},
        train_period=finetune_period,
        valid_period=None,
        test_period=test_period,
        **common_kwargs,
    )

    finalize_experiment_a_config(
        config,
        task_names=("streamflow",),
        loss_weights={"streamflow": 1.0},
        basin_file=basin_file,
        batch_size=args.batch_size,
        epochs=joint_epochs,
    )

    set_training_sample_filter(
        config,
        required_targets=("streamflow",),
    )

    save_config(
        config,
        output_dir / f"{experiment_name}.yaml",
        output_paths,
    )


    # ------------------------------------------------------------------
    # HPS and CGC trained from random initialization.
    # ------------------------------------------------------------------
    joint_tasks = task_list("streamflow", "ssm")
    joint_weights = task_weights(
        joint_tasks,
        streamflow=args.q_weight,
        ssm=args.ssm_weight,
    )

    joint_loss_weights = {
        "streamflow": float(args.q_weight),
        "ssm": float(args.ssm_weight),
    }

    for architecture in MULTI_TASK_ARCHITECTURES:
        config = deepcopy(template)
        experiment_name = build_experiment_name(
            run_tag=run_tag,
            suffix=f"{architecture}_qssm",
            seed=args.seed,
        )

        apply_common_config(
            config,
            experiment_name=experiment_name,
            architecture=architecture,
            tasks=joint_tasks,
            loss_weights=joint_weights,
            train_period=finetune_period,
            valid_period=None,
            test_period=test_period,
            **common_kwargs,
        )

        finalize_experiment_a_config(
            config,
            task_names=("streamflow", "ssm"),
            loss_weights=joint_loss_weights,
            basin_file=basin_file,
            batch_size=args.batch_size,
            epochs=joint_epochs,
        )

        save_config(
            config,
            output_dir / f"{experiment_name}.yaml",
            output_paths,
        )

    # ------------------------------------------------------------------
    # Q pretraining followed by Q+SSM fine-tuning.
    #
    # Both heads are retained during pretraining so that the checkpoint
    # remains structurally compatible with the fine-tuning model.
    # ------------------------------------------------------------------
    q_pretrain_weights = task_weights(
        joint_tasks,
        streamflow=1.0,
        ssm=0.0,
    )

    q_pretrain_loss_weights = {
        "streamflow": 1.0,
        "ssm": 0.0,
    }

    for architecture in MULTI_TASK_ARCHITECTURES:
        pretrain_config = deepcopy(template)
        pretrain_name = build_experiment_name(
            run_tag=run_tag,
            suffix=f"{architecture}_qpre",
            seed=args.seed,
        )

        apply_common_config(
            pretrain_config,
            experiment_name=pretrain_name,
            architecture=architecture,
            tasks=joint_tasks,
            loss_weights=q_pretrain_weights,
            train_period=pretrain_period,
            valid_period=None,
            test_period=finetune_period,
            **common_kwargs,
        )

        finalize_experiment_a_config(
            pretrain_config,
            task_names=("streamflow", "ssm"),
            loss_weights=q_pretrain_loss_weights,
            basin_file=basin_file,
            batch_size=args.batch_size,
            epochs=pretrain_epochs,
        )

        set_training_sample_filter(
            pretrain_config,
            required_targets=("streamflow",),
        )

        save_config(
            pretrain_config,
            output_dir / f"{pretrain_name}.yaml",
            output_paths,
        )

        finetune_config = deepcopy(template)
        finetune_name = build_experiment_name(
            run_tag=run_tag,
            suffix=(
                f"{architecture}_qpre_finetune_qssm"
            ),
            seed=args.seed,
        )

        apply_common_config(
            finetune_config,
            experiment_name=finetune_name,
            architecture=architecture,
            tasks=joint_tasks,
            loss_weights=joint_weights,
            train_period=finetune_period,
            valid_period=None,
            test_period=test_period,
            init_checkpoint=(
                f"__CHECKPOINT__:{pretrain_name}"
            ),
            **common_kwargs,
        )

        finalize_experiment_a_config(
            finetune_config,
            task_names=("streamflow", "ssm"),
            loss_weights=joint_loss_weights,
            basin_file=basin_file,
            batch_size=args.batch_size,
            epochs=joint_epochs,
        )

        save_config(
            finetune_config,
            output_dir / f"{finetune_name}.yaml",
            output_paths,
        )

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create reviewed Chapter 4 Experiment A "
            "Q-to-SSM configurations."
        )
    )

    parser.add_argument(
        "--template",
        default=Path("mtl_cgc/configs/default.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=Path("mtl_cgc/configs/ch4_qssm"),
        type=Path,
    )
    parser.add_argument(
        "--data-root",
        default=Path("output_592_basins"),
        type=Path,
    )
    parser.add_argument(
        "--basin-file",
        default=Path(
            "experiments/ch4_qssm/basin_splits/"
            "all_eligible_basins.txt"
        ),
        type=Path,
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--run-tag",
        default="",
        type=str,
        help=(
            "Optional tag inserted into experiment names. "
            "Use 'formal' for full experiments to avoid "
            "overwriting smoke-test artifacts."
        ),
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help=(
            "Legacy override applied to every stage. "
            "Use only for smoke tests."
        ),
    )
    parser.add_argument(
        "--joint-epochs",
        default=100,
        type=int,
        help=(
            "Epochs for STL, scratch MTL, and fine-tuning runs."
        ),
    )
    parser.add_argument(
        "--pretrain-epochs",
        default=200,
        type=int,
        help="Epochs for Q-pretraining runs.",
    )
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
    )
    parser.add_argument("--q-weight", default=0.5, type=float)
    parser.add_argument("--ssm-weight", default=0.5, type=float)

    parser.add_argument(
        "--pretrain-period",
        nargs=2,
        default=list(DEFAULT_PRETRAIN_PERIOD),
    )
    parser.add_argument(
        "--finetune-train-period",
        nargs=2,
        default=list(DEFAULT_FINETUNE_PERIOD),
    )
    parser.add_argument(
        "--test-period",
        nargs=2,
        default=list(DEFAULT_TEST_PERIOD),
    )

    return parser.parse_args()


def main() -> None:
    """Generate configurations and their manifest."""
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {args.batch_size}."
        )

    run_tag = normalize_run_tag(args.run_tag)
    joint_epochs, pretrain_epochs = resolve_epoch_plan(args)
    config_paths = create_q_to_ssm_configs(args)

    manifest = {
        "protocol": "q_to_ssm",
        "review_status": "experiment_a_config_reviewed",
        "run_tag": run_tag,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "joint_epochs": int(joint_epochs),
        "pretrain_epochs": int(pretrain_epochs),
        "q_weight": float(args.q_weight),
        "ssm_weight": float(args.ssm_weight),
        "basin_file": str(args.basin_file),
        "data_root": str(args.data_root),
        "pretrain_period": list(args.pretrain_period),
        "finetune_train_period": list(args.finetune_train_period),
        "test_period": list(args.test_period),
        "configs": [str(path) for path in config_paths],
    }

    manifest_name = (
        "ch4a_q_to_ssm_manifest.json"
        if not run_tag
        else (
            f"ch4a_q_to_ssm_{run_tag}_"
            f"seed{int(args.seed)}_manifest.json"
        )
    )
    manifest_path = args.out_dir / manifest_name
    write_json(manifest, manifest_path)

    print(
        f"Created {len(config_paths)} Experiment A configs "
        f"under {args.out_dir}"
    )
    print(f"Run tag: {run_tag or '<none>'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Joint/fine-tune epochs: {joint_epochs}")
    print(f"Q-pretrain epochs: {pretrain_epochs}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()