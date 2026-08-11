"""Generate Chapter 4B spatial PUB configurations from a frozen Ch4A base."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .constants import PUBScenario, ProtocolDefaults
from .io_utils import project_relative
from .paths import RUNS_DIR


def _target_map(config: dict[str, Any]) -> dict[str, tuple[int, dict[str, Any]]]:
    """Return configured targets keyed by normalized task name."""

    targets = config.get("data", {}).get("targets", [])
    result: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, target in enumerate(targets):
        name = str(target.get("name", "")).lower()
        if name:
            result[name] = (index, deepcopy(target))
    return result


def _task_towers_for_indices(
    model_cfg: dict[str, Any],
    indices: list[int],
) -> list[dict[str, Any]]:
    """Select task towers matching the selected output tasks."""

    towers = deepcopy(model_cfg.get("task_towers", []))
    if not towers:
        return []
    return [deepcopy(towers[min(index, len(towers) - 1)]) for index in indices]


def _configure_target(
    target: dict[str, Any],
    task: str,
    *,
    single_task: bool,
    defaults: ProtocolDefaults,
    run_profile: str = "formal",
) -> dict[str, Any]:
    """Apply Chapter 4A-compatible target and missing-data semantics."""

    target = deepcopy(target)
    if single_task:
        target["loss_weight"] = 1.0
    elif task == "streamflow":
        target["loss_weight"] = defaults.streamflow_weight
    else:
        target["loss_weight"] = defaults.ssm_weight

    if task == "ssm":
        target["interpolate_missing"] = False
        target.setdefault("unit_scale", 0.01)
        target.setdefault("output_unit", "m3 m-3")

    return target


def build_pub_config(
    base_config: dict[str, Any],
    scenario: PUBScenario,
    fold_id: int,
    seed: int,
    source_basin_file: Path,
    target_basin_file: Path,
    project_root: Path,
    defaults: ProtocolDefaults,
    run_profile: str = "formal",
) -> dict[str, Any]:
    """Create one formal PUB config from the frozen Chapter 4A Q-SSM template.

    Model architecture, forcing features, static attributes, optimizer settings,
    target scaling, and other common settings are inherited from the audited
    Chapter 4A configuration.  This function changes only the fields required
    by the Chapter 4B spatial data-limitation protocol.
    """

    config = deepcopy(base_config)
    targets = _target_map(config)

    missing_targets = {"streamflow", "ssm"} - set(targets)
    if missing_targets:
        raise ValueError(
            "The Chapter 4A base config must define streamflow and ssm targets. "
            f"Missing: {sorted(missing_targets)}"
        )

    selected_targets: list[dict[str, Any]] = []
    target_indices: list[int] = []
    single_task = len(scenario.active_tasks) == 1

    for task in scenario.active_tasks:
        index, target = targets[task]
        target_indices.append(index)
        selected_targets.append(
            _configure_target(
                target,
                task,
                single_task=single_task,
                defaults=defaults,
            )
        )

    profile = str(run_profile).strip().lower()
    if profile not in {"formal", "smoke"}:
        raise ValueError(f"Unsupported PUB run profile: {run_profile}")

    name = (
        f"ch4b_pub_{profile}_f{fold_id:02d}_{scenario.value}_seed{seed}"
    )

    experiment_cfg = config.setdefault("experiment", {})
    experiment_cfg.update(
        {
            "name": name,
            "task": "pub_q_from_auxiliary_ssm",
            "chapter": "ch4",
            "experiment_id": "ch4b",
            "save_dir": RUNS_DIR.as_posix(),
        }
    )

    data_cfg = config.setdefault("data", {})
    data_cfg["targets"] = selected_targets
    data_cfg["sequence_length"] = defaults.sequence_length
    data_cfg["forecast_history"] = defaults.sequence_length
    data_cfg["prediction_horizon"] = 1
    data_cfg["batch_size"] = defaults.batch_size
    data_cfg["num_workers"] = 0
    data_cfg["train_period"] = [defaults.pub_start, defaults.pub_end]
    data_cfg["val_period"] = None
    data_cfg["test_period"] = [defaults.pub_start, defaults.pub_end]
    data_cfg["spatial_split"] = True
    data_cfg["spatial_split_type"] = "explicit_pub_fold"
    data_cfg["basin_list_path"] = None

    # Role-aware sample selection is applied by the PUB data adapter.  A global
    # sample filter would incorrectly remove source-basin days with valid Q but
    # missing SSM, so it must remain disabled here.
    data_cfg["sample_filter"] = {
        "enabled": False,
        "required_valid_targets": [],
        "apply_to_modes": [],
    }

    model_cfg = config.setdefault("model", {})
    model_cfg["architecture"] = scenario.architecture
    selected_towers = _task_towers_for_indices(model_cfg, target_indices)
    if selected_towers:
        model_cfg["task_towers"] = selected_towers

    cgc_cfg = model_cfg.get("cgc")
    if isinstance(cgc_cfg, dict) and "task_experts" in cgc_cfg:
        existing = list(cgc_cfg.get("task_experts", [])) or [4]
        cgc_cfg["task_experts"] = [
            existing[min(index, len(existing) - 1)] for index in target_indices
        ]

    training_cfg = config.setdefault("training", {})
    training_cfg["epochs"] = defaults.epochs
    training_cfg["save_best_only"] = False
    training_cfg.setdefault("learning_rate", 0.001)
    training_cfg.setdefault("batch_progress", False)
    training_cfg["early_stopping"] = {
        "enabled": False,
        "patience": 1000000,
        "min_delta": 0.0,
    }
    training_cfg["monitor"] = {
        "name": "loss",
        "mode": "min",
        "min_delta": 0.0,
    }
    scheduler = training_cfg.get("scheduler")
    if isinstance(scheduler, dict):
        scheduler["mode"] = "min"

    reproducibility_cfg = config.setdefault("reproducibility", {})
    reproducibility_cfg["seed"] = int(seed)
    reproducibility_cfg["deterministic"] = True

    config["pub"] = {
        "enabled": True,
        "protocol_version": defaults.protocol_version,
        "run_profile": profile,
        "fold_id": int(fold_id),
        "scenario": scenario.value,
        "source_basin_file": project_relative(source_basin_file, project_root),
        "target_basin_file": project_relative(target_basin_file, project_root),
        "scaler_fit_scope": "source_only",
        "test_basin_scope": "target_only",
        "evaluation_task": "streamflow",
        "same_period_spatial_cv": True,
        "supervision": {
            "source": {
                "streamflow": "streamflow" in scenario.active_tasks,
                "ssm": "ssm" in scenario.active_tasks,
            },
            "target": {
                "streamflow": False,
                "ssm": scenario.target_ssm_supervision,
            },
        },
        "references": {
            "chapter3_summary": (
                "experiments/formal_ch3_modeling/06_summary/"
                "ch3_per_basin_with_metadata.csv"
            ),
            "chapter4a_protocol": "q_to_ssm",
            "chapter4a_seed42": True,
        },
    }

    evaluation = config.setdefault("evaluation_protocol", {})
    evaluation["metrics"] = ["nse", "kge", "rmse", "mae", "bias", "corr"]
    evaluation["primary_metric"] = "streamflow_nse_median"
    evaluation["export_csv"] = True
    evaluation["export_netcdf"] = True

    return config
