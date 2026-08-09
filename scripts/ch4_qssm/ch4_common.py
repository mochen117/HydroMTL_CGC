#!/usr/bin/env python3
"""Common helpers for Chapter 4 Q-SSM experiments.

The helpers are intentionally defensive because HydroMTL_CGC configuration
schemas may evolve. They update common key paths when present and create a
clear metadata block for every generated config.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


STREAMFLOW_ALIASES = (
    "streamflow",
    "usgsFlow",
    "usgs_flow",
    "q",
    "Q",
    "qobs",
    "q_obs",
    "runoff",
    "discharge",
)

SSM_ALIASES = (
    "ssm",
    "SSM",
    "surface_soil_moisture",
    "soil_moisture",
    "smap",
    "SMAP",
    "susm",
    "smp",
    "ssma",
    "susma",
)

ET_ALIASES = (
    "evapotranspiration",
    "ET",
    "et",
    "ET_sum",
    "LE",
)

MODEL_ARCH_VALUES = {
    "cgc": "CGC",
    "hard_mtl": "Hard_MTL",
    "stl": "STL",
}

# Candidate key paths used by different iterations of the project.
TASK_PATHS = [
    ("data", "targets"),
    ("data", "target_names"),
    ("data", "target_cols"),
    ("targets",),
    ("tasks",),
    ("model", "tasks"),
]

LOSS_WEIGHT_PATHS = [
    ("training", "loss_weights"),
    ("loss", "weights"),
    ("loss_weights",),
    ("training_params", "loss_weights"),
]

ARCH_PATHS = [
    ("model", "architecture"),
    ("model", "name"),
    ("model", "type"),
    ("architecture",),
    ("architecture", "type"),
    ("model_params", "architecture"),
]

DATA_ROOT_PATHS = [
    ("data", "data_root"),
    ("data", "root"),
    ("data", "path"),
    ("data", "data_dir"),
    ("data_params", "data_path"),
]

TRAIN_PERIOD_PATHS = [
    ("data", "train_period"),
    ("data", "periods", "train"),
    ("data_params", "t_range_train"),
]
VALID_PERIOD_PATHS = [
    ("data", "valid_period"),
    ("data", "validation_period"),
    ("data", "periods", "valid"),
    ("data_params", "t_range_valid"),
]
TEST_PERIOD_PATHS = [
    ("data", "test_period"),
    ("data", "periods", "test"),
    ("data_params", "t_range_test"),
]

BASIN_FILE_PATHS = [
    ("data", "basin_file"),
    ("data", "basins_file"),
    ("data", "basin_list_file"),
    ("data", "object_ids_file"),
    ("data_params", "gage_id_file"),
]
TRAIN_BASIN_FILE_PATHS = [
    ("data", "train_basin_file"),
    ("data", "train_basins_file"),
    ("split", "train_basin_file"),
]
VALID_BASIN_FILE_PATHS = [
    ("data", "valid_basin_file"),
    ("data", "validation_basin_file"),
    ("split", "valid_basin_file"),
]
TEST_BASIN_FILE_PATHS = [
    ("data", "test_basin_file"),
    ("data", "test_basins_file"),
    ("split", "test_basin_file"),
]

EPOCH_PATHS = [
    ("training", "epochs"),
    ("training_params", "epochs"),
]
BATCH_SIZE_PATHS = [
    ("data", "batch_size"),
    ("training", "batch_size"),
    ("training_params", "batch_size"),
]
SEED_PATHS = [
    ("seed",),
    ("training", "seed"),
    ("training", "random_seed"),
    ("training_params", "random_seed"),
]
EXPERIMENT_PATHS = [
    ("experiment_name",),
    ("experiment", "name"),
    ("run", "name"),
]
INIT_CKPT_PATHS = [
    ("init_checkpoint",),
    ("training", "init_checkpoint"),
    ("model", "init_checkpoint"),
    ("model", "checkpoint_path"),
    ("model_params", "weight_path"),
]
SCALER_STAT_PATHS = [
    ("data", "scaler_stat_file"),
    ("data", "stat_dict_file"),
    ("data_params", "stat_dict_file"),
    ("scaler", "stat_dict_file"),
]


@dataclass(frozen=True)
class Periods:
    pretrain: Tuple[str, str] = ("2005-10-01", "2015-10-01")
    finetune_train: Tuple[str, str] = ("2015-10-01", "2017-10-01")
    finetune_valid: Tuple[str, str] = ("2017-10-01", "2018-10-01")
    test: Tuple[str, str] = ("2018-10-01", "2021-10-01")


def project_root_from_file(file: str | Path) -> Path:
    p = Path(file).resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "main.py").exists() or (parent / "mtl_cgc").exists():
            return parent
    return Path.cwd().resolve()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)


def _has_path(d: Mapping[str, Any], path: Sequence[str]) -> bool:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return False
        cur = cur[key]
    return True


def _set_path(d: MutableMapping[str, Any], path: Sequence[str], value: Any, create: bool = True) -> bool:
    cur: MutableMapping[str, Any] = d
    for key in path[:-1]:
        if key not in cur:
            if not create:
                return False
            cur[key] = {}
        if not isinstance(cur[key], MutableMapping):
            if not create:
                return False
            cur[key] = {}
        cur = cur[key]
    if not create and path[-1] not in cur:
        return False
    cur[path[-1]] = value
    return True


def set_existing_or_first(cfg: MutableMapping[str, Any], paths: Sequence[Sequence[str]], value: Any) -> List[str]:
    changed = []
    for path in paths:
        if _has_path(cfg, path):
            _set_path(cfg, path, copy.deepcopy(value), create=True)
            changed.append(".".join(path))
    if not changed and paths:
        _set_path(cfg, paths[0], copy.deepcopy(value), create=True)
        changed.append(".".join(paths[0]))
    return changed


def set_optional_existing(cfg: MutableMapping[str, Any], paths: Sequence[Sequence[str]], value: Any) -> List[str]:
    changed = []
    for path in paths:
        if _has_path(cfg, path):
            _set_path(cfg, path, copy.deepcopy(value), create=True)
            changed.append(".".join(path))
    return changed


def normalize_task_name(task: str) -> str:
    t = task.strip()
    lower = t.lower()
    if lower in {"q", "flow", "runoff", "discharge", "usgsflow"}:
        return "streamflow"
    if lower in {"soil_moisture", "surface_soil_moisture", "smap", "susm", "smp", "ssma"}:
        return "ssm"
    if lower in {"et", "evap", "evapotranspiration"}:
        return "evapotranspiration"
    return t


def task_list(*tasks: str) -> List[str]:
    return [normalize_task_name(t) for t in tasks]


def task_weights(tasks: Sequence[str], streamflow: float = 0.0, ssm: float = 0.0, evapotranspiration: float = 0.0) -> Dict[str, float]:
    weights = {"streamflow": streamflow, "ssm": ssm, "evapotranspiration": evapotranspiration}
    return {normalize_task_name(t): float(weights.get(normalize_task_name(t), 0.0)) for t in tasks}


def apply_common_config(
    cfg: MutableMapping[str, Any],
    *,
    experiment_name: str,
    architecture: str,
    tasks: Sequence[str],
    loss_weights: Mapping[str, float],
    data_root: Optional[str] = None,
    train_period: Optional[Sequence[str]] = None,
    valid_period: Optional[Sequence[str]] = None,
    test_period: Optional[Sequence[str]] = None,
    basin_file: Optional[str] = None,
    train_basin_file: Optional[str] = None,
    valid_basin_file: Optional[str] = None,
    test_basin_file: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    scaler_stat_file: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Apply common modifications and return changed key paths."""
    changed: Dict[str, List[str]] = {}
    tasks = [normalize_task_name(t) for t in tasks]
    arch_value = MODEL_ARCH_VALUES.get(architecture, architecture)

    changed["experiment"] = set_existing_or_first(cfg, EXPERIMENT_PATHS, experiment_name)
    changed["architecture"] = set_existing_or_first(cfg, ARCH_PATHS, arch_value)
    changed["tasks"] = set_existing_or_first(cfg, TASK_PATHS, tasks)
    changed["loss_weights"] = set_existing_or_first(cfg, LOSS_WEIGHT_PATHS, dict(loss_weights))

    if data_root is not None:
        changed["data_root"] = set_existing_or_first(cfg, DATA_ROOT_PATHS, data_root)
    if train_period is not None:
        changed["train_period"] = set_existing_or_first(cfg, TRAIN_PERIOD_PATHS, list(train_period))
    if valid_period is not None:
        changed["valid_period"] = set_existing_or_first(cfg, VALID_PERIOD_PATHS, list(valid_period))
    if test_period is not None:
        changed["test_period"] = set_existing_or_first(cfg, TEST_PERIOD_PATHS, list(test_period))
    if basin_file is not None:
        changed["basin_file"] = set_existing_or_first(cfg, BASIN_FILE_PATHS, basin_file)
    if train_basin_file is not None:
        changed["train_basin_file"] = set_existing_or_first(cfg, TRAIN_BASIN_FILE_PATHS, train_basin_file)
    if valid_basin_file is not None:
        changed["valid_basin_file"] = set_existing_or_first(cfg, VALID_BASIN_FILE_PATHS, valid_basin_file)
    if test_basin_file is not None:
        changed["test_basin_file"] = set_existing_or_first(cfg, TEST_BASIN_FILE_PATHS, test_basin_file)
    if init_checkpoint is not None:
        changed["init_checkpoint"] = set_existing_or_first(cfg, INIT_CKPT_PATHS, init_checkpoint)
    if scaler_stat_file is not None:
        changed["scaler_stat_file"] = set_existing_or_first(cfg, SCALER_STAT_PATHS, scaler_stat_file)
    if epochs is not None:
        changed["epochs"] = set_existing_or_first(cfg, EPOCH_PATHS, int(epochs))
    if batch_size is not None:
        changed["batch_size"] = set_existing_or_first(cfg, BATCH_SIZE_PATHS, int(batch_size))
    if seed is not None:
        changed["seed"] = set_existing_or_first(cfg, SEED_PATHS, int(seed))

    # Explicit metadata block used by the Chapter 4 scripts. Core code can ignore it.
    _set_path(cfg, ("ch4_qssm",), {
        "experiment_name": experiment_name,
        "architecture": architecture,
        "tasks": tasks,
        "loss_weights": dict(loss_weights),
        "data_root": data_root,
        "train_period": list(train_period) if train_period is not None else None,
        "valid_period": list(valid_period) if valid_period is not None else None,
        "test_period": list(test_period) if test_period is not None else None,
        "basin_file": basin_file,
        "train_basin_file": train_basin_file,
        "valid_basin_file": valid_basin_file,
        "test_basin_file": test_basin_file,
        "init_checkpoint": init_checkpoint,
        "scaler_stat_file": scaler_stat_file,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
    })

    # Disable target interpolation in schemas that support such flags.
    interpolation_flags = [
        ("data", "interpolate_missing"),
        ("data", "target_interpolate"),
        ("data", "interpolate_targets"),
        ("data", "ssm_interpolate"),
        ("data_params", "target_rm_nan"),
    ]
    changed["disable_target_interpolation"] = set_optional_existing(cfg, interpolation_flags, False)
    return changed


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_basin_file(path: str | Path) -> List[str]:
    path = Path(path)
    basins = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split(",")[0]
            if not s or s.lower() in {"gage_id", "basin_id", "basin"}:
                continue
            basins.append(str(s).zfill(8) if s.isdigit() else str(s))
    return basins


def write_basin_file(basins: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for basin in basins:
            f.write(str(basin).strip() + "\n")


def find_latest_checkpoint(experiment_root: str | Path, pattern: str = "best_model.pth") -> Optional[Path]:
    root = Path(experiment_root)
    if not root.exists():
        return None
    candidates = list(root.rglob(pattern))
    if not candidates:
        candidates = list(root.rglob("*.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_scaler_stat(experiment_root: str | Path) -> Optional[Path]:
    root = Path(experiment_root)
    if not root.exists():
        return None
    names = [
        "dapengscaler_stat.json",
        "scaler_stats.json",
        "stat_dict.json",
        "target_scaler.pkl",
        "target_vars_scaler.pkl",
    ]
    for name in names:
        candidates = list(root.rglob(name))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def run_command(cmd: Sequence[str], cwd: Optional[str | Path] = None, dry_run: bool = False) -> int:
    printable = " ".join(str(x) for x in cmd)
    print(f"[CMD] {printable}")
    if dry_run:
        return 0
    proc = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
    return int(proc.returncode)


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def ensure_relative_to_project(path: str | Path, project_root: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(p)
