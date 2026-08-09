# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Modular execution pipeline for the HydroMTL framework.
# Provides leakage-safe training/testing, compact epoch-level reporting,
# validation/test exports, and online diagnostics for hydrological MTL experiments.
# ==============================================================================

import os
import sys
import gc
import json
import time
import ctypes
import random
import argparse
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Environment setup before importing torch
# ------------------------------------------------------------------------------
if os.environ.get("HYDRO_USE_PATCH", "1") == "1":
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        lib_path = os.path.join(conda_prefix, "lib")
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{old_ld}"

    libstdcxx_path = os.path.join(conda_prefix, "lib", "libstdc++.so.6") if conda_prefix else ""
    try:
        if libstdcxx_path:
            ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


def _preselect_device() -> str:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--device", type=str, default="auto")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.device == "auto" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"

    if pre_args.device == "auto":
        best_idx = 0
        try:
            import subprocess

            smi_query = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
            )
            free_mem = [int(x) for x in smi_query.strip().splitlines() if x.strip()]
            if free_mem:
                best_idx = free_mem.index(max(free_mem))
        except Exception:
            pass

        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
        return "cuda:0"

    if pre_args.device.startswith("cuda"):
        dev_parts = pre_args.device.split(":")
        dev_idx = dev_parts[1] if len(dev_parts) > 1 else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_idx)
        return "cuda:0"

    return "cpu"


TARGET_DEVICE = _preselect_device()

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------------------
# Imports after environment setup
# ------------------------------------------------------------------------------
import torch
import numpy as np
import pandas as pd
import yaml
from easydict import EasyDict as edict

from mtl_cgc.data.data_loaders import get_hydro_dataloaders
from mtl_cgc.data.data_sets import BasinSpatialSplitter
from mtl_cgc.core.cgc_models.mtl_model import build_model
from mtl_cgc.core.training.trainer import HydroTrainer
from mtl_cgc.core.evaluation.evaluator import (
    HydroEvaluator,
    ClimateSpecializationAnalyzer,
    compute_wilcoxon_paired_test,
)
from mtl_cgc.core.training.callbacks import EarlyStopping, ModelCheckpoint


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def release_memory() -> None:
    """Release Python and CUDA cache after large validation/test/export steps."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_memory_mb() -> float:
    """Return allocated CUDA memory in MB for lightweight monitoring."""
    if not torch.cuda.is_available():
        return 0.0

    return float(torch.cuda.memory_allocated() / 1024**2)


def parse_loss_weights(raw_items: Optional[List[str]]) -> Tuple[Dict[str, float], List[str]]:
    """Parse command-line loss weights formatted as task=weight."""
    if raw_items is None:
        return {}, []

    weight_dict: Dict[str, float] = {}
    messages: List[str] = []

    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --loss_weights item: {item}. Expected format: task=weight")

        key, value = item.split("=", 1)
        weight_dict[key.lower()] = float(value)

    return weight_dict, messages


def get_target_name(target: Any) -> str:
    """
    Return a lowercase target name from either a mapping-like target,
    an object with a `.name` attribute, or a plain string.
    """
    if isinstance(target, dict):
        value = target.get("name", target)
    else:
        value = getattr(target, "name", target)
    return str(value).strip().lower()


def period_is_missing(period: Any) -> bool:
    """Return True when a temporal period is absent."""
    if period is None:
        return True
    if isinstance(period, str) and period.strip().lower() in {"", "none", "null"}:
        return True
    if isinstance(period, (list, tuple)) and len(period) == 0:
        return True
    return False


def period_to_list(period: Any) -> Optional[List[str]]:
    """Convert a temporal period to a printable list."""
    if period_is_missing(period):
        return None
    return list(period)


def first_available_period(config_data: Any, *names: str) -> Any:
    """Return the first non-empty period from config.data."""
    for name in names:
        value = getattr(config_data, name, None)
        if not period_is_missing(value):
            return value
    return None


def validate_temporal_splits(config: Any) -> None:
    """
    Validate temporal splits.

    The validation period is optional. If validation is absent, the required
    order is:

        train_start <= train_end < test_start <= test_end

    If validation is present, the required order is:

        train_start <= train_end < val_start <= val_end < test_start <= test_end
    """

    def is_missing_period(period: Any) -> bool:
        if period is None:
            return True
        if isinstance(period, str) and period.lower() in {"", "none", "null"}:
            return True
        if isinstance(period, (list, tuple)) and len(period) == 0:
            return True
        return False

    def parse_period(period: Any, name: str):
        if is_missing_period(period):
            return None
        if not isinstance(period, (list, tuple)) or len(period) != 2:
            raise ValueError(f"{name} must be a two-element list, got: {period}")
        start, end = map(pd.to_datetime, period)
        return start, end

    train = parse_period(getattr(config.data, "train_period", None), "train_period")
    test = parse_period(getattr(config.data, "test_period", None), "test_period")

    validation_candidates = [
        getattr(config.data, "val_period", None),
        getattr(config.data, "valid_period", None),
        getattr(config.data, "validation_period", None),
    ]

    valid = None
    for candidate in validation_candidates:
        if not is_missing_period(candidate):
            valid = parse_period(candidate, "valid_period")
            break

    if train is None or test is None:
        raise ValueError("Both train_period and test_period are required.")

    train_start, train_end = train
    test_start, test_end = test

    if train_start > train_end:
        raise ValueError("Invalid train_period: train_start must be <= train_end.")

    if test_start > test_end:
        raise ValueError("Invalid test_period: test_start must be <= test_end.")

    if valid is None:
        if not (train_end < test_start):
            raise ValueError(
                "Invalid temporal split. Expected: "
                "train_start <= train_end < test_start <= test_end."
            )
        return

    val_start, val_end = valid

    if val_start > val_end:
        raise ValueError("Invalid valid_period: val_start must be <= val_end.")

    if not (train_start <= train_end < val_start <= val_end < test_start <= test_end):
        raise ValueError(
            "Invalid temporal split. Expected: "
            "train_start <= train_end < val_start <= val_end < test_start <= test_end."
        )
def discover_basin_ids(data_root: Path) -> List[str]:
    """Discover basin identifiers from gage_*.nc files."""
    basin_ids = sorted([f.stem.replace("gage_", "") for f in data_root.glob("gage_*.nc")])

    if not basin_ids:
        raise FileNotFoundError(f"No basin NetCDF files found under {data_root.resolve()}")

    return basin_ids


def load_ungauged_list(path: Optional[str]) -> Optional[List[str]]:
    """Load an optional text file listing ungauged basin ids."""
    if path is None:
        return None

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ungauged basin file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def build_spatial_split(config: Any, all_basin_ids: List[str]) -> Tuple[List[str], List[str], str]:
    """Build train/test basin lists. Test basins remain isolated for PUB evaluation."""
    spatial_split_enabled = bool(getattr(config.data, "spatial_split", False))
    spatial_split_type = getattr(config.data, "spatial_split_type", "random")

    if not spatial_split_enabled:
        return all_basin_ids, all_basin_ids, "none"

    splitter = BasinSpatialSplitter(
        all_basin_ids,
        random_seed=int(config.get("reproducibility", {}).get("seed", 42)),
    )

    metadata_path = Path(config.data.get("basin_metadata_path", ""))
    if spatial_split_type == "regional" and metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)
        metadata_df["basin_id"] = metadata_df["basin_id"].astype(str).str.zfill(8)

        region_col = getattr(config.data, "region_col", "huc_02")
        regional_splits = splitter.hydrologic_region_split(metadata_df, region_col=region_col)
        region_index = int(getattr(config.data, "region_split_index", 0))
        selected = regional_splits[region_index]

        return selected["train_basins"], selected["test_basins"], f"regional:{selected['test_region']}"

    train_basins, test_basins = splitter.random_kfold_split(
        n_splits=int(getattr(config.data, "n_splits", 5))
    )[0]

    return train_basins, test_basins, "random_kfold:first_fold"


def is_valid_metric(value: Any) -> bool:
    """Return True when a metric is finite."""
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def format_metric(value: Any, width: int = 7, precision: int = 4) -> str:
    """Format possibly missing metrics for aligned progress display."""
    try:
        value = float(value)
        if not np.isfinite(value):
            return f"{'nan':>{width}}"

        return f"{value:{width}.{precision}f}"
    except Exception:
        return f"{'nan':>{width}}"


def to_plain_dict(obj: Any) -> Any:
    """Convert nested EasyDict/list objects into JSON-serializable containers."""
    if isinstance(obj, dict):
        return {str(key): to_plain_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_plain_dict(value) for value in obj]
    return obj


def resolve_monitor_settings(config: Any, eval_cfg: Any) -> Tuple[str, str, float]:
    """Resolve the validation monitor shared by scheduler, checkpoint, and early stop."""
    monitor_cfg = config.training.get("monitor", {})
    fallback_name = eval_cfg.get("primary_metric", "streamflow_nse_median")

    monitor_name = str(monitor_cfg.get("name", fallback_name))
    default_mode = "min" if monitor_name.lower() in {"loss", "val_loss"} else "max"
    monitor_mode = str(monitor_cfg.get("mode", default_mode)).lower()
    min_delta = float(monitor_cfg.get("min_delta", config.training.get("min_delta", 1e-4)))

    if monitor_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported monitor mode: {monitor_mode}")

    return monitor_name, monitor_mode, min_delta


def compute_monitor_value(
    monitor_name: str,
    val_loss: float,
    val_metrics: Dict[str, float],
) -> float:
    """Compute the scalar monitor used for all training-control decisions."""
    name = monitor_name.lower()

    if name in {"loss", "val_loss"}:
        value = float(val_loss)
    elif name == "joint_nse":
        nse_values = [
            float(value)
            for key, value in val_metrics.items()
            if key.endswith("_nse_median") and np.isfinite(float(value))
        ]
        if not nse_values:
            raise ValueError("joint_nse requires at least one finite *_nse_median metric.")
        value = float(np.mean(nse_values))
    else:
        value = float(val_metrics.get(monitor_name, np.nan))

    if not np.isfinite(value):
        raise ValueError(f"Invalid monitor value: {monitor_name}={value}")

    return value


def monitor_is_better(
    value: float,
    best_value: Optional[float],
    mode: str,
    min_delta: float,
) -> bool:
    """Return True when a monitor value improves on the current best value."""
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value - min_delta
    if mode == "max":
        return value > best_value + min_delta
    raise ValueError(f"Unsupported monitor mode: {mode}")


def scheduler_step(scheduler: Optional[Any], monitor_value: float) -> None:
    """Step either ReduceLROnPlateau or epoch-based schedulers safely."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitor_value)
    else:
        scheduler.step()


def load_torch_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str,
) -> Any:
    """Load a trusted local checkpoint across PyTorch versions."""
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move all tensor-valued optimizer state to the active device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_rng_state(rng_state: Optional[Dict[str, Any]]) -> None:
    """Restore Python, NumPy, CPU, and CUDA random-number states."""
    if not rng_state:
        warnings.warn(
            "Resume checkpoint has no RNG state; continuation remains valid "
            "but will not be bitwise reproducible.",
            RuntimeWarning,
        )
        return

    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy")
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(
                    numpy_state["state"],
                    dtype=np.uint32,
                ),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )

    torch_cpu_state = rng_state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch_cpu_state.cpu())

    torch_cuda_state = rng_state.get("torch_cuda")
    if torch.cuda.is_available() and torch_cuda_state is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in torch_cuda_state]
        )


def _target_names_from_config(config_like: Any) -> List[str]:
    """Extract normalized target names from a config dictionary."""
    plain = to_plain_dict(config_like)
    targets = plain.get("data", {}).get("targets", [])
    names: List[str] = []

    for target in targets:
        if isinstance(target, dict):
            names.append(str(target.get("name", "")).lower())
        else:
            names.append(str(target).lower())

    return names


def validate_resume_compatibility(
    checkpoint: Dict[str, Any],
    config: Any,
    checkpoint_path: Path,
) -> None:
    """Reject accidental cross-experiment or cross-architecture resume."""
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError(
            "Resume checkpoint lacks its original config metadata: "
            f"{checkpoint_path}"
        )

    current = to_plain_dict(config)

    comparisons = {
        "experiment.name": (
            saved_config.get("experiment", {}).get("name"),
            current.get("experiment", {}).get("name"),
        ),
        "model.architecture": (
            str(
                saved_config.get("model", {}).get("architecture", "")
            ).lower(),
            str(
                current.get("model", {}).get("architecture", "")
            ).lower(),
        ),
        "reproducibility.seed": (
            saved_config.get("reproducibility", {}).get("seed"),
            current.get("reproducibility", {}).get("seed"),
        ),
        "data.targets": (
            _target_names_from_config(saved_config),
            _target_names_from_config(current),
        ),
    }

    mismatches = {
        key: values
        for key, values in comparisons.items()
        if values[0] != values[1]
    }

    if mismatches:
        raise ValueError(
            "Resume checkpoint is incompatible with the active config: "
            f"{checkpoint_path}; mismatches={mismatches}"
        )


def resume_training_checkpoint(
    *,
    checkpoint_path: Path,
    trainer: HydroTrainer,
    config: Any,
    device: torch.device,
    monitor_name: str,
    monitor_mode: str,
) -> Dict[str, Any]:
    """Restore model, optimizer, scheduler, scaler, and RNG state."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}"
        )

    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        device,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "A resumable checkpoint must be a dictionary: "
            f"{checkpoint_path}"
        )

    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
    }
    missing = sorted(
        key for key in required
        if checkpoint.get(key) is None
    )
    if missing:
        raise ValueError(
            "Checkpoint is not resumable because required fields are "
            f"missing: {missing}; path={checkpoint_path}"
        )

    validate_resume_compatibility(
        checkpoint,
        config,
        checkpoint_path,
    )

    saved_monitor_name = checkpoint.get("monitor_name")
    saved_monitor_mode = checkpoint.get("monitor_mode")
    if (
        saved_monitor_name not in {None, monitor_name}
        or saved_monitor_mode not in {None, monitor_mode}
    ):
        raise ValueError(
            "Resume monitor differs from the active config: "
            f"checkpoint=({saved_monitor_name}, {saved_monitor_mode}), "
            f"current=({monitor_name}, {monitor_mode})."
        )

    trainer.model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    trainer.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    move_optimizer_state_to_device(
        trainer.optimizer,
        device,
    )

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if trainer.scheduler is not None:
        if scheduler_state is None:
            raise ValueError(
                "Active config uses a scheduler but the resume checkpoint "
                f"does not contain scheduler state: {checkpoint_path}"
            )
        trainer.scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if trainer.amp_enabled and scaler_state is None:
        raise ValueError(
            "AMP is enabled but the resume checkpoint lacks scaler state: "
            f"{checkpoint_path}"
        )
    if scaler_state is not None:
        trainer.scaler.load_state_dict(scaler_state)

    restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def load_resume_history(
    *,
    checkpoint: Dict[str, Any],
    history_path: Path,
    checkpoint_epoch: int,
) -> List[Dict[str, Any]]:
    """Restore history from the checkpoint, falling back to the CSV."""
    checkpoint_history = checkpoint.get("history_records")

    if isinstance(checkpoint_history, list):
        frame = pd.DataFrame(checkpoint_history)
    elif history_path.exists():
        frame = pd.read_csv(history_path)
    else:
        frame = pd.DataFrame()

    if frame.empty:
        return []

    if "epoch" not in frame.columns:
        raise ValueError(
            f"Training history has no epoch column: {history_path}"
        )

    frame = frame[frame["epoch"] <= checkpoint_epoch]
    frame = (
        frame.drop_duplicates(subset=["epoch"], keep="last")
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    return frame.to_dict(orient="records")


def atomic_write_history(
    history_records: List[Dict[str, Any]],
    history_path: Path,
) -> None:
    """Atomically persist one deduplicated epoch history table."""
    frame = pd.DataFrame(history_records)
    if not frame.empty and "epoch" in frame.columns:
        frame = (
            frame.drop_duplicates(subset=["epoch"], keep="last")
            .sort_values("epoch")
            .reset_index(drop=True)
        )

    temporary_path = history_path.with_name(
        history_path.name + ".tmp"
    )
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(history_path)


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load old raw state_dict checkpoints and new full checkpoint payloads."""
    checkpoint = load_torch_checkpoint(checkpoint_path, device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    model.load_state_dict(checkpoint)
    return {
        "epoch": None,
        "monitor_name": None,
        "monitor_mode": None,
        "monitor_value": None,
        "model_state_dict": checkpoint,
    }


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Initialize model weights from a pretraining checkpoint.

    Only model parameters are restored. Optimizer and scheduler states are not
    loaded because the downstream Q+SSM stage is a new fine-tuning run.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Initialization checkpoint not found: {checkpoint_path}"
        )

    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=strict,
    )

    missing_keys = list(
        getattr(incompatible, "missing_keys", [])
    )
    unexpected_keys = list(
        getattr(incompatible, "unexpected_keys", [])
    )

    print(
        "Initialized model from checkpoint: "
        f"{checkpoint_path} | strict={strict} | "
        f"missing={len(missing_keys)} | "
        f"unexpected={len(unexpected_keys)}",
        flush=True,
    )

    if missing_keys:
        print(
            f"  Missing keys examples: {missing_keys[:8]}",
            flush=True,
        )

    if unexpected_keys:
        print(
            f"  Unexpected keys examples: {unexpected_keys[:8]}",
            flush=True,
        )

    if isinstance(checkpoint, dict):
        return checkpoint

    return {
        "epoch": None,
        "monitor_name": None,
        "monitor_mode": None,
        "monitor_value": None,
        "model_state_dict": state_dict,
    }


def resolve_initialization_checkpoint(
    config: Any,
    cli_checkpoint: Optional[str],
) -> Optional[Path]:
    """
    Resolve an optional initialization checkpoint.

    The command-line value has priority. Configuration values are accepted as
    a fallback, but unresolved ``__CHECKPOINT__:...`` placeholders must be
    handled by the Chapter 4 protocol runner before invoking ``main.py``.
    """
    candidates: List[Any] = [cli_checkpoint]

    for section_name, key_name in (
        (None, "init_checkpoint"),
        ("training", "init_checkpoint"),
        ("model", "init_checkpoint"),
        ("model", "checkpoint_path"),
        ("model_params", "weight_path"),
        ("ch4_qssm", "init_checkpoint"),
    ):
        if section_name is None:
            candidates.append(config.get(key_name))
            continue

        section = config.get(section_name, {})
        if isinstance(section, dict):
            candidates.append(section.get(key_name))

    for value in candidates:
        if value is None:
            continue

        text = str(value).strip()
        if not text or text.lower() in {"none", "null"}:
            continue

        if text.startswith("__CHECKPOINT__:"):
            raise ValueError(
                "Unresolved initialization checkpoint placeholder: "
                f"{text}. Run this configuration through "
                "run_ch4a_q_to_ssm_protocol.py so the Q-pretraining "
                "checkpoint is resolved first."
            )

        return Path(text)

    return None


def export_experiment_metadata(
    save_dir: Path,
    config: Any,
    args: argparse.Namespace,
    train_basins: List[str],
    test_basins: List[str],
    split_label: str,
) -> None:
    """Export minimal metadata needed for reproducibility."""
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "experiment_name": config.experiment.name,
        "mode": args.mode,
        "model_architecture": config.model.get("architecture", "unknown"),
        "device": TARGET_DEVICE,
        "split_label": split_label,
        "num_train_basins": len(train_basins),
        "num_test_basins": len(test_basins),
        "train_period": list(config.data.train_period),
        "val_period": period_to_list(getattr(config.data, "val_period", None)),
        "test_period": list(config.data.test_period),
        "loss_weights": args.loss_weights,
        "init_checkpoint": getattr(
            args,
            "init_checkpoint",
            None,
        ),
        "strict_init": bool(
            getattr(args, "strict_init", False)
        ),
    }

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def print_run_header(
    config: Any,
    args: argparse.Namespace,
    device: torch.device,
    all_basin_ids: List[str],
    train_basins: List[str],
    test_basins: List[str],
    split_label: str,
    override_msgs: List[str],
) -> None:
    """Print a compact and stable run summary."""
    arch = config.model.get("architecture", "unknown").upper()

    print("\n" + "=" * 112)
    print("HydroMTL Experiment Runner")
    print("-" * 112)
    print(f"Experiment        : {config.experiment.name}")
    print(f"Mode              : {args.mode.upper()}")
    print(f"Architecture      : {arch}")
    print(f"Device            : {device}")
    print(f"Split             : {split_label}")
    print(f"Basins            : total={len(all_basin_ids)} | train={len(train_basins)} | test={len(test_basins)}")
    print(f"Periods           : train={period_to_list(config.data.train_period)} | val={period_to_list(getattr(config.data, 'val_period', None))} | test={period_to_list(config.data.test_period)}")

    for msg in override_msgs:
        print(msg)

    print("=" * 112 + "\n")


def print_epoch_report(
    epoch: int,
    epochs: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
    grad_sims: Dict[str, float],
    best_streamflow_nse: float,
    best_evapotranspiration_nse: float,
    lr: float,
    elapsed: float,
) -> None:

    sf = {
        "NSE": val_metrics.get("streamflow_nse_median", np.nan),
        "KGE": val_metrics.get("streamflow_kge_median", np.nan),
        "RMSE": val_metrics.get("streamflow_rmse_median", np.nan),
        "MAE": val_metrics.get("streamflow_mae_median", np.nan),
        "Bias": val_metrics.get("streamflow_bias_median", np.nan),
        "Corr": val_metrics.get("streamflow_corr_median", np.nan),
    }

    et = {
        "NSE": val_metrics.get("evapotranspiration_nse_median", np.nan),
        "KGE": val_metrics.get("evapotranspiration_kge_median", np.nan),
        "RMSE": val_metrics.get("evapotranspiration_rmse_median", np.nan),
        "MAE": val_metrics.get("evapotranspiration_mae_median", np.nan),
        "Bias": val_metrics.get("evapotranspiration_bias_median", np.nan),
        "Corr": val_metrics.get("evapotranspiration_corr_median", np.nan),
    }

    has_streamflow = is_valid_metric(sf["NSE"])
    has_evapotranspiration = is_valid_metric(et["NSE"])
    enc_sim = grad_sims.get("Encoder", np.nan) if grad_sims else np.nan

    print("-" * 120, flush=True)

    print(
        f"[Epoch {epoch:03d}/{epochs:03d}] "
        f"Train={train_loss:>8.4f} | "
        f"Val={val_loss:>8.4f} | "
        f"LR={lr:.1e} | "
        f"Time={elapsed:>6.1f}s | "
        f"GPU={cuda_memory_mb():>7.1f}MB",
        flush=True,
    )

    if has_streamflow:
        print(
            f"{'Streamflow':<18} | "
            f"NSE={format_metric(sf['NSE'])} | "
            f"KGE={format_metric(sf['KGE'])} | "
            f"RMSE={format_metric(sf['RMSE'])} | "
            f"MAE={format_metric(sf['MAE'])} | "
            f"Bias={format_metric(sf['Bias'])} | "
            f"Corr={format_metric(sf['Corr'])} | "
            f"BestNSE={format_metric(best_streamflow_nse)}",
            flush=True,
        )

    if has_evapotranspiration:
        print(
            f"{'Evapotranspiration':<18} | "
            f"NSE={format_metric(et['NSE'])} | "
            f"KGE={format_metric(et['KGE'])} | "
            f"RMSE={format_metric(et['RMSE'])} | "
            f"MAE={format_metric(et['MAE'])} | "
            f"Bias={format_metric(et['Bias'])} | "
            f"Corr={format_metric(et['Corr'])} | "
            f"BestNSE={format_metric(best_evapotranspiration_nse)}",
            flush=True,
        )

    show_multitask_metrics = (
        has_streamflow
        and has_evapotranspiration
        and is_valid_metric(enc_sim)
    )

    if show_multitask_metrics:
        print(
            f"{'Multi-Task':<18} | "
            f"EncSim={format_metric(enc_sim)}",
            flush=True,
        )

    print("-" * 120, flush=True)


def print_final_metrics(title: str, metrics: Dict[str, float]) -> None:
    """Print final metrics as an aligned table."""
    print("\n" + "=" * 112)
    print(title)
    print("-" * 112)

    if not metrics:
        print("No metrics available.")
    else:
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, (float, int, np.floating, np.integer)):
                print(f"{key:<45} : {format_metric(value, width=12, precision=5)}")

    print("=" * 112 + "\n")


def print_diagnostics(epoch: int, diagnostics: Dict[str, Dict[str, Any]]) -> None:
    """Print optional routing diagnostics."""
    if not diagnostics:
        return

    print(f"[Routing Diagnostics | Epoch {epoch:03d}]")

    for gate_name, gate_info in diagnostics.items():
        entropy = gate_info.get("entropy", np.nan)
        utilization = gate_info.get("utilization", [])
        util_str = ", ".join([f"{float(u):.3f}" for u in utilization]) if len(utilization) else ""
        print(f"{gate_name:<28} | H={format_metric(entropy)} | Util=[{util_str}]")

    print("-" * 112)


def run_climate_diagnostics(config: Any, basin_ids: List[str], ds_export: Any, save_dir: Path) -> None:
    """Run optional climate-stratified gate-utilization diagnostics."""
    metadata_path = Path(config.data.get("basin_metadata_path", ""))

    if not metadata_path.exists() or ds_export is None:
        print("Climate diagnostics skipped: missing basin metadata or exported dataset.")
        return

    metadata_df = pd.read_csv(metadata_path)
    if "basin_id" not in metadata_df.columns:
        print("Climate diagnostics skipped: metadata is missing basin_id.")
        return

    metadata_df["basin_id"] = metadata_df["basin_id"].astype(str).str.zfill(8)
    aridity_dict = dict(zip(metadata_df["basin_id"], metadata_df.get("aridity_index", np.nan)))
    snow_dict = dict(zip(metadata_df["basin_id"], metadata_df.get("snow_fraction", np.nan)))

    analyzer = ClimateSpecializationAnalyzer(aridity_dict, snow_dict)
    target_names = [get_target_name(t) for t in config.data.targets]

    for task in target_names:
        gate_var = f"gate_{task}"

        if gate_var not in ds_export.data_vars:
            continue

        gate_weights = ds_export[gate_var].values
        expert_labels = [f"Expert_{i}" for i in range(gate_weights.shape[-1])]
        summary_df = analyzer.analyze_expert_utilization(basin_ids, gate_weights, expert_labels)
        out_path = save_dir / f"climate_specialization_{task}.csv"

        summary_df.to_csv(out_path)
        print(f"Climate diagnostics exported -> {out_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ungauged_file", type=str, default=None)
    parser.add_argument("--baseline_metrics_csv", type=str, default=None)
    parser.add_argument("--loss_weights", type=str, nargs="+", default=None, help="Override target loss weights: task=weight")
    parser.add_argument("--quiet_batches", action="store_true", help="Disable optional inner batch progress bars.")
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint used to initialize model weights "
            "before training."
        ),
    )
    parser.add_argument(
        "--strict_init",
        action="store_true",
        help=(
            "Require exact model-key matching when loading "
            "--init_checkpoint."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume this experiment from its last_model.pth checkpoint."
        ),
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help=(
            "Explicit full-state checkpoint used to resume the same "
            "experiment."
        ),
    )
    args = parser.parse_args()

    if args.resume and args.resume_checkpoint:
        raise ValueError(
            "Use either --resume or --resume_checkpoint, not both."
        )

    resume_requested = bool(
        args.resume or args.resume_checkpoint
    )

    if args.mode != "train" and resume_requested:
        raise ValueError(
            "Resume options are valid only with --mode train."
        )

    if args.init_checkpoint and resume_requested:
        raise ValueError(
            "--init_checkpoint cannot be combined with --resume or "
            "--resume_checkpoint."
        )

    with open(args.config, "r", encoding="utf-8") as f:
        config = edict(yaml.safe_load(f))

    if args.experiment_name:
        config.experiment.name = args.experiment_name

    if args.quiet_batches:
        config.training.batch_progress = False

    validate_temporal_splits(config)
    set_seed(int(config.get("reproducibility", {}).get("seed", 42)))

    weight_dict, override_msgs = parse_loss_weights(args.loss_weights)

    for target in config.data.targets:
        task_name = get_target_name(target)
        if task_name in weight_dict:
            target.loss_weight = weight_dict[task_name]
            override_msgs.append(f"Loss weight override: {task_name}={target.loss_weight}")

    eval_cfg = config.get("evaluation_protocol", config.get("evaluation", {}))
    device = torch.device(TARGET_DEVICE)

    data_root = Path(config.data.data_root)
    all_basin_ids = discover_basin_ids(data_root)
    basin_list_path = getattr(config.data, "basin_list_path", None)

    if basin_list_path is not None:
        basin_list_path = Path(basin_list_path)
        if not basin_list_path.exists():
            raise FileNotFoundError(f"Missing basin list file: {basin_list_path}")

        selected_basin_ids = [
            line.strip().replace(".0", "").zfill(8)
            for line in basin_list_path.read_text().splitlines()
            if line.strip()
        ]

        available = set(all_basin_ids)
        missing = sorted(set(selected_basin_ids) - available)

        if missing:
            raise ValueError(
                f"{len(missing)} selected basins were not found in data_root. "
                f"Examples: {missing[:10]}"
            )

        all_basin_ids = sorted(selected_basin_ids)
    train_basins, test_basins, split_label = build_spatial_split(config, all_basin_ids)

    search_cfg = config.get("hyperparameter_search", {})
    if search_cfg.get("enabled", False):
        max_train_basins = int(search_cfg.get("max_train_basins", 200))
        if len(train_basins) > max_train_basins:
            train_basins = train_basins[:max_train_basins]
            split_label = f"{split_label}:search_{max_train_basins}_basins"
            print(f"[Search Mode] Using {len(train_basins)} training basins for hyperparameter screening.", flush=True)

    spatial_split = bool(config.data.get("spatial_split", False))

    if spatial_split:
        overlap = set(train_basins).intersection(set(test_basins))
        assert len(overlap) == 0, (
            f"Spatial leakage detected: {len(overlap)} basins appear in both "
            "train and test sets."
        )
    else:
        print(
            "Temporal split detected: train/validation/test periods may share "
            "the same basins. Spatial leakage check is skipped.",
            flush=True,
        )

    ungauged_list = load_ungauged_list(args.ungauged_file)

    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    save_dir.mkdir(parents=True, exist_ok=True)

    export_experiment_metadata(save_dir, config, args, train_basins, test_basins, split_label)
    print_run_header(config, args, device, all_basin_ids, train_basins, test_basins, split_label, override_msgs)

    model = build_model(config)

    if args.mode == "train":
        resume_checkpoint_path: Optional[Path] = None

        if args.resume_checkpoint:
            resume_checkpoint_path = Path(
                args.resume_checkpoint
            ).expanduser()
            if not resume_checkpoint_path.is_absolute():
                resume_checkpoint_path = (
                    PROJECT_ROOT / resume_checkpoint_path
                ).resolve()
        elif args.resume:
            resume_checkpoint_path = (
                save_dir / "last_model.pth"
            ).resolve()

        init_checkpoint = None
        if resume_checkpoint_path is None:
            init_checkpoint = resolve_initialization_checkpoint(
                config=config,
                cli_checkpoint=args.init_checkpoint,
            )

        if init_checkpoint is not None:
            initialize_model_from_checkpoint(
                model=model,
                checkpoint_path=init_checkpoint,
                device=device,
                strict=bool(args.strict_init),
            )

        train_loader, val_loader, _, global_scaler = get_hydro_dataloaders(
            config,
            basin_ids=train_basins,
            mode="train",
            ungauged_basins=ungauged_list,
        )

        validation_period = first_available_period(
            config.data,
            "val_period",
            "valid_period",
            "validation_period",
        )
        use_validation = not period_is_missing(validation_period)

        print("\n" + "=" * 112)
        print("Dataset Diagnostics")
        print("-" * 112)
        print(f"Train Loader Batches  : {len(train_loader):,}")
        print(f"Validation Batches    : {len(val_loader) if val_loader is not None else 0:,}")
        print(f"Batch Size            : {train_loader.batch_size}")
        print("=" * 112 + "\n")

        monitor_name, monitor_mode, monitor_min_delta = resolve_monitor_settings(config, eval_cfg)
        sched_cfg = getattr(config.training, "scheduler", {})
        if sched_cfg and str(sched_cfg.get("type", "")).lower() in {"reduce_on_plateau", "reducelronplateau"}:
            config.training.scheduler["mode"] = monitor_mode

        evaluator = HydroEvaluator(config, train_basins, global_scaler)
        trainer = HydroTrainer(model=model, config=config, device=device, evaluator=evaluator)

        total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params:,}")
        print(f"Monitor             : {monitor_name} ({monitor_mode})")
        print("Optimization started.")
        print("-" * 112)

        epochs = int(getattr(config.training, "epochs", 100))
        best_monitor_value: Optional[float] = None
        best_streamflow_nse = -float("inf")
        best_evapotranspiration_nse = -float("inf")
        best_epoch = 0
        last_epoch = 0
        last_monitor_value = float("nan")

        best_metrics: Dict[str, float] = {}
        best_per_basin_metrics: Dict[str, Dict[str, float]] = {}
        last_val_metrics: Dict[str, float] = {}
        last_val_per_basin_metrics: Dict[str, Dict[str, float]] = {}

        early_cfg = config.training.get("early_stopping", {})
        patience_val = int(early_cfg.get("patience", config.training.get("patience", 15)))
        early_min_delta = float(early_cfg.get("min_delta", monitor_min_delta))

        early_stop = EarlyStopping(
            patience=patience_val,
            min_delta=early_min_delta,
            mode=monitor_mode,
            restore_best_weights=False,
        )
        ckpt = ModelCheckpoint(
            save_dir=str(save_dir),
            save_best_only=bool(config.training.get("save_best_only", True)),
            verbose=False,
        )

        diag_cfg = config.training.get("diagnostics", {})
        diag_enabled = bool(diag_cfg.get("enabled", True))
        diag_interval = int(diag_cfg.get("epoch_interval", getattr(config.training, "diagnostic_interval", 10)))
        diag_interval = max(1, diag_interval)

        history_path = save_dir / "training_history.csv"
        history_records: List[Dict[str, Any]] = []
        start_epoch = 1

        if resume_checkpoint_path is not None:
            resume_payload = resume_training_checkpoint(
                checkpoint_path=resume_checkpoint_path,
                trainer=trainer,
                config=config,
                device=device,
                monitor_name=monitor_name,
                monitor_mode=monitor_mode,
            )

            checkpoint_epoch = int(resume_payload["epoch"])
            if checkpoint_epoch > epochs:
                raise ValueError(
                    "Resume checkpoint exceeds target epochs: "
                    f"checkpoint={checkpoint_epoch}, target={epochs}."
                )

            start_epoch = checkpoint_epoch + 1
            last_epoch = checkpoint_epoch
            last_monitor_value = float(
                resume_payload.get(
                    "monitor_value",
                    float("nan"),
                )
            )
            best_monitor_value = resume_payload.get(
                "best_monitor_value",
                resume_payload.get("monitor_value"),
            )
            if best_monitor_value is not None:
                best_monitor_value = float(best_monitor_value)

            best_epoch = int(
                resume_payload.get(
                    "best_epoch",
                    checkpoint_epoch,
                )
            )
            best_streamflow_nse = float(
                resume_payload.get(
                    "best_streamflow_nse",
                    -float("inf"),
                )
            )
            best_evapotranspiration_nse = float(
                resume_payload.get(
                    "best_evapotranspiration_nse",
                    -float("inf"),
                )
            )
            best_metrics = dict(
                resume_payload.get("best_metrics", {})
            )
            best_per_basin_metrics = dict(
                resume_payload.get(
                    "best_per_basin_metrics",
                    {},
                )
            )
            last_val_metrics = dict(
                resume_payload.get("last_val_metrics", {})
            )
            last_val_per_basin_metrics = dict(
                resume_payload.get(
                    "last_val_per_basin_metrics",
                    {},
                )
            )

            history_records = load_resume_history(
                checkpoint=resume_payload,
                history_path=history_path,
                checkpoint_epoch=checkpoint_epoch,
            )

            saved_gradient_history = resume_payload.get(
                "gradient_history"
            )
            if isinstance(saved_gradient_history, list):
                trainer.gradient_history = list(
                    saved_gradient_history
                )

            early_state = resume_payload.get(
                "early_stopping_state"
            )
            if isinstance(early_state, dict):
                early_stop.load_state_dict(early_state)

            ckpt.best_value = best_monitor_value
            atomic_write_history(
                history_records,
                history_path,
            )

            print("Resuming training.")
            print(f"Checkpoint          : {resume_checkpoint_path}")
            print(f"Checkpoint epoch    : {checkpoint_epoch}")
            print(f"Start epoch         : {start_epoch}")
            print(f"Target epoch        : {epochs}")
            print("Model state         : restored")
            print("Optimizer state     : restored")
            print(
                "Scheduler state     : "
                + (
                    "restored"
                    if trainer.scheduler is not None
                    else "not configured"
                )
            )
            print(
                "AMP scaler state    : "
                + (
                    "restored"
                    if trainer.amp_enabled
                    else "disabled"
                )
            )

        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            last_epoch = epoch
            trainer.current_epoch = epoch
            log_gradients = diag_enabled and (epoch == 1 or epoch % diag_interval == 0)

            train_loss, task_losses, grad_sims = trainer.train_epoch(train_loader, log_gradients=log_gradients)
            if use_validation:
                val_loss, val_metrics, val_per_basin_metrics, _, val_diags = trainer.validate(
                    val_loader,
                    period_dates=validation_period,
                )
            else:
                val_loss = float(train_loss)
                val_metrics = {}
                val_per_basin_metrics = {}
                val_diags = {}

            last_val_metrics = val_metrics
            last_val_per_basin_metrics = val_per_basin_metrics

            current_streamflow_nse = val_metrics.get("streamflow_nse_median", np.nan)
            current_evapotranspiration_nse = val_metrics.get("evapotranspiration_nse_median", np.nan)

            if is_valid_metric(current_streamflow_nse):
                best_streamflow_nse = max(best_streamflow_nse, float(current_streamflow_nse))

            if is_valid_metric(current_evapotranspiration_nse):
                best_evapotranspiration_nse = max(best_evapotranspiration_nse, float(current_evapotranspiration_nse))

            monitor_value = compute_monitor_value(
                monitor_name=monitor_name,
                val_loss=val_loss,
                val_metrics=val_metrics,
            )
            last_monitor_value = monitor_value

            is_best = monitor_is_better(
                value=monitor_value,
                best_value=best_monitor_value,
                mode=monitor_mode,
                min_delta=monitor_min_delta,
            )

            if is_best:
                best_monitor_value = monitor_value
                best_metrics = dict(val_metrics)
                best_per_basin_metrics = dict(val_per_basin_metrics)
                best_epoch = epoch

            scheduler_step(trainer.scheduler, monitor_value)

            ckpt.step(
                model=trainer.model,
                epoch=epoch,
                monitor_value=monitor_value,
                is_best=is_best,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                monitor_name=monitor_name,
                monitor_mode=monitor_mode,
                extra={
                    "val_loss": float(val_loss),
                    "train_loss": float(train_loss),
                    "best_epoch": int(best_epoch),
                    "best_monitor_value": (
                        float(best_monitor_value)
                        if best_monitor_value is not None
                        else None
                    ),
                    "config": to_plain_dict(config),
                },
            )

            if bool(early_cfg.get("enabled", True)):
                early_stop.step(monitor_value)

            lr = trainer.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start

            print_epoch_report(
                epoch=epoch,
                epochs=epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                grad_sims=grad_sims,
                best_streamflow_nse=best_streamflow_nse,
                best_evapotranspiration_nse=best_evapotranspiration_nse,
                lr=lr,
                elapsed=elapsed,
            )

            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr,
                "monitor_name": monitor_name,
                "monitor_mode": monitor_mode,
                "monitor_value": monitor_value,
                "best_monitor_value": best_monitor_value,
                "is_best": is_best,
                "best_epoch": best_epoch,
                "encoder_grad_sim": grad_sims.get("Encoder", np.nan) if grad_sims else np.nan,
            }
            history_row.update({f"task_loss_{k}": v for k, v in task_losses.items()})
            history_row.update(grad_sims)
            history_row.update(val_metrics)
            history_records.append(history_row)
            atomic_write_history(
                history_records,
                history_path,
            )

            ckpt.save_last(
                model=trainer.model,
                epoch=epoch,
                monitor_value=monitor_value,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                monitor_name=monitor_name,
                monitor_mode=monitor_mode,
                filename="last_model.pth",
                extra={
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "best_epoch": int(best_epoch),
                    "best_monitor_value": (
                        float(best_monitor_value)
                        if best_monitor_value is not None
                        else None
                    ),
                    "best_streamflow_nse": float(best_streamflow_nse),
                    "best_evapotranspiration_nse": float(
                        best_evapotranspiration_nse
                    ),
                    "best_metrics": dict(best_metrics),
                    "best_per_basin_metrics": dict(
                        best_per_basin_metrics
                    ),
                    "last_val_metrics": dict(last_val_metrics),
                    "last_val_per_basin_metrics": dict(
                        last_val_per_basin_metrics
                    ),
                    "history_records": list(history_records),
                    "gradient_history": list(
                        trainer.gradient_history
                    ),
                    "early_stopping_state": early_stop.state_dict(),
                    "config": to_plain_dict(config),
                },
            )

            if log_gradients and epoch > 1:
                print_diagnostics(epoch, val_diags)

            release_memory()

            if bool(early_cfg.get("enabled", True)) and getattr(early_stop, "early_stop", False):
                print("Early stopping triggered.")
                break

        print("-" * 112)
        print(f"Best epoch: {best_epoch:03d}")

        final_checkpoint_extra = {
            "best_epoch": int(best_epoch),
            "best_monitor_value": (
                float(best_monitor_value)
                if best_monitor_value is not None
                else None
            ),
            "best_streamflow_nse": float(best_streamflow_nse),
            "best_evapotranspiration_nse": float(
                best_evapotranspiration_nse
            ),
            "best_metrics": dict(best_metrics),
            "best_per_basin_metrics": dict(
                best_per_basin_metrics
            ),
            "last_val_metrics": dict(last_val_metrics),
            "last_val_per_basin_metrics": dict(
                last_val_per_basin_metrics
            ),
            "history_records": list(history_records),
            "gradient_history": list(trainer.gradient_history),
            "early_stopping_state": early_stop.state_dict(),
            "training_complete": bool(last_epoch >= epochs),
            "config": to_plain_dict(config),
        }

        ckpt.save_last(
            model=trainer.model,
            epoch=last_epoch,
            monitor_value=last_monitor_value,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
            filename="last_model.pth",
            extra=final_checkpoint_extra,
        )
        ckpt.save_last(
            model=trainer.model,
            epoch=last_epoch,
            monitor_value=last_monitor_value,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
            filename="final_model.pth",
            extra=final_checkpoint_extra,
        )

        best_path = save_dir / "best_model.pth"
        if use_validation and best_path.exists():
            load_model_checkpoint(trainer.model, best_path, device)
            _, best_metrics, best_per_basin_metrics, _, _ = trainer.validate(
                val_loader,
                period_dates=validation_period,
            )

        final_metrics = dict(best_metrics if best_metrics else last_val_metrics)
        final_per_basin_metrics = best_per_basin_metrics if best_per_basin_metrics else last_val_per_basin_metrics

        input_encoder_sims = [
            h.get("grad_input_encoder_cosine", np.nan)
            for h in trainer.gradient_history
            if h and "grad_input_encoder_cosine" in h
        ]
        shared_expert_sims = [
            h.get("grad_shared_experts_cosine", np.nan)
            for h in trainer.gradient_history
            if h and "grad_shared_experts_cosine" in h
        ]

        final_metrics["monitor_name"] = monitor_name
        final_metrics["monitor_mode"] = monitor_mode
        final_metrics["best_monitor_value"] = float(best_monitor_value) if best_monitor_value is not None else np.nan
        final_metrics["best_epoch"] = int(best_epoch)
        final_metrics["encoder_grad_sim"] = float(np.nanmean(input_encoder_sims)) if input_encoder_sims else np.nan
        final_metrics["grad_input_encoder_cosine"] = final_metrics["encoder_grad_sim"]
        final_metrics["grad_shared_experts_cosine"] = (
            float(np.nanmean(shared_expert_sims)) if shared_expert_sims else np.nan
        )

        atomic_write_history(history_records, history_path)
        pd.DataFrame([final_metrics]).to_csv(save_dir / "validation_summary.csv", index=False)

        if final_per_basin_metrics:
            per_basin_df = pd.DataFrame.from_dict(final_per_basin_metrics, orient="index")
            per_basin_df.index.name = "gauge_id"
            per_basin_df.reset_index().to_csv(save_dir / "validation_per_basin_metrics.csv", index=False)

        print_final_metrics("Final validation metrics", final_metrics)
        print(f"Model training completed. Artifacts saved to: {save_dir}")

        del train_loader, val_loader, trainer, model
        release_memory()

    elif args.mode == "test":
        _, _, test_loader, global_scaler = get_hydro_dataloaders(
            config,
            basin_ids=test_basins,
            mode="test",
            ungauged_basins=ungauged_list,
            scaler_basin_ids=train_basins,
        )

        validation_period = first_available_period(
            config.data,
            "val_period",
            "valid_period",
            "validation_period",
        )

        if period_is_missing(validation_period):
            best_model = save_dir / "final_model.pth"
        else:
            best_model = save_dir / "best_model.pth"
            if not best_model.exists():
                best_model = save_dir / "final_model.pth"

        if not best_model.exists():
            print(f"[FATAL] No trained model found in {save_dir}. Run training first.")
            sys.exit(1)

        checkpoint_info = load_model_checkpoint(model, best_model, device)
        if checkpoint_info.get("monitor_name"):
            print(
                "Loaded checkpoint: "
                f"epoch={checkpoint_info.get('epoch')}, "
                f"{checkpoint_info.get('monitor_name')}={checkpoint_info.get('monitor_value')}",
                flush=True,
            )
        evaluator = HydroEvaluator(config, test_basins, global_scaler)
        trainer = HydroTrainer(model=model, config=config, device=device, evaluator=evaluator)

        print("Independent test evaluation started.")

        _, test_metrics, test_per_basin_metrics, ds_export, test_diags = trainer.validate(
            test_loader,
            period_dates=config.data.test_period,
        )

        print_final_metrics("Independent test metrics", test_metrics)

        export_csv_enabled = eval_cfg.get("export_csv", eval_cfg.get("plot_predictions", True))
        export_nc_enabled = eval_cfg.get("export_netcdf", eval_cfg.get("plot_predictions", True))

        if export_csv_enabled and test_per_basin_metrics:
            csv_path = save_dir / "test_per_basin_metrics.csv"
            per_basin_df = pd.DataFrame.from_dict(test_per_basin_metrics, orient="index")
            per_basin_df.index.name = "gauge_id"
            per_basin_df.reset_index().to_csv(csv_path, index=False)
            print(f"Spatial metrics exported -> {csv_path}")

        if export_nc_enabled and ds_export is not None:
            nc_path = save_dir / "test_predictions_and_weights.nc"
            encoding = {var: {"zlib": True, "complevel": 4, "shuffle": True} for var in ds_export.data_vars}
            ds_export.to_netcdf(nc_path, encoding=encoding)
            print(f"Prediction NetCDF exported -> {nc_path}")

        print_diagnostics(epoch=0, diagnostics=test_diags)
        run_climate_diagnostics(config, test_basins, ds_export, save_dir)

        if args.baseline_metrics_csv and Path(args.baseline_metrics_csv).exists():
            print("[Statistical Test] Wilcoxon paired comparison against baseline.")

            baseline_df = pd.read_csv(args.baseline_metrics_csv, index_col="gage_id")
            target_col = "streamflow_nse"
            current_scores = {
                b_id: test_per_basin_metrics[b_id].get(target_col, np.nan)
                for b_id in test_basins
                if b_id in test_per_basin_metrics
            }
            baseline_scores = baseline_df[target_col].to_dict() if target_col in baseline_df else {}

            if current_scores and baseline_scores:
                try:
                    stat, p_value, effect_size = compute_wilcoxon_paired_test(
                        current_scores,
                        baseline_scores,
                    )
                    print(f"  Metric       : {target_col.upper()}")
                    print(f"  Statistic    : {stat:.5f}")
                    print(f"  p-value      : {p_value:.6e}")
                    print(f"  Effect size  : {effect_size:.5f}")
                except Exception as exc:
                    print(f"  Wilcoxon test failed: {exc}")

        del test_loader, trainer, model
        release_memory()


if __name__ == "__main__":
    main()