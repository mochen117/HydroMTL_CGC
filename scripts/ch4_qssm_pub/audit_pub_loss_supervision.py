#!/usr/bin/env python3
"""
Audit effective Q/SSM supervision counts for Chapter 4B PUB training.

This script is intentionally read-only with respect to formal experiment
configurations, model code, checkpoints, and training outputs.

It reconstructs the target-day supervision masks from:
    1. the existing formal YAML configuration;
    2. the fixed PUB source/target basin lists;
    3. the original NetCDF basin data.

No model is instantiated, no optimizer is created, and no backward pass is
executed.

The audit reports:
    - candidate target-day samples;
    - retained training samples;
    - valid streamflow (Q) supervision;
    - valid surface soil moisture (SSM) supervision;
    - jointly supervised samples;
    - Q-only and SSM-only samples;
    - source- and target-basin counts separately;
    - samples consumed per epoch under shuffle=True and drop_last=True.

Important
---------
The task-specific counts reported here are exact data-side eligible supervision
counts under the current masking and sample-retention policy.

The actual training loss also requires finite model predictions. Under normal
finite model output, these eligible observation counts are the observations
participating in the task losses.

Because the training DataLoader shuffles samples and uses drop_last=True, the
identity of the few samples dropped at the end of each epoch can vary. The
script therefore reports rigorous lower/upper bounds for the task-specific
per-epoch counts rather than pretending to know the shuffled tail.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from mtl_cgc.data.data_loaders import load_nc_to_dict
from mtl_cgc.utils.temporal import (
    expand_period_for_sequence,
    normalize_period,
)


# =============================================================================
# Project constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "ch4_qssm_pub"
    / "protocol"
)

Q_TASK = "streamflow"
SSM_TASK = "ssm"

SUPPORTED_SCENARIOS = {
    "hps_target_ssm",
    "cgc_target_ssm",
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class PartitionCounts:
    """Supervision-count summary for one basin partition."""

    partition: str

    candidate_samples: int
    retained_samples: int
    discarded_samples: int

    q_valid: int
    ssm_valid: int

    both_valid: int
    q_only: int
    ssm_only: int

    candidate_neither_valid: int

    @property
    def q_coverage(self) -> float:
        """Fraction of retained samples carrying Q supervision."""
        return safe_ratio(self.q_valid, self.retained_samples)

    @property
    def ssm_coverage(self) -> float:
        """Fraction of retained samples carrying SSM supervision."""
        return safe_ratio(self.ssm_valid, self.retained_samples)


# =============================================================================
# Command-line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit Q/SSM supervision counts for a Chapter 4B "
            "target-SSM-assisted PUB configuration."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Formal HPS/CGC target-SSM PUB YAML configuration. "
            "Example: "
            "mtl_cgc/configs/ch4_qssm_pub/formal/seed42/fold01/"
            "ch4b_pub_formal_f01_hps_target_ssm_seed42.yaml"
        ),
    )

    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR,
        help=(
            "PUB protocol directory containing "
            "foldXX/source_basins.txt and foldXX/target_basins.txt."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path. "
            "If omitted, the script is print-only."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on source/target overlap, target-Q leakage, "
            "unsupported interpolation, or accounting inconsistency."
        ),
    )

    return parser.parse_args()


# =============================================================================
# General utilities
# =============================================================================

def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths against the HydroMTL_CGC project root."""

    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    if not isinstance(config, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}, "
            f"got {type(config).__name__}."
        )

    return config


def read_basin_ids(path: Path) -> List[str]:
    """Read one basin ID per line and normalize IDs to eight characters."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing basin-list file: {path}"
        )

    basin_ids: List[str] = []

    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            token = line.strip()

            if not token or token.startswith("#"):
                continue

            basin_id = token.split()[0].strip().zfill(8)
            basin_ids.append(basin_id)

    if not basin_ids:
        raise ValueError(
            f"No basin IDs found in {path}"
        )

    if len(basin_ids) != len(set(basin_ids)):
        raise ValueError(
            f"Duplicate basin IDs detected in {path}"
        )

    return basin_ids


def parse_fold_and_scenario(
    config_path: Path,
) -> Tuple[int, str]:
    """Infer PUB fold and scenario from the formal configuration filename."""

    pattern = re.compile(
        r"_f(?P<fold>\d{2})_"
        r"(?P<scenario>"
        r"stl_q|hps_target_ssm|cgc_target_ssm"
        r")_seed\d+$"
    )

    match = pattern.search(config_path.stem)

    if match is None:
        raise ValueError(
            "Could not infer fold/scenario from config filename.\n"
            "Expected a filename such as:\n"
            "  ch4b_pub_formal_f01_hps_target_ssm_seed42.yaml\n"
            f"Received:\n  {config_path.name}"
        )

    fold = int(match.group("fold"))
    scenario = match.group("scenario")

    return fold, scenario


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """Return a ratio or NaN when denominator is zero."""

    if denominator <= 0:
        return float("nan")

    return float(numerator) / float(denominator)


def format_int(value: int) -> str:
    """Format integer counts with thousands separators."""

    return f"{int(value):,}"


def inclusive_days(period: Sequence[str]) -> int:
    """Count inclusive calendar days in a two-element date period."""

    if len(period) != 2:
        raise ValueError(
            f"Expected two-element period, got: {period!r}"
        )

    start = date.fromisoformat(str(period[0]))
    end = date.fromisoformat(str(period[1]))

    if end < start:
        raise ValueError(
            f"Invalid period: {period!r}"
        )

    return (end - start).days + 1


# =============================================================================
# Configuration checks
# =============================================================================

def get_target_configs(
    data_cfg: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return target configurations keyed by lowercase task name."""

    targets = data_cfg.get("targets", [])

    if not isinstance(targets, list):
        raise TypeError(
            "config.data.targets must be a list."
        )

    result: Dict[str, Dict[str, Any]] = {}

    for target in targets:
        if not isinstance(target, Mapping):
            raise TypeError(
                "Each config.data.targets item must be a mapping."
            )

        target_dict = dict(target)

        task_name = str(
            target_dict.get("name", "")
        ).strip().lower()

        if not task_name:
            raise ValueError(
                "Found target without a valid 'name'."
            )

        result[task_name] = target_dict

    return result


def validate_target_configuration(
    data_cfg: Mapping[str, Any],
    *,
    strict: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Validate the Q/SSM target configuration.

    This audit intentionally does not silently reproduce target interpolation.
    The current Chapter 4 formal protocol is expected to use observed-value
    masking rather than SSM interpolation.
    """

    targets = get_target_configs(data_cfg)

    for task_name in (Q_TASK, SSM_TASK):
        if task_name not in targets:
            raise KeyError(
                f"Required target '{task_name}' is missing."
            )

        weight = float(
            targets[task_name].get(
                "loss_weight",
                1.0,
            )
        )

        if weight <= 0.0:
            raise ValueError(
                f"Target '{task_name}' has non-positive "
                f"loss weight: {weight}"
            )

        interpolate = bool(
            targets[task_name].get(
                "interpolate_missing",
                False,
            )
        )

        # Chapter 4B inherits the established streamflow missing-value
        # treatment from the Q-SSM configuration. Short Q gaps may therefore
        # be interpolated during training. SSM, however, must remain based on
        # observed values only.
        if task_name == SSM_TASK and interpolate:
            message = (
                "Target 'ssm' has interpolate_missing=True, but Chapter 4B "
                "requires observed-value SSM supervision with missing values "
                "masked from the SSM loss."
            )

            if strict:
                raise RuntimeError(message)

            print(f"[WARNING] {message}")

        if task_name == Q_TASK and interpolate:
            interpolation_limit = int(
                targets[task_name].get(
                    "interpolation_limit",
                    3,
                )
            )

            print(
                "[INFO] streamflow.interpolate_missing=True; "
                "the audit will reproduce the training-time Q "
                f"interpolation for validity counting (limit="
                f"{interpolation_limit})."
            )

    return targets


# =============================================================================
# Missing-value semantics
# =============================================================================

def get_missing_sentinel() -> Optional[float]:
    """
    Read the missing-value sentinel from the current loss implementation.

    If the class or attribute is unavailable, finite-value masking is still
    applied and None is returned.
    """

    try:
        from mtl_cgc.core.losses.crits import (
            DynamicMultiTaskLoss,
        )

        value = getattr(
            DynamicMultiTaskLoss,
            "_MISSING_SENTINEL",
            None,
        )

    except Exception:
        return None

    if value is None:
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def valid_target_mask(
    values: np.ndarray,
    missing_sentinel: Optional[float],
) -> np.ndarray:
    """
    Build the data-side validity mask used for supervision counting.

    The training loss additionally requires finite model predictions.
    """

    mask = np.isfinite(values)

    if (
        missing_sentinel is not None
        and np.isfinite(missing_sentinel)
    ):
        mask &= values != missing_sentinel

    return mask


# =============================================================================
# Raw-data loading and temporal alignment
# =============================================================================

def as_2d_target(
    values: np.ndarray,
    task_name: str,
) -> np.ndarray:
    """Normalize a target array to [basin, time]."""

    array = np.asarray(values)

    if (
        array.ndim == 3
        and array.shape[-1] == 1
    ):
        array = array[..., 0]

    if array.ndim != 2:
        raise ValueError(
            f"Expected '{task_name}' target with shape "
            "[basin, time] or [basin, time, 1], "
            f"got {array.shape}."
        )

    return array


def apply_training_target_interpolation(
    raw_data: Mapping[str, Any],
    target_configs: Mapping[str, Mapping[str, Any]],
) -> None:
    """
    Reproduce training-time target interpolation for supervision counting.

    Notes
    -----
    The production HydroDataset applies configured target interpolation only
    during training. This audit reproduces only the finite/NaN pattern needed
    for supervision accounting; it does not modify project files, NetCDF files,
    formal YAML configurations, checkpoints, or running experiments.

    Interpolation is applied to the complete context-expanded target series
    before the target-period slice is extracted. This matches the temporal
    ordering used by HydroDataset.

    A target-basin streamflow series that has already been fully masked to NaN
    remains entirely NaN after interpolation and therefore cannot contribute
    target-Q supervision.
    """

    y_dict = raw_data.get("y_dict")

    if not isinstance(y_dict, dict):
        raise TypeError(
            "raw_data['y_dict'] must be a mutable dictionary."
        )

    for task_name, target_cfg in target_configs.items():
        if not bool(
            target_cfg.get(
                "interpolate_missing",
                False,
            )
        ):
            continue

        if task_name not in y_dict:
            raise KeyError(
                f"Configured interpolation target '{task_name}' "
                "is missing from raw_data['y_dict']."
            )

        interpolation_limit = int(
            target_cfg.get(
                "interpolation_limit",
                3,
            )
        )

        if interpolation_limit <= 0:
            raise ValueError(
                f"interpolation_limit for target '{task_name}' "
                f"must be positive, got {interpolation_limit}."
            )

        values = np.asarray(
            y_dict[task_name]
        )

        original_ndim = values.ndim

        if values.ndim == 3 and values.shape[-1] == 1:
            work = values[..., 0].copy()
        elif values.ndim == 2:
            work = values.copy()
        else:
            raise ValueError(
                f"Expected target '{task_name}' with shape "
                "[basin, time] or [basin, time, 1], "
                f"got {values.shape}."
            )

        for basin_idx in range(work.shape[0]):
            series = pd.Series(
                work[basin_idx],
                copy=True,
            )

            interpolated = series.interpolate(
                method="linear",
                limit=interpolation_limit,
                limit_direction="forward",
            )

            work[basin_idx] = interpolated.to_numpy(
                dtype=work.dtype,
                copy=False,
            )

        if original_ndim == 3:
            y_dict[task_name] = work[..., np.newaxis]
        else:
            y_dict[task_name] = work


def target_period_view(
    raw_data: Mapping[str, Any],
    task_name: str,
    sequence_length: int,
    expected_target_days: int,
) -> np.ndarray:
    """
    Extract target-date observations from context-expanded raw arrays.

    Under N-to-1 semantics, the first rho-1 raw timesteps are historical
    context. The first prediction target is therefore at index rho-1.
    """

    y_dict = raw_data.get("y_dict")

    if not isinstance(y_dict, Mapping):
        raise KeyError(
            "Raw data does not contain a valid 'y_dict'."
        )

    if task_name not in y_dict:
        raise KeyError(
            f"Target '{task_name}' missing from y_dict. "
            f"Available: {sorted(y_dict)}"
        )

    values = as_2d_target(
        np.asarray(y_dict[task_name]),
        task_name,
    )

    context_steps = sequence_length - 1

    if context_steps < 0:
        raise ValueError(
            f"Invalid sequence_length={sequence_length}"
        )

    target_values = values[:, context_steps:]

    if target_values.shape[1] != expected_target_days:
        raise RuntimeError(
            f"Target-date alignment failure for '{task_name}'.\n"
            f"Raw shape            : {values.shape}\n"
            f"Context steps        : {context_steps}\n"
            f"Expected target days : {expected_target_days}\n"
            f"Observed target shape: {target_values.shape}"
        )

    return target_values


def load_partition_raw(
    *,
    data_root: Path,
    basin_ids: Sequence[str],
    data_cfg: Mapping[str, Any],
    read_period: Sequence[str],
    ungauged_basins: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """
    Load one partition with the project's existing NetCDF reader.

    For the target partition, passing all target basin IDs through
    ungauged_basins reproduces the Chapter 4B training-time target-Q mask.
    """

    return load_nc_to_dict(
        data_root=data_root,
        basin_ids=list(basin_ids),
        data_cfg=dict(data_cfg),
        split_period=list(read_period),
        split_name="train",
        ungauged_basins=(
            list(ungauged_basins)
            if ungauged_basins is not None
            else None
        ),
        mask_target=Q_TASK,
    )


# =============================================================================
# Supervision accounting
# =============================================================================

def count_source_partition(
    q_values: np.ndarray,
    ssm_values: np.ndarray,
    missing_sentinel: Optional[float],
) -> PartitionCounts:
    """
    Count source-basin supervision.

    In assisted multi-task training, a source sample is retained when at least
    one configured hydrological response (Q or SSM) is available.
    """

    q_valid = valid_target_mask(
        q_values,
        missing_sentinel,
    )

    ssm_valid = valid_target_mask(
        ssm_values,
        missing_sentinel,
    )

    keep = q_valid | ssm_valid

    both = q_valid & ssm_valid
    q_only = q_valid & ~ssm_valid
    ssm_only = ~q_valid & ssm_valid
    neither = ~q_valid & ~ssm_valid

    return PartitionCounts(
        partition="source",
        candidate_samples=int(q_valid.size),
        retained_samples=int(keep.sum()),
        discarded_samples=int((~keep).sum()),
        q_valid=int((q_valid & keep).sum()),
        ssm_valid=int((ssm_valid & keep).sum()),
        both_valid=int((both & keep).sum()),
        q_only=int((q_only & keep).sum()),
        ssm_only=int((ssm_only & keep).sum()),
        candidate_neither_valid=int(
            neither.sum()
        ),
    )


def count_target_partition(
    q_values: np.ndarray,
    ssm_values: np.ndarray,
    missing_sentinel: Optional[float],
) -> PartitionCounts:
    """
    Count target-basin supervision.

    For target-SSM-assisted PUB:
        - target Q must be fully masked;
        - a target sample is retained only when SSM is available.
    """

    q_valid = valid_target_mask(
        q_values,
        missing_sentinel,
    )

    ssm_valid = valid_target_mask(
        ssm_values,
        missing_sentinel,
    )

    keep = ssm_valid

    both = q_valid & ssm_valid
    q_only = q_valid & ~ssm_valid
    ssm_only = ~q_valid & ssm_valid
    neither = ~q_valid & ~ssm_valid

    return PartitionCounts(
        partition="target",
        candidate_samples=int(q_valid.size),
        retained_samples=int(keep.sum()),
        discarded_samples=int((~keep).sum()),
        q_valid=int((q_valid & keep).sum()),
        ssm_valid=int((ssm_valid & keep).sum()),
        both_valid=int((both & keep).sum()),
        q_only=int((q_only & keep).sum()),
        ssm_only=int((ssm_only & keep).sum()),
        candidate_neither_valid=int(
            neither.sum()
        ),
    )


def combine_counts(
    source: PartitionCounts,
    target: PartitionCounts,
) -> PartitionCounts:
    """Combine source and target supervision summaries."""

    return PartitionCounts(
        partition="combined",

        candidate_samples=(
            source.candidate_samples
            + target.candidate_samples
        ),

        retained_samples=(
            source.retained_samples
            + target.retained_samples
        ),

        discarded_samples=(
            source.discarded_samples
            + target.discarded_samples
        ),

        q_valid=(
            source.q_valid
            + target.q_valid
        ),

        ssm_valid=(
            source.ssm_valid
            + target.ssm_valid
        ),

        both_valid=(
            source.both_valid
            + target.both_valid
        ),

        q_only=(
            source.q_only
            + target.q_only
        ),

        ssm_only=(
            source.ssm_only
            + target.ssm_only
        ),

        candidate_neither_valid=(
            source.candidate_neither_valid
            + target.candidate_neither_valid
        ),
    )


def validate_accounting(
    *,
    source: PartitionCounts,
    target: PartitionCounts,
    combined: PartitionCounts,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    strict: bool,
) -> None:
    """Validate fold separation and supervision identities."""

    errors: List[str] = []

    overlap = sorted(
        set(source_ids).intersection(target_ids)
    )

    if overlap:
        errors.append(
            "Source/target basin overlap detected: "
            f"n={len(overlap)}, examples={overlap[:5]}"
        )

    for counts in (
        source,
        target,
        combined,
    ):
        categorized = (
            counts.both_valid
            + counts.q_only
            + counts.ssm_only
        )

        if categorized != counts.retained_samples:
            errors.append(
                f"{counts.partition}: "
                "retained sample accounting mismatch: "
                f"retained={counts.retained_samples}, "
                f"categorized={categorized}"
            )

        if (
            counts.retained_samples
            + counts.discarded_samples
            != counts.candidate_samples
        ):
            errors.append(
                f"{counts.partition}: "
                "candidate sample accounting mismatch."
            )

    if target.q_valid != 0:
        errors.append(
            "Target-basin Q leakage detected: "
            f"target q_valid={target.q_valid}"
        )

    if (
        target.both_valid != 0
        or target.q_only != 0
    ):
        errors.append(
            "Target partition contains retained "
            "Q-supervised samples."
        )

    if errors:
        message = "\n".join(
            f"  - {item}"
            for item in errors
        )

        if strict:
            raise RuntimeError(
                "Strict supervision audit failed:\n"
                + message
            )

        print(
            "\n[WARNING] Supervision audit "
            "found potential issues:"
        )
        print(message)


# =============================================================================
# Reporting
# =============================================================================

def print_partition(
    counts: PartitionCounts,
) -> None:
    """Print one partition-level supervision summary."""

    print(
        f"\n[{counts.partition.upper()}]"
    )

    print(
        "  candidate target-day samples : "
        f"{format_int(counts.candidate_samples)}"
    )

    print(
        "  retained training samples    : "
        f"{format_int(counts.retained_samples)}"
    )

    print(
        "  discarded samples            : "
        f"{format_int(counts.discarded_samples)}"
    )

    print(
        "  Q valid supervision          : "
        f"{format_int(counts.q_valid)}"
    )

    print(
        "  SSM valid supervision        : "
        f"{format_int(counts.ssm_valid)}"
    )

    print(
        "  both Q + SSM valid           : "
        f"{format_int(counts.both_valid)}"
    )

    print(
        "  Q only                       : "
        f"{format_int(counts.q_only)}"
    )

    print(
        "  SSM only                     : "
        f"{format_int(counts.ssm_only)}"
    )

    print(
        "  candidate neither valid      : "
        f"{format_int(counts.candidate_neither_valid)}"
    )

    print(
        "  Q coverage / retained        : "
        f"{100.0 * counts.q_coverage:8.3f}%"
    )

    print(
        "  SSM coverage / retained      : "
        f"{100.0 * counts.ssm_coverage:8.3f}%"
    )


def print_epoch_accounting(
    combined: PartitionCounts,
    batch_size: int,
) -> Dict[str, int]:
    """
    Report training-loader accounting for shuffle=True/drop_last=True.

    Task-specific exact counts within the dropped tail cannot be known without
    replaying the epoch sampler, so rigorous bounds are reported instead.
    """

    if batch_size <= 0:
        raise ValueError(
            f"Invalid batch_size={batch_size}"
        )

    num_batches = (
        combined.retained_samples
        // batch_size
    )

    consumed = (
        num_batches
        * batch_size
    )

    dropped = (
        combined.retained_samples
        - consumed
    )

    q_min = max(
        0,
        combined.q_valid - dropped,
    )

    q_max = combined.q_valid

    ssm_min = max(
        0,
        combined.ssm_valid - dropped,
    )

    ssm_max = combined.ssm_valid

    print(
        "\n[TRAIN-LOADER EPOCH ACCOUNTING]"
    )

    print(
        f"  batch size                    : "
        f"{batch_size}"
    )

    print(
        "  shuffle                       : "
        "True"
    )

    print(
        "  drop_last                     : "
        "True"
    )

    print(
        "  full batches / epoch          : "
        f"{format_int(num_batches)}"
    )

    print(
        "  samples consumed / epoch      : "
        f"{format_int(consumed)}"
    )

    print(
        "  shuffled tail dropped / epoch : "
        f"{format_int(dropped)}"
    )

    print(
        "  Q supervision / epoch         : "
        f"[{format_int(q_min)}, "
        f"{format_int(q_max)}]"
    )

    print(
        "  SSM supervision / epoch       : "
        f"[{format_int(ssm_min)}, "
        f"{format_int(ssm_max)}]"
    )

    print(
        "  note                           : "
        "exact task counts depend on the "
        "shuffled tail."
    )

    return {
        "batch_size": batch_size,
        "num_batches_per_epoch": num_batches,
        "samples_consumed_per_epoch": consumed,
        "samples_dropped_per_epoch": dropped,

        "q_supervision_per_epoch_min": q_min,
        "q_supervision_per_epoch_max": q_max,

        "ssm_supervision_per_epoch_min": ssm_min,
        "ssm_supervision_per_epoch_max": ssm_max,
    }


def write_csv(
    path: Path,
    *,
    fold: int,
    scenario: str,
    source: PartitionCounts,
    target: PartitionCounts,
    combined: PartitionCounts,
    epoch_info: Mapping[str, int],
) -> None:
    """Write optional audit results outside formal run directories."""

    output_path = resolve_project_path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: List[Dict[str, Any]] = []

    for counts in (
        source,
        target,
        combined,
    ):
        row: Dict[str, Any] = {
            "fold": fold,
            "scenario": scenario,
            **asdict(counts),
            "q_coverage": counts.q_coverage,
            "ssm_coverage": counts.ssm_coverage,
        }

        if counts.partition == "combined":
            row.update(epoch_info)

        rows.append(row)

    fieldnames: List[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_obj:

        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nAudit CSV written to:\n"
        f"  {output_path}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run the read-only PUB loss-supervision audit."""

    args = parse_args()

    config_path = resolve_project_path(
        args.config
    )

    protocol_dir = resolve_project_path(
        args.protocol_dir
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config does not exist: "
            f"{config_path}"
        )

    fold, scenario = parse_fold_and_scenario(
        config_path
    )

    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(
            "This script audits target-SSM-assisted "
            "multi-task scenarios only.\n"
            f"Received: {scenario}\n"
            f"Expected: {sorted(SUPPORTED_SCENARIOS)}"
        )

    config = load_yaml(
        config_path
    )

    data_cfg = config.get("data")

    if not isinstance(data_cfg, dict):
        raise KeyError(
            "Config is missing a valid "
            "'data' mapping."
        )

    target_configs = (
        validate_target_configuration(
            data_cfg,
            strict=args.strict,
        )
    )

    sequence_length = int(
        data_cfg.get(
            "sequence_length",
            0,
        )
    )

    if sequence_length <= 0:
        raise ValueError(
            "Invalid data.sequence_length: "
            f"{sequence_length}"
        )

    batch_size = int(
        data_cfg.get(
            "batch_size",
            64,
        )
    )

    train_period = normalize_period(
        data_cfg["train_period"],
        name="train_period",
    )

    train_read_period = (
        expand_period_for_sequence(
            train_period,
            sequence_length,
        )
    )

    target_days = inclusive_days(
        train_period
    )

    data_root = resolve_project_path(
        Path(
            str(data_cfg["data_root"])
        )
    )

    fold_dir = (
        protocol_dir
        / f"fold{fold:02d}"
    )

    source_ids = read_basin_ids(
        fold_dir
        / "source_basins.txt"
    )

    target_ids = read_basin_ids(
        fold_dir
        / "target_basins.txt"
    )

    q_weight = float(
        target_configs[Q_TASK].get(
            "loss_weight",
            1.0,
        )
    )

    ssm_weight = float(
        target_configs[SSM_TASK].get(
            "loss_weight",
            1.0,
        )
    )

    print("=" * 96)
    print(
        "Chapter 4B PUB loss-supervision audit"
    )
    print("-" * 96)

    print(
        f"Config              : "
        f"{config_path}"
    )

    print(
        f"Scenario            : "
        f"{scenario}"
    )

    print(
        f"Fold                : "
        f"{fold:02d}"
    )

    print(
        f"Source basins       : "
        f"{len(source_ids)}"
    )

    print(
        f"Target basins       : "
        f"{len(target_ids)}"
    )

    print(
        f"Data root           : "
        f"{data_root}"
    )

    print(
        f"Train target period : "
        f"{train_period}"
    )

    print(
        f"Read period         : "
        f"{train_read_period}"
    )

    print(
        f"Target days         : "
        f"{target_days}"
    )

    print(
        f"Sequence length     : "
        f"{sequence_length}"
    )

    print(
        f"Loss weights        : "
        f"Q={q_weight}, "
        f"SSM={ssm_weight}"
    )

    print("=" * 96)

    missing_sentinel = (
        get_missing_sentinel()
    )

    # -------------------------------------------------------------------------
    # Source partition
    # -------------------------------------------------------------------------
    print(
        "\nLoading source partition "
        "(read-only)..."
    )

    source_raw = load_partition_raw(
        data_root=data_root,
        basin_ids=source_ids,
        data_cfg=data_cfg,
        read_period=train_read_period,
        ungauged_basins=None,
    )

    # Reproduce the training-time finite/NaN target pattern before counting.
    apply_training_target_interpolation(
        source_raw,
        target_configs,
    )

    source_q = target_period_view(
        source_raw,
        Q_TASK,
        sequence_length,
        target_days,
    )

    source_ssm = target_period_view(
        source_raw,
        SSM_TASK,
        sequence_length,
        target_days,
    )

    source_counts = (
        count_source_partition(
            source_q,
            source_ssm,
            missing_sentinel,
        )
    )

    # Release source raw arrays before loading target basins to reduce RAM use.
    del source_raw
    del source_q
    del source_ssm
    gc.collect()

    # -------------------------------------------------------------------------
    # Target partition
    # -------------------------------------------------------------------------
    print(
        "Loading target partition "
        "with training-time Q mask "
        "(read-only)..."
    )

    target_raw = load_partition_raw(
        data_root=data_root,
        basin_ids=target_ids,
        data_cfg=data_cfg,
        read_period=train_read_period,

        # This reproduces the PUB training-time target-Q mask.
        ungauged_basins=target_ids,
    )

    # Apply interpolation only after target Q has been fully masked.
    # An all-NaN target-Q sequence must remain all-NaN.
    apply_training_target_interpolation(
        target_raw,
        target_configs,
    )

    target_q = target_period_view(
        target_raw,
        Q_TASK,
        sequence_length,
        target_days,
    )

    target_ssm = target_period_view(
        target_raw,
        SSM_TASK,
        sequence_length,
        target_days,
    )

    target_counts = (
        count_target_partition(
            target_q,
            target_ssm,
            missing_sentinel,
        )
    )

    del target_raw
    del target_q
    del target_ssm
    gc.collect()

    # -------------------------------------------------------------------------
    # Combined accounting
    # -------------------------------------------------------------------------
    combined_counts = combine_counts(
        source_counts,
        target_counts,
    )

    validate_accounting(
        source=source_counts,
        target=target_counts,
        combined=combined_counts,
        source_ids=source_ids,
        target_ids=target_ids,
        strict=args.strict,
    )

    print_partition(
        source_counts
    )

    print_partition(
        target_counts
    )

    print_partition(
        combined_counts
    )

    epoch_info = (
        print_epoch_accounting(
            combined_counts,
            batch_size,
        )
    )

    print(
        "\n[INTERPRETATION]"
    )

    print(
        "  1. The dataset-level Q/SSM counts are "
        "data-side eligible supervision counts."
    )

    print(
        "  2. Target-basin Q must be zero after the "
        "PUB training mask."
    )

    print(
        "  3. Target-basin SSM is retained as auxiliary "
        "supervision."
    )

    print(
        "  4. These counts quantify supervision availability, "
        "not gradient contribution."
    )

    print(
        "  5. Actual task influence also depends on loss scale, "
        "loss weight, gradient magnitude, and gradient alignment."
    )

    print(
        "  6. No model, checkpoint, optimizer, formal YAML, "
        "or running experiment was modified."
    )

    if args.output_csv is not None:
        write_csv(
            args.output_csv,
            fold=fold,
            scenario=scenario,
            source=source_counts,
            target=target_counts,
            combined=combined_counts,
            epoch_info=epoch_info,
        )


if __name__ == "__main__":
    main()
