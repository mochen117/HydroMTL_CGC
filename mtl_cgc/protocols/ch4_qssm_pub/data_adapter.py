"""PUB-specific DataLoader adapter that leaves the frozen core loader unchanged.

This module exists to preserve reproducibility of completed Chapter 3 and
Chapter 4A experiments.  The standard ``mtl_cgc.data.data_loaders`` module is
not modified.  Instead, Chapter 4B injects a role-aware training loader that:

1. loads source + target basins for target-SSM-assisted MTL scenarios;
2. masks target-basin streamflow labels for the entire PUB period;
3. keeps target-basin SSM labels as auxiliary supervision;
4. fits normalization statistics on source basins only; and
5. removes basin-days that have no supervised target under the PUB role policy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .native_runtime import bootstrap_native_runtime

bootstrap_native_runtime(strict=True)

import numpy as np
import torch
from torch.utils.data import DataLoader

from .protocol import PUBProtocol


@dataclass(frozen=True)
class PUBDataPlan:
    """Explicit basin-role plan used by one PUB run."""

    training_basins: tuple[str, ...]
    scaler_basins: tuple[str, ...]
    masked_streamflow_basins: tuple[str, ...]
    evaluation_basins: tuple[str, ...]

    @property
    def training_count(self) -> int:
        return len(self.training_basins)

    @property
    def scaler_count(self) -> int:
        return len(self.scaler_basins)

    @property
    def evaluation_count(self) -> int:
        return len(self.evaluation_basins)


def build_data_plan(protocol: PUBProtocol) -> PUBDataPlan:
    """Resolve source/target roles into explicit train, scaler, and test sets."""

    return PUBDataPlan(
        training_basins=tuple(protocol.effective_training_basins),
        scaler_basins=tuple(sorted(protocol.source_basins)),
        masked_streamflow_basins=tuple(protocol.masked_streamflow_basins),
        evaluation_basins=tuple(sorted(protocol.target_basins)),
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(
    config: Any,
    *,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator,
) -> dict[str, Any]:
    data_cfg = config["data"]
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(data_cfg.get("batch_size", 64)),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "num_workers": num_workers,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
        "generator": generator,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(
            data_cfg.get("persistent_workers", False)
        )
    return kwargs


def _subset_raw_by_basin(
    raw_data: dict[str, Any],
    positions: list[int],
) -> dict[str, Any]:
    """Copy a basin subset from dense raw arrays without re-reading NetCDF files."""

    index = np.asarray(positions, dtype=int)
    return {
        "dyn": np.take(raw_data["dyn"], index, axis=0),
        "s_num": np.take(raw_data["s_num"], index, axis=0),
        "s_cat": (
            np.take(raw_data["s_cat"], index, axis=0)
            if raw_data.get("s_cat") is not None
            else None
        ),
        "y_dict": {
            task: np.take(values, index, axis=0)
            for task, values in raw_data["y_dict"].items()
        },
    }


def _target_only_view(
    raw_data: dict[str, Any],
    sequence_length: int,
) -> dict[str, Any]:
    """Remove historical context before fitting source-only scaler statistics."""

    context_steps = int(sequence_length) - 1
    return {
        "dyn": raw_data["dyn"][:, context_steps:, :],
        "s_num": raw_data["s_num"],
        "s_cat": raw_data.get("s_cat"),
        "y_dict": {
            task: values[:, context_steps:]
            for task, values in raw_data["y_dict"].items()
        },
    }


def _apply_role_aware_sample_filter(
    dataset: Any,
    protocol: PUBProtocol,
) -> dict[str, int]:
    """Remove basin-days with no valid supervised target under PUB semantics.

    Source basins keep a sample when at least one configured source task is
    observed.  Target basins in assisted MTL scenarios keep only dates with a
    finite SSM observation because target Q is deliberately masked.
    """

    basin_ids = [str(item) for item in dataset.basin_ids]
    basin_to_index = {basin_id: idx for idx, basin_id in enumerate(basin_ids)}
    source_indices = [
        basin_to_index[item]
        for item in protocol.source_basins
        if item in basin_to_index
    ]
    target_indices = [
        basin_to_index[item]
        for item in protocol.target_basins
        if item in basin_to_index
    ]

    target_start = int(dataset.rho) - 1
    target_stop = target_start + int(dataset.num_time_steps)
    keep = np.zeros(
        (int(dataset.num_basins), int(dataset.num_time_steps)),
        dtype=bool,
    )

    source_tasks = [
        task for task in protocol.scenario.active_tasks if task in dataset.y_dict
    ]
    for basin_idx in source_indices:
        valid = np.zeros(int(dataset.num_time_steps), dtype=bool)
        for task in source_tasks:
            values = dataset.y_dict[task][basin_idx, target_start:target_stop]
            valid |= np.isfinite(values)
        keep[basin_idx] = valid

    if protocol.scenario.include_target_during_training:
        if "ssm" not in dataset.y_dict:
            raise KeyError("Target-SSM-assisted PUB requires the 'ssm' task.")
        for basin_idx in target_indices:
            values = dataset.y_dict["ssm"][basin_idx, target_start:target_stop]
            keep[basin_idx] = np.isfinite(values)

    basin_index, time_index = np.where(keep)
    dataset.basin_index = basin_index.astype(np.int32, copy=False)
    dataset.time_index = time_index.astype(np.int32, copy=False)
    dataset.num_samples = int(dataset.basin_index.size)

    if dataset.num_samples == 0:
        raise RuntimeError("PUB role-aware filtering removed every training sample.")

    source_samples = int(keep[source_indices].sum()) if source_indices else 0
    target_samples = int(keep[target_indices].sum()) if target_indices else 0
    return {
        "source_samples": source_samples,
        "target_samples": target_samples,
        "total_samples": int(keep.sum()),
    }


def build_pub_train_bundle(
    config: Any,
    protocol: PUBProtocol,
):
    """Build the PUB training loader with source-only normalization statistics."""

    from mtl_cgc.data.data_loaders import load_nc_to_dict
    from mtl_cgc.data.data_scalers import HydroScaler
    from mtl_cgc.data.data_sets import HydroDataset
    from mtl_cgc.utils.temporal import expand_period_for_sequence, normalize_period

    data_cfg = config["data"]
    data_root = Path(data_cfg["data_root"])
    sequence_length = int(data_cfg.get("sequence_length", 365))
    train_period = normalize_period(data_cfg["train_period"], name="train_period")
    train_read_period = expand_period_for_sequence(train_period, sequence_length)

    plan = build_data_plan(protocol)
    training_ids = list(plan.training_basins)
    scaler_ids = list(plan.scaler_basins)

    train_raw = load_nc_to_dict(
        data_root=data_root,
        basin_ids=training_ids,
        data_cfg=data_cfg,
        split_period=train_read_period,
        split_name="train",
        ungauged_basins=list(plan.masked_streamflow_basins),
        mask_target="streamflow",
    )

    positions = {basin_id: idx for idx, basin_id in enumerate(training_ids)}
    missing_scaler = [item for item in scaler_ids if item not in positions]
    if missing_scaler:
        raise RuntimeError(
            "Source-only scaler basins are absent from the effective training set: "
            f"{missing_scaler[:10]}"
        )

    scaler_raw = _subset_raw_by_basin(
        train_raw,
        [positions[item] for item in scaler_ids],
    )
    scaler_raw = _target_only_view(scaler_raw, sequence_length)

    scaler = HydroScaler(data_cfg)
    scaler.fit_transform(
        scaler_raw["dyn"],
        scaler_raw["s_num"],
        scaler_raw.get("s_cat"),
        scaler_raw["y_dict"],
    )

    train_dataset = HydroDataset(
        raw_data=train_raw,
        data_params=data_cfg,
        basin_ids=training_ids,
        target_period=train_period,
        mode="train",
        scaler=scaler,
    )
    train_dataset.config = config
    filter_counts = _apply_role_aware_sample_filter(train_dataset, protocol)

    seed = int(config.get("reproducibility", {}).get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        **_loader_kwargs(
            config,
            shuffle=True,
            drop_last=True,
            generator=generator,
        ),
    )

    print("PUB training-data semantics")
    print(f"  effective training basins : {plan.training_count}")
    print(f"  source scaler basins       : {plan.scaler_count}")
    print(f"  masked target-Q basins     : {len(plan.masked_streamflow_basins)}")
    print(f"  source supervised samples  : {filter_counts['source_samples']:,}")
    print(f"  target SSM samples         : {filter_counts['target_samples']:,}")
    print(f"  total training samples     : {filter_counts['total_samples']:,}")

    return train_loader, None, None, scaler


def make_pub_loader(
    original_loader: Callable[..., Any],
    protocol: PUBProtocol,
) -> Callable[..., Any]:
    """Return a loader function compatible with the repository's ``main.py``."""

    plan = build_data_plan(protocol)

    def pub_loader(
        config: Any,
        basin_ids: list[str],
        mode: str = "train",
        ungauged_basins: Optional[list[str]] = None,
        mask_target: str = "streamflow",
        scaler_basin_ids: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        del basin_ids, ungauged_basins, mask_target, scaler_basin_ids
        normalized_mode = str(mode).strip().lower()

        if normalized_mode == "train":
            return build_pub_train_bundle(config=config, protocol=protocol)

        if normalized_mode == "test":
            return original_loader(
                config,
                basin_ids=list(plan.evaluation_basins),
                mode="test",
                ungauged_basins=None,
                scaler_basin_ids=list(plan.scaler_basins),
                **kwargs,
            )

        raise ValueError(f"Unsupported PUB loader mode: {mode}")

    return pub_loader
