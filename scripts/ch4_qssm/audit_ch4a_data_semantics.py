#!/usr/bin/env python3
"""
Audit runtime target-missing and sample-filter semantics for Experiment A.

The script compares raw target masks with the transformed HydroDataset and
verifies that the number of indexed training samples follows data.sample_filter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from mtl_cgc.data.data_loaders import (
    get_hydro_dataloaders,
    load_nc_to_dict,
)
from mtl_cgc.utils.temporal import (
    expand_period_for_sequence,
    normalize_period,
)


ConfigDict = Dict[str, Any]


def require(condition: bool, message: str) -> None:
    """Raise a descriptive runtime-audit error."""
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit Experiment A runtime data semantics."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--max-basins",
        default=32,
        type=int,
    )

    return parser.parse_args()


def main() -> None:
    """Run the real-data semantic audit."""
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]

    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()

    require(
        config_path.exists(),
        f"Configuration not found: {config_path}",
    )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    require(
        isinstance(config, dict),
        "Configuration must be a dictionary.",
    )

    data_config = config["data"]

    basin_file_value = (
        data_config.get("basin_list_path")
        or data_config.get("basin_file")
    )
    require(
        basin_file_value is not None,
        "No basin list path was found.",
    )

    basin_file = Path(basin_file_value)
    if not basin_file.is_absolute():
        basin_file = project_root / basin_file

    basin_ids = [
        line.strip()
        for line in basin_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ][: args.max_basins]

    require(
        len(basin_ids) > 0,
        "The selected basin list is empty.",
    )

    sequence_length = int(
        data_config["sequence_length"]
    )

    train_period = normalize_period(
        data_config["train_period"],
        name="train_period",
    )
    train_read_period = expand_period_for_sequence(
        train_period,
        sequence_length,
    )

    data_root = Path(data_config["data_root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root

    raw_data = load_nc_to_dict(
        data_root=data_root,
        basin_ids=basin_ids,
        data_cfg=data_config,
        split_period=train_read_period,
        split_name="runtime_audit",
    )

    train_loader, _, _, _ = get_hydro_dataloaders(
        config=config,
        basin_ids=basin_ids,
        mode="train",
    )

    dataset = train_loader.dataset
    target_start = sequence_length - 1

    print("=" * 100)
    print("Configuration       :", config_path)
    print("Basins audited      :", len(basin_ids))
    print("Training period     :", train_period)
    print("Sequence length     :", sequence_length)
    print("Dataset samples     :", len(dataset))
    print(
        "Full basin-day count:",
        len(basin_ids) * dataset.num_time_steps,
    )

    dataset_target_views: Dict[str, np.ndarray] = {}

    for target_config in data_config.get(
        "targets",
        [],
    ):
        task_name = str(
            target_config["name"]
        ).strip().lower()

        raw_target = raw_data["y_dict"][task_name][
            :,
            target_start:,
        ]

        dataset_target = dataset.y_dict[task_name][
            :,
            target_start:,
        ]

        require(
            raw_target.shape == dataset_target.shape,
            (
                f"Target shape mismatch for {task_name}: "
                f"raw={raw_target.shape}, "
                f"dataset={dataset_target.shape}"
            ),
        )

        dataset_target_views[task_name] = (
            dataset_target
        )

        raw_mask = np.isfinite(raw_target)
        dataset_mask = np.isfinite(
            dataset_target
        )

        print("-" * 100)
        print("Target              :", task_name)
        print(
            "interpolate_missing :",
            target_config.get(
                "interpolate_missing"
            ),
        )
        print(
            "Raw valid ratio     :",
            float(raw_mask.mean()),
        )
        print(
            "Dataset valid ratio :",
            float(dataset_mask.mean()),
        )
        print(
            "Raw valid count     :",
            int(raw_mask.sum()),
        )
        print(
            "Dataset valid count :",
            int(dataset_mask.sum()),
        )

        if not bool(
            target_config.get(
                "interpolate_missing",
                False,
            )
        ):
            require(
                np.array_equal(
                    raw_mask,
                    dataset_mask,
                ),
                (
                    f"Missing-value mask changed for "
                    f"non-interpolated target {task_name}."
                ),
            )

    sample_filter = data_config.get(
        "sample_filter"
    )

    use_filter = (
        isinstance(sample_filter, dict)
        and bool(
            sample_filter.get(
                "enabled",
                False,
            )
        )
        and "train"
        in {
            str(mode).strip().lower()
            for mode in sample_filter.get(
                "apply_to_modes",
                [],
            )
        }
    )

    full_sample_count = (
        len(basin_ids)
        * dataset.num_time_steps
    )

    if use_filter:
        required_targets = [
            str(task).strip().lower()
            for task in sample_filter.get(
                "required_valid_targets",
                [],
            )
        ]

        require(
            len(required_targets) > 0,
            (
                "sample_filter is enabled but "
                "required_valid_targets is empty."
            ),
        )

        valid_samples = np.ones(
            (
                len(basin_ids),
                dataset.num_time_steps,
            ),
            dtype=bool,
        )

        for task_name in required_targets:
            require(
                task_name in dataset_target_views,
                (
                    f"Filtered target {task_name} "
                    "is not available."
                ),
            )
            valid_samples &= np.isfinite(
                dataset_target_views[task_name]
            )

        expected_sample_count = int(
            valid_samples.sum()
        )
    else:
        required_targets = []
        expected_sample_count = (
            full_sample_count
        )

    print("-" * 100)
    print("Sample filter       :", sample_filter)
    print("Required targets    :", required_targets)
    print(
        "Expected samples    :",
        expected_sample_count,
    )
    print("Actual samples      :", len(dataset))

    require(
        len(dataset) == expected_sample_count,
        (
            "Dataset sample count does not match "
            f"the configured filtering semantics: "
            f"expected={expected_sample_count}, "
            f"actual={len(dataset)}."
        ),
    )

    print("=" * 100)
    print("Runtime data semantic audit: PASS")


if __name__ == "__main__":
    main()
