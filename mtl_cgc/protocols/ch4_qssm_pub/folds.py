"""Deterministic basin-fold construction for PUB experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from .io_utils import normalize_basin_id


def build_balanced_folds(
    basin_ids: Iterable[str],
    n_folds: int,
    split_seed: int,
    region_by_basin: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assign each basin to exactly one target fold.

    When region labels are available, basins are allocated by region while
    balancing both regional and total fold counts. The algorithm is fully
    deterministic for a fixed seed.
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    normalized = [normalize_basin_id(item) for item in basin_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Input basin identifiers contain duplicates.")

    rng = np.random.default_rng(split_seed)
    region_lookup = region_by_basin or {}
    groups: dict[str, list[str]] = defaultdict(list)

    for basin_id in normalized:
        region = str(region_lookup.get(basin_id, "UNKNOWN"))
        groups[region].append(basin_id)

    total_counts = [0] * n_folds
    regional_counts: dict[str, list[int]] = {
        region: [0] * n_folds for region in groups
    }
    assignments: dict[str, int] = {}

    for region, members in sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        members = sorted(members)
        rng.shuffle(members)

        for basin_id in members:
            candidates = list(range(n_folds))
            rng.shuffle(candidates)
            chosen = min(
                candidates,
                key=lambda fold: (
                    regional_counts[region][fold],
                    total_counts[fold],
                    fold,
                ),
            )
            assignments[basin_id] = chosen + 1
            regional_counts[region][chosen] += 1
            total_counts[chosen] += 1

    rows = [
        {
            "basin_id": basin_id,
            "target_fold": assignments[basin_id],
            "region": str(region_lookup.get(basin_id, "UNKNOWN")),
            "split_seed": split_seed,
        }
        for basin_id in sorted(normalized)
    ]
    return pd.DataFrame(rows)


def validate_fold_assignment(
    assignment: pd.DataFrame,
    expected_basin_ids: Iterable[str],
    n_folds: int,
) -> None:
    """Validate coverage, uniqueness, and fold labels."""

    required = {"basin_id", "target_fold"}
    missing_columns = required - set(assignment.columns)
    if missing_columns:
        raise ValueError(f"Missing assignment columns: {sorted(missing_columns)}")

    expected = {normalize_basin_id(item) for item in expected_basin_ids}
    observed = {
        normalize_basin_id(item) for item in assignment["basin_id"].tolist()
    }

    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Fold coverage mismatch. Missing={missing[:10]}, extra={extra[:10]}"
        )

    if assignment["basin_id"].astype(str).duplicated().any():
        raise ValueError("A basin appears more than once in fold assignments.")

    invalid_folds = sorted(
        set(assignment["target_fold"].astype(int)) - set(range(1, n_folds + 1))
    )
    if invalid_folds:
        raise ValueError(f"Invalid fold labels: {invalid_folds}")
