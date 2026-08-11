#!/usr/bin/env python3
"""Audit basin-fold artifacts for the Chapter 4 PUB experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.folds import validate_fold_assignment  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    load_json,
    read_basin_ids,
)
from mtl_cgc.protocols.ch4_qssm_pub.paths import FOLD_MANIFEST  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=FOLD_MANIFEST,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = (
        args.fold_manifest
        if args.fold_manifest.is_absolute()
        else PROJECT_ROOT / args.fold_manifest
    )
    manifest = load_json(manifest_path)
    eligible = read_basin_ids(Path(manifest["eligible_basin_file"]))
    assignment = pd.read_csv(manifest["assignment_csv"])
    n_folds = int(manifest["n_folds"])

    validate_fold_assignment(assignment, eligible, n_folds)

    eligible_set = set(eligible)
    target_union: set[str] = set()
    target_seen: set[str] = set()

    for fold in manifest["folds"]:
        fold_id = int(fold["fold_id"])
        source = set(read_basin_ids(Path(fold["source_basin_file"])))
        target = set(read_basin_ids(Path(fold["target_basin_file"])))

        if source & target:
            raise RuntimeError(f"Fold {fold_id}: source/target overlap detected.")
        if source | target != eligible_set:
            raise RuntimeError(f"Fold {fold_id}: source/target union is incomplete.")
        if target_seen & target:
            raise RuntimeError(
                f"Fold {fold_id}: target basins overlap a previous fold."
            )

        target_seen |= target
        target_union |= target
        print(
            f"Fold {fold_id:02d}: source={len(source):3d}, "
            f"target={len(target):3d}, overlap=0"
        )

    if target_union != eligible_set:
        raise RuntimeError("The union of target folds does not cover all basins.")

    print("PUB fold audit passed.")


if __name__ == "__main__":
    main()
