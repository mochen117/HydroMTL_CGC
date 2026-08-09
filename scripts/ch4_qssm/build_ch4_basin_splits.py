#!/usr/bin/env python3
"""Build basin lists for Chapter 4 Q-SSM protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ch4_common import read_basin_file, write_basin_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all-basin, smoke, and PUB fold basin lists.")
    parser.add_argument("--eligible-basins", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("experiments/ch4_qssm/basin_splits"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--k-folds", default=5, type=int)
    parser.add_argument("--smoke-n", default=20, type=int)
    parser.add_argument("--valid-fraction", default=0.1, type=float, help="Validation fraction within training basins for PUB folds.")
    args = parser.parse_args()

    basins = np.array(read_basin_file(args.eligible_basins), dtype=str)
    if basins.size < 10:
        raise ValueError("Too few eligible basins for Chapter 4 experiments.")
    rng = np.random.default_rng(args.seed)
    shuffled = basins.copy()
    rng.shuffle(shuffled)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_basin_file(shuffled, args.out_dir / "all_eligible_basins.txt")
    write_basin_file(shuffled[: min(args.smoke_n, len(shuffled))], args.out_dir / "smoke_basins.txt")

    folds = np.array_split(shuffled, args.k_folds)
    meta = {
        "seed": args.seed,
        "k_folds": args.k_folds,
        "n_basins": int(len(shuffled)),
        "valid_fraction": args.valid_fraction,
        "folds": [],
    }
    for k, test in enumerate(folds):
        train_valid = np.setdiff1d(shuffled, test, assume_unique=False)
        rng_fold = np.random.default_rng(args.seed + k + 1000)
        train_valid = train_valid.copy()
        rng_fold.shuffle(train_valid)
        n_valid = max(1, int(round(len(train_valid) * args.valid_fraction)))
        valid = train_valid[:n_valid]
        train = train_valid[n_valid:]
        fold_dir = args.out_dir / f"fold_{k:02d}"
        write_basin_file(train, fold_dir / "train_basins.txt")
        write_basin_file(valid, fold_dir / "valid_basins.txt")
        write_basin_file(test, fold_dir / "test_basins.txt")
        meta["folds"].append({
            "fold": k,
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
            "n_test": int(len(test)),
            "train_file": str(fold_dir / "train_basins.txt"),
            "valid_file": str(fold_dir / "valid_basins.txt"),
            "test_file": str(fold_dir / "test_basins.txt"),
        })
    write_json(meta, args.out_dir / "split_metadata.json")
    print(f"Wrote basin splits to {args.out_dir}")
    print(f"All eligible basins: {len(shuffled)}")
    print(f"Smoke basins: {min(args.smoke_n, len(shuffled))}")
    print(f"PUB folds: {args.k_folds}")


if __name__ == "__main__":
    main()
