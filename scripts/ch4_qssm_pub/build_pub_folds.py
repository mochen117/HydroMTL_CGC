#!/usr/bin/env python3
"""Build deterministic HUC2-balanced folds for Chapter 4B PUB.

The default metadata source is the frozen Chapter 3 per-basin summary so that
Chapter 4B uses the same basin identifiers and HUC2 labels as the earlier
analysis.  Eligible basins can be supplied explicitly or discovered from the
prepared ``output_592_basins/gage_*.nc`` files.  Every eligible basin becomes a
target basin in exactly one fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.constants import ProtocolDefaults  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.folds import (  # noqa: E402
    build_balanced_folds,
    validate_fold_assignment,
)
from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    atomic_write_json,
    normalize_basin_id,
    read_basin_ids,
    write_basin_ids,
)
from mtl_cgc.protocols.ch4_qssm_pub.paths import (  # noqa: E402
    CH3_SUMMARY,
    PROTOCOL_DIR,
)


DEFAULT_DATA_ROOT = Path("output_592_basins")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eligible-basins",
        type=Path,
        default=None,
        help=(
            "Optional text file containing eligible basin ids. If omitted, "
            "basins are discovered from --data-root/gage_*.nc."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Prepared per-basin NetCDF directory used when --eligible-basins is omitted.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=CH3_SUMMARY,
        help="Frozen Chapter 3 per-basin summary used for HUC2 balancing.",
    )
    parser.add_argument(
        "--basin-id-column",
        default="auto",
        help="Metadata basin-id column. 'auto' tries gauge_id, basin_id, gage_id, site_no.",
    )
    parser.add_argument("--region-column", default="huc_02")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260701)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROTOCOL_DIR,
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_basin_column(frame: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in frame.columns:
            raise ValueError(f"Metadata is missing basin-id column: {requested}")
        return requested

    for candidate in ("gauge_id", "basin_id", "gage_id", "site_no"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Could not auto-detect a basin-id column. Tried gauge_id, basin_id, "
        "gage_id, and site_no."
    )


def discover_basin_ids(data_root: Path) -> list[str]:
    files = sorted(data_root.glob("gage_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No gage_*.nc files were found under data root: {data_root}"
        )
    basin_ids = [normalize_basin_id(path.stem.replace("gage_", "", 1)) for path in files]
    if len(set(basin_ids)) != len(basin_ids):
        raise ValueError("Duplicate basin ids were discovered from NetCDF filenames.")
    return basin_ids


def load_region_lookup(
    metadata_csv: Path,
    basin_id_column: str,
    region_column: str,
) -> tuple[dict[str, str], str]:
    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)

    metadata = pd.read_csv(metadata_csv)
    basin_col = resolve_basin_column(metadata, basin_id_column)
    if region_column not in metadata.columns:
        raise ValueError(f"Metadata is missing region column: {region_column}")

    subset = metadata[[basin_col, region_column]].copy()
    subset[basin_col] = subset[basin_col].map(normalize_basin_id)
    subset[region_column] = subset[region_column].astype(str)
    subset = subset.drop_duplicates(subset=[basin_col])
    return dict(zip(subset[basin_col], subset[region_column])), basin_col


def main() -> None:
    args = parse_args()
    defaults = ProtocolDefaults()
    metadata_csv = resolve(args.metadata_csv)
    data_root = resolve(args.data_root)
    output_dir = resolve(args.output_dir)

    if args.eligible_basins is None:
        basin_ids = discover_basin_ids(data_root)
        eligible_source = f"discovered from {data_root}/gage_*.nc"
    else:
        eligible_path = resolve(args.eligible_basins)
        basin_ids = read_basin_ids(eligible_path)
        eligible_source = str(eligible_path)

    region_lookup, basin_col = load_region_lookup(
        metadata_csv=metadata_csv,
        basin_id_column=args.basin_id_column,
        region_column=args.region_column,
    )

    missing_metadata = sorted(set(basin_ids) - set(region_lookup))
    if missing_metadata:
        raise ValueError(
            f"{len(missing_metadata)} eligible basins are absent from Chapter 3 "
            f"metadata. Examples: {missing_metadata[:10]}"
        )

    assignment = build_balanced_folds(
        basin_ids=basin_ids,
        n_folds=args.n_folds,
        split_seed=args.split_seed,
        region_by_basin=region_lookup,
    )
    validate_fold_assignment(
        assignment=assignment,
        expected_basin_ids=basin_ids,
        n_folds=args.n_folds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_path = output_dir / "eligible_basins.txt"
    write_basin_ids(eligible_path, sorted(basin_ids))

    assignment_path = output_dir / "pub_fold_assignments.csv"
    assignment.to_csv(assignment_path, index=False)

    all_basins = set(basin_ids)
    fold_records: list[dict[str, object]] = []
    for fold_id in range(1, args.n_folds + 1):
        fold_dir = output_dir / f"fold{fold_id:02d}"
        target = sorted(
            assignment.loc[
                assignment["target_fold"].astype(int) == fold_id,
                "basin_id",
            ].astype(str)
        )
        source = sorted(all_basins - set(target))

        source_path = fold_dir / "source_basins.txt"
        target_path = fold_dir / "target_basins.txt"
        write_basin_ids(source_path, source)
        write_basin_ids(target_path, target)

        fold_records.append(
            {
                "fold_id": fold_id,
                "source_basin_file": str(source_path),
                "target_basin_file": str(target_path),
                "source_basin_count": len(source),
                "target_basin_count": len(target),
            }
        )

    manifest = {
        "protocol_version": defaults.protocol_version,
        "n_folds": args.n_folds,
        "split_seed": args.split_seed,
        "eligible_basin_source": eligible_source,
        "eligible_basin_file": str(eligible_path),
        "eligible_basin_count": len(basin_ids),
        "chapter3_metadata_csv": str(metadata_csv),
        "chapter3_basin_id_column": basin_col,
        "region_column": args.region_column,
        "assignment_csv": str(assignment_path),
        "folds": fold_records,
    }
    atomic_write_json(output_dir / "pub_fold_manifest.json", manifest)

    counts = assignment.groupby("target_fold").size().to_dict()
    print(json.dumps({"target_fold_counts": counts}, indent=2))
    print(f"Eligible basins: {len(basin_ids)}")
    print(f"Eligible source: {eligible_source}")
    print(f"Chapter 3 metadata: {metadata_csv}")
    print(f"Fold artifacts exported to: {output_dir}")


if __name__ == "__main__":
    main()
