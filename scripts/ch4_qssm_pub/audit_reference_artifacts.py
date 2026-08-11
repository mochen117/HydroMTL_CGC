#!/usr/bin/env python3
"""Audit frozen Chapter 3 and Chapter 4A artifacts required by Chapter 4B.

The audit is intentionally read-only.  It verifies that the completed Chapter 3
summary and Chapter 4A formal seed-42 result tables describe the same basin set
before Chapter 4B uses them for fold construction and cross-experiment analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


CH4A_REQUIRED = {
    "STL-SSM": "ch4a_formal_stl_ssm_seed42",
    "Hard-MTL": "ch4a_formal_hps_qssm_seed42",
    "CGC": "ch4a_formal_cgc_qssm_seed42",
    "Hard-pretrained": "ch4a_formal_hps_qpre_finetune_qssm_seed42",
    "CGC-pretrained": "ch4a_formal_cgc_qpre_finetune_qssm_seed42",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ch3-summary",
        type=Path,
        default=Path(
            "experiments/formal_ch3_modeling/06_summary/"
            "ch3_per_basin_with_metadata.csv"
        ),
    )
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--expected-basins",
        type=int,
        default=592,
        help="Expected formal basin count; set 0 to disable the count check.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(8)


def basin_set(frame: pd.DataFrame) -> set[str]:
    for candidate in ("gauge_id", "basin_id", "gage_id", "Unnamed: 0", frame.columns[0]):
        if candidate in frame.columns:
            return {normalize_id(value) for value in frame[candidate].tolist()}
    raise ValueError("Cannot identify basin-id column.")


def main() -> None:
    args = parse_args()
    ch3_path = resolve(args.ch3_summary)
    exp_root = resolve(args.experiments_root)
    errors: list[str] = []
    reference_basins: set[str] = set()

    base_config = PROJECT_ROOT / (
        "mtl_cgc/configs/ch4_qssm_formal/seed42/q_to_ssm/"
        "ch4a_formal_cgc_qssm_seed42.yaml"
    )
    data_root = PROJECT_ROOT / "output_592_basins"
    if not base_config.exists():
        errors.append(f"Missing frozen Chapter 4A base config: {base_config}")
    else:
        print(f"Chapter 4A base config: {base_config}")
    if not data_root.exists():
        errors.append(f"Missing prepared basin data root: {data_root}")
    else:
        nc_count = len(list(data_root.glob("gage_*.nc")))
        if args.expected_basins and nc_count != args.expected_basins:
            errors.append(
                f"Prepared NetCDF basin count is {nc_count}, expected {args.expected_basins}."
            )
        print(f"Prepared basin NetCDFs: {nc_count} -> {data_root}")

    if not ch3_path.exists():
        errors.append(f"Missing Chapter 3 summary: {ch3_path}")
    else:
        ch3 = pd.read_csv(ch3_path)
        required = {"gauge_id", "huc_02", "aridity", "frac_snow"}
        missing = required - set(ch3.columns)
        if missing:
            errors.append(f"Chapter 3 summary missing columns: {sorted(missing)}")
        reference_basins = basin_set(ch3)
        if len(reference_basins) != len(ch3):
            errors.append("Chapter 3 summary contains duplicate basin identifiers.")
        if args.expected_basins and len(reference_basins) != args.expected_basins:
            errors.append(
                f"Chapter 3 basin count is {len(reference_basins)}, "
                f"expected {args.expected_basins}."
            )
        print(
            f"Chapter 3 summary: rows={len(ch3)}, unique_basins={len(reference_basins)}, "
            f"columns={len(ch3.columns)}"
        )

    for label, experiment in CH4A_REQUIRED.items():
        path = exp_root / experiment / "test_per_basin_metrics.csv"
        if not path.exists():
            errors.append(f"Missing Chapter 4A {label}: {path}")
            continue
        frame = pd.read_csv(path)
        if "ssm_nse" not in frame.columns:
            errors.append(f"Chapter 4A {label} lacks ssm_nse: {path}")
        ids = basin_set(frame)
        if len(ids) != len(frame):
            errors.append(f"Chapter 4A {label} contains duplicate basin ids: {path}")
        if reference_basins and ids != reference_basins:
            missing_ids = sorted(reference_basins - ids)
            extra_ids = sorted(ids - reference_basins)
            errors.append(
                f"Chapter 4A {label} basin set differs from Chapter 3: "
                f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
            )
        print(
            f"Chapter 4A {label:<16s}: rows={len(frame):3d}, "
            f"unique_basins={len(ids):3d} -> {path}"
        )

    if errors:
        print("\nReference artifact audit: FAIL")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print("\nReference artifact audit: PASS")
    print("Frozen Chapter 3 / Chapter 4A artifacts are aligned for Chapter 4B.")


if __name__ == "__main__":
    main()
