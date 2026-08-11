#!/usr/bin/env python3
"""Inspect whether the frozen HydroMTL_CGC core can host the Chapter 4B overlay."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.native_runtime import bootstrap_native_runtime  # noqa: E402

bootstrap_native_runtime(strict=True)


def import_main():
    path = PROJECT_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("hydromtl_compat_main", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    errors: list[str] = []

    project_main = import_main()
    for name in ("build_spatial_split", "get_hydro_dataloaders", "validate_temporal_splits", "main"):
        if not hasattr(project_main, name):
            errors.append(f"main.py missing required symbol: {name}")

    from mtl_cgc.data import data_loaders, data_sets

    if not hasattr(data_loaders, "load_nc_to_dict"):
        errors.append("mtl_cgc.data.data_loaders.load_nc_to_dict is required.")

    loader_sig = inspect.signature(data_loaders.get_hydro_dataloaders)
    if "scaler_basin_ids" not in loader_sig.parameters:
        errors.append(
            "Core test loader must support scaler_basin_ids for source-only test scaling."
        )

    dataset_source = inspect.getsource(data_sets.HydroDataset)
    if "interpolate_missing" not in dataset_source:
        errors.append(
            "HydroDataset must support target-level interpolate_missing semantics from Chapter 4A."
        )

    split_source = inspect.getsource(data_loaders.assert_temporal_splits)
    if "spatial_split" not in split_source:
        errors.append(
            "Core DataLoader temporal guard must recognize spatial_split. "
            "The PUB wrapper overrides main.validate_temporal_splits, but the "
            "frozen DataLoader still needs its own spatial-split support."
        )

    if errors:
        print("PUB compatibility check: FAIL")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print("PUB compatibility check: PASS")
    print("The Chapter 4B overlay can run without replacing frozen core files.")


if __name__ == "__main__":
    main()
