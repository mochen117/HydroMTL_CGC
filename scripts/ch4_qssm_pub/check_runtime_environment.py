#!/usr/bin/env python3
"""Validate the native NetCDF runtime used by Chapter 4B PUB.

The check applies the same process-local conda ``libstdc++`` bootstrap used by
PUB training and data-audit entry points, then imports netCDF4 and xarray.  It
never modifies the conda environment on disk.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.native_runtime import (  # noqa: E402
    bootstrap_native_runtime,
    loaded_libstdcxx_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Inspect the raw process environment without the HydroMTL runtime bootstrap.",
    )
    return parser.parse_args()


def glibcxx_versions(lib_path: Path) -> list[tuple[int, ...]]:
    """Return numeric GLIBCXX versions visible in one libstdc++ binary."""

    if not lib_path.exists():
        return []
    try:
        result = subprocess.run(
            ["strings", str(lib_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []

    versions: set[tuple[int, ...]] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("GLIBCXX_"):
            continue
        suffix = line.removeprefix("GLIBCXX_")
        try:
            versions.add(tuple(int(part) for part in suffix.split(".")))
        except ValueError:
            continue
    return sorted(versions)


def format_version(version: tuple[int, ...] | None) -> str:
    if not version:
        return "<unknown>"
    return "GLIBCXX_" + ".".join(str(part) for part in version)


def main() -> None:
    args = parse_args()

    if args.no_bootstrap:
        os.environ["HYDRO_USE_PATCH"] = "0"
    report = bootstrap_native_runtime(strict=not args.no_bootstrap)

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_lib = Path(conda_prefix) / "lib" / "libstdc++.so.6" if conda_prefix else None
    versions = glibcxx_versions(conda_lib) if conda_lib is not None else []

    print("=" * 96)
    print("Chapter 4B native runtime check")
    print("-" * 96)
    print(f"Python                : {sys.executable}")
    print(f"CONDA_PREFIX          : {conda_prefix or '<unset>'}")
    print(f"Bootstrap enabled     : {report.enabled}")
    print(f"Preload attempted     : {report.preload_attempted}")
    print(f"Preload succeeded     : {report.preload_succeeded}")
    print(f"Conda libstdc++       : {report.libstdcxx_path or '<unset>'}")
    print(f"Mapped libstdc++      : {loaded_libstdcxx_path() or '<not detected>'}")
    print(f"Latest conda GLIBCXX  : {format_version(versions[-1] if versions else None)}")
    print(f"LD_LIBRARY_PATH       : {os.environ.get('LD_LIBRARY_PATH', '<unset>')}")
    print(f"Bootstrap message     : {report.message}")
    print("-" * 96)

    failures: list[str] = []

    try:
        import netCDF4
        print(f"netCDF4               : PASS ({netCDF4.__version__})")
    except Exception as exc:  # noqa: BLE001 - diagnostic entry point
        print(f"netCDF4               : FAIL ({exc})")
        failures.append(f"netCDF4: {exc}")

    try:
        import xarray as xr
        print(f"xarray                : PASS ({xr.__version__})")
    except Exception as exc:  # noqa: BLE001 - diagnostic entry point
        print(f"xarray                : FAIL ({exc})")
        failures.append(f"xarray: {exc}")

    if not failures:
        # Open one real CAMELS NetCDF file when available.  This catches backend
        # linkage errors that a bare Python import can miss.
        data_root = PROJECT_ROOT / "output_592_basins"
        candidates = sorted(data_root.glob("gage_*.nc")) if data_root.exists() else []
        if candidates:
            try:
                import xarray as xr
                with xr.open_dataset(candidates[0]) as ds:
                    _ = tuple(ds.dims)
                print(f"NetCDF open smoke     : PASS ({candidates[0].name})")
            except Exception as exc:  # noqa: BLE001
                print(f"NetCDF open smoke     : FAIL ({exc})")
                failures.append(f"open_dataset: {exc}")
        else:
            print("NetCDF open smoke     : SKIP (output_592_basins/gage_*.nc not found)")

    print("=" * 96)
    if failures:
        print("Native runtime check: FAIL")
        for failure in failures:
            print(" -", failure)
        if any("GLIBCXX_" in item for item in failures):
            print(
                "The process still selected an incompatible C++ runtime. "
                "Do not start PUB training until this check passes."
            )
        raise SystemExit(1)

    print("Native runtime check: PASS")


if __name__ == "__main__":
    main()
