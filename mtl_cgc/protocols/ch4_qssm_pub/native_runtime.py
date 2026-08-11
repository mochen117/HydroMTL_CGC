"""Native-library bootstrap for Chapter 4B PUB execution.

The HydroMTL_CGC server environment may expose an older system ``libstdc++``
before the active conda environment's runtime.  Packages such as ``netCDF4``
can then fail with a ``GLIBCXX_* not found`` error when they load ICU/HDF5
shared libraries.

This module provides a small, process-local bootstrap that must run before
importing compiled third-party packages such as torch, xarray, or netCDF4.
It does not modify the conda environment on disk and does not replace any
frozen Chapter 3 / Chapter 4A core file.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


_BOOTSTRAPPED = False
_LAST_REPORT: "NativeRuntimeReport | None" = None


@dataclass(frozen=True)
class NativeRuntimeReport:
    """Result of one process-local native-runtime bootstrap attempt."""

    enabled: bool
    conda_prefix: str
    conda_lib_dir: str
    libstdcxx_path: str
    ld_library_path: str
    preload_attempted: bool
    preload_succeeded: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prepend_path(value: str, current: str) -> str:
    """Prepend ``value`` to a colon-separated environment path once."""

    parts = [item for item in current.split(":") if item]
    if value in parts:
        parts.remove(value)
    return ":".join([value, *parts])


def _loaded_libstdcxx_path() -> str | None:
    """Return the currently mapped libstdc++ path on Linux when available."""

    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return None
    try:
        for line in maps_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "libstdc++.so.6" in line:
                candidate = line.split()[-1]
                if candidate.startswith("/"):
                    return candidate
    except OSError:
        return None
    return None


def bootstrap_native_runtime(*, strict: bool = False) -> NativeRuntimeReport:
    """Prefer the active conda C++ runtime before compiled package imports.

    The behavior mirrors the runtime patch already used by the frozen
    ``main.py``: ``$CONDA_PREFIX/lib`` is placed first in ``LD_LIBRARY_PATH``
    and the conda ``libstdc++.so.6`` is loaded with ``RTLD_GLOBAL``.

    Parameters
    ----------
    strict:
        If True, raise when an active conda environment contains a
        ``libstdc++.so.6`` that cannot be preloaded.  Missing ``CONDA_PREFIX``
        remains a no-op because a system Python installation may be valid.
    """

    global _BOOTSTRAPPED, _LAST_REPORT

    if _BOOTSTRAPPED and _LAST_REPORT is not None:
        return _LAST_REPORT

    enabled = os.environ.get("HYDRO_USE_PATCH", "1") == "1"
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    conda_lib_dir = str(Path(conda_prefix) / "lib") if conda_prefix else ""
    libstdcxx_path = (
        str(Path(conda_lib_dir) / "libstdc++.so.6") if conda_lib_dir else ""
    )

    if not enabled:
        report = NativeRuntimeReport(
            enabled=False,
            conda_prefix=conda_prefix,
            conda_lib_dir=conda_lib_dir,
            libstdcxx_path=libstdcxx_path,
            ld_library_path=os.environ.get("LD_LIBRARY_PATH", ""),
            preload_attempted=False,
            preload_succeeded=False,
            message="Native runtime bootstrap disabled by HYDRO_USE_PATCH=0.",
        )
        _BOOTSTRAPPED = True
        _LAST_REPORT = report
        return report

    if not conda_prefix:
        report = NativeRuntimeReport(
            enabled=True,
            conda_prefix="",
            conda_lib_dir="",
            libstdcxx_path="",
            ld_library_path=os.environ.get("LD_LIBRARY_PATH", ""),
            preload_attempted=False,
            preload_succeeded=False,
            message="CONDA_PREFIX is unset; native runtime bootstrap is a no-op.",
        )
        _BOOTSTRAPPED = True
        _LAST_REPORT = report
        return report

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = _prepend_path(conda_lib_dir, old_ld)
    os.environ["LD_LIBRARY_PATH"] = new_ld

    lib_path = Path(libstdcxx_path)
    if not lib_path.exists():
        message = f"Conda libstdc++ not found: {lib_path}"
        if strict:
            raise RuntimeError(message)
        report = NativeRuntimeReport(
            enabled=True,
            conda_prefix=conda_prefix,
            conda_lib_dir=conda_lib_dir,
            libstdcxx_path=libstdcxx_path,
            ld_library_path=new_ld,
            preload_attempted=False,
            preload_succeeded=False,
            message=message,
        )
        _BOOTSTRAPPED = True
        _LAST_REPORT = report
        return report

    try:
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        ctypes.CDLL(str(lib_path), mode=mode)
    except OSError as exc:
        message = f"Failed to preload conda libstdc++: {lib_path}: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        report = NativeRuntimeReport(
            enabled=True,
            conda_prefix=conda_prefix,
            conda_lib_dir=conda_lib_dir,
            libstdcxx_path=libstdcxx_path,
            ld_library_path=new_ld,
            preload_attempted=True,
            preload_succeeded=False,
            message=message,
        )
        _BOOTSTRAPPED = True
        _LAST_REPORT = report
        return report

    mapped = _loaded_libstdcxx_path()
    message = "Conda libstdc++ preloaded with RTLD_GLOBAL."
    if mapped:
        message += f" mapped={mapped}"

    report = NativeRuntimeReport(
        enabled=True,
        conda_prefix=conda_prefix,
        conda_lib_dir=conda_lib_dir,
        libstdcxx_path=libstdcxx_path,
        ld_library_path=new_ld,
        preload_attempted=True,
        preload_succeeded=True,
        message=message,
    )
    _BOOTSTRAPPED = True
    _LAST_REPORT = report
    return report


def loaded_libstdcxx_path() -> str | None:
    """Expose the mapped libstdc++ path for runtime diagnostics."""

    return _loaded_libstdcxx_path()
