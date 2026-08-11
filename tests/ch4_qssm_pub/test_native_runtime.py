"""Regression tests for the Chapter 4B native-runtime bootstrap."""

from __future__ import annotations

import importlib
from pathlib import Path

from mtl_cgc.protocols.ch4_qssm_pub import native_runtime


def test_prepend_path_is_deduplicated() -> None:
    assert native_runtime._prepend_path("/conda/lib", "/usr/lib:/conda/lib:/opt/lib") == (
        "/conda/lib:/usr/lib:/opt/lib"
    )


def test_bootstrap_can_be_disabled_without_native_load(monkeypatch) -> None:
    monkeypatch.setenv("HYDRO_USE_PATCH", "0")
    module = importlib.reload(native_runtime)
    report = module.bootstrap_native_runtime(strict=True)
    assert report.enabled is False
    assert report.preload_attempted is False


def test_runtime_bootstrap_precedes_compiled_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    checks = {
        root / "scripts/ch4_qssm_pub/pub_main.py": [
            "bootstrap_native_runtime(strict=True)",
            "import yaml",
            "from mtl_cgc.protocols.ch4_qssm_pub.data_adapter",
        ],
        root / "scripts/ch4_qssm_pub/audit_pub_data_semantics.py": [
            "bootstrap_native_runtime(strict=True)",
            "import numpy as np",
            "from mtl_cgc.data.data_loaders import load_nc_to_dict",
        ],
        root / "scripts/ch4_qssm_pub/audit_pub_outputs.py": [
            "bootstrap_native_runtime(strict=True)",
            "import xarray as xr",
        ],
        root / "scripts/ch4_qssm_pub/ensemble_pub_predictions.py": [
            "bootstrap_native_runtime(strict=True)",
            "import xarray as xr",
        ],
    }

    for path, markers in checks.items():
        source = path.read_text(encoding="utf-8")
        positions = [source.index(marker) for marker in markers]
        assert positions == sorted(positions), f"Import order regression in {path}"
