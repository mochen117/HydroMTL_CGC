#!/usr/bin/env python3
"""Protocol checks for Chapter 4 Q-SSM configs and outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ch4_common import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Check generated Chapter 4 configs for common leakage risks.")
    parser.add_argument("--config-dir", default=Path("mtl_cgc/configs/ch4_qssm"), type=Path)
    args = parser.parse_args()
    issues = []
    for cfg_path in sorted(args.config_dir.rglob("*.yaml")):
        cfg = load_yaml(cfg_path)
        meta = cfg.get("ch4_qssm", {}) if isinstance(cfg.get("ch4_qssm", {}), dict) else {}
        name = meta.get("experiment_name", cfg_path.stem)
        tasks = meta.get("tasks", [])
        if "ssm" in tasks:
            # Flags that should not be true.
            for path in [("data", "interpolate_missing"), ("data", "target_interpolate"), ("data", "interpolate_targets"), ("data_params", "target_rm_nan")]:
                cur = cfg
                ok = True
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        ok = False; break
                    cur = cur[key]
                if ok and cur is True:
                    issues.append((str(cfg_path), f"{'.'.join(path)} is True; SSM/PUB labels should not be interpolated."))
        if meta.get("protocol") == "pub" and not meta.get("test_basin_file"):
            issues.append((str(cfg_path), "PUB config has no test_basin_file in ch4_qssm metadata."))
        print(f"OK checked: {cfg_path} ({name})")
    if issues:
        print("\nIssues:")
        for path, msg in issues:
            print(f"- {path}: {msg}")
        raise SystemExit(1)
    print("\nNo obvious protocol issues detected.")


if __name__ == "__main__":
    main()
