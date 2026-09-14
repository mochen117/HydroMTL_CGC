#!/usr/bin/env python3
"""
Chapter 3 independent-test launcher.

Protocol
--------
- Models:
    STL-Q, STL-ET, Hard-MTL, MMoE, CGC
- Test period:
    2016-10-01 to 2021-09-30
- Checkpoint:
    best_model.pth ONLY
- Existing test results:
    validated and skipped by default
- --force:
    rerun independent testing

This script never trains a model and never falls back to final_model.pth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ch3.run_ch3_models import (  # noqa: E402
    BASE_CONFIG_PATH,
    MODEL_RUNS,
    apply_common_config,
    load_yaml,
    save_yaml,
)

RESULT_ROOT = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
CONFIG_DIR = RESULT_ROOT / "test_configs"

EXPECTED_TEST_PERIOD = ["2016-10-01", "2021-09-30"]

NAME_TO_KEY = {
    "ch3_stl_q_seed42": "stl_q",
    "ch3_stl_et_seed42": "stl_et",
    "ch3_hard_mtl_seed42": "hard_mtl",
    "ch3_mmoe_mtl_seed42": "mmoe",
    "ch3_cgc_mtl_seed42": "cgc",
}

MODEL_ORDER = [
    "stl_q",
    "stl_et",
    "hard_mtl",
    "mmoe",
    "cgc",
]

METRIC_SUFFIXES = (
    "_nse",
    "_kge",
    "_rmse",
    "_mae",
    "_bias",
    "_corr",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_save_dir(cfg: Dict[str, Any]) -> Path:
    save_dir = Path(cfg["experiment"]["save_dir"])
    experiment_name = cfg["experiment"]["name"]

    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir

    # Formal Chapter 3 outputs are stored under:
    # <save_dir>/<experiment_name>/
    if save_dir.name != experiment_name:
        save_dir = save_dir / experiment_name

    return save_dir.resolve()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_test_summary(metrics_path: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_path)

    row: Dict[str, Any] = {
        "n_basins": len(df),
    }

    for col in df.columns:
        if not col.endswith(METRIC_SUFFIXES):
            continue

        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            continue

        row[f"{col}_n"] = int(len(x))
        row[f"{col}_median"] = float(x.median())
        row[f"{col}_mean"] = float(x.mean())
        row[f"{col}_q25"] = float(x.quantile(0.25))
        row[f"{col}_q75"] = float(x.quantile(0.75))
        row[f"{col}_min"] = float(x.min())
        row[f"{col}_max"] = float(x.max())

    pd.DataFrame([row]).to_csv(output_path, index=False)


def validate_result(
    save_dir: Path,
    targets: List[str],
    expected_basins: int,
) -> None:
    metrics_path = save_dir / "test_per_basin_metrics.csv"
    pred_path = save_dir / "test_predictions_and_weights.nc"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing independent-test metrics: {metrics_path}"
        )

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing independent-test predictions: {pred_path}"
        )

    df = pd.read_csv(metrics_path)

    if len(df) != expected_basins:
        raise RuntimeError(
            f"{save_dir.name}: expected {expected_basins} basins, "
            f"found {len(df)}."
        )

    for target in targets:
        col = f"{target}_nse"

        if col not in df.columns:
            raise RuntimeError(
                f"{save_dir.name}: required metric column missing: {col}"
            )

        n_valid = pd.to_numeric(
            df[col], errors="coerce"
        ).notna().sum()

        if n_valid != expected_basins:
            raise RuntimeError(
                f"{save_dir.name}: {col} has "
                f"{n_valid}/{expected_basins} valid values."
            )

    write_test_summary(
        metrics_path,
        save_dir / "test_summary.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/audit Chapter 3 independent tests."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all"] + MODEL_ORDER,
        help="Models to test. Default: all.",
    )

    parser.add_argument(
        "--expected-basins",
        type=int,
        default=592,
        help="Expected number of CAMELS-US basins.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun test even when test outputs already exist.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands and protocol checks.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected = (
        MODEL_ORDER
        if "all" in args.models
        else args.models
    )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(BASE_CONFIG_PATH)

    runs_by_key: Dict[str, Dict[str, Any]] = {}

    for run in MODEL_RUNS:
        name = run["name"]

        if name not in NAME_TO_KEY:
            continue

        runs_by_key[NAME_TO_KEY[name]] = run

    missing_keys = [
        key for key in MODEL_ORDER
        if key not in runs_by_key
    ]

    if missing_keys:
        raise RuntimeError(
            f"MODEL_RUNS missing expected models: {missing_keys}"
        )

    manifest: Dict[str, Any] = {
        "protocol": "chapter3_independent_test",
        "git_commit": git_commit(),
        "test_period": EXPECTED_TEST_PERIOD,
        "expected_basins": args.expected_basins,
        "checkpoint_policy": "best_model.pth_only",
        "models": [],
    }

    print("=" * 88)
    print("CHAPTER 3 INDEPENDENT TEST")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Git commit   : {manifest['git_commit']}")
    print(
        "Test period  : "
        f"{EXPECTED_TEST_PERIOD[0]} -> "
        f"{EXPECTED_TEST_PERIOD[1]}"
    )
    print("Checkpoint   : best_model.pth ONLY")
    print()

    for key in selected:
        run = runs_by_key[key]
        cfg = apply_common_config(base_cfg, run)

        actual_test_period = list(cfg["data"]["test_period"])

        if actual_test_period != EXPECTED_TEST_PERIOD:
            raise RuntimeError(
                f"{run['name']}: unexpected test period "
                f"{actual_test_period}; expected "
                f"{EXPECTED_TEST_PERIOD}"
            )

        save_dir = resolve_save_dir(cfg)
        checkpoint = save_dir / "best_model.pth"

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{run['name']}: required checkpoint missing:\n"
                f"  {checkpoint}\n"
                "Independent paper evaluation does not fall back "
                "to final_model.pth."
            )

        checkpoint_hash_before = sha256_file(checkpoint)

        config_path = (
            CONFIG_DIR /
            f"{run['name']}_test.yaml"
        )
        save_yaml(cfg, config_path)

        metrics_path = (
            save_dir /
            "test_per_basin_metrics.csv"
        )

        print("-" * 88)
        print(f"Model          : {key}")
        print(f"Experiment     : {run['name']}")
        print(f"Architecture   : {run['architecture']}")
        print(f"Targets        : {run['targets']}")
        print(
            "Primary metric : "
            f"{cfg['evaluation_protocol']['primary_metric']}"
        )
        print(f"Checkpoint     : {checkpoint}")
        print(
            "SHA256         : "
            f"{checkpoint_hash_before[:16]}..."
        )

        status = "pending"

        if args.dry_run:
            print(
                "DRY RUN        : "
                f"{sys.executable} main.py "
                f"--config {config_path} --mode test"
            )
            status = "dry_run"

        elif metrics_path.exists() and not args.force:
            print(
                "Status         : existing test result found; "
                "audit only (no rerun)"
            )

            validate_result(
                save_dir,
                run["targets"],
                args.expected_basins,
            )

            status = "validated_existing"

        else:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "--config",
                str(config_path),
                "--mode",
                "test",
            ]

            print(
                "Status         : running independent test"
            )

            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
            )

            validate_result(
                save_dir,
                run["targets"],
                args.expected_basins,
            )

            status = "completed"

        checkpoint_hash_after = sha256_file(checkpoint)

        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError(
                f"{run['name']}: checkpoint SHA256 changed "
                "during evaluation."
            )

        if not args.dry_run:
            print("Result check   : PASS")
            print(
                "test_summary   : "
                f"{save_dir / 'test_summary.csv'}"
            )

        manifest["models"].append(
            {
                "model_key": key,
                "experiment_name": run["name"],
                "architecture": run["architecture"],
                "targets": run["targets"],
                "primary_metric": (
                    cfg["evaluation_protocol"][
                        "primary_metric"
                    ]
                ),
                "config": relative_to_project(
                    config_path
                ),
                "save_dir": relative_to_project(
                    save_dir
                ),
                "checkpoint": relative_to_project(
                    checkpoint
                ),
                "checkpoint_sha256": (
                    checkpoint_hash_before
                ),
                "test_metrics": relative_to_project(
                    save_dir /
                    "test_per_basin_metrics.csv"
                ),
                "test_predictions": relative_to_project(
                    save_dir /
                    "test_predictions_and_weights.nc"
                ),
                "status": status,
            }
        )

    manifest_path = (
        RESULT_ROOT /
        "ch3_independent_test_manifest.json"
    )

    if not args.dry_run:
        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("\n" + "=" * 88)
        print("INDEPENDENT TEST AUDIT COMPLETE")
        print("=" * 88)
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
