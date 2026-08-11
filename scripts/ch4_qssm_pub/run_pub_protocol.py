#!/usr/bin/env python3
"""Sequential, resumable runner for Chapter 4 PUB experiments."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.checkpoints import inspect_experiment  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    atomic_write_json,
    load_json,
    load_yaml,
    resolve_project_path,
)
from mtl_cgc.protocols.ch4_qssm_pub.paths import MANIFEST_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument(
        "--subset",
        choices=["core", "ablation", "all"],
        default="core",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--quiet-batches", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(lock_path: Path, manifest_path: Path) -> None:
    """Create a manifest lock and reject a concurrent active runner."""

    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = int(existing.get("pid", -1))
        except Exception:
            existing_pid = -1

        if existing_pid > 0 and pid_is_alive(existing_pid):
            raise RuntimeError(
                "Another active Chapter 4B PUB runner is using this manifest: "
                f"pid={existing_pid}, lock={lock_path}"
            )
        lock_path.unlink()

    atomic_write_json(
        lock_path,
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": utc_now(),
            "manifest": str(manifest_path),
        },
    )


def select_entries(
    entries: list[dict[str, Any]],
    subset: str,
    folds: list[int] | None,
) -> list[dict[str, Any]]:
    selected = []
    for entry in entries:
        if subset != "all" and entry.get("group") != subset:
            continue
        if folds is not None and int(entry["fold_id"]) not in set(folds):
            continue
        selected.append(entry)
    return selected


def test_is_complete(experiment_dir: Path) -> bool:
    return (
        (experiment_dir / "test_per_basin_metrics.csv").exists()
        and (experiment_dir / "test_predictions_and_weights.nc").exists()
    )


def build_command(
    args: argparse.Namespace,
    config_path: Path,
    resume_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/ch4_qssm_pub/pub_main.py"),
        "--config",
        str(config_path),
        "--mode",
        args.mode,
        "--device",
        args.device,
    ]
    if args.quiet_batches:
        command.append("--quiet_batches")
    if resume_checkpoint is not None:
        command.extend(["--resume_checkpoint", str(resume_checkpoint)])
    return command


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest, PROJECT_ROOT)
    manifest = load_json(manifest_path)
    entries = select_entries(
        list(manifest.get("entries", [])),
        subset=args.subset,
        folds=args.folds,
    )
    if not entries:
        raise ValueError("No PUB configurations match the requested selection.")

    run_dir = PROJECT_ROOT / MANIFEST_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / f"{manifest_path.stem}_{args.mode}_{args.subset}.lock"
    run_manifest_path = (
        run_dir / f"{manifest_path.stem}_run_{args.mode}_{args.subset}.json"
    )

    if not args.dry_run:
        acquire_lock(lock_path, manifest_path)

    records: list[dict[str, Any]] = []
    exit_code = 0

    try:
        for index, entry in enumerate(entries, start=1):
            config_path = resolve_project_path(entry["config"], PROJECT_ROOT)
            config = load_yaml(config_path)
            experiment_name = str(config["experiment"]["name"])
            target_epoch = int(config["training"]["epochs"])
            save_root = Path(config["experiment"].get("save_dir", "experiments"))
            if not save_root.is_absolute():
                save_root = PROJECT_ROOT / save_root
            experiment_dir = save_root / experiment_name

            resume_checkpoint: Path | None = None
            action = "run"
            status_text = "pending"

            if args.mode == "train":
                checkpoint_status = inspect_experiment(
                    experiment_dir=experiment_dir,
                    target_epoch=target_epoch,
                )
                status_text = checkpoint_status.state

                if checkpoint_status.state == "invalid":
                    raise RuntimeError(
                        f"Invalid checkpoint for {experiment_name}: "
                        f"{checkpoint_status.message}"
                    )
                if (
                    checkpoint_status.state == "completed"
                    and args.skip_completed
                ):
                    action = "skipped_completed"
                elif (
                    checkpoint_status.state == "partial"
                    and args.resume_partial
                ):
                    resume_checkpoint = checkpoint_status.checkpoint_path
                    action = "resume"
            else:
                if test_is_complete(experiment_dir) and args.skip_completed:
                    action = "skipped_completed"
                    status_text = "completed"

            command = build_command(args, config_path, resume_checkpoint)

            print("\n" + "=" * 112)
            print(f"Chapter 4B PUB run {index}/{len(entries)}")
            print("-" * 112)
            print(f"Fold       : {int(entry['fold_id']):02d}")
            print(f"Scenario   : {entry['scenario']}")
            print(f"Config     : {config_path.name}")
            print(f"Experiment : {experiment_name}")
            print(f"Status     : {status_text}")
            print(f"Action     : {action}")
            print("Command    : " + " ".join(command))
            print("=" * 112)

            record = {
                "fold_id": int(entry["fold_id"]),
                "scenario": entry["scenario"],
                "group": entry["group"],
                "config": str(config_path),
                "experiment_name": experiment_name,
                "mode": args.mode,
                "status_before": status_text,
                "action": action,
                "resume_checkpoint": (
                    str(resume_checkpoint) if resume_checkpoint else None
                ),
                "command": command,
                "started_at": utc_now(),
                "return_code": None,
            }

            if action == "skipped_completed" or args.dry_run:
                record["return_code"] = 0
                record["finished_at"] = utc_now()
                records.append(record)
                atomic_write_json(run_manifest_path, {"records": records})
                continue

            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            record["return_code"] = int(result.returncode)
            record["finished_at"] = utc_now()
            records.append(record)
            atomic_write_json(run_manifest_path, {"records": records})

            if result.returncode != 0:
                exit_code = int(result.returncode)
                raise RuntimeError(
                    f"PUB command failed with return code {result.returncode}: "
                    + " ".join(command)
                )
    finally:
        atomic_write_json(
            run_manifest_path,
            {
                "manifest": str(manifest_path),
                "mode": args.mode,
                "subset": args.subset,
                "dry_run": args.dry_run,
                "records": records,
                "updated_at": utc_now(),
            },
        )
        if lock_path.exists() and not args.dry_run:
            lock_path.unlink()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
