#!/usr/bin/env python3
"""
Check the training status of Chapter 4 Experiment A models.

The script reads a protocol manifest, inspects each experiment directory,
loads the latest available checkpoint, and reports whether each model is
completed, running, partially completed, pending, or invalid.

Examples
--------
Check the default seed-42 manifest:

    python scripts/ch4_qssm/check_ch4a_training_status.py

Check another manifest:

    python scripts/ch4_qssm/check_ch4a_training_status.py \
        --manifest path/to/manifest.json

Refresh the status every 60 seconds:

    python scripts/ch4_qssm/check_ch4a_training_status.py --watch 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_MANIFEST = (
    "mtl_cgc/configs/ch4_qssm_formal/seed42/"
    "ch4a_q_to_ssm_formal_seed42_manifest.json"
)


@dataclass(frozen=True)
class ModelStatus:
    """Status information for one configured experiment."""

    name: str
    config_path: str
    target_epoch: int
    current_epoch: int
    status: str
    checkpoint_path: str | None
    active_process: bool
    message: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect checkpoints and active processes for Chapter 4 "
            "Experiment A."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(DEFAULT_MANIFEST),
        help=(
            "Path to the experiment manifest. Relative paths are resolved "
            "from the project root."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root directory. By default, it is inferred from the "
            "location of this script."
        ),
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help=(
            "Refresh the status every N seconds. Use 0 for a one-time check."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for writing the latest status summary as JSON.",
    )
    parser.add_argument(
        "--show-processes",
        action="store_true",
        help="Print matching runner and main.py processes.",
    )
    return parser.parse_args()


def infer_project_root(explicit_root: Path | None) -> Path:
    """Return the absolute project root."""

    if explicit_root is not None:
        return explicit_root.expanduser().resolve()

    # scripts/ch4_qssm/check_ch4a_training_status.py -> project root
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path, project_root: Path, fallback_root: Path) -> Path:
    """Resolve a potentially relative path against known roots."""

    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    project_candidate = (project_root / path).resolve()
    if project_candidate.exists():
        return project_candidate

    fallback_candidate = (fallback_root / path).resolve()
    if fallback_candidate.exists():
        return fallback_candidate

    # Return the expected project-relative location for clear diagnostics.
    return project_candidate


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted local PyTorch checkpoint on the CPU."""

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must contain a dictionary, got {type(checkpoint)!r}"
        )

    return checkpoint


def read_checkpoint_epoch(path: Path) -> int:
    """Read and validate the epoch stored in a checkpoint."""

    checkpoint = load_checkpoint(path)

    if "epoch" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain an 'epoch' field: {path}")

    epoch = int(checkpoint["epoch"])
    if epoch < 0:
        raise ValueError(f"Invalid checkpoint epoch {epoch}: {path}")

    return epoch


def get_active_process_lines() -> list[str]:
    """Return active runner and training-process command lines."""

    try:
        result = subprocess.run(
            [
                "pgrep",
                "-af",
                "run_ch4a_q_to_ssm_protocol.py|main.py",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    if result.returncode not in {0, 1}:
        return []

    current_pid = str(os.getpid())

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith(f"{current_pid} ")
    ]


def process_matches_experiment(
    process_lines: list[str],
    experiment_name: str,
    config_path: Path,
) -> bool:
    """Return whether an active process appears to run this experiment."""

    config_name = config_path.name
    config_text = str(config_path)

    return any(
        experiment_name in line
        or config_name in line
        or config_text in line
        for line in process_lines
    )


def load_manifest(
    manifest_path: Path,
    project_root: Path,
) -> tuple[Path, list[Path]]:
    """Load and resolve all config paths from a protocol manifest."""

    resolved_manifest = resolve_path(
        manifest_path,
        project_root=project_root,
        fallback_root=Path.cwd(),
    )

    if not resolved_manifest.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {resolved_manifest}"
        )

    manifest = json.loads(
        resolved_manifest.read_text(encoding="utf-8")
    )

    config_values = manifest.get("configs")
    if not isinstance(config_values, list) or not config_values:
        raise ValueError(
            "Manifest field 'configs' must be a non-empty list."
        )

    config_paths = [
        resolve_path(
            Path(value),
            project_root=project_root,
            fallback_root=resolved_manifest.parent,
        )
        for value in config_values
    ]

    return resolved_manifest, config_paths


def inspect_model(
    config_path: Path,
    project_root: Path,
    process_lines: list[str],
) -> ModelStatus:
    """Inspect one configured experiment."""

    if not config_path.exists():
        return ModelStatus(
            name=config_path.stem,
            config_path=str(config_path),
            target_epoch=0,
            current_epoch=0,
            status="ERROR",
            checkpoint_path=None,
            active_process=False,
            message="Config file does not exist.",
        )

    try:
        config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        name = str(config["experiment"]["name"])
        target_epoch = int(config["training"]["epochs"])
    except Exception as exc:
        return ModelStatus(
            name=config_path.stem,
            config_path=str(config_path),
            target_epoch=0,
            current_epoch=0,
            status="ERROR",
            checkpoint_path=None,
            active_process=False,
            message=f"Invalid config: {exc}",
        )

    experiment_dir = project_root / "experiments" / name
    final_path = experiment_dir / "final_model.pth"
    last_path = experiment_dir / "last_model.pth"

    active = process_matches_experiment(
        process_lines=process_lines,
        experiment_name=name,
        config_path=config_path,
    )

    checkpoint_candidates: list[tuple[str, Path, int]] = []
    checkpoint_errors: list[str] = []

    for checkpoint_kind, checkpoint_path in (
        ("final", final_path),
        ("last", last_path),
    ):
        if not checkpoint_path.exists():
            continue

        try:
            epoch = read_checkpoint_epoch(checkpoint_path)
            checkpoint_candidates.append(
                (checkpoint_kind, checkpoint_path, epoch)
            )
        except Exception as exc:
            checkpoint_errors.append(
                f"{checkpoint_kind}_model.pth: {exc}"
            )

    final_entries = [
        item for item in checkpoint_candidates if item[0] == "final"
    ]
    last_entries = [
        item for item in checkpoint_candidates if item[0] == "last"
    ]

    final_epoch = final_entries[0][2] if final_entries else None
    last_epoch = last_entries[0][2] if last_entries else None

    if final_epoch is not None and final_epoch >= target_epoch:
        return ModelStatus(
            name=name,
            config_path=str(config_path),
            target_epoch=target_epoch,
            current_epoch=final_epoch,
            status="COMPLETED",
            checkpoint_path=str(final_path),
            active_process=active,
        )

    valid_epochs = [
        epoch
        for epoch in (final_epoch, last_epoch)
        if epoch is not None
    ]
    current_epoch = max(valid_epochs, default=0)

    selected_checkpoint: Path | None = None
    if last_epoch is not None and last_epoch == current_epoch:
        selected_checkpoint = last_path
    elif final_epoch is not None:
        selected_checkpoint = final_path

    if active:
        status = "RUNNING"
    elif current_epoch > 0:
        status = "PARTIAL"
    elif checkpoint_errors:
        status = "ERROR"
    else:
        status = "PENDING"

    message = "; ".join(checkpoint_errors) if checkpoint_errors else None

    return ModelStatus(
        name=name,
        config_path=str(config_path),
        target_epoch=target_epoch,
        current_epoch=current_epoch,
        status=status,
        checkpoint_path=(
            str(selected_checkpoint)
            if selected_checkpoint is not None
            else None
        ),
        active_process=active,
        message=message,
    )


def print_report(
    manifest_path: Path,
    statuses: list[ModelStatus],
    process_lines: list[str],
    show_processes: bool,
) -> dict[str, Any]:
    """Print a formatted status report and return a serializable summary."""

    counts = {
        "COMPLETED": 0,
        "RUNNING": 0,
        "PARTIAL": 0,
        "PENDING": 0,
        "ERROR": 0,
    }

    total_target_epochs = 0
    total_completed_epochs = 0

    print("=" * 118)
    print(f"Manifest: {manifest_path}")
    print("-" * 118)

    for item in statuses:
        counts[item.status] = counts.get(item.status, 0) + 1

        total_target_epochs += max(item.target_epoch, 0)
        total_completed_epochs += min(
            max(item.current_epoch, 0),
            max(item.target_epoch, 0),
        )

        progress_text = (
            f"{item.current_epoch:3d}/{item.target_epoch:<3d}"
            if item.target_epoch > 0
            else "  -/-  "
        )

        print(
            f"{item.name:66s} "
            f"{item.status:10s} "
            f"epoch={progress_text}"
        )

        if item.message:
            print(f"    Warning: {item.message}")

    print("=" * 118)

    model_count = len(statuses)
    epoch_progress = (
        100.0 * total_completed_epochs / total_target_epochs
        if total_target_epochs > 0
        else 0.0
    )

    print(
        "Models: "
        f"completed={counts['COMPLETED']}/{model_count} | "
        f"running={counts['RUNNING']} | "
        f"partial={counts['PARTIAL']} | "
        f"pending={counts['PENDING']} | "
        f"errors={counts['ERROR']}"
    )
    print(
        "Epoch progress: "
        f"{total_completed_epochs}/{total_target_epochs} "
        f"({epoch_progress:.1f}%)"
    )

    if show_processes:
        print("-" * 118)
        print("Matching processes:")
        if process_lines:
            for line in process_lines:
                print(f"  {line}")
        else:
            print("  None")

    return {
        "manifest": str(manifest_path),
        "model_count": model_count,
        "counts": counts,
        "epoch_progress": {
            "completed": total_completed_epochs,
            "target": total_target_epochs,
            "percent": round(epoch_progress, 3),
        },
        "models": [asdict(item) for item in statuses],
        "processes": process_lines,
    }


def write_json_summary(
    output_path: Path,
    summary: dict[str, Any],
    project_root: Path,
) -> None:
    """Write the status summary atomically as JSON."""

    resolved_output = (
        output_path.expanduser().resolve()
        if output_path.is_absolute()
        else (project_root / output_path).resolve()
    )

    resolved_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = resolved_output.with_suffix(
        resolved_output.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(resolved_output)


def run_once(args: argparse.Namespace, project_root: Path) -> int:
    """Run one status inspection."""

    manifest_path, config_paths = load_manifest(
        manifest_path=args.manifest,
        project_root=project_root,
    )

    process_lines = get_active_process_lines()

    statuses = [
        inspect_model(
            config_path=config_path,
            project_root=project_root,
            process_lines=process_lines,
        )
        for config_path in config_paths
    ]

    summary = print_report(
        manifest_path=manifest_path,
        statuses=statuses,
        process_lines=process_lines,
        show_processes=args.show_processes,
    )

    if args.json_output is not None:
        write_json_summary(
            output_path=args.json_output,
            summary=summary,
            project_root=project_root,
        )

    return 1 if any(item.status == "ERROR" for item in statuses) else 0


def main() -> int:
    """Program entry point."""

    args = parse_args()

    if args.watch < 0:
        raise ValueError("--watch must be zero or a positive integer.")

    project_root = infer_project_root(args.project_root)

    if args.watch == 0:
        return run_once(args=args, project_root=project_root)

    exit_code = 0
    try:
        while True:
            # Clear the terminal before each refresh.
            print("\033[2J\033[H", end="")
            exit_code = run_once(
                args=args,
                project_root=project_root,
            )
            print(
                f"\nRefreshing every {args.watch} seconds. "
                "Press Ctrl+C to stop."
            )
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nStatus monitoring stopped.")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
