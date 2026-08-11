#!/usr/bin/env python3
"""Report checkpoint and process status for a PUB manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtl_cgc.protocols.ch4_qssm_pub.checkpoints import inspect_experiment  # noqa: E402
from mtl_cgc.protocols.ch4_qssm_pub.io_utils import (  # noqa: E402
    load_json,
    load_yaml,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    parser.add_argument("--show-processes", action="store_true")
    return parser.parse_args()


def process_lines() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "run_pub_protocol.py|pub_main.py|main.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def run_once(args: argparse.Namespace) -> None:
    manifest = load_json(resolve_project_path(args.manifest, PROJECT_ROOT))
    processes = process_lines()
    counts = {"COMPLETED": 0, "RUNNING": 0, "PARTIAL": 0, "PENDING": 0, "INVALID": 0}
    completed_epochs = 0
    target_epochs = 0

    print("=" * 122)
    for entry in manifest["entries"]:
        config_path = resolve_project_path(entry["config"], PROJECT_ROOT)
        config = load_yaml(config_path)
        name = str(config["experiment"]["name"])
        target_epoch = int(config["training"]["epochs"])
        save_root = Path(config["experiment"].get("save_dir", "experiments"))
        if not save_root.is_absolute():
            save_root = PROJECT_ROOT / save_root
        status = inspect_experiment(save_root / name, target_epoch)
        active = any(name in line or config_path.name in line for line in processes)

        if status.state == "completed":
            label = "COMPLETED"
        elif status.state == "partial" and active:
            label = "RUNNING"
        elif status.state == "partial":
            label = "PARTIAL"
        elif status.state == "invalid":
            label = "INVALID"
        else:
            label = "PENDING"

        counts[label] += 1
        completed_epochs += max(0, min(status.epoch, target_epoch))
        target_epochs += target_epoch
        print(
            f"fold={int(entry['fold_id']):02d} "
            f"{entry['scenario']:<18s} {label:<10s} "
            f"epoch={max(status.epoch, 0):3d}/{target_epoch:<3d} {name}"
        )
        if status.message:
            print(f"    {status.message}")

    print("=" * 122)
    percent = 100.0 * completed_epochs / target_epochs if target_epochs else 0.0
    print(
        "Models: "
        + " | ".join(f"{key.lower()}={value}" for key, value in counts.items())
    )
    print(
        f"Epoch progress: {completed_epochs}/{target_epochs} ({percent:.1f}%)"
    )

    if args.show_processes:
        print("-" * 122)
        print("Matching processes:")
        for line in processes or ["None"]:
            print(f"  {line}")


def main() -> None:
    args = parse_args()
    if args.watch < 0:
        raise ValueError("--watch must be non-negative.")

    if args.watch == 0:
        run_once(args)
        return

    try:
        while True:
            print("\033[2J\033[H", end="")
            run_once(args)
            print(f"\nRefreshing every {args.watch} seconds. Press Ctrl+C to stop.")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nStatus monitoring stopped.")


if __name__ == "__main__":
    main()
