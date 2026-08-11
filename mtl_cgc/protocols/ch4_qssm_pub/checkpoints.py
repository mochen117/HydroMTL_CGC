"""Checkpoint inspection helpers for PUB runners and status reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .native_runtime import bootstrap_native_runtime

bootstrap_native_runtime(strict=True)

import torch


@dataclass(frozen=True)
class CheckpointStatus:
    """Training status inferred from final and last checkpoints."""

    state: str
    epoch: int
    target_epoch: int
    checkpoint_path: Path | None
    message: str | None = None


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted local checkpoint on CPU."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def checkpoint_epoch(path: Path) -> int:
    """Read the epoch stored in a checkpoint payload."""

    checkpoint = load_checkpoint(path)
    if "epoch" not in checkpoint:
        raise KeyError(
            f"Checkpoint has no epoch field. PUB formal runs require the "
            f"resume-capable checkpoint format: {path}"
        )
    return int(checkpoint["epoch"])


def inspect_experiment(
    experiment_dir: Path,
    target_epoch: int,
) -> CheckpointStatus:
    """Classify one experiment as completed, partial, pending, or invalid."""

    final_path = experiment_dir / "final_model.pth"
    last_path = experiment_dir / "last_model.pth"

    try:
        if final_path.exists():
            epoch = checkpoint_epoch(final_path)
            if epoch >= target_epoch:
                return CheckpointStatus(
                    state="completed",
                    epoch=epoch,
                    target_epoch=target_epoch,
                    checkpoint_path=final_path,
                )

        if last_path.exists():
            epoch = checkpoint_epoch(last_path)
            return CheckpointStatus(
                state="partial",
                epoch=epoch,
                target_epoch=target_epoch,
                checkpoint_path=last_path,
            )

        if final_path.exists():
            epoch = checkpoint_epoch(final_path)
            return CheckpointStatus(
                state="partial",
                epoch=epoch,
                target_epoch=target_epoch,
                checkpoint_path=final_path,
            )

        return CheckpointStatus(
            state="pending",
            epoch=0,
            target_epoch=target_epoch,
            checkpoint_path=None,
        )
    except Exception as exc:
        return CheckpointStatus(
            state="invalid",
            epoch=-1,
            target_epoch=target_epoch,
            checkpoint_path=None,
            message=str(exc),
        )
