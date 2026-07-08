# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Mode-aware checkpointing and early stopping callbacks for HydroMTL.
# These callbacks use one scalar validation monitor consistently across learning
# rate scheduling, checkpoint selection, and early stopping.
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


class EarlyStopping:
    """Mode-aware early stopping without implicit weight restoration.

    The callback only tracks whether training should stop. Model persistence is
    handled by ``ModelCheckpoint`` so that the saved best model, best epoch, and
    validation summary all refer to the same monitor.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best_weights: bool = False,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported early-stopping mode: {mode}")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best_weights = bool(restore_best_weights)

        self.best_value: Optional[float] = None
        self.counter = 0
        self.early_stop = False

        # Kept only for backward compatibility with older callers. The new
        # training loop disables implicit restoration and loads best_model.pth
        # explicitly when needed.
        self.best_weights: Optional[Dict[str, Any]] = None

    def is_better(self, value: float) -> bool:
        """Return True when ``value`` improves the current best monitor."""
        value = float(value)
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < self.best_value - self.min_delta

        return value > self.best_value + self.min_delta

    def step(self, value: float, model: Optional[torch.nn.Module] = None) -> bool:
        """Update early-stopping state and return whether the value improved."""
        value = float(value)

        if self.is_better(value):
            self.best_value = value
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)

        return False


class ModelCheckpoint:
    """Save best and optional epoch checkpoints with full training metadata."""

    def __init__(
        self,
        save_dir: str,
        save_best_only: bool = True,
        verbose: bool = False,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_best_only = bool(save_best_only)
        self.verbose = bool(verbose)
        self.best_value: Optional[float] = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_payload(
        model: torch.nn.Module,
        epoch: int,
        monitor_value: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        monitor_name: str = "monitor",
        monitor_mode: str = "min",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a checkpoint payload compatible with future audits."""
        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "monitor_name": str(monitor_name),
            "monitor_mode": str(monitor_mode),
            "monitor_value": float(monitor_value),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        }

        if extra:
            payload.update(extra)

        return payload

    def step(
        self,
        model: torch.nn.Module,
        epoch: int,
        monitor_value: float,
        is_best: bool,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        monitor_name: str = "monitor",
        monitor_mode: str = "min",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the current checkpoint and update ``best_model.pth`` if needed."""
        payload = self._build_payload(
            model=model,
            epoch=epoch,
            monitor_value=monitor_value,
            optimizer=optimizer,
            scheduler=scheduler,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
            extra=extra,
        )

        if not self.save_best_only:
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(payload, epoch_path)

        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(payload, best_path)
            self.best_value = float(monitor_value)
            if self.verbose:
                print(
                    f"Epoch {epoch:03d}: best model saved "
                    f"({monitor_name}={monitor_value:.6f}, mode={monitor_mode})."
                )

    def save_last(
        self,
        model: torch.nn.Module,
        epoch: int,
        monitor_value: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        monitor_name: str = "monitor",
        monitor_mode: str = "min",
        filename: str = "last_model.pth",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save the final epoch checkpoint and return the path."""
        payload = self._build_payload(
            model=model,
            epoch=epoch,
            monitor_value=monitor_value,
            optimizer=optimizer,
            scheduler=scheduler,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
            extra=extra,
        )
        out_path = self.save_dir / filename
        torch.save(payload, out_path)
        return out_path
