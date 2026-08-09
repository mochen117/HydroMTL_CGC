# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Mode-aware checkpointing and early stopping callbacks for HydroMTL.
# These callbacks use one scalar validation monitor consistently across learning
# rate scheduling, checkpoint selection, and early stopping.
# ==============================================================================

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
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
            if (
                self.restore_best_weights
                and model is not None
                and self.best_weights is not None
            ):
                model.load_state_dict(self.best_weights)

        return False

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable early-stopping state."""
        return {
            "best_value": self.best_value,
            "counter": int(self.counter),
            "early_stop": bool(self.early_stop),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore early-stopping state."""
        self.best_value = state.get("best_value")
        self.counter = int(state.get("counter", 0))
        self.early_stop = bool(state.get("early_stop", False))


class ModelCheckpoint:
    """Save best, latest, and final checkpoints with resumable state."""

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
    def _capture_rng_state() -> Dict[str, Any]:
        """Capture Python, NumPy, CPU, and CUDA random-number states."""
        numpy_state = np.random.get_state()

        return {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": str(numpy_state[0]),
                "state": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    @staticmethod
    def _atomic_torch_save(
        payload: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Write a checkpoint atomically to avoid partially written files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)

    @classmethod
    def _build_payload(
        cls,
        model: torch.nn.Module,
        epoch: int,
        monitor_value: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        monitor_name: str = "monitor",
        monitor_mode: str = "min",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a full checkpoint payload for evaluation and resumption."""
        payload: Dict[str, Any] = {
            "checkpoint_format_version": 2,
            "epoch": int(epoch),
            "monitor_name": str(monitor_name),
            "monitor_mode": str(monitor_mode),
            "monitor_value": float(monitor_value),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "scaler_state_dict": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "rng_state": cls._capture_rng_state(),
        }

        if extra:
            overlap = set(payload).intersection(extra)
            if overlap:
                raise ValueError(
                    "Checkpoint extra metadata must not overwrite core fields: "
                    f"{sorted(overlap)}"
                )
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
        scaler: Optional[Any] = None,
        monitor_name: str = "monitor",
        monitor_mode: str = "min",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save an optional epoch checkpoint and update ``best_model.pth``."""
        payload = self._build_payload(
            model=model,
            epoch=epoch,
            monitor_value=monitor_value,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
            extra=extra,
        )

        if not self.save_best_only:
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            self._atomic_torch_save(payload, epoch_path)

        if is_best:
            best_path = self.save_dir / "best_model.pth"
            self._atomic_torch_save(payload, best_path)
            self.best_value = float(monitor_value)
            if self.verbose:
                print(
                    f"Epoch {epoch:03d}: best model saved "
                    f"({monitor_name}={monitor_value:.6f}, "
                    f"mode={monitor_mode})."
                )

    def save_last(
        self,
        model: torch.nn.Module,
        epoch: int,
        monitor_value: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        monitor_name: str = "monitor",
        monitor_mode: str = "min",
        filename: str = "last_model.pth",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a named full-state checkpoint and return its path."""
        payload = self._build_payload(
            model=model,
            epoch=epoch,
            monitor_value=monitor_value,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
            extra=extra,
        )
        output_path = self.save_dir / filename
        self._atomic_torch_save(payload, output_path)
        return output_path
