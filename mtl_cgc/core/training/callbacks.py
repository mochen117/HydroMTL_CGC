# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Model checkpointing and adaptive early stopping callbacks.
# Synchronizes weight tracking to maintain the best validation epoch.
# ==============================================================================

import torch
from pathlib import Path
from typing import Dict, Any, Optional

class EarlyStopping:
    """Stops training execution when validation metrics stop improving."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_weights: Optional[Dict[str, Any]] = None

    def step(self, val_loss: float, model: torch.nn.Module):
        """Monitors evaluation metrics of the epoch to track convergence."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)


class ModelCheckpoint:
    """Saves optimized parameters securely onto the local storage."""
    def __init__(self, save_dir: str, save_best_only: bool = True, verbose: bool = False):
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_loss = float('inf')
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def step(self, model: torch.nn.Module, epoch: int, val_loss: float, is_best: bool, optimizer: Optional[torch.optim.Optimizer] = None):
        """Saves weights and tracking records onto the checkpoint path."""
        if not self.save_best_only:
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), epoch_path)
            
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            self.best_loss = val_loss
            if self.verbose:
                print(f"Epoch {epoch:02d}: Best model weights saved with validation score: {val_loss:.4f}")