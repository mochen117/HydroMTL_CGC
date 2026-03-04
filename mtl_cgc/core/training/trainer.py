"""
Training module for HydroMTL_CGC model
Handles training loop, validation, checkpointing, and learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
import sys
from pathlib import Path
import wandb

from mtl_cgc.utils.logger import setup_logger
from .losses import get_loss_function
from .metrics import compute_metrics
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# ------------------------------------------------------------
# Force console logging: ensure logger outputs to stdout
# ------------------------------------------------------------
logger = setup_logger(__name__)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
# ------------------------------------------------------------


class HydroTrainer:
    """Main trainer class for HydroMTL_CGC model"""

    def __init__(self, model: nn.Module, config: Any,  # config is ExperimentConfig
                 device: torch.device, use_wandb: bool = False):
        """
        Initialize trainer

        Args:
            model: HydroMTL_CGC model
            config: ExperimentConfig object (not dict)
            device: Training device (cuda/cpu)
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        self.model = self.model.to(device)

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup loss function
        self.criterion = get_loss_function(
            self.config.training.loss,                     # loss config dict
            [t['name'] for t in self.config.data.targets]  # task names
        )

        # Store task names for later use
        self.task_names = [t['name'] for t in self.config.data.targets]

        # Setup learning rate scheduler
        self.lr_scheduler = self._setup_scheduler()

        # Setup callbacks
        self.callbacks = self._setup_callbacks()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        # Gradient clipping
        self.clip_grad_norm = getattr(self.config.training, 'clip_grad_norm', 1.0)

        logger.info(f"Initialized trainer on device: {device}")
        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Using loss function: {self.criterion.__class__.__name__}")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration"""
        opt_cfg = self.config.training
        optimizer_name = getattr(opt_cfg, 'optimizer', 'adam').lower()
        learning_rate = getattr(opt_cfg, 'learning_rate', 0.001)
        weight_decay = getattr(opt_cfg, 'weight_decay', 0.0)

        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        scheduler_config = getattr(self.config.training, 'scheduler', {})

        if not scheduler_config or scheduler_config.get('type') is None:
            return None

        scheduler_type = scheduler_config['type'].lower()

        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_callbacks(self) -> Dict[str, Any]:
        """Setup training callbacks"""
        callbacks = {}

        # Early stopping
        early_stop_config = getattr(self.config.training, 'early_stopping', {})
        if early_stop_config.get('enabled', False):
            callbacks['early_stopping'] = EarlyStopping(
                patience=early_stop_config.get('patience', 30),
                min_delta=early_stop_config.get('min_delta', 1e-4)
            )

        # Model checkpoint
        checkpoint_config = getattr(self.config.training, 'checkpoint', {})
        if checkpoint_config.get('enabled', True):
            save_dir = Path(self.config.experiment['save_dir']) / 'checkpoints'
            callbacks['checkpoint'] = ModelCheckpoint(
                save_dir=save_dir,
                save_best_only=checkpoint_config.get('save_best_only', True),
                save_frequency=checkpoint_config.get('save_frequency', 10)
            )

        return callbacks

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss and metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        # Use a dictionary to collect metrics dynamically
        batch_metrics = {}

        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}",
                    file=sys.stdout)  # force output to stdout

        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            targets = {}
            if 'streamflow' in batch:
                targets['streamflow'] = batch['streamflow'].to(self.device)
            if 'evapotranspiration' in batch:
                targets['evapotranspiration'] = batch['evapotranspiration'].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)
            # Pass task_names as keyword argument
            loss = self.criterion(predictions, targets, task_names=self.task_names)
            loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )

            self.optimizer.step()

            total_loss += loss.item()

            # Compute metrics for this batch
            batch_metrics_batch = compute_metrics(
                predictions, targets,
                self.config.evaluation['metrics']
            )

            # Collect metric values dynamically
            for key, value in batch_metrics_batch.items():
                if value is not None:
                    if key not in batch_metrics:
                        batch_metrics[key] = []
                    batch_metrics[key].append(value)

            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })

            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })

        avg_loss = total_loss / len(train_loader)
        # Compute average for each metric
        avg_metrics = {}
        for key, values in batch_metrics.items():
            avg_metrics[key] = np.nanmean(values) if values else np.nan

        return avg_loss, avg_metrics

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        batch_metrics = {}

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = {}
                if 'streamflow' in batch:
                    targets['streamflow'] = batch['streamflow'].to(self.device)
                if 'evapotranspiration' in batch:
                    targets['evapotranspiration'] = batch['evapotranspiration'].to(self.device)

                predictions = self.model(features)
                # Pass task_names as keyword argument
                loss = self.criterion(predictions, targets, task_names=self.task_names)
                total_loss += loss.item()

                batch_metrics_batch = compute_metrics(
                    predictions, targets,
                    self.config.evaluation['metrics']
                )

                for key, value in batch_metrics_batch.items():
                    if value is not None:
                        if key not in batch_metrics:
                            batch_metrics[key] = []
                        batch_metrics[key].append(value)

        avg_loss = total_loss / len(val_loader)
        avg_metrics = {}
        for key, values in batch_metrics.items():
            avg_metrics[key] = np.nanmean(values) if values else np.nan

        return avg_loss, avg_metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        logger.info("Starting training...")
        start_time = time.time()

        if self.use_wandb:
            wandb.init(project="HydroMTL_CGC", config=self.config)
            wandb.watch(self.model)

        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch

            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics = self.validate(val_loader)

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_metrics'].append(train_metrics)
            self.train_history['val_metrics'].append(val_metrics)

            logger.info(f"Epoch {epoch+1}/{self.config.training.epochs}: "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Log all metrics (they already include task prefixes)
            for metric_name in train_metrics.keys():
                train_metric = train_metrics.get(metric_name, np.nan)
                val_metric = val_metrics.get(metric_name, np.nan)
                logger.info(f"  {metric_name}: Train={train_metric:.4f}, Val={val_metric:.4f}")

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'train/metrics': train_metrics,
                    'val/metrics': val_metrics,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })

            if 'checkpoint' in self.callbacks:
                self.callbacks['checkpoint'].step(
                    model=self.model,
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=(val_loss < self.best_val_loss)
                )

            if 'early_stopping' in self.callbacks:
                if self.callbacks['early_stopping'].step(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        self.save_checkpoint(is_final=True)

        if self.use_wandb:
            wandb.finish()

        return self.train_history

    def save_checkpoint(self, checkpoint_path: Optional[str] = None,
                       is_final: bool = False) -> None:
        """
        Save training checkpoint

        Args:
            checkpoint_path: Path to save checkpoint
            is_final: Whether this is the final checkpoint
        """
        if checkpoint_path is None:
            save_dir = Path(self.config.experiment['save_dir'])
            if is_final:
                checkpoint_path = save_dir / 'final_model.pth'
            else:
                checkpoint_path = save_dir / 'checkpoint.pth'

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.lr_scheduler and checkpoint['scheduler_state_dict']:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")

    def predict(self, data_loader: DataLoader,
                return_analysis: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate predictions

        Args:
            data_loader: Data loader for prediction
            return_analysis: Whether to return gate analysis

        Returns:
            Dictionary of predictions for each task
        """
        self.model.eval()
        predictions = {}

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                model_output = self.model(features, return_gate_analysis=return_analysis)

                for task_name, task_pred in model_output.items():
                    if task_name == 'gate_analysis' and not return_analysis:
                        continue

                    if task_name not in predictions:
                        predictions[task_name] = []

                    if isinstance(task_pred, dict):
                        for key, value in task_pred.items():
                            if key not in predictions:
                                predictions[f"{task_name}_{key}"] = []
                            predictions[f"{task_name}_{key}"].append(
                                value.cpu().numpy()
                            )
                    else:
                        predictions[task_name].append(task_pred.cpu().numpy())

        for key in predictions:
            predictions[key] = np.concatenate(predictions[key], axis=0)

        return predictions