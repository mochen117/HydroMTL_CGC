import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 verbose: bool = True, restore_best_weights: bool = True):
        """
        Initialize early stopping callback
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state_dict = None
        
    def step(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if early stopping condition is met
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from (if restore_best_weights is True)
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            
            if self.restore_best_weights and model is not None:
                self.best_state_dict = {
                    'model_state_dict': model.state_dict(),
                    'best_loss': self.best_loss
                }
            
            if self.verbose:
                logger.info(f'EarlyStopping: Validation loss improved to {val_loss:.6f}')
                
        else:
            # No improvement
            self.counter += 1
            
            if self.verbose:
                logger.info(f'EarlyStopping: No improvement for {self.counter}/{self.patience} epochs')
            
            if self.counter >= self.patience:
                self.early_stop = True
                
                if self.verbose:
                    logger.info(f'EarlyStopping: Triggered after {self.patience} epochs without improvement')
                
                if self.restore_best_weights and self.best_state_dict is not None:
                    logger.info('EarlyStopping: Restoring best model weights')
                    model.load_state_dict(self.best_state_dict['model_state_dict'])
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping counter"""
        self.counter = 0
        self.early_stop = False


class ModelCheckpoint:
    """Callback to save model checkpoints during training"""
    
    def __init__(self, save_dir: str, save_best_only: bool = True, 
                 save_frequency: int = 1, verbose: bool = True):
        """
        Initialize model checkpoint callback
        
        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Whether to save only when validation loss improves
            save_frequency: Frequency of saving (every N epochs)
            verbose: Whether to print messages
        """
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.verbose = verbose
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        
    def step(self, model: torch.nn.Module, epoch: int, val_loss: float, 
             is_best: bool, optimizer=None, scheduler=None, **kwargs):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            epoch: Current epoch number
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
            optimizer: Optimizer to save (optional)
            scheduler: Scheduler to save (optional)
            **kwargs: Additional items to save
        """
        # Check if we should save this epoch
        should_save = False
        
        if self.save_best_only and is_best:
            should_save = True
            if self.verbose:
                logger.info(f'ModelCheckpoint: Best model improved (loss: {val_loss:.6f}), saving...')
        
        elif not self.save_best_only and epoch % self.save_frequency == 0:
            should_save = True
        
        if should_save:
            # Prepare checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                **kwargs
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save checkpoint
            if self.save_best_only and is_best:
                checkpoint_path = self.save_dir / 'best_model.pth'
                checkpoint['best'] = True
            else:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch:04d}.pth'
            
            torch.save(checkpoint, checkpoint_path)
            
            if self.verbose:
                logger.info(f'ModelCheckpoint: Saved checkpoint to {checkpoint_path}')
        
        # Update best loss
        if is_best:
            self.best_loss = val_loss


class LearningRateScheduler:
    """Callback for custom learning rate scheduling"""
    
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, 
                 verbose: bool = True):
        """
        Initialize learning rate scheduler callback
        
        Args:
            scheduler: PyTorch learning rate scheduler
            verbose: Whether to print messages
        """
        self.scheduler = scheduler
        self.verbose = verbose
        self.current_lr = None
        
    def step(self, metrics=None):
        """
        Step the learning rate scheduler
        
        Args:
            metrics: Metrics for schedulers like ReduceLROnPlateau
        """
        if metrics is not None and hasattr(self.scheduler, 'step'):
            # For ReduceLROnPlateau
            self.scheduler.step(metrics)
        elif hasattr(self.scheduler, 'step'):
            # For other schedulers
            self.scheduler.step()
        
        # Get current learning rate
        if hasattr(self.scheduler.optimizer, 'param_groups'):
            self.current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            
            if self.verbose:
                logger.info(f'LearningRateScheduler: Learning rate updated to {self.current_lr:.6f}')
    
    def get_lr(self):
        """Get current learning rate"""
        return self.current_lr


class CallbackHandler:
    """Handler for managing multiple callbacks"""
    
    def __init__(self, callbacks=None):
        """
        Initialize callback handler
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks if callbacks is not None else []
        
    def add_callback(self, callback):
        """Add a callback to the handler"""
        self.callbacks.append(callback)
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, logs)
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs)
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(logs)
    
    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of each batch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_begin'):
                callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(batch, logs)