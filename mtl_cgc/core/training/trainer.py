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
from tqdm import tqdm

from mtl_cgc.utils.logger import setup_logger
from .losses import get_loss_function
from .metrics import compute_metrics
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

logger = setup_logger(__name__)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.squeeze()
    target = target.squeeze()
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=pred.device)
    return torch.nn.functional.mse_loss(pred[mask], target[mask])

class HydroTrainer:
    def __init__(self, model: nn.Module, config: Any, device: torch.device,
                 use_wandb: bool = False, basin_scalers=None):
        self.model = model
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.basin_scalers = basin_scalers
        self.model = self.model.to(device)

        self.task_names = [t['name'] for t in self.config.data.targets]
        self.task_weights = {t['name']: float(t.get('loss_weight', 1.0)) for t in self.config.data.targets}
        
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_scheduler()
        self.callbacks = self._setup_callbacks()

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': []
        }
        self.clip_grad_norm = float(getattr(self.config.training, 'clip_grad_norm', 1.0))

    def _setup_optimizer(self) -> optim.Optimizer:
        opt_cfg = self.config.training
        optimizer_name = getattr(opt_cfg, 'optimizer', 'adam').lower()
        
        learning_rate = float(getattr(opt_cfg, 'learning_rate', 0.001))
        weight_decay = float(getattr(opt_cfg, 'weight_decay', 0.0))

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adadelta':
            return optim.Adadelta(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        scheduler_config = getattr(self.config.training, 'scheduler', {})
        if not scheduler_config or scheduler_config.get('type') is None:
            return None
        
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=float(scheduler_config.get('factor', 0.5)),
                patience=int(scheduler_config.get('patience', 10)),
                min_lr=float(scheduler_config.get('min_lr', 1e-6))
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(self.config.training.epochs),
                eta_min=float(scheduler_config.get('min_lr', 1e-6))
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=int(scheduler_config.get('step_size', 30)),
                gamma=float(scheduler_config.get('gamma', 0.1))
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _setup_callbacks(self) -> Dict[str, Any]:
        callbacks = {}
        early_stop_config = getattr(self.config.training, 'early_stopping', {})
        if early_stop_config.get('enabled', False):
            callbacks['early_stopping'] = EarlyStopping(
                patience=int(early_stop_config.get('patience', 30)),
                min_delta=float(early_stop_config.get('min_delta', 1e-4))
            )
        checkpoint_config = getattr(self.config.training, 'checkpoint', {})
        if checkpoint_config.get('enabled', True):
            save_dir = Path(self.config.experiment.get('save_dir', './output')) / 'checkpoints'
            callbacks['checkpoint'] = ModelCheckpoint(
                save_dir=save_dir,
                save_best_only=checkpoint_config.get('save_best_only', True),
                save_frequency=int(checkpoint_config.get('save_frequency', 10))
            )
        return callbacks

    def _compute_loss(self, predictions: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        valid_tasks = 0
        for task_name in self.task_names:
            if task_name in predictions and task_name in targets:
                weight = self.task_weights.get(task_name, 1.0)
                if weight > 0.0:
                    loss = masked_mse(predictions[task_name], targets[task_name])
                    if not torch.isnan(loss):
                        total_loss += weight * loss
                        valid_tasks += 1
                    
        if valid_tasks == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return total_loss

    def _safe_inverse_transform(self, scaler, data: np.ndarray, task_name: str, basin_idx: int) -> np.ndarray:
        if scaler is None:
            return data
        if np.abs(scaler.scale_) < 1e-6:
            return data
        try:
            transformed = scaler.inverse_transform(data)
        except Exception as e:
            logger.warning(f"Basin {basin_idx}, task {task_name}: inverse_transform failed: {e}")
            return data
            
        if np.isnan(transformed).any() or np.isinf(transformed).any():
            return data
        return transformed

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        
        task_losses = {task_name: 0.0 for task_name in self.task_names}
        task_batches = {task_name: 0 for task_name in self.task_names}
        
        all_predictions = {}
        all_targets = {}
        all_basin_idxs = []

        epochs = int(self.config.training.epochs)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout)
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            cat_x = batch.get('categorical_features')
            if cat_x is not None:
                cat_x = cat_x.to(self.device)
                
            basin_idxs = batch['basin_idx'].cpu().numpy()
            targets = {name: batch[name].to(self.device) for name in self.task_names if name in batch}

            self.optimizer.zero_grad()
            predictions = self.model(features, categorical_features=cat_x)
            
            if torch.isnan(features).any():
                logger.error("\n[Fatal] Features contain NaN! Check data imputation.")
                sys.exit(1)
            if cat_x is not None and torch.isnan(cat_x).any():
                logger.error("\n[Fatal] Categorical features contain NaN!")
                sys.exit(1)
            if any(torch.isnan(p).any() for p in predictions.values()):
                logger.error("\n[Fatal] Model predicted NaN! Possible exploding gradients.")
                sys.exit(1)
            
            loss = self._compute_loss(predictions, targets)
            
            if loss.item() == 0.0 or torch.isnan(loss):
                pbar.set_postfix({'avg_loss': f"{(total_loss / max(1, batch_idx)):.4f}"})
                continue
                
            loss.backward()
            
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
            self.optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                for task_name in self.task_names:
                    weight = self.task_weights.get(task_name, 1.0)
                    if weight > 0.0 and task_name in predictions and task_name in targets:
                        t_loss = masked_mse(predictions[task_name], targets[task_name])
                        if not torch.isnan(t_loss):
                            task_losses[task_name] += t_loss.item()
                            task_batches[task_name] += 1
                            
                        pred = predictions[task_name].detach().cpu()
                        targ = targets[task_name].detach().cpu()
                        
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(1)
                        elif pred.dim() == 3:
                            pred = pred.view(-1, pred.size(-1))
                            
                        if targ.dim() == 1:
                            targ = targ.unsqueeze(1)
                        elif targ.dim() == 3:
                            targ = targ.view(-1, targ.size(-1))
                            
                        all_predictions.setdefault(task_name, []).append(pred)
                        all_targets.setdefault(task_name, []).append(targ)
                
                all_basin_idxs.append(basin_idxs)

            pbar.set_postfix({'avg_loss': f"{(total_loss / (batch_idx + 1)):.4f}"})
            
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(), 
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
        
        if not all_predictions:
            return avg_loss, {f'{t}_loss': float('nan') for t in self.task_names}

        for task_name in all_predictions:
            all_predictions[task_name] = torch.cat(all_predictions[task_name], dim=0)
            all_targets[task_name] = torch.cat(all_targets[task_name], dim=0)
            
        all_basin_idxs = np.concatenate(all_basin_idxs)

        if self.basin_scalers is not None:
            pred_orig = {}
            target_orig = {}
            for task_name in all_predictions:
                pred_orig[task_name] = torch.zeros_like(all_predictions[task_name])
                target_orig[task_name] = torch.zeros_like(all_targets[task_name])

            for basin_idx, task_scalers in enumerate(self.basin_scalers):
                if task_scalers is None:
                    continue
                mask = (all_basin_idxs == basin_idx)
                if mask.sum() == 0:
                    continue
                
                for task_name in all_predictions:
                    scaler = task_scalers.get(task_name)
                    if scaler is None:
                        pred_orig[task_name][mask] = all_predictions[task_name][mask]
                        target_orig[task_name][mask] = all_targets[task_name][mask]
                        continue
                    
                    pred_np = all_predictions[task_name][mask].numpy()
                    target_np = all_targets[task_name][mask].numpy()
                    
                    pred_inv = self._safe_inverse_transform(scaler, pred_np, task_name, basin_idx)
                    target_inv = self._safe_inverse_transform(scaler, target_np, task_name, basin_idx)
                    
                    pred_orig[task_name][mask] = torch.from_numpy(pred_inv)
                    target_orig[task_name][mask] = torch.from_numpy(target_inv)

            all_predictions = pred_orig
            all_targets = target_orig

        train_metrics = compute_metrics(all_predictions, all_targets, self.config.evaluation['metrics'])
        
        for task_name in self.task_names:
            if task_batches[task_name] > 0:
                train_metrics[f'{task_name}_loss'] = task_losses[task_name] / task_batches[task_name]
            else:
                train_metrics[f'{task_name}_loss'] = float('nan')

        return avg_loss, train_metrics

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        
        task_losses = {task_name: 0.0 for task_name in self.task_names}
        task_batches = {task_name: 0 for task_name in self.task_names}
        
        all_predictions = {}
        all_targets = {}
        all_basin_idxs = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                cat_x = batch.get('categorical_features')
                if cat_x is not None:
                    cat_x = cat_x.to(self.device)
                    
                basin_idxs = batch['basin_idx'].cpu().numpy()
                targets = {name: batch[name].to(self.device) for name in self.task_names if name in batch}

                predictions = self.model(features, categorical_features=cat_x)
                loss = self._compute_loss(predictions, targets)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()

                for task_name in self.task_names:
                    weight = self.task_weights.get(task_name, 1.0)
                    if weight > 0.0 and task_name in predictions and task_name in targets:
                        t_loss = masked_mse(predictions[task_name], targets[task_name])
                        if not torch.isnan(t_loss):
                            task_losses[task_name] += t_loss.item()
                            task_batches[task_name] += 1
                            
                        pred = predictions[task_name].cpu()
                        targ = targets[task_name].cpu()
                        
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(1)
                        elif pred.dim() == 3:
                            pred = pred.view(-1, pred.size(-1))
                            
                        if targ.dim() == 1:
                            targ = targ.unsqueeze(1)
                        elif targ.dim() == 3:
                            targ = targ.view(-1, targ.size(-1))
                            
                        all_predictions.setdefault(task_name, []).append(pred)
                        all_targets.setdefault(task_name, []).append(targ)

                all_basin_idxs.append(basin_idxs)

        if not all_predictions:
            return float('nan'), {f'{t}_nse': float('nan') for t in self.task_names}

        for task_name in all_predictions:
            all_predictions[task_name] = torch.cat(all_predictions[task_name], dim=0)
            all_targets[task_name] = torch.cat(all_targets[task_name], dim=0)
            
        all_basin_idxs = np.concatenate(all_basin_idxs)

        if self.basin_scalers is not None:
            pred_orig = {}
            target_orig = {}
            for task_name in all_predictions:
                pred_orig[task_name] = torch.zeros_like(all_predictions[task_name])
                target_orig[task_name] = torch.zeros_like(all_targets[task_name])

            for basin_idx, task_scalers in enumerate(self.basin_scalers):
                if task_scalers is None:
                    continue
                    
                mask = (all_basin_idxs == basin_idx)
                if mask.sum() == 0:
                    continue
                    
                for task_name in all_predictions:
                    scaler = task_scalers.get(task_name)
                    if scaler is None:
                        pred_orig[task_name][mask] = all_predictions[task_name][mask]
                        target_orig[task_name][mask] = all_targets[task_name][mask]
                        continue

                    pred_np = all_predictions[task_name][mask].numpy()
                    target_np = all_targets[task_name][mask].numpy()

                    pred_inv = self._safe_inverse_transform(scaler, pred_np, task_name, basin_idx)
                    target_inv = self._safe_inverse_transform(scaler, target_np, task_name, basin_idx)

                    pred_orig[task_name][mask] = torch.from_numpy(pred_inv)
                    target_orig[task_name][mask] = torch.from_numpy(target_inv)

            all_predictions = pred_orig
            all_targets = target_orig

        global_metrics = compute_metrics(all_predictions, all_targets, self.config.evaluation['metrics'])
        
        for task_name in self.task_names:
            if task_batches[task_name] > 0:
                global_metrics[f'{task_name}_loss'] = task_losses[task_name] / task_batches[task_name]
            else:
                global_metrics[f'{task_name}_loss'] = float('nan')

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
        return avg_loss, global_metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(self.config.experiment.get('save_dir', './output')) / 'tensorboard'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        start_time = time.time()

        if self.use_wandb:
            wandb.init(project="HydroMTL_CGC", config=self.config)
            wandb.watch(self.model)

        epochs = int(self.config.training.epochs)
        for epoch in range(self.current_epoch, epochs):
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

            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n" + "=" * 95)
            print(f"Epoch {epoch+1}/{epochs} Summary | Total Train Loss: {train_loss:.4f} | Total Val Loss: {val_loss:.4f} | LR: {current_lr:.6e}")
            print("=" * 95)
            
            metrics_to_print = ['loss', 'nse', 'kge', 'rmse', 'bias', 'corr']
            active_tasks = [t for t in self.task_names if self.task_weights.get(t, 1.0) > 0.0]
            
            for i, task_name in enumerate(active_tasks):
                print(f"[{task_name.upper()}]")
                
                train_str = []
                for m in metrics_to_print:
                    val_val = train_metrics.get(f'{task_name}_{m}', float('nan'))
                    train_str.append(f"{m.upper()}: {val_val:>8.4f}" if not np.isnan(val_val) else f"{m.upper()}:      NaN")
                print("  Train -> " + " | ".join(train_str))
                
                val_str = []
                for m in metrics_to_print:
                    val_val = val_metrics.get(f'{task_name}_{m}', float('nan'))
                    val_str.append(f"{m.upper()}: {val_val:>8.4f}" if not np.isnan(val_val) else f"{m.upper()}:      NaN")
                print("  Val   -> " + " | ".join(val_str))
                
                if i < len(active_tasks) - 1:
                    print("-" * 95)
            
            print("=" * 95 + "\n")
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
            
            for task_name in active_tasks:
                for m in metrics_to_print:
                    train_val = train_metrics.get(f'{task_name}_{m}', float('nan'))
                    val_val = val_metrics.get(f'{task_name}_{m}', float('nan'))
                    
                    if not np.isnan(train_val):
                        writer.add_scalar(f'{m.upper()}/{task_name}/train', train_val, epoch)
                    if not np.isnan(val_val):
                        writer.add_scalar(f'{m.upper()}/{task_name}/val', val_val, epoch)

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch, 
                    'train/loss': train_loss, 
                    'val/loss': val_loss,
                    'train/learning_rate': current_lr,
                    'train/metrics': train_metrics,
                    'val/metrics': val_metrics
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
                    print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
                    break
                    
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

        writer.close()
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.save_checkpoint(is_final=True)
        
        if self.use_wandb:
            wandb.finish()
            
        return self.train_history

    def save_checkpoint(self, checkpoint_path: Optional[str] = None, is_final: bool = False) -> None:
        if checkpoint_path is None:
            save_dir = Path(self.config.experiment.get('save_dir', './output'))
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_dir / ('final_model.pth' if is_final else 'checkpoint.pth')
            
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

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler and checkpoint.get('scheduler_state_dict'):
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', {})

    def predict(self, data_loader: DataLoader, return_analysis: bool = False) -> Dict[str, np.ndarray]:
        self.model.eval()
        predictions = {}
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                cat_x = batch.get('categorical_features')
                if cat_x is not None:
                    cat_x = cat_x.to(self.device)
                    
                model_output = self.model(features, categorical_features=cat_x, return_gate_analysis=return_analysis)
                
                for task_name, task_pred in model_output.items():
                    if task_name == 'gate_analysis' and not return_analysis:
                        continue
                    predictions.setdefault(task_name, []).append(task_pred.cpu().numpy())
                    
        for key in predictions:
            predictions[key] = np.concatenate(predictions[key], axis=0)
            
        return predictions