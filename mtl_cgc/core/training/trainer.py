# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Advanced Model Trainer for Hydrological MTL Framework.
# Features: Masked RMSE computation, Mixed Precision Training (AMP), 
# robust progress bar management, TensorBoard tracking, and per-basin metrics.
# Refactored for maximum modularity and adherence to the DRY principle.
# ==============================================================================

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from mtl_cgc.core.evaluation.metrics import compute_metrics
try:
    from mtl_cgc.core.training.callbacks import (
        EarlyStopping, 
        ModelCheckpoint, 
        LearningRateScheduler, 
        CallbackHandler
    )
except ImportError:
    pass


def masked_rmse(pred: Union[torch.Tensor, Dict], target: torch.Tensor) -> torch.Tensor:
    """
    Robust Masked Root Mean Squared Error Loss.
    Safely ignores NaNs, prevents extreme gradient explosions via clamping,
    and adds epsilon to sqrt to prevent NaN gradients near zero.
    """
    if isinstance(pred, dict) and 'means' in pred:
        val = torch.sum(pred['means'].squeeze(-1) * pred['weights'], dim=1)
    else:
        val = pred.squeeze()
        
    target = target.squeeze()
    
    if val.shape != target.shape:
        try:
            val = val.view_as(target)
        except Exception:
            pass 
            
    mask = torch.isfinite(target) & torch.isfinite(val)
    if mask.sum() == 0:
        return (torch.nan_to_num(val) * 0.0).sum()
        
    diff = torch.clamp(val[mask] - target[mask], min=-1000.0, max=1000.0)
    mse_loss = torch.mean(diff ** 2)
    return torch.sqrt(mse_loss + 1e-8)


class HydroTrainer:
    """
    Orchestrates the training, validation, and testing loops.
    Expected Model Forward Signature:
        forward(...) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    """
    def __init__(
        self, 
        model: nn.Module, 
        config: Any, 
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        scaler: Any = None,
        use_wandb: bool = False,
        basin_scalers: Optional[List] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = criterion
        self.scaler = scaler
        self.use_wandb = use_wandb
        self.basin_scalers = basin_scalers

        self.targets_cfg = self.config.data.get('targets',[])
        self.task_names =[str(t.get('name', '')).lower() for t in self.targets_cfg]
        self.task_weights = {str(t.get('name', '')).lower(): float(t.get('loss_weight', 1.0)) for t in self.targets_cfg}
        
        self.clip_grad_norm = float(getattr(self.config.training, 'clip_grad_norm', 1.0))
        self.use_amp = getattr(self.config.training, 'use_amp', False)
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = {'train_loss':[], 'val_loss': [], 'train_metrics': [], 'val_metrics':[]}

        self.optimizer = self._setup_optimizer()
        base_scheduler = self._setup_scheduler()
        
        self.callbacks = CallbackHandler()
        self._setup_callbacks(base_scheduler)
        
        self.writer = None
        logging_cfg = getattr(self.config, 'logging', {})
        if isinstance(logging_cfg, dict) and logging_cfg.get('tensorboard', False):
            tb_dir = Path(self.config.experiment.get('save_dir', './output')) / 'tensorboard'
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))

    def _setup_optimizer(self) -> optim.Optimizer:
        opt_cfg = self.config.training
        opt_name = getattr(opt_cfg, 'optimizer', 'adamw').lower()
        lr = float(getattr(opt_cfg, 'learning_rate', 0.001))
        wd = float(getattr(opt_cfg, 'weight_decay', 0.001))

        if opt_name == 'adam': 
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd': 
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        elif opt_name == 'adadelta': 
            return optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=wd)
        else: 
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        sched_cfg = getattr(self.config.training, 'scheduler', {})
        if not sched_cfg or not sched_cfg.get('type'): 
            return None
        s_type = sched_cfg['type'].lower()
        if s_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=float(sched_cfg.get('factor', 0.5)),
                patience=int(sched_cfg.get('patience', 10)), min_lr=float(sched_cfg.get('min_lr', 1e-6))
            )
        return None

    def _setup_callbacks(self, base_scheduler):
        early_stop_cfg = getattr(self.config.training, 'early_stopping', {})
        if early_stop_cfg.get('enabled', True):
            self.callbacks.add_callback(EarlyStopping(
                patience=int(early_stop_cfg.get('patience', 15)),
                min_delta=float(early_stop_cfg.get('min_delta', 1e-4)), 
                restore_best_weights=True, verbose=False
            ))
            
        checkpoint_cfg = getattr(self.config.training, 'checkpoint', {})
        if checkpoint_cfg.get('enabled', True):
            save_dir = Path(self.config.experiment.get('save_dir', './output')) / 'checkpoints'
            self.callbacks.add_callback(ModelCheckpoint(save_dir=str(save_dir), save_best_only=True, verbose=False))
            
        if base_scheduler is not None:
            self.callbacks.add_callback(LearningRateScheduler(base_scheduler, verbose=False))

    def _unpack_batch(self, batch_data: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict, Optional[np.ndarray]]:
        """Extracts and formats batch elements into appropriate structures."""
        if not isinstance(batch_data, dict):
            raise ValueError("Dataset must return a dictionary format mapping inputs and targets.")

        targets_dict = {}
        dyn_x = batch_data.get('features', batch_data.get('dyn'))
        stat_num = batch_data.get('static_num', batch_data.get('s_num'))
        stat_cat = batch_data.get('categorical_features', batch_data.get('s_cat'))
        b_idx_raw = batch_data.get('basin_idx')
        
        basin_idx = b_idx_raw.cpu().numpy() if isinstance(b_idx_raw, torch.Tensor) else np.array(b_idx_raw)
        
        for k, v in batch_data.items():
            k_lower = str(k).lower()
            if k_lower in self.task_names:
                targets_dict[k_lower] = v
                
        if basin_idx is None and stat_num is not None:
            basin_idx = np.zeros(dyn_x.shape[0])
            
        return dyn_x, stat_num, stat_cat, targets_dict, basin_idx

    def _to_tensor(self, v: Any) -> Optional[torch.Tensor]:
        if v is None: 
            return None
        if isinstance(v, torch.Tensor): 
            return v.to(self.device).float()
        if isinstance(v, np.ndarray): 
            return torch.from_numpy(v).to(self.device).float()
        return torch.tensor(v, device=self.device, dtype=torch.float32)

    def _sanitize_predictions(self, preds_raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Clamps raw model predictions to prevent NaN/Inf propagation during metric computation."""
        preds_dict = {}
        for k, v in preds_raw.items():
            k_lower = str(k).lower()
            if isinstance(v, dict):
                preds_dict[k_lower] = {
                    'means': torch.clamp(torch.nan_to_num(v['means']), -10.0, 10.0), 
                    'weights': torch.clamp(torch.nan_to_num(v['weights']), 0.0, 1.0)
                }
            else:
                preds_dict[k_lower] = torch.clamp(torch.nan_to_num(v), -10.0, 10.0)
        return preds_dict

    def _compute_batch_loss(self, preds_dict: Dict, targets_dev: Dict, stat_num: Optional[torch.Tensor]) -> torch.Tensor:
        """Computes multi-task loss utilizing either a custom criterion or masked RMSE."""
        if self.criterion is not None:
            if hasattr(self.criterion, 'stat') and stat_num is not None:
                return self.criterion(preds_dict, targets_dev, stat_num)
            return self.criterion(preds_dict, targets_dev)
            
        valid_tasks =[t for t in self.task_names if t in preds_dict and t in targets_dev]
        if valid_tasks:
            return sum([self.task_weights.get(t, 1.0) * masked_rmse(preds_dict[t], targets_dev[t]) for t in valid_tasks])
        
        # Fallback to prevent graph disconnection
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _collect_batch_metrics(self, preds_dict: Dict, targets_dev: Dict, task_loss_sums: Dict, all_preds: Dict, all_targets: Dict):
        """Extracts and flattens batch predictions for subsequent spatial evaluation."""
        for t in self.task_names:
            if t in preds_dict and t in targets_dev:
                t_loss = masked_rmse(preds_dict[t], targets_dev[t])
                task_loss_sums[t] += float(torch.nan_to_num(t_loss).item())
                
                p_val = preds_dict[t]
                if isinstance(p_val, dict) and 'means' in p_val:
                    p_val = torch.sum(p_val['means'].squeeze(-1) * p_val['weights'], dim=1)
                else:
                    p_val = p_val.squeeze(-1)
                    
                all_preds[t].append(p_val.detach().cpu().numpy().flatten())
                all_targets[t].append(targets_dev[t].detach().cpu().numpy().flatten())

    def train_epoch(self, loader: DataLoader, pbar: Optional[tqdm] = None) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        task_loss_sums = {t: 0.0 for t in self.task_names}
        valid_batches = 0
        current_lr = self.optimizer.param_groups[0]['lr']
        
        all_preds = {t: [] for t in self.task_names}
        all_targets = {t:[] for t in self.task_names}
        all_basin_idxs, all_stat_nums = [],[]

        for batch_data in loader:
            try:
                dyn_x, stat_num, stat_cat, targets_dict, basin_idx = self._unpack_batch(batch_data)
                
                if dyn_x is None or not targets_dict:
                    continue

                dyn_x = torch.nan_to_num(self._to_tensor(dyn_x), nan=0.0)
                stat_num_tensor = torch.nan_to_num(self._to_tensor(stat_num), nan=0.0)
                if stat_cat is not None: 
                    stat_cat = stat_cat.to(self.device, dtype=torch.long)
                
                targets_dev = {k: self._to_tensor(v) for k, v in targets_dict.items() if k in self.task_names}

                self.optimizer.zero_grad()
                
                # Forward and Loss with AMP
                with autocast(enabled=self.use_amp):
                    preds_raw, _ = self.model(dyn_x, stat_num_tensor, stat_cat)
                    preds_dict = self._sanitize_predictions(preds_raw)
                    
                    loss = self._compute_batch_loss(preds_dict, targets_dev, stat_num_tensor)
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
                    
                    if not loss.requires_grad:
                        loss.requires_grad = True

                # Backward and Optimization using GradScaler
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
                
                loss_val = float(loss.item())
                total_loss += loss_val
                valid_batches += 1

                # Metric extraction
                with torch.no_grad():
                    self._collect_batch_metrics(preds_dict, targets_dev, task_loss_sums, all_preds, all_targets)
                    if basin_idx is not None: 
                        all_basin_idxs.append(basin_idx.flatten())
                    if stat_num_tensor is not None: 
                        all_stat_nums.append(stat_num_tensor.detach().cpu().numpy())
                
                if pbar is not None:
                    disp_loss = total_loss / valid_batches
                    pbar.set_postfix({'loss': f"{disp_loss:.4f}"})

            finally:
                if pbar is not None:
                    pbar.update(1)

        # Sequence flattening and metric computation
        preds_concat = {t: (np.concatenate(all_preds[t]) if all_preds[t] else np.array([])) for t in self.task_names}
        targets_concat = {t: (np.concatenate(all_targets[t]) if all_targets[t] else np.array([])) for t in self.task_names}
        basin_idxs_concat = np.concatenate(all_basin_idxs) if len(all_basin_idxs) > 0 else np.array([])
        stat_nums_concat = np.concatenate(all_stat_nums, axis=0) if all_stat_nums else None

        p_phys, t_phys = self._apply_inverse_scaling(preds_concat, targets_concat, basin_idxs_concat, stat_nums_concat)
        train_metrics, _ = self._compute_spatial_median_metrics(p_phys, t_phys, basin_idxs_concat)

        valid_batches = max(1, valid_batches) 
        return total_loss / valid_batches, {t: v / valid_batches for t, v in task_loss_sums.items()}, train_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, pbar: Optional[tqdm] = None, desc: str = "Evaluating") -> Tuple:
        """
        Executes validation or independent test loop.
        Returns:
            (avg_loss, avg_tasks, final_metrics, p_phys, t_phys, basin_idxs, gates_concat, per_basin_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        task_loss_sums = {t: 0.0 for t in self.task_names}
        valid_batches = 0
        
        all_preds = {t:[] for t in self.task_names}
        all_targets = {t:[] for t in self.task_names}
        all_basin_idxs, all_stat_nums = [],[]
        all_gates = {}

        internal_pbar = False
        if pbar is None:
            pbar = tqdm(total=len(loader), desc=desc, leave=True, file=sys.stdout, ncols=100, mininterval=10.0)
            internal_pbar = True

        for batch_data in loader:
            try:
                dyn_x, stat_num, stat_cat, targets_dict, basin_idx = self._unpack_batch(batch_data)

                if dyn_x is None or not targets_dict:
                    continue

                dyn_x = torch.nan_to_num(self._to_tensor(dyn_x), nan=0.0)
                stat_num_tensor = torch.nan_to_num(self._to_tensor(stat_num), nan=0.0)
                if stat_cat is not None: 
                    stat_cat = stat_cat.to(self.device, dtype=torch.long)
                targets_dev = {k: self._to_tensor(v) for k, v in targets_dict.items() if k in self.task_names}

                with autocast(enabled=self.use_amp):
                    preds_raw, gates_raw = self.model(dyn_x, stat_num_tensor, stat_cat)
                    preds_dict = self._sanitize_predictions(preds_raw)
                    loss = self._compute_batch_loss(preds_dict, targets_dev, stat_num_tensor)
                    
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
                total_loss += float(loss.item())
                valid_batches += 1

                # Capture gate weights for interpretability
                for k, v in gates_raw.items():
                    if k not in all_gates: 
                        all_gates[k] = []
                    all_gates[k].append(v.cpu().numpy())

                self._collect_batch_metrics(preds_dict, targets_dev, task_loss_sums, all_preds, all_targets)
                
                if basin_idx is not None: 
                    all_basin_idxs.append(basin_idx.flatten())
                if stat_num_tensor is not None: 
                    all_stat_nums.append(stat_num_tensor.detach().cpu().numpy())
                    
                if not internal_pbar and pbar is not None:
                    disp_loss = total_loss / valid_batches
                    pbar.set_postfix({'loss': f"{disp_loss:.4f}"})

            finally:
                if pbar is not None:
                    pbar.update(1)

        if internal_pbar and pbar is not None: 
            pbar.close()

        preds_concat = {t: (np.concatenate(all_preds[t]) if all_preds[t] else np.array([])) for t in self.task_names}
        targets_concat = {t: (np.concatenate(all_targets[t]) if all_targets[t] else np.array([])) for t in self.task_names}
        basin_idxs_concat = np.concatenate(all_basin_idxs) if len(all_basin_idxs) > 0 else np.array([])
        stat_nums_concat = np.concatenate(all_stat_nums, axis=0) if all_stat_nums else None
        gates_concat = {k: np.concatenate(v, axis=0) for k, v in all_gates.items()} if all_gates else {}

        p_phys, t_phys = self._apply_inverse_scaling(preds_concat, targets_concat, basin_idxs_concat, stat_nums_concat)
        final_metrics, per_basin_metrics = self._compute_spatial_median_metrics(p_phys, t_phys, basin_idxs_concat)

        valid_batches = max(1, valid_batches) 
        return (total_loss / valid_batches, {t: v / valid_batches for t, v in task_loss_sums.items()}, 
                final_metrics, p_phys, t_phys, basin_idxs_concat, gates_concat, per_basin_metrics)

    def _safe_inverse_transform(self, scaler, data: np.ndarray, task_name: str, basin_idx: int) -> np.ndarray:
        if scaler is None: 
            return data
        if hasattr(scaler, 'scale_') and np.abs(scaler.scale_) < 1e-6: 
            return data
        try:
            transformed = scaler.inverse_transform(data)
        except Exception:
            return data
        if np.isnan(transformed).any() or np.isinf(transformed).any():
            return data
        return transformed

    def _apply_inverse_scaling(self, preds_concat, targets_concat, basin_idxs_concat, stat_nums_concat=None):
        if getattr(self, 'basin_scalers', None) is not None:
            p_phys = {t: np.zeros_like(preds_concat[t]) for t in self.task_names}
            t_phys = {t: np.zeros_like(targets_concat[t]) for t in self.task_names}
            for basin_idx, task_scalers in enumerate(self.basin_scalers):
                if task_scalers is None: 
                    continue
                mask = (basin_idxs_concat == basin_idx)
                if mask.sum() == 0: 
                    continue
                
                for t in self.task_names:
                    scaler = task_scalers.get(t)
                    if scaler is None:
                        p_phys[t][mask] = preds_concat[t][mask]
                        t_phys[t][mask] = targets_concat[t][mask]
                        continue
                    
                    try:
                        p_transformed = scaler.inverse_transform(preds_concat[t][mask])
                        t_transformed = scaler.inverse_transform(targets_concat[t][mask])
                        
                        p_phys[t][mask] = p_transformed if not np.isnan(p_transformed).any() else preds_concat[t][mask]
                        t_phys[t][mask] = t_transformed if not np.isnan(t_transformed).any() else targets_concat[t][mask]
                    except Exception:
                        p_phys[t][mask] = preds_concat[t][mask]
                        t_phys[t][mask] = targets_concat[t][mask]
            return p_phys, t_phys
        
        elif self.scaler is not None:
            if stat_nums_concat is not None and preds_concat.get(self.task_names[0],[]).size > 0:
                return self.scaler.inverse_transform_target(preds_concat, stat_nums_concat), self.scaler.inverse_transform_target(targets_concat, stat_nums_concat)
        
        return preds_concat, targets_concat

    def _compute_spatial_median_metrics(self, p_phys: Dict, t_phys: Dict, basin_idxs: np.ndarray) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        unique_basins = np.unique(basin_idxs) if len(basin_idxs) > 0 else np.array([])
        raw_metrics = getattr(self.config.evaluation, 'metrics',['nse', 'kge', 'rmse', 'bias', 'corr'])
        target_metrics =[m.lower() for m in raw_metrics]
        
        final_metrics = {}
        per_basin_metrics = {}
        global_nse_list =[]

        for task in self.task_names:
            if task not in p_phys or task not in t_phys or p_phys[task].size == 0: 
                continue
            
            basin_metrics_list = {m:[] for m in target_metrics}
            for b_idx in unique_basins:
                mask = (basin_idxs == b_idx)
                if mask.sum() < 1: 
                    continue
                
                p_b_valid = p_phys[task][mask].flatten()
                t_b_valid = t_phys[task][mask].flatten()
                
                valid_idx = ~np.isnan(t_b_valid) & ~np.isnan(p_b_valid)
                if valid_idx.sum() < 1: 
                    continue
                
                res = compute_metrics({task: torch.from_numpy(p_b_valid[valid_idx])}, {task: torch.from_numpy(t_b_valid[valid_idx])}, target_metrics)
                
                b_idx_int = int(b_idx)
                if b_idx_int not in per_basin_metrics:
                    per_basin_metrics[b_idx_int] = {}
                    
                for m in target_metrics:
                    val = res.get(f"{task}_{m}")
                    if val is not None and not np.isnan(val):
                        basin_metrics_list[m].append(val)
                        per_basin_metrics[b_idx_int][f"{task}_{m}"] = val
                        if m == 'nse': 
                            global_nse_list.append(val)
                        
            for m in target_metrics:
                if basin_metrics_list[m]:
                    arr = np.array(basin_metrics_list[m])
                    final_metrics[f"{task}_{m}"] = float(np.median(arr))
                    final_metrics[f"{task}_{m}_75th"] = float(np.percentile(arr, 75))
                    final_metrics[f"{task}_{m}_pos_ratio"] = float(np.mean(arr > 0) * 100.0)
                else:
                    final_metrics[f"{task}_{m}"] = float('nan')
                    final_metrics[f"{task}_{m}_75th"] = float('nan')
                    final_metrics[f"{task}_{m}_pos_ratio"] = float('nan')
                    
        return final_metrics, per_basin_metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        save_path = Path(self.config.experiment.get('save_dir', './output'))
        save_path.mkdir(parents=True, exist_ok=True)

        epochs = int(getattr(self.config.training, 'epochs', 100))
        total_batches = len(train_loader) + len(val_loader)
        
        try:
            for epoch in range(self.current_epoch + 1, epochs + 1):
                self.current_epoch = epoch
                pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}/{epochs} [Train]", leave=True, file=sys.stdout, ncols=100, mininterval=10.0)
                
                t_loss, t_tasks, t_metrics = self.train_epoch(train_loader, pbar)
                
                pbar.set_description(f"Epoch {epoch}/{epochs} [Valid]")
                v_loss, v_tasks, v_metrics, _, _, _, _, _ = self.evaluate(val_loader, pbar)
                pbar.close()

                for cb in self.callbacks.callbacks:
                    if isinstance(cb, LearningRateScheduler): 
                        cb.step(v_loss)
                    elif isinstance(cb, EarlyStopping): 
                        cb.step(v_loss, self.model)
                    elif isinstance(cb, ModelCheckpoint): 
                        cb.step(self.model, epoch, v_loss, (v_loss < getattr(self, 'best_val_loss', float('inf'))), self.optimizer)

                if not np.isnan(v_loss) and v_loss < self.best_val_loss:
                    self.best_val_loss = v_loss

                current_lr = self.optimizer.param_groups[0]['lr']
                self.train_history['train_loss'].append(t_loss)
                self.train_history['val_loss'].append(v_loss)
                self.train_history['train_metrics'].append(t_metrics)
                self.train_history['val_metrics'].append(v_metrics)

                self._print_summary(epoch, epochs, current_lr, t_loss, t_tasks, v_loss, v_tasks, t_metrics, v_metrics)
                
                if self.writer is not None:
                    self.writer.add_scalar('Global/Train_Loss', t_loss, epoch)
                    self.writer.add_scalar('Global/Valid_Loss', v_loss, epoch)
                    self.writer.add_scalar('Global/Learning_Rate', current_lr, epoch)
                    
                    for t in self.task_names:
                        if f"{t}_nse" in t_metrics and not np.isnan(t_metrics[f"{t}_nse"]):
                            self.writer.add_scalar(f'{t.upper()}/Train_NSE', t_metrics[f"{t}_nse"], epoch)
                        if f"{t}_nse" in v_metrics and not np.isnan(v_metrics[f"{t}_nse"]):
                            self.writer.add_scalar(f'{t.upper()}/Valid_NSE', v_metrics[f"{t}_nse"], epoch)
                            
                        if f"{t}_rmse" in t_metrics and not np.isnan(t_metrics[f"{t}_rmse"]):
                            self.writer.add_scalar(f'{t.upper()}/Train_RMSE', t_metrics[f"{t}_rmse"], epoch)
                        if f"{t}_rmse" in v_metrics and not np.isnan(v_metrics[f"{t}_rmse"]):
                            self.writer.add_scalar(f'{t.upper()}/Valid_RMSE', v_metrics[f"{t}_rmse"], epoch)

                early_stopper = next((cb for cb in self.callbacks.callbacks if isinstance(cb, EarlyStopping)), None)
                if early_stopper and getattr(early_stopper, 'early_stop', False):
                    print("\n[INFO] Early stopping triggered. Terminating training.")
                    break

        finally:
            if self.writer is not None:
                self.writer.close()

        return self.train_history

    def _print_summary(self, ep: int, total: int, lr: float, t_l: float, t_tk: Dict, v_l: float, v_tk: Dict, t_mets: Dict, v_mets: Dict):
        print(f"\n{'='*85}")
        print(f" Epoch {ep}/{total} Summary | LR: {lr:.2e} | Train Loss: {t_l:.4f} | Val Loss: {v_l:.4f}")
        print(f"{'='*85}")
        
        raw_metrics = getattr(self.config.evaluation, 'metrics', ['nse', 'kge', 'rmse', 'bias', 'corr'])
        metrics_to_print = [m.lower() for m in raw_metrics]
        
        for i, t in enumerate(self.task_names):
            print(f" Task: {t.upper()}")
            
            t_str = "   ".join([f"{m.upper()}: {t_mets.get(f'{t}_{m}', float('nan')):>6.3f}" for m in metrics_to_print])
            print(f"   Train  Loss: {t_tk.get(t, 0.0):>6.4f}   {t_str}")
            
            v_str = "   ".join([f"{m.upper()}: {v_mets.get(f'{t}_{m}', float('nan')):>6.3f}" for m in metrics_to_print])
            print(f"   Valid  Loss: {v_tk.get(t, 0.0):>6.4f}   {v_str}")
            
            nse_med = v_mets.get(f'{t}_nse', float('nan'))
            nse_75 = v_mets.get(f'{t}_nse_75th', float('nan'))
            nse_pos = v_mets.get(f'{t}_nse_pos_ratio', 0.0)
            print(f"   Stats  Val NSE Median: {nse_med:>6.3f}   (75th: {nse_75:>6.3f})   NSE>0 Ratio: {nse_pos:>5.1f}%")
            
            if i < len(self.task_names) - 1: 
                print("")
                
        print(f"{'='*85}\n")