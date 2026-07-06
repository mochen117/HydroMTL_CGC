# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Multi-task training engine for HydroMTL.
# Provides masked losses, gradient-similarity diagnostics, routing summaries,
# and PyTorch-version-compatible mixed precision support.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Any, Union
from collections import defaultdict

from mtl_cgc.core.losses.crits import DynamicMultiTaskLoss


def masked_rmse(
    pred: Union[torch.Tensor, Dict[str, Any]],
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute masked RMSE with strict shape checking to prevent silent broadcasting."""
    if isinstance(pred, dict) and "means" in pred and "weights" in pred:
        pred_value = torch.sum(
            pred["means"].squeeze(-1) * pred["weights"],
            dim=1,
        ).reshape(-1)
    else:
        pred_value = pred.reshape(-1)

    target = target.reshape(-1)

    if pred_value.shape[0] != target.shape[0]:
        raise ValueError(
            f"Shape mismatch in masked_rmse: "
            f"pred length={pred_value.shape[0]}, target length={target.shape[0]}"
        )

    mask = torch.isfinite(target) & torch.isfinite(pred_value) & (target != -9999.0)

    if mask.sum() == 0:
        return (torch.nan_to_num(pred_value) * 0.0).sum()

    mse = torch.mean((pred_value[mask] - target[mask]) ** 2)
    return torch.sqrt(mse + 1e-8)


class HydroTrainer:
    """Trainer supporting hydrological multi-task optimization and diagnostics."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        evaluator: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.evaluator = evaluator
        self.current_epoch = 1

        self.targets_cfg = self.config.data.get("targets", [])
        self.task_names = [str(t.get("name", "")).lower() for t in self.targets_cfg]
        self.task_weights = {
            str(t.get("name", "")).lower(): float(t.get("loss_weight", 1.0))
            for t in self.targets_cfg
        }

        self.clip_norm = float(getattr(self.config.training, "clip_grad_norm", 1.0))
        self.use_amp = bool(getattr(self.config.training, "use_amp", False))
        self.amp_enabled = self.use_amp and self.device.type == "cuda"

        self.scaler = GradScaler(enabled=self.amp_enabled)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        stat_dict = {}
        if evaluator is not None and getattr(evaluator, "scaler", None) is not None:
            stat_dict = getattr(evaluator.scaler, "stat_dict", {})

        self.criterion = DynamicMultiTaskLoss(config, stat_dict)
        self.gradient_history: List[Dict[str, float]] = []

    # --------------------------------------------------------------------------
    # Builders
    # --------------------------------------------------------------------------
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from training configuration."""
        train_cfg = self.config.training
        optimizer_name = str(getattr(train_cfg, "optimizer", "adamw")).lower()
        lr = float(getattr(train_cfg, "learning_rate", 1e-3))
        weight_decay = float(getattr(train_cfg, "weight_decay", 1e-4))

        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )

        return optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _build_scheduler(self) -> Optional[Any]:
        """Build optional learning-rate scheduler."""
        sched_cfg = getattr(self.config.training, "scheduler", {})
        if not sched_cfg or not sched_cfg.get("type"):
            return None

        sched_type = str(sched_cfg.get("type")).lower()

        if sched_type in {"reduce_on_plateau", "reducelronplateau"}:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(sched_cfg.get("factor", 0.5)),
                patience=int(sched_cfg.get("patience", 5)),
                min_lr=float(sched_cfg.get("min_lr", 1e-6)),
            )

        if sched_type in {"multistep", "multi_step"}:
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=list(sched_cfg.get("milestones", [15, 25])),
                gamma=float(sched_cfg.get("gamma", 0.2)),
            )

        if sched_type in {"step", "steplr"}:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(sched_cfg.get("step_size", 20)),
                gamma=float(sched_cfg.get("gamma", 0.5)),
            )

        return None

    # --------------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------------
    def compute_gradient_similarity(
        self,
        loss_q: torch.Tensor,
        loss_et: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute task-gradient cosine similarity for selected parameter groups."""
        param_groups = {
            "Encoder": [
                p
                for name, p in self.model.named_parameters()
                if (
                    "shared_encoder" in name
                    or "encoder" in name
                    or "lstm" in name
                )
                and p.requires_grad
            ],
            "CGCBlock": [
                p
                for name, p in self.model.named_parameters()
                if "cgc_layer" in name and p.requires_grad
            ],
            "Gate": [
                p
                for name, p in self.model.named_parameters()
                if "gate" in name.lower() and p.requires_grad
            ],
        }

        similarities: Dict[str, float] = {}


        for group_name, params in param_groups.items():
            if not params:
                continue

            grads_q = torch.autograd.grad(
                loss_q,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            grads_et = torch.autograd.grad(
                loss_et,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )

            flat_q = []
            flat_et = []

            for grad_q, grad_et in zip(grads_q, grads_et):
                if grad_q is not None and grad_et is not None:
                    flat_q.append(grad_q.detach().flatten())
                    flat_et.append(grad_et.detach().flatten())

            if flat_q and flat_et:
                vec_q = torch.cat(flat_q)
                vec_et = torch.cat(flat_et)
                sim = F.cosine_similarity(vec_q, vec_et, dim=0)
                similarities[group_name] = float(sim.item())
            else:
                similarities[group_name] = 0.0

            del grads_q, grads_et, flat_q, flat_et

        return similarities

    @staticmethod
    def summarize_gate_tensor(gate_array: np.ndarray) -> Dict[str, Any]:
        """Summarize routing probabilities using entropy and expert utilization."""
        if gate_array.size == 0:
            return {"entropy": np.nan, "utilization": []}

        gate_array = np.asarray(gate_array)

        if gate_array.ndim < 2:
            return {"entropy": np.nan, "utilization": []}

        mask = np.isfinite(gate_array).all(axis=-1)
        valid = gate_array[mask]

        if valid.size == 0:
            return {"entropy": np.nan, "utilization": []}

        valid = np.clip(valid, 1e-12, 1.0)
        entropy = -np.mean(np.sum(valid * np.log(valid), axis=-1))
        utilization = np.mean(valid, axis=0)

        return {
            "entropy": float(entropy),
            "utilization": [float(x) for x in utilization],
        }

    def _summarize_collected_gates(
        self,
        collected_gates: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Dict[str, Any]]:
        """Create routing diagnostics from collected gate arrays."""
        diagnostics: Dict[str, Dict[str, Any]] = {}

        for gate_name, gate_list in collected_gates.items():
            if not gate_list:
                continue

            try:
                gate_array = np.concatenate(gate_list, axis=0)
                diagnostics[gate_name] = self.summarize_gate_tensor(gate_array)
            except Exception:
                diagnostics[gate_name] = {
                    "entropy": np.nan,
                    "utilization": [],
                }

        return diagnostics

    # --------------------------------------------------------------------------
    # Epoch loops
    # --------------------------------------------------------------------------
    def train_epoch(
        self,
        loader: DataLoader,
        log_gradients: bool = False,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Run one optimization epoch.

        Displays one compact batch-level progress bar per epoch when
        config.training.batch_progress is enabled.
        """
        self.model.train()

        epoch_loss = 0.0
        task_loss_sums = defaultdict(float)
        grad_sim_sums = defaultdict(list)
        total_batches = 0

        show_batch_bar = bool(getattr(self.config.training, "batch_progress", False))

        iterator = loader
        if show_batch_bar:
            iterator = tqdm(
                loader,
                desc=f"Epoch {self.current_epoch:03d}/{self.config.training.epochs:03d}",
                leave=False,
                dynamic_ncols=True,
                mininterval=2.0,
                smoothing=0.1,
                ascii=False,
            )

        running_loss = 0.0

        for batch_idx, batch in enumerate(iterator, start=1):
            x = batch["features"].to(self.device).float()
            static_num = batch["static_num"].to(self.device).float()

            static_cat = None
            if "categorical_features" in batch:
                static_cat = batch["categorical_features"].to(self.device).long()

            targets = {
                task: batch[task].to(self.device).float()
                for task in self.task_names
                if task in batch
            }

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp_enabled):
                preds, _ = self.model(x, static_num, static_cat)

                losses = {
                    task: masked_rmse(preds[task], targets[task])
                    for task in self.task_names
                    if task in preds and task in targets
                }

                total_loss = self.criterion(preds, targets, static_num)

            if (
                log_gradients
                and batch_idx % 50 == 0
                and "streamflow" in losses
                and "evapotranspiration" in losses
            ):
                try:
                    sim_dict = self.compute_gradient_similarity(
                        losses["streamflow"],
                        losses["evapotranspiration"],
                    )
                    for key, value in sim_dict.items():
                        grad_sim_sums[key].append(value)
                except Exception:
                    pass

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_value = float(total_loss.detach().cpu().item())
            epoch_loss += loss_value
            running_loss = loss_value

            for task, loss_tensor in losses.items():
                task_loss_sums[task] += float(loss_tensor.detach().cpu().item())

            total_batches += 1

            if show_batch_bar and batch_idx % 100 == 0:
                iterator.set_postfix(
                    loss=f"{running_loss:.4f}",
                    refresh=False,
                )

        avg_loss = epoch_loss / max(1, total_batches)

        avg_task_losses = {
            task: value / max(1, total_batches)
            for task, value in task_loss_sums.items()
        }

        avg_grad_sims = {
            key: float(np.mean(values))
            for key, values in grad_sim_sums.items()
            if values
        }

        if avg_grad_sims:
            avg_grad_sims["Epoch_Num"] = float(self.current_epoch)
            self.gradient_history.append(avg_grad_sims)

        return avg_loss, avg_task_losses, avg_grad_sims

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
        period_dates: List[str],
    ) -> Tuple[
        float,
        Dict[str, float],
        Dict[str, Dict[str, float]],
        Optional[Any],
        Dict[str, Dict[str, Any]],
    ]:
        """Evaluate model and return global metrics, basin metrics, exports, and diagnostics."""
        self.model.eval()

        epoch_loss = 0.0
        total_batches = 0

        collected = {
            "preds": {task: [] for task in self.task_names},
            "targets": {task: [] for task in self.task_names},
            "gates": defaultdict(list),
            "basin_idx": [],
            "time_idx": [],
            "stat_num": [],
        }

        for batch in loader:
            x = batch["features"].to(self.device).float()
            static_num = batch["static_num"].to(self.device).float()

            static_cat = None
            if "categorical_features" in batch:
                static_cat = batch["categorical_features"].to(self.device).long()

            targets = {
                task: batch[task].to(self.device).float()
                for task in self.task_names
                if task in batch
            }

            with autocast(enabled=self.amp_enabled):
                preds, gates = self.model(x, static_num, static_cat)
                total_loss = self.criterion(preds, targets, static_num)

            epoch_loss += float(total_loss.detach().cpu().item())
            total_batches += 1

            for task in self.task_names:
                if task not in preds or task not in targets:
                    continue

                pred_value = preds[task]

                if isinstance(pred_value, dict) and "means" in pred_value and "weights" in pred_value:
                    pred_value = torch.sum(
                        pred_value["means"].squeeze(-1) * pred_value["weights"],
                        dim=1,
                    ).reshape(-1)
                else:
                    pred_value = pred_value.reshape(-1)

                target_value = targets[task].reshape(-1)

                collected["preds"][task].append(
                    pred_value.detach().cpu().numpy()[:, None]
                )
                collected["targets"][task].append(
                    target_value.detach().cpu().numpy()[:, None]
                )

            if gates:
                for gate_name, gate_value in gates.items():
                    collected["gates"][gate_name].append(
                        gate_value.detach().cpu().numpy()
                    )

            collected["basin_idx"].append(batch["basin_idx"].cpu().numpy())
            collected["time_idx"].append(batch["time_idx"].cpu().numpy())
            collected["stat_num"].append(static_num.detach().cpu().numpy())

        avg_loss = epoch_loss / max(1, total_batches)

        if self.evaluator is None:
            return avg_loss, {}, {}, None, {}

        global_metrics, per_basin_metrics, ds_export = self.evaluator.process_and_evaluate(
            collected,
            period_dates,
        )

        diagnostics = self._summarize_collected_gates(collected["gates"])

        return avg_loss, global_metrics, per_basin_metrics, ds_export, diagnostics