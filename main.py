# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Main execution pipeline for HydroMTL_CGC.
# Orchestrates training, independent testing, and NetCDF exports.
# Supports dynamic CLI overrides for ablation studies (STL vs Hard-MTL vs CGC).
# ==============================================================================

import os
import sys
import ctypes
import argparse
import random
import warnings
from pathlib import Path
import multiprocessing as mp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    lib_path = os.path.join(conda_prefix, 'lib')
    old_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + old_ld

libstdcxx_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6') if conda_prefix else ""
try:
    ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass 

try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from easydict import EasyDict as edict

from mtl_cgc.data.data_loaders import get_hydro_dataloaders
from mtl_cgc.core.cgc_models.mtl_model import HydroMTL_CGC
from mtl_cgc.core.training.trainer import HydroTrainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_best_gpu() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    max_mem = 0
    best_idx = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory
        if mem > max_mem:
            max_mem = mem
            best_idx = i
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    return torch.device(f"cuda:{best_idx}")


def load_config(config_path: str) -> edict:
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))
    return config


def main():
    parser = argparse.ArgumentParser(description="Train HydroMTL_CGC Models")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    # Dynamic Override Arguments for Ablation Studies
    parser.add_argument("--loss_weights", type=str, nargs='+', default=None)
    parser.add_argument("--targets", type=str, nargs='+', default=None, help="Override targets, e.g., streamflow evapotranspiration")
    parser.add_argument("--task_experts", type=int, nargs='+', default=None, help="Override task experts, e.g., 0 0 for Hard-MTL")
    
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    
    if args.experiment_name:
        config.experiment.name = args.experiment_name

    override_msgs =[]
    
    # 1. Override Targets (For STL experiments)
    if args.targets is not None:
        valid_targets = [t for t in config.data.targets if t.name.lower() in[tgt.lower() for tgt in args.targets]]
        config.data.targets = valid_targets
        override_msgs.append(f"Targets filtered -> {[t.name for t in valid_targets]}")

    # 2. Override Loss Weights
    if args.loss_weights is not None:
        try:
            weight_dict = {item.split('=')[0]: float(item.split('=')[1]) for item in args.loss_weights}
            for target in config.data.targets:
                t_name = target.name.lower()
                for k, v in weight_dict.items():
                    if k.lower() == t_name:
                        target.loss_weight = v
                        override_msgs.append(f"Target '{target.name}' weight -> {v}")
        except Exception:
            sys.exit(1)

    # 3. Override Task Experts (For Hard-MTL baseline)
    if args.task_experts is not None:
        config.model.cgc.task_experts = args.task_experts
        override_msgs.append(f"Task Experts -> {args.task_experts}")

    device = get_best_gpu() if args.device == "auto" else torch.device(args.device)

    data_root = Path(config.data.data_root)
    nc_files = list(data_root.glob("gage_*.nc"))
    basin_ids =[f.stem.replace("gage_", "") for f in nc_files]
    basin_ids.sort()

    print(f"\n{'='*85}")
    print(f" HydroMTL_CGC Framework | Mode: {args.mode.upper()} | Device: {device}")
    print(f"{'-'*85}")
    print(f" Discovered {len(basin_ids)} basins in dataset.")
    for msg in override_msgs:
        print(f" Override | {msg}")
    print(f"{'='*85}\n")

    train_loader, val_loader, test_loader, global_scaler = get_hydro_dataloaders(config, basin_ids, mode=args.mode)
    
    model = HydroMTL_CGC(config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Total trainable parameters: {trainable_params:,}\n")

    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    config.experiment.save_dir = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = HydroTrainer(
        model=model,
        config=config,
        device=device,
        criterion=None,
        scaler=global_scaler,
        use_wandb=False
    )

    epochs_config = int(getattr(config.training, 'epochs', 100))

    if args.mode == "train":
        print("Commencing training phase...\n")
        history = trainer.fit(train_loader, val_loader)
        
        model_path = save_dir / 'final_model.pth'
        torch.save(model.state_dict(), model_path)
        
        print("\nEvaluating final model on the independent validation set...")
        _, _, val_metrics, _, _, _, _, _ = trainer.evaluate(val_loader, desc="Final Validation")
        
        print(f"\n{'='*85}\n FINAL VALIDATION REPORT (Computed in mm/day)\n{'='*85}")
        for task in config.data.targets:
            t_name = task.name
            print(f"[{t_name.upper():<18}] NSE: {val_metrics.get(f'{t_name}_nse', float('nan')):>6.3f} | RMSE: {val_metrics.get(f'{t_name}_rmse', float('nan')):>6.3f}")
        print("="*85)

    elif args.mode == "test":
        model_path = save_dir / 'best_model.pth'
        if not model_path.exists():
            model_path = save_dir / 'final_model.pth'
            
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found in {save_dir}. Please run mode 'train' first.")

        model.load_state_dict(torch.load(model_path, map_location=device))
        
        print("\nCommencing rigorous evaluation on unseen TEST split...")
        _, _, metrics, p_phys, t_phys, b_idxs, gates, per_basin_metrics = trainer.evaluate(test_loader, desc="Independent Test")
        
        print(f"\n{'='*85}\n INDEPENDENT TEST REPORT (mm/day)\n{'='*85}")
        for k, v in metrics.items():
            print(f"  {k:<30}: {v:.4f}")
        print("="*85)

        print("\nExporting per-basin performance metrics to CSV...")
        if per_basin_metrics:
            df_metrics = pd.DataFrame.from_dict(per_basin_metrics, orient='index')
            df_metrics.index = [basin_ids[i] for i in df_metrics.index]
            df_metrics.index.name = 'gage_id'
            
            csv_save_path = save_dir / "test_per_basin_metrics.csv"
            df_metrics.to_csv(csv_save_path)
            print(f"Successfully saved to: {csv_save_path}")

        print("Exporting predictions and gate routing weights to NetCDF...")
        
        seq_len = config.data.get('sequence_length', 365)
        test_start = pd.to_datetime(config.data.test_period[0])
        test_end = pd.to_datetime(config.data.test_period[1])
        pred_start = test_start + pd.Timedelta(days=seq_len - 1)
        time_index = pd.date_range(start=pred_start, end=test_end, freq='D')
        
        num_basins = len(basin_ids)
        num_days = len(time_index)
        expected_size = num_basins * num_days
        
        ds_dict = {}
        
        for task in config.data.targets:
            t_name = task.name.lower()
            if t_name in p_phys and p_phys[t_name].size > 0:
                actual_size = p_phys[t_name].size
                if actual_size == expected_size:
                    ds_dict[f"{t_name}_sim"] = (["basin", "time"], p_phys[t_name].reshape(num_basins, num_days))
                    ds_dict[f"{t_name}_obs"] = (["basin", "time"], t_phys[t_name].reshape(num_basins, num_days))
                else:
                    print(f"  [WARNING] Shape mismatch for {t_name}: Expected {expected_size}, got {actual_size}.")
                    
        for g_name, g_arr in gates.items():
            actual_size = g_arr.shape[0]
            if actual_size == expected_size:
                num_experts = g_arr.shape[-1]
                dim_name = f"expert_{g_name}" 
                ds_dict[g_name] = (["basin", "time", dim_name], g_arr.reshape(num_basins, num_days, num_experts))
                
        if ds_dict:
            ds = xr.Dataset(
                data_vars=ds_dict,
                coords={
                    "basin": basin_ids,
                    "time": time_index
                }
            )
            ds.attrs['description'] = "HydroMTL_CGC Predictions and Internal Gate Weights"
            ds.attrs['units'] = "mm/day"
            
            nc_save_path = save_dir / "test_predictions_and_weights.nc"
            ds.to_netcdf(nc_save_path)
            print(f"Successfully saved to: {nc_save_path}")

if __name__ == "__main__":
    main()