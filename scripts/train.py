#!/usr/bin/env python3
import os
import sys
import ctypes
import argparse
import logging
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
else:
    print("[ERROR] CONDA_PREFIX not set. Activate conda environment first.")
    sys.exit(1)

libstdcxx_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
try:
    ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    pass 

try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from easydict import EasyDict as edict

from mtl_cgc.data.data_loader import create_data_loaders
from mtl_cgc.core.cgc_models.mtl_model import HydroMTL_CGC

# --- [FIXED] 导入重构后正确的 Hard_MTL_Model ---
from mtl_cgc.core.cgc_models.baselines import Hard_MTL_Model

def get_best_gpu():
    if not torch.cuda.is_available():
        print("No GPU found, using CPU.")
        return torch.device('cpu')
    
    max_mem = 0
    best_idx = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory
        if mem > max_mem:
            max_mem = mem
            best_idx = i
            
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    device = torch.device(f"cuda:{best_idx}")
    return device

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))
    return config

def main():
    parser = argparse.ArgumentParser(description="Train HydroMTL Models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    
    parser.add_argument("--model_type", type=str, default="mtl_cgc", 
                        choices=["mtl_cgc", "mtl_hard"],
                        help="Choose architecture to train")
                        
    parser.add_argument("--loss_weights", type=float, nargs='+', default=None,
                        help="Loss weights for targets in order. e.g., 1.0 0.0 for STL streamflow")
    
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.epochs is not None:
        config.training.epochs = args.epochs

    if args.loss_weights is not None:
        targets = config.data.targets
        if len(args.loss_weights) != len(targets):
            raise ValueError(f"Expected {len(targets)} loss weights, got {len(args.loss_weights)}")
            
        for i, weight in enumerate(args.loss_weights):
            targets[i].loss_weight = weight
            print(f"[CONFIG OVERRIDE] Set '{targets[i].name}' loss_weight to {weight}")

    if args.device == "auto":
        device = get_best_gpu()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    data_root = Path(config.data.data_root)
    nc_files = list(data_root.glob("gage_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No gage_*.nc files found in {data_root}")
    basin_ids = [f.stem.replace("gage_", "") for f in nc_files]
    print(f"Found {len(basin_ids)} basins")

    data_loaders = create_data_loaders(config.data, basin_ids)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    basin_scalers = data_loaders['basin_scalers']

    print(f"Building Model Architecture: {args.model_type.upper()}")
    
    if args.model_type == "mtl_cgc":
        model = HydroMTL_CGC(config)
    elif args.model_type == "mtl_hard":
        # --- [FIXED] 极其清爽的实例化，维度计算已内聚到 baselines.py 中 ---
        model = Hard_MTL_Model(config)
    else:
        raise ValueError("Invalid model_type")

    weight_str = "default"
    if args.loss_weights is not None:
        weight_str = "_".join([str(w).replace('.', 'p') for w in args.loss_weights])
    save_suffix = f"{args.model_type}_w_{weight_str}"
    
    original_save_dir = Path(config.experiment.save_dir)
    save_dir = original_save_dir / save_suffix
    config.experiment.save_dir = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment outputs will be saved to: {save_dir}")
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    from mtl_cgc.core.training.trainer import HydroTrainer
    trainer = HydroTrainer(
        model=model,
        config=config,
        device=device,
        use_wandb=False,
        basin_scalers=basin_scalers
    )

    history = trainer.fit(train_loader, val_loader)

    torch.save(model.state_dict(), save_dir / 'final_model.pth')
    print(f"Model saved to {save_dir / 'final_model.pth'}")

    final_metrics = history['val_metrics'][-1]
    print("\nFinal validation metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()