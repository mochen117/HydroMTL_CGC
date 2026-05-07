#!/usr/bin/env python3
import os
import sys
import ctypes
import argparse
import warnings
from pathlib import Path
import multiprocessing as mp

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    lib_path = os.path.join(conda_prefix, 'lib')
    old_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + old_ld

libstdcxx_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6') if conda_prefix else ''
try: ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
except Exception: pass 
try: mp.set_start_method('fork', force=True)
except RuntimeError: pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from easydict import EasyDict as edict

from mtl_cgc.data.data_loaders import get_hydro_dataloaders
from mtl_cgc.core.cgc_models.mtl_model import HydroMTL_CGC
from mtl_cgc.core.cgc_models.baselines import Hard_MTL_Model
from mtl_cgc.core.losses.crits import DynamicMultiTaskLoss
from mtl_cgc.core.training.trainer import HydroTrainer

def get_best_gpu():
    if not torch.cuda.is_available(): return torch.device('cpu')
    max_mem, best_idx = 0, 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory
        if mem > max_mem: max_mem, best_idx = mem, i
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    return torch.device(f"cuda:{best_idx}")

def main():
    parser = argparse.ArgumentParser(description="Train HydroMTL_CGC")
    parser.add_argument("--config", type=str, default="default.yaml", help="Path to config YAML file")
    parser.add_argument("--model_type", type=str, default="mtl_cgc", choices=["mtl_cgc", "mtl_hard"])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = edict(yaml.safe_load(f))

    device = get_best_gpu()
    print(f"--- Pipeline Initializing on {device} ---")

    data_root = Path(config.data.data_root)
    basin_ids = [f.stem.replace("gage_", "") for f in list(data_root.glob("gage_*.nc"))]
    print(f"Discovered {len(basin_ids)} basin files.")

    # 1. Architecture Assembly
    train_loader, val_loader, test_loader, scaler = get_hydro_dataloaders(config, basin_ids)
    model = HydroMTL_CGC(config) if args.model_type == "mtl_cgc" else Hard_MTL_Model(config)
    criterion = DynamicMultiTaskLoss(config, stat_dict=scaler.stat_dict)

    # 2. Output Directory
    save_dir = Path(config.experiment.save_dir) / config.experiment.name
    config.experiment.save_dir = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. Training execution
    trainer = HydroTrainer(model, config, device, criterion, scaler)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()