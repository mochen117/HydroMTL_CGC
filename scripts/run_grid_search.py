# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Advanced Hyperparameter Grid Search script for HydroMTL_CGC.
# Expands search space to 32 permutations to meet top-tier journal standards.
# Automatically injects 'Fast-Search' constraints (50 epochs) to save time.
# ==============================================================================

import os
import yaml
import itertools
import subprocess
from copy import deepcopy
from pathlib import Path

# Dynamically resolve absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
BASE_CONFIG_PATH = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"

# Define the comprehensive 5D search space
GRID_SPACE = {
    'seq_len': [180, 365],
    'hidden_dim': [128, 256],
    'shared_experts':[2, 4],
    'dropout': [0.3, 0.5],
    'lr':[0.001, 0.0005]
}

# Fixed loss weights during grid search to maintain consistent gradient scale
FIXED_WEIGHTS = "streamflow=1.0 evapotranspiration=0.1"


def load_base_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_temp_config(config_dict: dict, path: Path):
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def run_command(command: str):
    result = subprocess.run(command, shell=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command execution failed: {command}")


def main():
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Base configuration not found at {BASE_CONFIG_PATH}")

    base_config = load_base_config(BASE_CONFIG_PATH)
    
    keys = list(GRID_SPACE.keys())
    combinations = list(itertools.product(*(GRID_SPACE[k] for k in keys)))
    total_runs = len(combinations)
    
    # 强制刷新：flush=True
    print(f"\n{'='*80}", flush=True)
    print(f"[INFO] Commencing Comprehensive Grid Search across {total_runs} configurations.", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    for idx, combo in enumerate(combinations, start=1):
        params = dict(zip(keys, combo))
        sl = params['seq_len']
        hd = params['hidden_dim']
        se = params['shared_experts']
        drop = params['dropout']
        lr = params['lr']
        
        exp_name = f"grid_sl{sl}_hd{hd}_se{se}_dp{int(drop*10)}_lr{str(lr).split('.')[-1]}"
        temp_yaml_path = PROJECT_ROOT / f"temp_{exp_name}.yaml"
        
        print(f"\n{'='*80}", flush=True)
        print(f"[RUN {idx:02d}/{total_runs:02d}] Evaluating: {exp_name}", flush=True)
        print(f" Parameters -> Seq: {sl} | Hidden: {hd} | Experts: {se} | Dropout: {drop} | LR: {lr}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        cfg = deepcopy(base_config)
        cfg['experiment']['name'] = exp_name
        
        cfg['data']['sequence_length'] = sl
        cfg['data']['forecast_history'] = sl
        cfg['model']['cgc']['expert_hidden_dim'] = hd
        cfg['model']['cgc']['shared_experts'] = se
        cfg['model']['cgc']['dropout_rate'] = drop
        cfg['model']['encoder']['hidden_dim'] = hd
        cfg['training']['learning_rate'] = lr
        cfg['training']['dropout'] = drop
        
        # Fast-Search Constraints
        cfg['training']['epochs'] = 50
        if 'early_stopping' not in cfg['training']:
            cfg['training']['early_stopping'] = {}
        cfg['training']['early_stopping']['enabled'] = True
        cfg['training']['early_stopping']['patience'] = 15
        
        save_temp_config(cfg, temp_yaml_path)
        
        try:
            print(f"[INFO] Launching Training Phase for {exp_name}...", flush=True)
            train_cmd = f"python {str(MAIN_SCRIPT)} --config {str(temp_yaml_path)} --mode train --loss_weights {FIXED_WEIGHTS}"
            run_command(train_cmd)
            
            print(f"\n[INFO] Launching Testing Phase for {exp_name}...", flush=True)
            test_cmd = f"python {str(MAIN_SCRIPT)} --config {str(temp_yaml_path)} --mode test --loss_weights {FIXED_WEIGHTS}"
            run_command(test_cmd)
            
        except Exception as e:
            print(f"\n[ERROR] Run {exp_name} failed. Details: {e}", flush=True)
            print(f"[INFO] Skipping to the next configuration...\n", flush=True)
            
        finally:
            if temp_yaml_path.exists():
                os.remove(temp_yaml_path)
                
    print(f"\n{'='*80}", flush=True)
    print("[SUCCESS] Comprehensive Hyperparameter Grid Search Completed.", flush=True)
    print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    main()