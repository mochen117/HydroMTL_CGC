# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Randomized Hyperparameter Search script for HydroMTL_CGC.
# Adopts standard hydrological ranges (Seq: 180/270/365, Batch: 64/100/256, etc.).
# Explores task-specific decoupling capacities (task_experts).
# Randomly samples 40 configurations from the 1296-dimensional discrete search space 
# to ensure computational efficiency while mathematically guaranteeing optimal boundaries.
# Reference: Bergstra & Bengio (2012) Random Search for Hyper-Parameter Optimization.
# ==============================================================================

import os
import yaml
import itertools
import random
import subprocess
from copy import deepcopy
from pathlib import Path

# Dynamically resolve absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
BASE_CONFIG_PATH = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"

# Define the comprehensive search space with physically justified bounds
SEARCH_SPACE = {
    'seq_len': [180, 270, 365],         # Temporal context lengths
    'batch_size': [64, 100, 256],       # Batch sizes for gradient noise control
    'hidden_dim': [64, 128, 256],       # Model capacities (Power of 2)
    'shared_experts': [4, 6, 8],        # Macro hydro-climatic zoning complexity
    'task_experts': [[2, 1], [4, 2]],   # Independent routing capacity for Q and ET
    'dropout': [0.2, 0.3, 0.4, 0.5],    # Regularization intensity
    'lr': [0.001, 0.0005]               # AdamW optimal initial steps
}

# Number of experiments to randomly sample from the full Cartesian product
NUM_SAMPLES = 40
FIXED_WEIGHTS = "streamflow=1.0 evapotranspiration=0.1"


def load_base_config(path: Path) -> dict:
    """Loads the base YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_temp_config(config_dict: dict, path: Path):
    """Saves the mutated configuration to a temporary YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def run_command(command: str):
    """Executes a shell command from the project root directory."""
    result = subprocess.run(command, shell=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command execution failed: {command}")


def main():
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Base configuration not found at {BASE_CONFIG_PATH}")

    base_config = load_base_config(BASE_CONFIG_PATH)
    
    # 1. Generate full Cartesian product of the search space
    keys = list(SEARCH_SPACE.keys())
    all_combinations = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
    total_possible = len(all_combinations)
    
    # 2. Randomly sample N configurations
    random.seed(42)  # For rigorous reproducibility in academic paper
    sampled_combos = random.sample(all_combinations, min(NUM_SAMPLES, total_possible))
    
    print(f"\n{'='*85}", flush=True)
    print(f"[INFO] Total possible configurations in search space: {total_possible}", flush=True)
    print(f"[INFO] Commencing Randomized Search on {len(sampled_combos)} selected configurations.", flush=True)
    print(f"{'='*85}\n", flush=True)
    
    for idx, combo in enumerate(sampled_combos, start=1):
        params = dict(zip(keys, combo))
        sl = params['seq_len']
        bs = params['batch_size']
        hd = params['hidden_dim']
        se = params['shared_experts']
        te = params['task_experts']
        drop = params['dropout']
        lr = params['lr']
        
        # Compact naming convention (e.g., te21 for task_experts [2, 1])
        te_str = f"{te[0]}{te[1]}"
        exp_name = f"rand_sl{sl}_bs{bs}_hd{hd}_se{se}_te{te_str}_dp{int(drop*10)}_lr{str(lr).split('.')[-1]}"
        temp_yaml_path = PROJECT_ROOT / f"temp_{exp_name}.yaml"
        
        print(f"\n{'='*85}", flush=True)
        print(f" [RUN {idx:02d}/{len(sampled_combos):02d}] Evaluating: {exp_name}", flush=True)
        print(f" Params -> Seq: {sl} | Batch: {bs} | Hidden: {hd} | Shared Exp: {se} | Task Exp: {te} | Drop: {drop} | LR: {lr}", flush=True)
        print(f"{'='*85}\n", flush=True)
        
        # 3. Mutate the configuration dict safely based on the search space
        cfg = deepcopy(base_config)
        cfg['experiment']['name'] = exp_name
        
        cfg['data']['sequence_length'] = sl
        cfg['data']['forecast_history'] = sl
        cfg['data']['batch_size'] = bs
        
        cfg['model']['cgc']['expert_hidden_dim'] = hd
        cfg['model']['cgc']['shared_experts'] = se
        cfg['model']['cgc']['task_experts'] = te
        cfg['model']['cgc']['dropout_rate'] = drop
        cfg['model']['encoder']['hidden_dim'] = hd
        
        cfg['training']['learning_rate'] = lr
        cfg['training']['dropout'] = drop
        
        # ====================================================================
        # Fast-Search Constraints (Max 50 epochs, early stop at 15)
        # ====================================================================
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
                
    print(f"\n{'='*85}", flush=True)
    print("[SUCCESS] Randomized Hyperparameter Search Completed.")
    print(f"{'='*85}\n", flush=True)


if __name__ == "__main__":
    main()