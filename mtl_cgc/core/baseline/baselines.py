# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Automated module for executing baseline ablation studies.
# Dynamically generates STL-Q, STL-ET, and Hard-MTL (Zero Task-Experts) models
# to benchmark against the HydroMTL_CGC architecture.
# ==============================================================================

import os
import yaml
import subprocess
from copy import deepcopy
from pathlib import Path

# Dynamically resolve absolute paths based on the new deep directory structure
# File: mtl_cgc/core/baseline/baselines.py
# Root: Parent x 3 -> core -> mtl_cgc -> HydroMTL_CGC
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

MAIN_SCRIPT = PROJECT_ROOT / "main.py"
BASE_CONFIG_PATH = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"


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
    
    # --------------------------------------------------------------------------
    # Define Baseline Experiments
    # --------------------------------------------------------------------------
    experiments =[]

    # 1. STL-Q (Single Task: Streamflow)
    cfg_stl_q = deepcopy(base_config)
    cfg_stl_q['experiment']['name'] = "baseline_stl_q"
    # Isolate streamflow target
    cfg_stl_q['data']['targets'] =[t for t in cfg_stl_q['data']['targets'] if 'streamflow' in t['name'].lower()]
    # Dummy expert value since it's a single task
    cfg_stl_q['model']['cgc']['task_experts'] =[2] 
    experiments.append({
        "name": "STL-Streamflow",
        "cfg": cfg_stl_q,
        "weights": "streamflow=1.0"
    })

    # 2. STL-ET (Single Task: Evapotranspiration)
    cfg_stl_et = deepcopy(base_config)
    cfg_stl_et['experiment']['name'] = "baseline_stl_et"
    # Isolate evapotranspiration target
    cfg_stl_et['data']['targets'] =[t for t in cfg_stl_et['data']['targets'] if 'evapotranspiration' in t['name'].lower()]
    cfg_stl_et['model']['cgc']['task_experts'] = [2]
    experiments.append({
        "name": "STL-Evapotranspiration",
        "cfg": cfg_stl_et,
        "weights": "evapotranspiration=1.0"
    })

    # 3. Hard-MTL (Traditional Hard Parameter Sharing)
    cfg_hard = deepcopy(base_config)
    cfg_hard['experiment']['name'] = "baseline_hard_mtl"
    # CRITICAL: Setting task-specific experts to 0 forces the model to degenerate 
    # into a conventional hard-sharing network (utilizing ONLY shared experts).
    cfg_hard['model']['cgc']['task_experts'] = [0, 0]
    experiments.append({
        "name": "Hard-Sharing MTL",
        "cfg": cfg_hard,
        # Maintain consistent weights with the primary CGC model
        "weights": "streamflow=1.0 evapotranspiration=0.1" 
    })

    # --------------------------------------------------------------------------
    # Execute Experiments
    # --------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"[INFO] Commencing Baseline Ablation Studies ({len(experiments)} Models).")
    print(f"       Project Root: {PROJECT_ROOT}")
    print(f"{'='*80}\n")

    for idx, exp in enumerate(experiments, start=1):
        exp_title = exp["name"]
        cfg = exp["cfg"]
        weights = exp["weights"]
        
        exp_dir_name = cfg['experiment']['name']
        temp_yaml_path = PROJECT_ROOT / f"temp_{exp_dir_name}.yaml"
        
        print(f"\n{'='*80}")
        print(f" [RUN {idx:02d}/{len(experiments):02d}] Evaluating Baseline: {exp_title}")
        print(f" Loss Weights: {weights}")
        print(f"{'='*80}\n")
        
        save_temp_config(cfg, temp_yaml_path)
        
        try:
            # Execute Training Phase
            print(f"[INFO] Launching Training Phase for {exp_title}...")
            train_cmd = f"python {str(MAIN_SCRIPT)} --config {str(temp_yaml_path)} --mode train --loss_weights {weights}"
            run_command(train_cmd)
            
            # Execute Testing Phase
            print(f"\n[INFO] Launching Testing Phase & NetCDF Export...")
            test_cmd = f"python {str(MAIN_SCRIPT)} --config {str(temp_yaml_path)} --mode test --loss_weights {weights}"
            run_command(test_cmd)
            
        except Exception as e:
            print(f"\n[ERROR] Baseline {exp_title} failed. Details: {e}")
        finally:
            # Ensure workspace remains clean
            if temp_yaml_path.exists():
                os.remove(temp_yaml_path)
                
    print(f"\n{'='*80}")
    print("[SUCCESS] All baseline experiments completed successfully.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()