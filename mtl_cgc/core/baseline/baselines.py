# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Automated module for executing baseline ablation studies.
# Evaluates STL-Q, STL-ET, Hard-MTL, and MMoE.
# Aligns command executions to list-based subprocesses to prevent shell injections.
# ==============================================================================

import os
import yaml
import subprocess
from copy import deepcopy
from pathlib import Path

# Resolve project root dynamically by searching upwards for main.py to prevent path offsets
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
for _ in range(5):
    if (PROJECT_ROOT / "main.py").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

MAIN_SCRIPT = PROJECT_ROOT / "main.py"
BASE_CONFIG_PATH = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"


def load_base_config(path: Path) -> dict:
    """Loads the base YAML configuration."""
    if not path.exists():
        raise FileNotFoundError(f"[FATAL] Base configuration not found at {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_temp_config(config_dict: dict, path: Path):
    """Saves the mutated configuration to a temporary YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def run_command(command_list: list):
    """Executes a subprocess list command robustly from the project root directory without shell=True."""
    result = subprocess.run(command_list, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command execution failed with return code {result.returncode}:\n{command_list}")


def main():
    base_config = load_base_config(BASE_CONFIG_PATH)
    FIXED_MTL_WEIGHTS = ["streamflow=1.0", "evapotranspiration=0.1"]
    
    experiments = []

    # --- 1. STL (Single Task: Streamflow Baseline) ---
    cfg_stl = deepcopy(base_config)
    cfg_stl['experiment']['name'] = "baseline_stl"
    cfg_stl['model']['architecture'] = "stl"
    experiments.append({
        "name": "Single-Task Learning (STL)",
        "cfg": cfg_stl,
        "weights": ["streamflow=1.0", "evapotranspiration=0.0"]
    })

    # --- 2. HPS (Hard Parameter Sharing) ---
    cfg_hps = deepcopy(base_config)
    cfg_hps['experiment']['name'] = "baseline_hps"
    cfg_hps['model']['architecture'] = "hps"
    experiments.append({
        "name": "Hard Parameter Sharing (HPS)",
        "cfg": cfg_hps,
        "weights": FIXED_MTL_WEIGHTS
    })

    # --- 3. MMoE (Multi-gate Mixture-of-Experts) ---
    cfg_mmoe = deepcopy(base_config)
    cfg_mmoe['experiment']['name'] = "baseline_mmoe"
    cfg_mmoe['model']['architecture'] = "mmoe"
    experiments.append({
        "name": "MMoE (Multi-gate Mixture-of-Experts)",
        "cfg": cfg_mmoe,
        "weights": FIXED_MTL_WEIGHTS
    })

    # --- 4. CGC (Customized Gate Control) ---
    cfg_cgc = deepcopy(base_config)
    cfg_cgc['experiment']['name'] = "baseline_cgc"
    cfg_cgc['model']['architecture'] = "cgc"
    experiments.append({
        "name": "Customized Gate Control (CGC)",
        "cfg": cfg_cgc,
        "weights": FIXED_MTL_WEIGHTS
    })

    print(f"\n{'='*85}")
    print(f"[INFO] Commencing Comprehensive Baseline Ablation Studies ({len(experiments)} Models).")
    print(f"       Project Root: {PROJECT_ROOT}")
    print(f"{'='*85}\n")

    for idx, exp in enumerate(experiments, start=1):
        exp_title = exp["name"]
        cfg = exp["cfg"]
        weights = exp["weights"]
        
        exp_dir_name = cfg['experiment']['name']
        temp_yaml_path = PROJECT_ROOT / f"temp_{exp_dir_name}.yaml"
        
        print(f"\n{'='*85}")
        print(f" [RUN {idx:02d}/{len(experiments):02d}] Evaluating Architecture: {exp_title}")
        print(f" Loss Weights Alignment: {weights}")
        print(f"{'='*85}\n")
        
        save_temp_config(cfg, temp_yaml_path)
        
        try:
            # Enforce non-shell list based execution for enhanced pipeline safety
            print(f"[INFO] Launching Training Phase for {exp_title}...")
            train_cmd = ["python", str(MAIN_SCRIPT), "--config", str(temp_yaml_path), "--mode", "train", "--loss_weights"] + weights
            run_command(train_cmd)
            
            print(f"\n[INFO] Launching Testing Phase & NetCDF/CSV Export...")
            test_cmd = ["python", str(MAIN_SCRIPT), "--config", str(temp_yaml_path), "--mode", "test", "--loss_weights"] + weights
            run_command(test_cmd)
            
            print(f"\n[SUCCESS] Baseline '{exp_title}' executed successfully.")
            
        except Exception as e:
            print(f"\n[ERROR] Architecture {exp_title} failed during execution. Details: {e}")
            print(f"[INFO] Proceeding to the next baseline...\n")
            
        finally:
            if temp_yaml_path.exists():
                try:
                    os.remove(temp_yaml_path)
                except Exception:
                    pass
                
    print(f"\n{'='*85}")
    print("[SUCCESS] All baseline ablation experiments completed and artifacts saved.")
    print(f"{'='*85}\n")

if __name__ == "__main__":
    main()