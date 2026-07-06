# ==============================================================================
# Description:
#   Formal Chapter 3 training launcher for HydroMTL_CGC.
#
# Purpose:
#   Train the final Chapter 3 models after Stage-1, Stage-2, and Stage-3
#   hyperparameter searches.
#
# Models:
#   1. STL_Q
#   2. STL_ET
#   3. Hard-MTL
#   4. MMoE
#   5. CGC
# ==============================================================================

import os
import gc
import yaml
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except Exception:
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

for _ in range(6):
    if (PROJECT_ROOT / "main.py").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "main.py").exists():
    raise FileNotFoundError("Project root not found. main.py is missing.")

MAIN_SCRIPT = PROJECT_ROOT / "main.py"
BASE_CONFIG_PATH = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"

CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
LOG_DIR = CH3_DIR / "logs"
SUMMARY_DIR = CH3_DIR / "06_summary"

CH3_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 20

FINAL_BACKBONE = {
    "sequence_length": 365,
    "hidden_dim": 64,
    "batch_size": 64,
    "learning_rate": 0.001,
}

FINAL_LOSS_WEIGHTS = {
    "streamflow": 1.0,
    "evapotranspiration": 0.1,
}

FINAL_CGC = {
    "shared_experts": 4,
    "task_experts": [4, 4],
    "expert_hidden_dim": 256,
    "temperature": 1.0,
}

MODEL_RUNS = [
    {
        "name": "ch3_stl_q_seed42",
        "folder": "01_stl_q",
        "architecture": "stl",
        "targets": ["streamflow"],
        "loss_weights": {"streamflow": 1.0},
    },
    {
        "name": "ch3_stl_et_seed42",
        "folder": "02_stl_et",
        "architecture": "stl",
        "targets": ["evapotranspiration"],
        "loss_weights": {"evapotranspiration": 1.0},
    },
    {
        "name": "ch3_hard_mtl_seed42",
        "folder": "03_hard_mtl",
        "architecture": "hps",
        "targets": ["streamflow", "evapotranspiration"],
        "loss_weights": FINAL_LOSS_WEIGHTS,
    },
    {
        "name": "ch3_mmoe_mtl_seed42",
        "folder": "04_mmoe_mtl",
        "architecture": "mmoe",
        "targets": ["streamflow", "evapotranspiration"],
        "loss_weights": FINAL_LOSS_WEIGHTS,
    },
    {
        "name": "ch3_cgc_mtl_seed42",
        "folder": "05_cgc_mtl",
        "architecture": "cgc",
        "targets": ["streamflow", "evapotranspiration"],
        "loss_weights": FINAL_LOSS_WEIGHTS,
    },
]


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: Path) -> None:
    """Save a YAML configuration file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)


def release_memory() -> None:
    """Release Python and CUDA memory."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def filter_targets(
    targets: List[Dict[str, Any]],
    selected_names: List[str],
    loss_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Keep selected targets and update loss weights."""
    selected = {name.lower() for name in selected_names}
    filtered = []

    for target in targets:
        task_name = str(target["name"]).lower()
        if task_name not in selected:
            continue

        new_target = deepcopy(target)
        new_target["loss_weight"] = float(loss_weights.get(task_name, 1.0))
        filtered.append(new_target)

    if not filtered:
        raise ValueError(f"No target matched selected names: {selected_names}")

    return filtered


def apply_common_config(base_cfg: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """Apply final Chapter 3 settings to a copied base config."""
    cfg = deepcopy(base_cfg)

    run_root = CH3_DIR / run["folder"]

    cfg["experiment"]["name"] = run["name"]
    cfg["experiment"]["save_dir"] = str(run_root)

    cfg.setdefault("reproducibility", {})
    cfg["reproducibility"]["seed"] = SEED

    cfg["data"]["sequence_length"] = FINAL_BACKBONE["sequence_length"]
    cfg["data"]["batch_size"] = FINAL_BACKBONE["batch_size"]
    cfg["data"]["targets"] = filter_targets(
        cfg["data"]["targets"],
        run["targets"],
        run["loss_weights"],
    )

    cfg["model"]["architecture"] = run["architecture"]
    cfg["model"]["encoder"]["hidden_dim"] = FINAL_BACKBONE["hidden_dim"]

    cfg["training"]["epochs"] = EPOCHS
    cfg["training"]["learning_rate"] = FINAL_BACKBONE["learning_rate"]
    cfg["training"]["batch_progress"] = False

    if "early_stopping" in cfg["training"]:
        cfg["training"]["early_stopping"]["enabled"] = True
        cfg["training"]["early_stopping"]["patience"] = EARLY_STOPPING_PATIENCE

    cfg.setdefault("experiment_tracking", {})
    cfg["experiment_tracking"]["save_predictions"] = True
    cfg["experiment_tracking"]["save_per_basin_metrics"] = True
    cfg["experiment_tracking"]["save_gate_weights"] = run["architecture"] in {"mmoe", "cgc"}
    cfg["experiment_tracking"]["save_gradient_diagnostics"] = run["architecture"] in {"hps", "mmoe", "cgc"}

    cfg.setdefault("evaluation_protocol", {})
    cfg["evaluation_protocol"]["primary_metric"] = "streamflow_nse_median"

    if run["architecture"] == "cgc":
        cfg.setdefault("model", {})
        cfg["model"].setdefault("cgc", {})
        cfg["model"]["cgc"].update(FINAL_CGC)

    return cfg


def build_loss_weight_args(weights: Dict[str, float]) -> List[str]:
    """Build command-line loss weight arguments."""
    return [f"{task}={float(weight)}" for task, weight in weights.items()]


def run_command(command: List[str], log_path: Path) -> None:
    """Run command and write output to log file."""

    env = os.environ.copy()

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}"

    with open(log_path, "w", encoding="utf-8") as log_file:

        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=log_file,
            text=True,
            env=env,
        )

        process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed. See log: {log_path}"
        )

def expected_run_dir(run: Dict[str, Any]) -> Path:
    """Return expected output directory for one run."""
    return CH3_DIR / run["folder"] / run["name"]


def main() -> None:
    base_cfg = load_yaml(BASE_CONFIG_PATH)

    print("\n" + "=" * 120)
    print("Formal Chapter 3 model training")
    print("-" * 120)
    print("Backbone : sequence_length=365, hidden_dim=64, batch_size=64, lr=0.001")
    print("MTL loss : streamflow=1.0, evapotranspiration=0.1")
    print("CGC arch : shared_experts=4, task_experts=[4,4], expert_hidden_dim=256, temperature=1.0")
    print("Models   : STL_Q, STL_ET, Hard-MTL, MMoE, CGC")
    print("=" * 120 + "\n")

    for run in MODEL_RUNS:
        run_dir = expected_run_dir(run)
        log_path = LOG_DIR / f"{run['name']}.log"
        temp_cfg_path = PROJECT_ROOT / f"temp_{run['name']}.yaml"

        if (run_dir / "validation_summary.csv").exists():
            print(f"[SKIP] Existing run detected: {run['name']}")
            continue

        print("\n" + "-" * 120)
        print(f"[RUN] {run['name']}")
        print(f"Architecture : {run['architecture']}")
        print(f"Targets      : {run['targets']}")
        print(f"Output       : {run_dir}")
        print("-" * 120)

        try:
            cfg = apply_common_config(base_cfg, run)
            save_yaml(cfg, temp_cfg_path)

            command = [
                "python",
                "-u",
                str(MAIN_SCRIPT),
                "--config",
                str(temp_cfg_path),
                "--mode",
                "train",
                "--loss_weights",
                *build_loss_weight_args(run["loss_weights"]),
            ]

            run_command(command, log_path)

        finally:
            if temp_cfg_path.exists():
                temp_cfg_path.unlink()
            release_memory()

    print("\nChapter 3 formal model training finished.")


if __name__ == "__main__":
    main()