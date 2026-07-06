# ==============================================================================
# Description:
#   Run Chapter 4 STL-Q climate-consistency controlled experiments.
#
# Purpose:
#   Provide single-task streamflow baselines for evaluating CGC robustness under
#   different train-test climate consistency groups.
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import subprocess
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"

CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
CONFIG_DIR = CH4_DIR / "configs"
LOG_DIR = CH4_DIR / "logs"
GROUP_DIR = CH4_DIR / "basin_groups"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GROUP_FILES: Dict[str, Path] = {
    "low": GROUP_DIR / "consistency_low.txt",
    "medium": GROUP_DIR / "consistency_medium.txt",
    "high": GROUP_DIR / "consistency_high.txt",
}


def load_yaml(path: Path) -> dict:
    """Load YAML file."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(obj: dict, path: Path) -> None:
    """Save YAML file."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(obj, file, sort_keys=False, allow_unicode=False)


def apply_stlq_config(cfg: dict) -> dict:
    """Convert a multi-task config into an STL-Q streamflow-only config."""
    cfg["model"]["architecture"] = "stl"

    cfg["data"]["targets"] = [
        {
            "name": "streamflow",
            "type": "regression",
            "loss_weight": 1.0,
            "constraint": "non_negative",
        }
    ]

    cfg["evaluation_protocol"]["primary_metric"] = "streamflow_nse_median"
    cfg["experiment_tracking"]["save_gate_weights"] = False

    return cfg


def build_config(group_name: str, basin_list_path: Path) -> Path:
    """Build one temporary STL-Q config for one climate-consistency group."""
    cfg = load_yaml(BASE_CONFIG)
    cfg = apply_stlq_config(cfg)

    run_name = f"ch4_consistency_{group_name}_stlq_seed42"

    cfg["experiment"]["name"] = f"formal_ch4_training_experiments/{run_name}"
    cfg["data"]["basin_list_path"] = str(basin_list_path)

    out_path = CONFIG_DIR / f"{run_name}.yaml"
    save_yaml(cfg, out_path)
    return out_path


def run_command(cmd: List[str], log_path: Path) -> None:
    """Run one command and write stdout/stderr to a log file."""
    print("Running:", " ".join(cmd))

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(f"Command failed. Check log: {log_path}")


def main() -> None:
    """Run all STL-Q climate-consistency experiments."""
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Missing base config: {BASE_CONFIG}")

    for group_name, basin_list_path in GROUP_FILES.items():
        if not basin_list_path.exists():
            raise FileNotFoundError(
                f"Missing basin list: {basin_list_path}. "
                "Run scripts/ch4/build_data_consistency_groups.py first."
            )

        config_path = build_config(group_name, basin_list_path)
        log_path = LOG_DIR / f"{config_path.stem}.log"

        cmd = [
            "python",
            "-u",
            "main.py",
            "--config",
            str(config_path),
            "--mode",
            "train",
            "--device",
            "auto",
            "--quiet_batches",
        ]

        run_command(cmd, log_path)

    print("All Chapter 4 STL-Q climate-consistency experiments completed.")


if __name__ == "__main__":
    main()