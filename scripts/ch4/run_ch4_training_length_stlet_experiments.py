# ==============================================================================
# Description:
#   Run Chapter 4 STL-ET training-length controlled experiments.
#
# Purpose:
#   Provide single-task evapotranspiration baselines for evaluating whether CGC
#   evapotranspiration performance remains beneficial under different training
#   data lengths.
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

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_LENGTHS: Dict[str, List[str]] = {
    "train_1yr": ["2010-10-01", "2011-09-30"],
    "train_3yr": ["2008-10-01", "2011-09-30"],
    "train_5yr": ["2006-10-01", "2011-09-30"],
    "train_7yr": ["2004-10-01", "2011-09-30"],
    "train_10yr": ["2001-10-01", "2011-09-30"],
}


def load_yaml(path: Path) -> dict:
    """Load YAML file."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(obj: dict, path: Path) -> None:
    """Save YAML file."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(obj, file, sort_keys=False, allow_unicode=False)


def apply_stlet_config(cfg: dict) -> dict:
    """Convert a multi-task config into an STL-ET evapotranspiration-only config."""
    cfg["model"]["architecture"] = "stl"

    cfg["data"]["targets"] = [
        {
            "name": "evapotranspiration",
            "type": "regression",
            "loss_weight": 1.0,
            "constraint": "non_negative",
        }
    ]

    cfg["evaluation_protocol"]["primary_metric"] = "evapotranspiration_nse_median"
    cfg["experiment_tracking"]["save_gate_weights"] = False

    return cfg


def build_config(length_name: str, train_period: List[str]) -> Path:
    """Build one temporary STL-ET config for one training-length experiment."""
    cfg = load_yaml(BASE_CONFIG)
    cfg = apply_stlet_config(cfg)

    run_name = f"ch4_length_{length_name}_stlet_seed42"

    cfg["experiment"]["name"] = f"formal_ch4_training_experiments/{run_name}"
    cfg["data"]["train_period"] = train_period
    cfg["data"]["basin_list_path"] = None

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
    """Run all STL-ET training-length experiments."""
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Missing base config: {BASE_CONFIG}")

    for length_name, train_period in TRAINING_LENGTHS.items():
        config_path = build_config(length_name, train_period)
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

    print("All Chapter 4 STL-ET training-length experiments completed.")


if __name__ == "__main__":
    main()