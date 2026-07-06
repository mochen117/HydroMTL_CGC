# ==============================================================================
# Description:
#   Run Chapter 4 training-length controlled experiments.
#
# Purpose:
#   Evaluate how training data length affects CGC streamflow-ET multi-task
#   learning performance. Validation and test periods are kept unchanged.
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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    """Save YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def build_config(length_name: str, train_period: List[str]) -> Path:
    """Build one temporary config for one training-length experiment."""
    cfg = load_yaml(BASE_CONFIG)
    run_name = f"ch4_length_{length_name}_cgc_seed42"

    cfg["experiment"]["name"] = f"formal_ch4_training_experiments/{run_name}"
    cfg["data"]["train_period"] = train_period
    cfg["data"]["basin_list_path"] = None
    cfg["model"]["architecture"] = "cgc"

    out_path = CONFIG_DIR / f"{run_name}.yaml"
    save_yaml(cfg, out_path)
    return out_path


def run_command(cmd: List[str], log_path: Path) -> None:
    """Run one command and write output to a log file."""
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
    """Run all training-length experiments."""
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Missing base config: {BASE_CONFIG}")

    for length_name, train_period in TRAINING_LENGTHS.items():
        config_path = build_config(length_name, train_period)
        log_path = LOG_DIR / f"{config_path.stem}.log"

        cmd = [
            "python",
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

    print("All Chapter 4 training-length experiments completed.")


if __name__ == "__main__":
    main()