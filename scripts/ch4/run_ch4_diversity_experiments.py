# ==============================================================================
# Description:
#   Run Chapter 4 training-basin diversity experiments.
#
# Purpose:
#   Train CGC models using basin subsets with different HUC2 regional diversity
#   levels. The experiment evaluates whether hydrologically diverse training
#   basins improve multi-task model generalization.
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
    "low": GROUP_DIR / "diversity_low.txt",
    "medium": GROUP_DIR / "diversity_medium.txt",
    "high": GROUP_DIR / "diversity_high.txt",
}


def load_yaml(path: Path) -> dict:
    """Load YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    """Save YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def build_config(group_name: str, basin_list_path: Path) -> Path:
    """Build one temporary config for one basin-diversity group."""
    cfg = load_yaml(BASE_CONFIG)
    run_name = f"ch4_diversity_{group_name}_cgc_seed42"

    cfg["experiment"]["name"] = f"formal_ch4_training_experiments/{run_name}"
    cfg["data"]["basin_list_path"] = str(basin_list_path)
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
    """Run all basin-diversity experiments."""
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Missing base config: {BASE_CONFIG}")

    for group_name, basin_list_path in GROUP_FILES.items():
        if not basin_list_path.exists():
            raise FileNotFoundError(
                f"Missing basin list: {basin_list_path}. "
                "Run scripts/ch4/build_basin_diversity_groups.py first."
            )

        config_path = build_config(group_name, basin_list_path)
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

    print("All Chapter 4 basin-diversity experiments completed.")


if __name__ == "__main__":
    main()