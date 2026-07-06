# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Stage 2 - Multi-task loss-weight search for CGC.
# Protocol:
#   Fixed Stage-1 backbone parameters are used.
#   Only streamflow/evapotranspiration loss weights are searched.
# Ranking:
#   1) Maximize streamflow validation median NSE
#   2) Maximize evapotranspiration validation median NSE
#   3) Maximize streamflow validation median KGE
#   4) Minimize streamflow NSE IQR
# Resume:
#   Completed trials in the leaderboard are skipped automatically.
# ==============================================================================

import os
import gc
import yaml
import hashlib
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None


# ==============================================================================
# Paths
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR
for _ in range(5):
    if (PROJECT_ROOT / "main.py").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "main.py").exists():
    raise FileNotFoundError(
        "Unable to locate project root. main.py was not found within 5 parent levels."
    )

MAIN_SCRIPT = PROJECT_ROOT / "main.py"

BASE_CONFIG_PATH = (
    PROJECT_ROOT
    / "mtl_cgc"
    / "configs"
    / "default.yaml"
)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
STAGE2_DIR = EXPERIMENTS_DIR / "stage2_mtl_weight_search"

STAGE2_DIR.mkdir(parents=True, exist_ok=True)

LEADERBOARD_CSV = (
    STAGE2_DIR
    / "stage2_mtl_weight_leaderboard.csv"
)


# ==============================================================================
# Search protocol
# ==============================================================================

SEED = 42
MAX_EPOCHS = 80
SEARCH_MAX_BASINS = 200

FIXED_BACKBONE = {
    "sequence_length": 365,
    "hidden_dim": 64,
    "batch_size": 64,
    "learning_rate": 0.001,
}

WEIGHT_CANDIDATES: List[Dict[str, float]] = [
    {"streamflow": 1.0, "evapotranspiration": 0.05},
    {"streamflow": 1.0, "evapotranspiration": 0.10},
    {"streamflow": 1.0, "evapotranspiration": 0.30},
    {"streamflow": 1.0, "evapotranspiration": 0.50},
    {"streamflow": 1.0, "evapotranspiration": 1.00},
    {"streamflow": 1.0, "evapotranspiration": 2.00},
]

# ==============================================================================
# Basic utilities
# ==============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: Path) -> None:
    """Save YAML configuration."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)


def make_trial_id(weights: Dict[str, float]) -> str:
    """Create deterministic trial id for one weight-search candidate."""
    text = (
        f"q={weights['streamflow']};"
        f"et={weights['evapotranspiration']};"
        f"seed={SEED}"
    )
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def run_command(command: List[str]) -> None:
    """Run subprocess safely without shell=True."""
    env = os.environ.copy()

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}"

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {command}")


def release_memory() -> None:
    """Release Python and CUDA cache after each trial."""
    gc.collect()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# Leaderboard utilities
# ==============================================================================

def load_leaderboard() -> pd.DataFrame:
    """Load leaderboard if it exists."""
    if LEADERBOARD_CSV.exists():
        return pd.read_csv(LEADERBOARD_CSV)

    return pd.DataFrame()


def status_rank(status: str) -> int:
    """Completed trials rank before failed/running trials."""
    mapping = {
        "completed": 0,
        "failed": 1,
        "running": 2,
    }
    return mapping.get(str(status).lower(), 9)


def completed_trial_ids() -> set:
    """Return completed trial ids from the leaderboard."""
    df = load_leaderboard()

    if df.empty or "trial_id" not in df.columns:
        return set()

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "completed"]

    return set(df["trial_id"].astype(str).tolist())


def trial_finished_on_disk(run_dir: Path) -> bool:
    """Check whether a run has valid finished artifacts on disk."""
    summary_csv = run_dir / "validation_summary.csv"
    best_model = run_dir / "best_model.pth"
    final_model = run_dir / "final_model.pth"

    return summary_csv.exists() and (best_model.exists() or final_model.exists())


def sort_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Sort leaderboard by multi-task hydrological validation quality."""
    sort_cols = []
    ascending = []

    for col, asc in [
        ("status_rank", True),
        ("Val_Q_NSE_Median", False),
        ("Val_ET_NSE_Median", False),
        ("Val_Q_KGE_Median", False),
        ("Val_Q_NSE_IQR", True),
        ("Val_Q_Fail_Rate", True),
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(asc)

    if sort_cols:
        return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return df.reset_index(drop=True)


def append_or_update_record(record: Dict[str, Any]) -> None:
    """Append or update one leaderboard record."""
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([record])

    if LEADERBOARD_CSV.exists():
        df_old = pd.read_csv(LEADERBOARD_CSV)

        if "trial_id" in df_old.columns:
            df_old = df_old[df_old["trial_id"].astype(str) != str(record["trial_id"])]

        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all = sort_leaderboard(df_all)
    df_all.to_csv(LEADERBOARD_CSV, index=False)


# ==============================================================================
# Config mutation
# ==============================================================================

def apply_stage2_config(
    base_cfg: Dict[str, Any],
    weights: Dict[str, float],
    run_name: str,
) -> Dict[str, Any]:
    """Create one CGC multi-task training config for Stage-2 search."""
    cfg = deepcopy(base_cfg)

    cfg["experiment"]["name"] = run_name
    cfg["experiment"]["save_dir"] = str(STAGE2_DIR)

    cfg.setdefault("reproducibility", {})
    cfg["reproducibility"]["seed"] = SEED

    cfg["model"]["architecture"] = "cgc"
    cfg["model"]["encoder"]["hidden_dim"] = int(FIXED_BACKBONE["hidden_dim"])

    cfg["data"]["sequence_length"] = int(FIXED_BACKBONE["sequence_length"])
    cfg["data"]["batch_size"] = int(FIXED_BACKBONE["batch_size"])

    cfg["training"]["learning_rate"] = float(FIXED_BACKBONE["learning_rate"])
    cfg["training"]["epochs"] = MAX_EPOCHS
    cfg["training"]["batch_progress"] = False

    cfg.setdefault("hyperparameter_search", {})
    cfg["hyperparameter_search"]["enabled"] = True
    cfg["hyperparameter_search"]["max_train_basins"] = SEARCH_MAX_BASINS

    # Ensure both tasks are kept and assign loss weights.
    target_names = {str(t["name"]).lower() for t in cfg["data"]["targets"]}
    required = {"streamflow", "evapotranspiration"}

    if not required.issubset(target_names):
        raise ValueError(
            "Stage-2 requires both streamflow and evapotranspiration targets."
        )

    for target in cfg["data"]["targets"]:
        task_name = str(target["name"]).lower()

        if task_name in weights:
            target["loss_weight"] = float(weights[task_name])

    cfg.setdefault("evaluation_protocol", {})
    cfg["evaluation_protocol"]["primary_metric"] = "streamflow_nse_median"

    return cfg


# ==============================================================================
# Metrics
# ==============================================================================

def compute_robustness_metrics(csv_path: Path) -> Dict[str, float]:
    """Compute basin-level robustness statistics from validation metrics."""
    result = {
        "Val_Q_NSE_IQR": np.nan,
        "Val_Q_Fail_Rate": np.nan,
        "Val_Q_Weak_Rate": np.nan,
        "Val_ET_NSE_IQR": np.nan,
        "Val_ET_Fail_Rate": np.nan,
        "Val_ET_Weak_Rate": np.nan,
    }

    if not csv_path.exists():
        return result

    df = pd.read_csv(csv_path, index_col=0)

    if "streamflow_nse" in df.columns:
        values = df["streamflow_nse"].dropna().values

        if len(values) > 0:
            result["Val_Q_NSE_IQR"] = float(np.percentile(values, 75) - np.percentile(values, 25))
            result["Val_Q_Fail_Rate"] = float(np.mean(values < 0.0))
            result["Val_Q_Weak_Rate"] = float(np.mean(values < 0.5))

    if "evapotranspiration_nse" in df.columns:
        values = df["evapotranspiration_nse"].dropna().values

        if len(values) > 0:
            result["Val_ET_NSE_IQR"] = float(np.percentile(values, 75) - np.percentile(values, 25))
            result["Val_ET_Fail_Rate"] = float(np.mean(values < 0.0))
            result["Val_ET_Weak_Rate"] = float(np.mean(values < 0.5))

    return result


def read_trial_metrics(run_dir: Path) -> Dict[str, Any]:
    """Read validation summary and per-basin robustness metrics."""
    summary_csv = run_dir / "validation_summary.csv"
    basin_csv = run_dir / "validation_per_basin_metrics.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_csv}")

    summary = pd.read_csv(summary_csv).iloc[0].to_dict()
    robust_stats = compute_robustness_metrics(basin_csv)

    return {
        "Val_Q_NSE_Median": float(summary.get("streamflow_nse_median", np.nan)),
        "Val_Q_KGE_Median": float(summary.get("streamflow_kge_median", np.nan)),
        "Val_Q_RMSE_Median": float(summary.get("streamflow_rmse_median", np.nan)),
        "Val_Q_Bias_Median": float(summary.get("streamflow_bias_median", np.nan)),
        "Val_Q_Corr_Median": float(summary.get("streamflow_corr_median", np.nan)),
        "Val_ET_NSE_Median": float(summary.get("evapotranspiration_nse_median", np.nan)),
        "Val_ET_KGE_Median": float(summary.get("evapotranspiration_kge_median", np.nan)),
        "Val_ET_RMSE_Median": float(summary.get("evapotranspiration_rmse_median", np.nan)),
        "Val_ET_Bias_Median": float(summary.get("evapotranspiration_bias_median", np.nan)),
        "Val_ET_Corr_Median": float(summary.get("evapotranspiration_corr_median", np.nan)),
        "Encoder_Grad_Sim": float(summary.get("encoder_grad_sim", np.nan)),
        **robust_stats,
    }


def make_record(
    trial_id: str,
    run_name: str,
    weights: Dict[str, float],
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    """Build one leaderboard row."""
    record = {
        "trial_id": trial_id,
        "run_name": run_name,
        "status": status,
        "status_rank": status_rank(status),
        "seed": SEED,
        "sequence_length": FIXED_BACKBONE["sequence_length"],
        "hidden_dim": FIXED_BACKBONE["hidden_dim"],
        "batch_size": FIXED_BACKBONE["batch_size"],
        "learning_rate": FIXED_BACKBONE["learning_rate"],
        "streamflow_weight": weights["streamflow"],
        "evapotranspiration_weight": weights["evapotranspiration"],
        "error": error,
    }

    if metrics:
        record.update(metrics)

    return record


def print_current_best() -> None:
    """Print current best completed configuration."""
    df = load_leaderboard()

    if df.empty or "status" not in df.columns:
        return

    df = df[df["status"].astype(str).str.lower() == "completed"]

    if df.empty:
        return

    df = df.sort_values(
        by=[
            "Val_Q_NSE_Median",
            "Val_ET_NSE_Median",
            "Val_Q_KGE_Median",
            "Val_Q_NSE_IQR",
            "Val_Q_Fail_Rate",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    best = df.iloc[0]

    print("\nCurrent best completed Stage-2 configuration")
    print("-" * 96)
    print(f"Run Name      : {best['run_name']}")
    print(f"Q Weight      : {float(best['streamflow_weight'])}")
    print(f"ET Weight     : {float(best['evapotranspiration_weight'])}")
    print(f"Q NSE Median  : {float(best['Val_Q_NSE_Median']):.4f}")
    print(f"ET NSE Median : {float(best['Val_ET_NSE_Median']):.4f}")
    print(f"Q KGE Median  : {float(best['Val_Q_KGE_Median']):.4f}")
    print(f"Q NSE IQR     : {float(best['Val_Q_NSE_IQR']):.4f}")
    print(f"Q Fail Rate   : {float(best['Val_Q_Fail_Rate']):.4f}")
    print("-" * 96 + "\n")


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(BASE_CONFIG_PATH)
    completed = completed_trial_ids()

    print("\n" + "=" * 96)
    print("Stage 2 Multi-Task Loss-Weight Search: CGC")
    print("-" * 96)
    print("Protocol : Fixed Stage-1 backbone + controlled loss-weight search")
    print("Ranking  : Q NSE -> ET NSE -> Q KGE -> Q NSE IQR -> Q Failure Rate")
    print(f"Trials   : {len(WEIGHT_CANDIDATES)}")
    print(f"Epochs   : {MAX_EPOCHS}")
    print(f"Resume   : {len(completed)} completed trial(s) found")
    print("=" * 96 + "\n")

    for idx, weights in enumerate(WEIGHT_CANDIDATES, start=1):
        trial_id = make_trial_id(weights)
        run_name = (
            f"stage2_cgc_w_q{weights['streamflow']}_"
            f"et{weights['evapotranspiration']}_{trial_id}"
        )
        run_dir = STAGE2_DIR / run_name
        temp_cfg_path = PROJECT_ROOT / f"temp_{run_name}.yaml"

        if trial_id in completed or trial_finished_on_disk(run_dir):
            print(f"[SKIP] Completed: {run_name}")
            continue

        print("\n" + "-" * 96)
        print(f"[Stage 2 | Candidate {idx}/{len(WEIGHT_CANDIDATES)}]")
        print(f"Run Name : {run_name}")
        print(f"Weights  : {weights}")
        print("-" * 96)

        running_record = make_record(
            trial_id=trial_id,
            run_name=run_name,
            weights=weights,
            status="running",
        )
        append_or_update_record(running_record)

        try:
            cfg = apply_stage2_config(base_cfg, weights, run_name)
            save_yaml(cfg, temp_cfg_path)

            train_cmd = [
                "python",
                str(MAIN_SCRIPT),
                "--config",
                str(temp_cfg_path),
                "--mode",
                "train",
                "--loss_weights",
                f"streamflow={weights['streamflow']}",
                f"evapotranspiration={weights['evapotranspiration']}",
            ]

            run_command(train_cmd)

            metrics = read_trial_metrics(run_dir)

            completed_record = make_record(
                trial_id=trial_id,
                run_name=run_name,
                weights=weights,
                status="completed",
                metrics=metrics,
            )
            append_or_update_record(completed_record)
            completed.add(trial_id)

            print("\n[COMPLETED]")
            print(f"Run Name   : {run_name}")
            print(f"Q NSE      : {metrics['Val_Q_NSE_Median']:.4f}")
            print(f"ET NSE     : {metrics['Val_ET_NSE_Median']:.4f}")
            print(f"Q KGE      : {metrics['Val_Q_KGE_Median']:.4f}")
            print(f"Q IQR      : {metrics['Val_Q_NSE_IQR']:.4f}")
            print(f"Q FailRate : {metrics['Val_Q_Fail_Rate']:.4f}")

        except Exception as exc:
            failed_record = make_record(
                trial_id=trial_id,
                run_name=run_name,
                weights=weights,
                status="failed",
                error=str(exc),
            )
            append_or_update_record(failed_record)

            print("\n[FAILED]")
            print(f"Run Name : {run_name}")
            print(f"Error    : {exc}")

        finally:
            if temp_cfg_path.exists():
                os.remove(temp_cfg_path)

            release_memory()

        print_current_best()

    print("\n" + "=" * 96)
    print("Stage 2 multi-task loss-weight search completed.")
    print(f"Leaderboard saved to: {LEADERBOARD_CSV}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()