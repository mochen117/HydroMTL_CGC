# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Stage 1 - Backbone hyperparameter search for STL-Q.
# Protocol: Hierarchical grid search for hydrological backbone calibration.
# Search order:
#   1) sequence_length
#   2) hidden_dim
#   3) batch_size
#   4) learning_rate
# Ranking:
#   1) Maximize validation median NSE
#   2) Maximize validation median KGE
#   3) Minimize validation NSE IQR
#   4) Minimize validation failure rate
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
from typing import Dict, Any, List, Tuple, Optional

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

# ==============================================================================
# Experiment Directories
# ==============================================================================

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
STAGE1_DIR = EXPERIMENTS_DIR / "stage1_backbone_search"

STAGE1_DIR.mkdir(parents=True, exist_ok=True)

LEADERBOARD_CSV = (
    STAGE1_DIR
    / "stage1_backbone_leaderboard.csv"
)


# ==============================================================================
# Search protocol
# ==============================================================================
SEED = 42
MAX_EPOCHS = 30
SEARCH_MAX_BASINS = 200

BASE_PARAMS = {
    "sequence_length": 270,
    "hidden_dim": 128,
    "batch_size": 64,
    "learning_rate": 0.001,
}

SEARCH_STAGES: List[Tuple[str, List[Any]]] = [
    ("sequence_length", [180, 270, 365]),
    ("hidden_dim", [64, 128, 256]),
    ("batch_size", [64, 128, 256]),
    ("learning_rate", [0.001, 0.0005]),
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


def make_trial_id(stage_name: str, params: Dict[str, Any]) -> str:
    """Create deterministic trial id for one hierarchical-search candidate."""
    text = (
        f"stage={stage_name};"
        f"seq={params['sequence_length']};"
        f"hidden={params['hidden_dim']};"
        f"batch={params['batch_size']};"
        f"lr={params['learning_rate']};"
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
# Resume and leaderboard
# ==============================================================================
def load_leaderboard() -> pd.DataFrame:
    """Load leaderboard if it exists."""
    if LEADERBOARD_CSV.exists():
        return pd.read_csv(LEADERBOARD_CSV)
    return pd.DataFrame()


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
    """Sort leaderboard by hydrological validation quality."""
    sort_cols = []
    ascending = []

    for col, asc in [
        ("stage_order", True),
        ("status_rank", True),
        ("Val_NSE_Median", False),
        ("Val_KGE_Median", False),
        ("Val_NSE_IQR", True),
        ("Val_NSE_Fail_Rate", True),
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(asc)

    if sort_cols:
        return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return df.reset_index(drop=True)


def status_rank(status: str) -> int:
    """Completed trials rank before failed/running trials."""
    mapping = {
        "completed": 0,
        "failed": 1,
        "running": 2,
    }
    return mapping.get(str(status).lower(), 9)


def append_or_update_record(record: Dict[str, Any]) -> None:
    """Append or update one leaderboard record."""
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)

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


def make_record(
    trial_id: str,
    run_name: str,
    stage_name: str,
    stage_order: int,
    params: Dict[str, Any],
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    """Build one leaderboard row."""
    record = {
        "trial_id": trial_id,
        "run_name": run_name,
        "stage": stage_name,
        "stage_order": stage_order,
        "status": status,
        "status_rank": status_rank(status),
        "seed": SEED,
        "sequence_length": params["sequence_length"],
        "hidden_dim": params["hidden_dim"],
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
        "error": error,
    }

    if metrics:
        record.update(metrics)

    return record


# ==============================================================================
# Config mutation
# ==============================================================================
def apply_stage1_config(
    base_cfg: Dict[str, Any],
    params: Dict[str, Any],
    run_name: str,
) -> Dict[str, Any]:
    """Create one STL-Q training config for Stage-1 search."""
    cfg = deepcopy(base_cfg)

    cfg["experiment"]["name"] = run_name
    cfg["experiment"]["save_dir"] = str(STAGE1_DIR)

    cfg.setdefault("reproducibility", {})
    cfg["reproducibility"]["seed"] = SEED

    cfg["model"]["architecture"] = "stl"
    cfg["model"]["encoder"]["hidden_dim"] = int(params["hidden_dim"])

    cfg["data"]["sequence_length"] = int(params["sequence_length"])
    cfg["data"]["batch_size"] = int(params["batch_size"])

    # Strict STL-Q search: keep only streamflow as the target task.
    cfg["data"]["targets"] = [
        t for t in cfg["data"]["targets"]
        if "streamflow" in str(t["name"]).lower()
    ]

    if not cfg["data"]["targets"]:
        raise ValueError("No streamflow target found in config.data.targets.")

    cfg["training"]["learning_rate"] = float(params["learning_rate"])
    cfg["training"]["epochs"] = MAX_EPOCHS
    cfg.setdefault("hyperparameter_search", {})
    cfg["hyperparameter_search"]["enabled"] = True
    cfg["hyperparameter_search"]["max_train_basins"] = SEARCH_MAX_BASINS
    cfg["training"]["optimizer"] = cfg["training"].get("optimizer", "adam")

    cfg["training"]["scheduler"] = {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-5,
    }

    cfg["training"]["early_stopping"] = {
        "patience": 10,
        "min_delta": 1e-4,
    }

    cfg.setdefault("evaluation_protocol", {})
    cfg["evaluation_protocol"]["primary_metric"] = "streamflow_nse_median"

    return cfg


# ==============================================================================
# Metrics
# ==============================================================================
def compute_iqr_and_failure_rate(csv_path: Path) -> Dict[str, float]:
    """Compute basin-level robustness statistics from validation metrics."""
    result = {
        "Val_NSE_IQR": np.nan,
        "Val_NSE_Fail_Rate": np.nan,
        "Val_NSE_Weak_Rate": np.nan,
        "Val_Corr_Median": np.nan,
        "Val_Bias_Median": np.nan,
    }

    if not csv_path.exists():
        return result

    df = pd.read_csv(csv_path, index_col=0)

    if "streamflow_nse" in df.columns:
        values = df["streamflow_nse"].dropna().values
        if len(values) > 0:
            result["Val_NSE_IQR"] = float(np.percentile(values, 75) - np.percentile(values, 25))
            result["Val_NSE_Fail_Rate"] = float(np.mean(values < 0.0))
            result["Val_NSE_Weak_Rate"] = float(np.mean(values < 0.5))

    if "streamflow_corr" in df.columns:
        corr_values = df["streamflow_corr"].dropna().values
        if len(corr_values) > 0:
            result["Val_Corr_Median"] = float(np.median(corr_values))

    if "streamflow_bias" in df.columns:
        bias_values = df["streamflow_bias"].dropna().values
        if len(bias_values) > 0:
            result["Val_Bias_Median"] = float(np.median(bias_values))

    return result


def read_trial_metrics(run_dir: Path) -> Dict[str, Any]:
    """Read validation summary and per-basin robustness metrics."""
    summary_csv = run_dir / "validation_summary.csv"
    basin_csv = run_dir / "validation_per_basin_metrics.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_csv}")

    summary = pd.read_csv(summary_csv).iloc[0].to_dict()
    robust_stats = compute_iqr_and_failure_rate(basin_csv)

    return {
        "Val_NSE_Median": float(summary.get("streamflow_nse_median", np.nan)),
        "Val_KGE_Median": float(summary.get("streamflow_kge_median", np.nan)),
        "Val_RMSE_Median": float(summary.get("streamflow_rmse_median", np.nan)),
        "Val_MAE_Median": float(summary.get("streamflow_mae_median", np.nan)),
        "Encoder_Grad_Sim": float(summary.get("encoder_grad_sim", np.nan)),
        **robust_stats,
    }


# ==============================================================================
# Hierarchical search logic
# ==============================================================================
def best_params_from_previous_stage(
    stage_name: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """Select best params from one completed stage."""
    df = load_leaderboard()

    if df.empty or "stage" not in df.columns or "status" not in df.columns:
        return dict(fallback)

    stage_df = df[
        (df["stage"].astype(str) == stage_name)
        & (df["status"].astype(str).str.lower() == "completed")
    ].copy()

    if stage_df.empty:
        return dict(fallback)

    stage_df = stage_df.sort_values(
        by=["Val_NSE_Median", "Val_KGE_Median", "Val_NSE_IQR", "Val_NSE_Fail_Rate"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best = stage_df.iloc[0]

    return {
        "sequence_length": int(best["sequence_length"]),
        "hidden_dim": int(best["hidden_dim"]),
        "batch_size": int(best["batch_size"]),
        "learning_rate": float(best["learning_rate"]),
    }


def build_stage_candidates(
    stage_name: str,
    candidate_values: List[Any],
    anchor_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build controlled-variable candidates for one stage."""
    candidates = []

    for value in candidate_values:
        params = dict(anchor_params)
        params[stage_name] = value
        candidates.append(params)

    return candidates


def print_stage_header(
    stage_order: int,
    stage_name: str,
    candidates: List[Dict[str, Any]],
) -> None:
    """Print stage information."""
    print("\n" + "=" * 96)
    print(f"Stage {stage_order}: search {stage_name}")
    print("-" * 96)
    for idx, params in enumerate(candidates, start=1):
        print(f"{idx:02d}. {params}")
    print("=" * 96 + "\n")


def print_current_best() -> None:
    """Print current best completed configuration."""
    df = load_leaderboard()

    if df.empty or "status" not in df.columns:
        return

    df = df[df["status"].astype(str).str.lower() == "completed"]

    if df.empty:
        return

    df = df.sort_values(
        by=["Val_NSE_Median", "Val_KGE_Median", "Val_NSE_IQR", "Val_NSE_Fail_Rate"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best = df.iloc[0]

    print("\nCurrent best completed configuration")
    print("-" * 96)
    print(f"Run Name      : {best['run_name']}")
    print(f"Stage         : {best['stage']}")
    print(f"SeqLen        : {int(best['sequence_length'])}")
    print(f"Hidden Dim    : {int(best['hidden_dim'])}")
    print(f"Batch Size    : {int(best['batch_size'])}")
    print(f"Learning Rate : {float(best['learning_rate'])}")
    print(f"NSE Median    : {float(best['Val_NSE_Median']):.4f}")
    print(f"KGE Median    : {float(best['Val_KGE_Median']):.4f}")
    print(f"NSE IQR       : {float(best['Val_NSE_IQR']):.4f}")
    print(f"Fail Rate     : {float(best['Val_NSE_Fail_Rate']):.4f}")
    print("-" * 96 + "\n")


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(BASE_CONFIG_PATH)
    completed = completed_trial_ids()

    print("\n" + "=" * 96)
    print("Stage 1 Backbone Search: STL-Q Hierarchical Grid Search")
    print("-" * 96)
    print("Protocol : Stepwise controlled-variable grid search")
    print("Order    : sequence_length -> hidden_dim -> batch_size -> learning_rate")
    print("Ranking  : Median NSE -> Median KGE -> NSE IQR -> Failure Rate")
    print(f"Trials   : 3 + 3 + 3 + 2 = 11")
    print(f"Epochs   : {MAX_EPOCHS}")
    print(f"Resume   : {len(completed)} completed trial(s) found")
    print("=" * 96 + "\n")

    anchor_params = dict(BASE_PARAMS)

    for stage_order, (stage_name, values) in enumerate(SEARCH_STAGES, start=1):
        if stage_order > 1:
            previous_stage_name = SEARCH_STAGES[stage_order - 2][0]
            anchor_params = best_params_from_previous_stage(
                previous_stage_name,
                fallback=anchor_params,
            )

        candidates = build_stage_candidates(stage_name, values, anchor_params)
        print_stage_header(stage_order, stage_name, candidates)

        for candidate_idx, params in enumerate(candidates, start=1):
            trial_id = make_trial_id(stage_name, params)
            run_name = f"stage1_stl_q_{stage_name}_{trial_id}"
            run_dir = STAGE1_DIR / run_name
            temp_cfg_path = PROJECT_ROOT / f"temp_{run_name}.yaml"

            if trial_id in completed or trial_finished_on_disk(run_dir):
                print(f"[SKIP] Completed: {run_name}")
                continue

            print("\n" + "-" * 96)
            print(f"[Stage {stage_order} | Candidate {candidate_idx}/{len(candidates)}]")
            print(f"Run Name : {run_name}")
            print(f"Params   : {params}")
            print("-" * 96)

            running_record = make_record(
                trial_id=trial_id,
                run_name=run_name,
                stage_name=stage_name,
                stage_order=stage_order,
                params=params,
                status="running",
            )
            append_or_update_record(running_record)

            try:
                cfg = apply_stage1_config(base_cfg, params, run_name)
                save_yaml(cfg, temp_cfg_path)

                train_cmd = [
                    "python",
                    str(MAIN_SCRIPT),
                    "--config",
                    str(temp_cfg_path),
                    "--mode",
                    "train",
                    "--loss_weights",
                    "streamflow=1.0",
                ]

                run_command(train_cmd)

                metrics = read_trial_metrics(run_dir)

                completed_record = make_record(
                    trial_id=trial_id,
                    run_name=run_name,
                    stage_name=stage_name,
                    stage_order=stage_order,
                    params=params,
                    status="completed",
                    metrics=metrics,
                )
                append_or_update_record(completed_record)
                completed.add(trial_id)

                print("\n[COMPLETED]")
                print(f"Run Name  : {run_name}")
                print(f"NSE       : {metrics['Val_NSE_Median']:.4f}")
                print(f"KGE       : {metrics['Val_KGE_Median']:.4f}")
                print(f"IQR       : {metrics['Val_NSE_IQR']:.4f}")
                print(f"Fail Rate : {metrics['Val_NSE_Fail_Rate']:.4f}")

            except Exception as exc:
                failed_record = make_record(
                    trial_id=trial_id,
                    run_name=run_name,
                    stage_name=stage_name,
                    stage_order=stage_order,
                    params=params,
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
    print("Stage 1 backbone hierarchical search completed.")
    print(f"Leaderboard saved to: {LEADERBOARD_CSV}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()