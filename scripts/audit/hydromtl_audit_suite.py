#!/usr/bin/env python3
"""HydroMTL_CGC audit suite.

Read-only audit utilities for training protocol, existing histories,
units/scalers, gradient parameter groups, and checkpoints.

Typical usage from the project root:

    python hydromtl_audit_suite.py active-code --project-root . --out-dir experiments/audit_code
    python hydromtl_audit_suite.py static-protocol --project-root . --out-dir experiments/audit_code
    python hydromtl_audit_suite.py histories --experiments-root experiments --out-dir experiments/audit_ch3
    python hydromtl_audit_suite.py units --project-root . --data-root <netcdf_root> --out-dir experiments/audit_units
    python hydromtl_audit_suite.py gradients --project-root . --config mtl_cgc/configs/default.yaml --out-dir experiments/audit_gradients
    python hydromtl_audit_suite.py checkpoints --experiments-root experiments/formal_ch3_modeling --out-dir experiments/audit_checkpoints
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROTOCOL_PATTERNS = [
    "early_stop.step",
    "scheduler.step",
    "ckpt.step",
    "train_epoch",
    "diagnostic_interval",
    "gradient_batch_interval",
    "compute_gradient_similarity",
    "encoder_grad_sim",
]

STATIC_CHECKS = [
    {
        "id": "MONITOR_001",
        "severity": "high",
        "pattern": "scheduler.step(val_loss)",
        "message": "Scheduler is hard-coded to val_loss. Use one unified monitor value.",
    },
    {
        "id": "MONITOR_002",
        "severity": "high",
        "pattern": "early_stop.step(val_loss",
        "message": "Early stopping is hard-coded to val_loss. Use one unified monitor value.",
    },
    {
        "id": "MONITOR_003",
        "severity": "high",
        "pattern": "ckpt.step(trainer.model, epoch, -current_metric",
        "message": "Checkpoint may be selected by a metric while scheduler/early stop use val_loss.",
    },
    {
        "id": "GRAD_001",
        "severity": "medium",
        "pattern": "diagnostic_interval",
        "message": "Gradient diagnostics are epoch-throttled. Confirm this is intended.",
    },
    {
        "id": "GRAD_002",
        "severity": "medium",
        "pattern": "batch_idx % 1000",
        "message": "Gradient diagnostics are batch-throttled every 1000 batches; likely too sparse.",
    },
    {
        "id": "GRAD_003",
        "severity": "high",
        "pattern": "except Exception:\n                    pass",
        "message": "Gradient diagnostic failures may be silently ignored.",
    },
    {
        "id": "GRAD_004",
        "severity": "medium",
        "pattern": "encoder_grad_sim",
        "message": "Only encoder_grad_sim may be stored; shared expert diagnostics may be lost.",
    },
    {
        "id": "SCALER_001",
        "severity": "high",
        "pattern": "np.power(10.0, q_log) - self.LOG_EPSILON) ** 2",
        "message": "Streamflow inverse transform can square negative low-flow values. Clamp before squaring.",
    },
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(args: Sequence[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"[ERROR] {' '.join(args)} failed: {exc}"


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def import_module_file(module_name: str, project_root: Path) -> str:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        module: ModuleType = __import__(module_name, fromlist=["__name__"])
        return inspect.getfile(module)
    except Exception as exc:
        return f"[IMPORT_FAILED] {module_name}: {exc}"


def iter_source_files(project_root: Path) -> Iterable[Path]:
    suffixes = {".py", ".yaml", ".yml", ".sh"}
    ignored = {".git", "__pycache__", ".ipynb_checkpoints"}
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in ignored for part in path.parts):
            continue
        yield path


def scan_patterns(project_root: Path, patterns: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in iter_source_files(project_root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        rel = str(path.relative_to(project_root))
        for line_no, line in enumerate(lines, start=1):
            for pattern in patterns:
                if pattern in line:
                    rows.append({"file": rel, "line": line_no, "pattern": pattern, "text": line.strip()})
    return rows


def cmd_active_code(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    out_dir = ensure_dir(args.out_dir.resolve())
    module_files = {
        "trainer": import_module_file("mtl_cgc.core.training.trainer", project_root),
        "callbacks": import_module_file("mtl_cgc.core.training.callbacks", project_root),
        "data_scalers": import_module_file("mtl_cgc.data.data_scalers", project_root),
        "data_loaders": import_module_file("mtl_cgc.data.data_loaders", project_root),
        "data_sets": import_module_file("mtl_cgc.data.data_sets", project_root),
        "evaluator": import_module_file("mtl_cgc.core.evaluation.evaluator", project_root),
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "git_commit": run_command(["git", "rev-parse", "HEAD"], project_root),
        "git_status_short": run_command(["git", "status", "--short"], project_root),
        "module_files": module_files,
    }
    (out_dir / "active_code_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = scan_patterns(project_root, PROTOCOL_PATTERNS)
    write_csv(out_dir / "protocol_pattern_locations.csv", rows, ["file", "line", "pattern", "text"])
    md = [
        "# Active Code Manifest",
        "",
        f"- Project root: `{project_root}`",
        f"- Git commit: `{manifest['git_commit']}`",
        "",
        "## Imported module files",
    ]
    for key, value in module_files.items():
        md.append(f"- {key}: `{value}`")
    md += ["", "## Git status --short", "```text", manifest["git_status_short"], "```", "", f"Pattern CSV: `{out_dir / 'protocol_pattern_locations.csv'}`"]
    (out_dir / "active_code_manifest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Active-code audit written to {out_dir}")


def find_exact_pattern_locations(project_root: Path, pattern: str) -> List[Tuple[str, int, str]]:
    rows: List[Tuple[str, int, str]] = []
    first_token = pattern.splitlines()[0]
    for path in iter_source_files(project_root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        text = "\n".join(lines)
        if pattern not in text:
            continue
        for line_no, line in enumerate(lines, start=1):
            if first_token in line:
                rows.append((str(path.relative_to(project_root)), line_no, line.strip()))
    return rows


def cmd_static_protocol(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    out_dir = ensure_dir(args.out_dir.resolve())
    findings: List[Dict[str, str]] = []
    for check in STATIC_CHECKS:
        locations = find_exact_pattern_locations(project_root, check["pattern"])
        if not locations:
            findings.append({
                "check_id": check["id"], "severity": check["severity"], "status": "not_found",
                "file": "", "line": "", "text": "", "message": check["message"],
            })
            continue
        for file_name, line_no, text in locations:
            findings.append({
                "check_id": check["id"], "severity": check["severity"], "status": "found",
                "file": file_name, "line": str(line_no), "text": text, "message": check["message"],
            })
    write_csv(out_dir / "training_protocol_static_findings.csv", findings)
    high = sum(1 for row in findings if row["severity"] == "high" and row["status"] == "found")
    medium = sum(1 for row in findings if row["severity"] == "medium" and row["status"] == "found")
    md = [
        "# Training Protocol Static Audit",
        "",
        f"- Project root: `{project_root}`",
        f"- High-severity findings: {high}",
        f"- Medium-severity findings: {medium}",
        "",
        "## Recommended actions",
        "1. Replace scheduler/checkpoint/early-stop calls with one unified `monitor_value`.",
        "2. Move gradient diagnostics to fixed batches and save all shared-parameter metrics.",
        "3. Replace silent diagnostic exceptions with warnings or fail-fast behavior.",
        "4. Clamp `sqrt_q_ratio` before squaring in streamflow inverse transform.",
    ]
    (out_dir / "training_protocol_static_findings.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Static-protocol audit written to {out_dir}")
    if high:
        print(f"[WARN] Found {high} high-severity issue(s).")


def safe_epoch_at_min(df: pd.DataFrame, column: str) -> Optional[int]:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return None
    return int(df.loc[values.idxmin(), "epoch"])


def safe_epoch_at_max(df: pd.DataFrame, column: str) -> Optional[int]:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return None
    return int(df.loc[values.idxmax(), "epoch"])


def lr_drop_epochs(df: pd.DataFrame) -> str:
    if "learning_rate" not in df.columns or "epoch" not in df.columns:
        return ""
    lr = pd.to_numeric(df["learning_rate"], errors="coerce")
    epochs = pd.to_numeric(df["epoch"], errors="coerce")
    drops = epochs[lr.diff() < 0].dropna().astype(int).tolist()
    return ",".join(map(str, drops))


def audit_one_history(path: Path, root: Path) -> Dict[str, object]:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"{path} does not contain an epoch column.")
    result: Dict[str, object] = {
        "run_dir": str(path.parent.relative_to(root)),
        "history_path": str(path.relative_to(root)),
        "num_epochs_recorded": int(len(df)),
        "final_epoch": int(pd.to_numeric(df["epoch"], errors="coerce").max()),
        "min_val_loss_epoch": safe_epoch_at_min(df, "val_loss"),
        "lr_drop_epochs": lr_drop_epochs(df),
    }
    for col in [c for c in df.columns if c.endswith("_nse_median")]:
        result[f"best_{col}_epoch"] = safe_epoch_at_max(df, col)
        result[f"best_{col}_value"] = float(pd.to_numeric(df[col], errors="coerce").max())
    if {"streamflow_nse_median", "evapotranspiration_nse_median"}.issubset(df.columns):
        joint = (pd.to_numeric(df["streamflow_nse_median"], errors="coerce") + pd.to_numeric(df["evapotranspiration_nse_median"], errors="coerce")) / 2.0
        result["best_joint_q_et_nse_epoch"] = int(df.loc[joint.idxmax(), "epoch"])
        result["best_joint_q_et_nse_value"] = float(joint.max())
    if "is_best" in df.columns:
        mask = df["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])
        result["is_best_epochs"] = ",".join(map(str, df.loc[mask, "epoch"].astype(int).tolist()))
    if "best_epoch" in df.columns:
        best_series = pd.to_numeric(df["best_epoch"], errors="coerce").dropna()
        if not best_series.empty:
            result["last_reported_best_epoch"] = int(best_series.iloc[-1])
    grad_cols = [col for col in df.columns if "grad" in col.lower()]
    result["gradient_columns"] = ",".join(grad_cols)
    for col in grad_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        result[f"{col}_non_nan_count"] = int(values.notna().sum())
        result[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
    if isinstance(result.get("min_val_loss_epoch"), int):
        result["final_minus_min_loss_epoch"] = int(result["final_epoch"] - int(result["min_val_loss_epoch"]))
    return result


def cmd_histories(args: argparse.Namespace) -> None:
    root = args.experiments_root.resolve()
    out_dir = ensure_dir(args.out_dir.resolve())
    histories = sorted(root.rglob("training_history.csv"))
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    for path in histories:
        try:
            rows.append(audit_one_history(path, root))
        except Exception as exc:
            errors.append({"history_path": str(path), "error": str(exc)})
    pd.DataFrame(rows).to_csv(out_dir / "training_history_audit.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "training_history_audit_errors.csv", index=False)
    md = [
        "# Training History Audit",
        "",
        f"- Experiments root: `{root}`",
        f"- Histories found: {len(histories)}",
        f"- Histories audited: {len(rows)}",
        f"- Errors: {len(errors)}",
        "",
        "Check whether min-val-loss epoch, max-task-NSE epoch, joint-NSE epoch, and reported best_epoch agree.",
    ]
    (out_dir / "training_history_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Audited {len(rows)} histories -> {out_dir / 'training_history_audit.csv'}")


def find_variable(ds, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for name in ds.data_vars:
        lower = name.lower()
        if any(candidate.lower() in lower for candidate in candidates):
            return name
    return None


def numeric_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"finite_count": 0, "nan_ratio": float(np.isnan(arr).mean()), "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan}
    valid = arr[finite]
    return {
        "finite_count": int(finite.sum()), "nan_ratio": float(np.isnan(arr).mean()),
        "min": float(np.min(valid)), "max": float(np.max(valid)),
        "mean": float(np.mean(valid)), "median": float(np.median(valid)),
    }


def median_valid_interval_days(ds, variable: str) -> float:
    if "time" not in ds.coords:
        return np.nan
    values = np.asarray(ds[variable].values, dtype=float)
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1)
        valid = np.isfinite(values).any(axis=1)
    else:
        valid = np.isfinite(values)
    times = pd.to_datetime(ds["time"].values)
    valid_times = times[valid[: len(times)]]
    if len(valid_times) < 2:
        return np.nan
    intervals = np.diff(valid_times.values).astype("timedelta64[D]").astype(int)
    return float(np.median(intervals))


def audit_netcdf_file(path: Path) -> Dict[str, object]:
    import xarray as xr
    row: Dict[str, object] = {"file": str(path)}
    try:
        ds = xr.open_dataset(path)
    except Exception as exc:
        row["error"] = f"open_failed: {exc}"
        return row
    try:
        q_var = find_variable(ds, ["streamflow", "qobs", "q", "discharge"])
        et_var = find_variable(ds, ["evapotranspiration", "et", "aet", "mod16"])
        row["streamflow_var"] = q_var or ""
        row["et_var"] = et_var or ""
        if q_var:
            row.update({f"streamflow_{k}": v for k, v in numeric_summary(ds[q_var].values).items()})
            row["streamflow_attrs"] = json.dumps(dict(ds[q_var].attrs), ensure_ascii=False)
        if et_var:
            row.update({f"et_{k}": v for k, v in numeric_summary(ds[et_var].values).items()})
            row["et_attrs"] = json.dumps(dict(ds[et_var].attrs), ensure_ascii=False)
            row["et_median_valid_interval_days"] = median_valid_interval_days(ds, et_var)
        for required in ["area_gages2", "p_mean"]:
            row[f"has_{required}"] = bool(required in ds.data_vars or required in ds.coords or required in ds.attrs)
    finally:
        ds.close()
    return row


def audit_scaler_source(project_root: Path) -> List[Dict[str, str]]:
    scaler_path = project_root / "mtl_cgc" / "data" / "data_scalers.py"
    if not scaler_path.exists():
        return [{"check": "scaler_file", "status": "missing", "detail": str(scaler_path)}]
    text = scaler_path.read_text(encoding="utf-8", errors="replace")
    checks: List[Dict[str, str]] = []
    def record(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "status": "ok" if ok else "warning", "detail": detail})
    record("cfs_to_m3s_factor", "0.0283168" in text, "Expected if raw streamflow is cfs.")
    record("low_flow_inverse_clamp", "sqrt_q_ratio" in text and "np.maximum" in text, "Expected clamp before squaring.")
    record("dangerous_low_flow_square_pattern", "np.power(10.0, q_log) - self.LOG_EPSILON) ** 2" not in text, "Avoid squaring negative sqrt-flow term.")
    record("returns_m3s", "q_m3s" in text, "Expected if final streamflow unit is m3/s.")
    return checks


def cmd_units(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    out_dir = ensure_dir(args.out_dir.resolve())
    nc_files = sorted(data_root.rglob("*.nc"))[: args.max_files]
    rows = [audit_netcdf_file(path) for path in nc_files]
    pd.DataFrame(rows).to_csv(out_dir / "netcdf_unit_audit.csv", index=False)
    pd.DataFrame(audit_scaler_source(project_root)).to_csv(out_dir / "scaler_source_audit.csv", index=False)
    md = [
        "# Unit and Scaler Audit", "", f"- Project root: `{project_root}`", f"- Data root: `{data_root}`",
        f"- NetCDF files inspected: {len(nc_files)}", "", "Key checks:",
        "- Streamflow raw unit must match scaler assumption.",
        "- Streamflow inverse transform must clamp before squaring low-flow values.",
        "- ET must be confirmed as daily mm/day; 8-day products need aggregation/masking.",
        "- `area_gages2` and `p_mean` must be present and positive.",
    ]
    (out_dir / "unit_and_scaler_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Unit/scaler audit written to {out_dir}")


def to_edict(obj):
    try:
        from easydict import EasyDict as edict
    except Exception:
        return obj
    if isinstance(obj, dict):
        return edict({key: to_edict(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [to_edict(value) for value in obj]
    return obj


def infer_model_type(config) -> str:
    for attr_path in [("model", "architecture"), ("model", "type")]:
        parent = getattr(config, attr_path[0], None)
        value = getattr(parent, attr_path[1], None) if parent is not None else None
        if value:
            return str(value).lower()
    value = getattr(config, "architecture", None)
    return str(value).lower() if value else "unknown"


def group_parameters(name: str, model_type: str) -> List[str]:
    groups: List[str] = []
    if "encoder" in name or "lstm" in name:
        groups.append("current_encoder_rule")
    if "shared_expert" in name:
        groups.append("recommended_shared_experts")
    if "encoder" in name and "head" not in name and "tower" not in name:
        groups.append("recommended_input_encoder")
    if "gate" in name:
        groups.append("gate_parameters_review_only")
    if any(token in name for token in ["task_specific", "private", "specific_expert"]):
        groups.append("task_private_excluded_from_conflict")
    if model_type in {"hps", "hard", "hard_mtl"} and ("encoder" in name or "lstm" in name):
        groups.append("recommended_hard_shared_encoder")
    return groups


def cmd_gradients(args: argparse.Namespace) -> None:
    import yaml
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    out_dir = ensure_dir(args.out_dir.resolve())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from mtl_cgc.core.cgc_models.mtl_model import build_model
    config = to_edict(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    model_type = infer_model_type(config)
    model = build_model(config)
    rows: List[Dict[str, object]] = []
    for name, parameter in model.named_parameters():
        groups = group_parameters(name, model_type)
        rows.append({
            "parameter_name": name,
            "num_parameters": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
            "groups": ";".join(groups),
            "is_current_encoder_rule": "current_encoder_rule" in groups,
            "is_recommended_shared_conflict": any(g in groups for g in ["recommended_shared_experts", "recommended_input_encoder", "recommended_hard_shared_encoder"]),
            "is_private_excluded": "task_private_excluded_from_conflict" in groups,
        })
    write_csv(out_dir / "gradient_parameter_group_audit.csv", rows)
    total = sum(int(r["num_parameters"]) for r in rows if r["requires_grad"])
    current = sum(int(r["num_parameters"]) for r in rows if r["is_current_encoder_rule"] and r["requires_grad"])
    recommended = sum(int(r["num_parameters"]) for r in rows if r["is_recommended_shared_conflict"] and r["requires_grad"])
    md = [
        "# Gradient Parameter Group Audit", "", f"- Config: `{config_path}`", f"- Inferred model type: `{model_type}`",
        f"- Trainable parameters: {total:,}", f"- Current encoder-rule parameters: {current:,}",
        f"- Recommended shared-conflict parameters: {recommended:,}", "", "Inspect CSV to ensure private experts are excluded from shared-gradient conflict analysis.",
    ]
    (out_dir / "gradient_parameter_group_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Gradient group audit written to {out_dir}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_summary(path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {"exists": path.exists(), "path": str(path), "size_bytes": path.stat().st_size if path.exists() else 0, "sha256": sha256_file(path) if path.exists() else "", "format": "", "num_keys": 0, "epoch": "", "monitor_name": "", "monitor_value": ""}
    if not path.exists():
        return row
    import torch
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
        row.update({"format": "full_checkpoint", "epoch": obj.get("epoch", ""), "monitor_name": obj.get("monitor_name", ""), "monitor_value": obj.get("monitor_value", "")})
    elif isinstance(obj, dict):
        state = obj
        row["format"] = "raw_state_dict"
    else:
        state = {}
        row["format"] = type(obj).__name__
    row["num_keys"] = len(state)
    return row


def compare_state_dicts(best_path: Path, final_path: Path) -> Dict[str, object]:
    result: Dict[str, object] = {"common_keys": 0, "best_only_keys": 0, "final_only_keys": 0, "mean_abs_parameter_difference_sample": np.nan}
    if not best_path.exists() or not final_path.exists():
        return result
    import torch
    best_obj = torch.load(best_path, map_location="cpu")
    final_obj = torch.load(final_path, map_location="cpu")
    best_state = best_obj.get("model_state_dict", best_obj) if isinstance(best_obj, dict) else {}
    final_state = final_obj.get("model_state_dict", final_obj) if isinstance(final_obj, dict) else {}
    best_keys = set(best_state.keys())
    final_keys = set(final_state.keys())
    common = sorted(best_keys & final_keys)
    result.update({"common_keys": len(common), "best_only_keys": len(best_keys - final_keys), "final_only_keys": len(final_keys - best_keys)})
    diffs: List[float] = []
    for key in common[:50]:
        a, b = best_state[key], final_state[key]
        if hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape and a.numel() > 0:
            diffs.append(float(torch.mean(torch.abs(a.float() - b.float())).item()))
    if diffs:
        result["mean_abs_parameter_difference_sample"] = sum(diffs) / len(diffs)
    return result


def cmd_checkpoints(args: argparse.Namespace) -> None:
    root = args.experiments_root.resolve()
    out_dir = ensure_dir(args.out_dir.resolve())
    run_dirs = sorted({path.parent for path in root.rglob("*.pth")})
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        best_path = run_dir / "best_model.pth"
        final_path = run_dir / "final_model.pth"
        best = checkpoint_summary(best_path)
        final = checkpoint_summary(final_path)
        row: Dict[str, object] = {"run_dir": str(run_dir.relative_to(root))}
        row.update({f"best_{key}": value for key, value in best.items()})
        row.update({f"final_{key}": value for key, value in final.items()})
        row.update(compare_state_dicts(best_path, final_path))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "checkpoint_audit.csv", index=False)
    md = ["# Checkpoint Audit", "", f"- Experiments root: `{root}`", f"- Run directories with checkpoints: {len(run_dirs)}", "", "Raw state_dict checkpoints cannot reveal epoch or monitor metadata. Future experiments should save full checkpoints."]
    (out_dir / "checkpoint_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Checkpoint audit written to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("active-code")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/audit_code"))
    p.set_defaults(func=cmd_active_code)
    p = sub.add_parser("static-protocol")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/audit_code"))
    p.set_defaults(func=cmd_static_protocol)
    p = sub.add_parser("histories")
    p.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/audit_ch3"))
    p.set_defaults(func=cmd_histories)
    p = sub.add_parser("units")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/audit_units"))
    p.add_argument("--max-files", type=int, default=20)
    p.set_defaults(func=cmd_units)
    p = sub.add_parser("gradients")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/audit_gradients"))
    p.set_defaults(func=cmd_gradients)
    p = sub.add_parser("checkpoints")
    p.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/audit_checkpoints"))
    p.set_defaults(func=cmd_checkpoints)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
