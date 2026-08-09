#!/usr/bin/env python3
"""Collect and compare Chapter 4 Q-SSM experiment outputs.

The evaluator is file-layout tolerant. It searches experiment folders for
summary/per-basin CSV files and computes model comparisons where possible.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def infer_protocol_model(exp_name: str) -> dict:
    s = exp_name.lower()
    protocol = "q_to_ssm" if "ch4a" in s else "pub" if "ch4b" in s else "unknown"
    if "stl" in s:
        model = "STL"
    elif "hard_mtl" in s:
        model = "Hard_MTL"
    elif "cgc" in s:
        model = "CGC"
    else:
        model = "Unknown"
    pretrained = "qpre_finetune" in s or "qpre" in s and "finetune" in s
    fold = None
    m = re.search(r"fold_\d+", s)
    if m:
        fold = m.group(0)
    return {"protocol": protocol, "model": model, "pretrained": pretrained, "fold": fold}


def find_csvs(root: Path) -> List[Path]:
    patterns = ["*summary*.csv", "*metrics*.csv", "*per_basin*.csv", "training_history.csv"]
    out = []
    for pat in patterns:
        out.extend(root.rglob(pat))
    return sorted(set(out))


def read_summary_from_csv(path: Path) -> Optional[dict]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    row = df.iloc[-1]
    result = {"source_csv": str(path)}
    # Direct summary columns, e.g. Val_Q_NSE_Median or test_ssm_nse_median.
    for c in df.columns:
        lc = c.lower()
        if any(tok in lc for tok in ["nse", "kge", "rmse", "bias", "corr"]):
            val = row[c]
            if pd.api.types.is_number(val):
                result[c] = float(val)
    # Per-basin metric columns.
    metric_col = None
    for cand in ("NSE", "nse"):
        if cand in df.columns:
            metric_col = cand
            break
    if metric_col is not None:
        task_col = None
        for cand in ("task", "target", "variable", "var"):
            if cand in df.columns:
                task_col = cand
                break
        if task_col is None:
            result["NSE_median"] = float(np.nanmedian(df[metric_col].values))
        else:
            for task, sub in df.groupby(task_col):
                result[f"{task}_NSE_median"] = float(np.nanmedian(sub[metric_col].values))
                result[f"{task}_NSE_improvement_base"] = np.nan
    return result if len(result) > 1 else None


def collect(experiments_root: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted([p for p in experiments_root.iterdir() if p.is_dir()]):
        if not exp_dir.name.startswith("ch4"):
            continue
        csvs = find_csvs(exp_dir)
        if not csvs:
            continue
        merged = {"experiment": exp_dir.name, **infer_protocol_model(exp_dir.name)}
        for csv in csvs:
            res = read_summary_from_csv(csv)
            if res:
                # Prefer the richest result.
                if len(res) > len(merged):
                    merged.update(res)
        rows.append(merged)
    return pd.DataFrame(rows)


def compute_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_candidates = [c for c in df.columns if "nse" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    group_cols = [c for c in ["protocol", "fold", "pretrained"] if c in df.columns]
    for keys, group in df.groupby(group_cols, dropna=False) if group_cols else [((), df)]:
        stl = group[group["model"] == "STL"] if "model" in group else pd.DataFrame()
        for _, r in group.iterrows():
            out = r.to_dict()
            if not stl.empty:
                base = stl.iloc[0]
                for m in metric_candidates:
                    if m in r and m in base and pd.notna(r[m]) and pd.notna(base[m]):
                        out[f"delta_vs_STL__{m}"] = float(r[m] - base[m])
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chapter 4 Q-SSM experiments.")
    parser.add_argument("--experiments-root", default=Path("experiments"), type=Path)
    parser.add_argument("--out-dir", default=Path("experiments/ch4_qssm/evaluation"), type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args.experiments_root)
    df.to_csv(args.out_dir / "ch4_raw_collected_results.csv", index=False)
    comp = compute_pairwise(df)
    comp.to_csv(args.out_dir / "ch4_model_comparison.csv", index=False)
    print(f"Collected {len(df)} experiment rows.")
    print(f"Wrote: {args.out_dir / 'ch4_raw_collected_results.csv'}")
    print(f"Wrote: {args.out_dir / 'ch4_model_comparison.csv'}")
    if not comp.empty:
        print(comp[[c for c in ["experiment", "protocol", "fold", "model", "pretrained"] if c in comp.columns]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
