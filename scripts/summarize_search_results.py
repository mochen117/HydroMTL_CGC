# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Utility script to summarize hyperparameter search results.
# Supports both Grid Search ('grid_*') and Randomized Search ('rand_*').
# Parses per-basin metrics from directories and identifies the top-performing 
# architectures based on Streamflow NSE spatial median.
# Automatically exports a consolidated summary report to a CSV file.
# ==============================================================================

import pandas as pd
from pathlib import Path
import sys

# Dynamically resolve absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def summarize_results(exp_dir: Path) -> pd.DataFrame:
    """
    Scans the experiments directory for hyperparameter search outputs, calculates 
    spatial medians for core metrics, and returns a sorted DataFrame.
    """
    results =[]
    
    if not exp_dir.exists():
        print(f"[ERROR] Experiments directory not found at: {exp_dir}")
        sys.exit(1)

    print(f"[INFO] Scanning for hyperparameter search results in: {exp_dir}")
    
    # Iterate through all directories matching either grid or random search prefixes
    for d in exp_dir.iterdir():
        if d.is_dir() and (d.name.startswith('grid_') or d.name.startswith('rand_')):
            csv_path = d / 'test_per_basin_metrics.csv'
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Ensure the expected columns exist before computing medians
                    if 'streamflow_nse' in df.columns and 'evapotranspiration_nse' in df.columns:
                        median_q_nse = df['streamflow_nse'].median()
                        median_et_nse = df['evapotranspiration_nse'].median()
                        
                        results.append({
                            'Experiment_Name': d.name, 
                            'Streamflow_NSE_Median': median_q_nse,
                            'ET_NSE_Median': median_et_nse
                        })
                except Exception as e:
                    print(f"  [WARNING] Failed to parse {csv_path}. Details: {e}")

    if not results:
        print("[WARNING] No valid test_per_basin_metrics.csv files found.")
        return None

    # Convert to DataFrame and sort by Streamflow NSE in descending order
    df_results = pd.DataFrame(results).sort_values(by='Streamflow_NSE_Median', ascending=False)
    df_results.reset_index(drop=True, inplace=True)
    
    return df_results


def main():
    df_results = summarize_results(EXPERIMENTS_DIR)
    
    if df_results is not None:
        print(f"\n{'='*85}")
        print(" 🏆 TOP 5 ARCHITECTURES (Ranked by Streamflow NSE Spatial Median) 🏆")
        print(f"{'='*85}")
        
        # Display top 5 results cleanly in the terminal
        print(df_results.head(5).to_string(index=True))
        
        print(f"{'='*85}\n")
        
        # Export the comprehensive summary report
        summary_save_path = EXPERIMENTS_DIR / "hyperparameter_search_summary.csv"
        df_results.to_csv(summary_save_path, index=False)
        print(f"[SUCCESS] Full summary report exported to: {summary_save_path}")


if __name__ == "__main__":
    main()