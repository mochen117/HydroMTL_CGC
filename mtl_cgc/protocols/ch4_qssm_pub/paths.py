"""Canonical project-relative paths for Chapter 4 Experiment B.

All runtime artifacts are written under ``experiments/ch4_qssm_pub``.  Source
code, configurations, scripts, and tests live outside the runtime results tree.
"""

from pathlib import Path

RESULTS_ROOT = Path("experiments/ch4_qssm_pub")
PROTOCOL_DIR = RESULTS_ROOT / "protocol"
FOLD_MANIFEST = PROTOCOL_DIR / "pub_fold_manifest.json"
RUNS_DIR = RESULTS_ROOT / "runs"
MANIFEST_DIR = RESULTS_ROOT / "manifests"
ENSEMBLE_DIR = RESULTS_ROOT / "ensemble"
SUMMARY_DIR = RESULTS_ROOT / "summary"

CONFIG_ROOT = Path("mtl_cgc/configs/ch4_qssm_pub")
FORMAL_CONFIG_DIR = CONFIG_ROOT / "formal"
TEMPLATE_DIR = CONFIG_ROOT / "templates"

CH3_SUMMARY = Path(
    "experiments/formal_ch3_modeling/06_summary/ch3_per_basin_with_metadata.csv"
)
CH4A_BASE_CONFIG = Path(
    "mtl_cgc/configs/ch4_qssm_formal/seed42/q_to_ssm/"
    "ch4a_formal_cgc_qssm_seed42.yaml"
)
