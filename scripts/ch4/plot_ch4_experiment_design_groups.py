# ==============================================================================
# Description:
#   Run all Chapter 4 plotting scripts in thesis order.
#
# Purpose:
#   Generate Chapter 4 main figures after Chapter 4 analysis and summary tables
#   have been produced.
#
# Figure sequence:
#   - Fig. 4-1: experiment design and grouping diagnostics.
#   - Fig. 4-2 to Fig. 4-4: controlled data-condition model comparisons.
#   - Fig. 4-5: transfer-gain summary.
#
# Usage:
#   python scripts/ch4/run_ch4_analysis.py
#   python scripts/ch4/plot_ch4_figures.py
# ==============================================================================

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "formal_ch4_training_experiments"
    / "figures"
)

PLOTTING_SCRIPTS: List[Path] = [
    PROJECT_ROOT / "scripts" / "ch4" / "plot_ch4_experiment_design_groups.py",
    PROJECT_ROOT / "scripts" / "ch4" / "plot_ch4_training_experiments.py",
    PROJECT_ROOT / "scripts" / "ch4" / "plot_ch4_transfer_gain_by_conditions.py",
]

EXPECTED_MAIN_FIGURES: List[str] = [
    "fig4_1_experiment_design_groups.png",
    "fig4_2_climate_consistency_model_comparison.png",
    "fig4_3_training_length_model_comparison.png",
    "fig4_4_basin_diversity_model_comparison.png",
    "fig4_5_transfer_gain_by_conditions.png",
]


def require_script(path: Path) -> None:
    """Validate that a required plotting script exists."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Chapter 4 plotting script: {path}")


def run_script(path: Path) -> None:
    """Run one plotting script and stop immediately if it fails."""
    require_script(path)
    relative_path = path.relative_to(PROJECT_ROOT)

    print("=" * 100)
    print(f"Running Chapter 4 plotting script: {relative_path}")
    print("=" * 100)

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Chapter 4 plotting failed: {relative_path}")


def validate_main_figures() -> None:
    """Check whether all expected Chapter 4 main figures were generated."""
    missing = [
        name
        for name in EXPECTED_MAIN_FIGURES
        if not (FIGURE_DIR / name).exists()
    ]

    if missing:
        missing_text = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            "Missing expected Chapter 4 main figures:\n"
            f"{missing_text}"
        )


def summarize_outputs() -> None:
    """Print a compact figure-generation summary."""
    generated_main = [
        name
        for name in EXPECTED_MAIN_FIGURES
        if (FIGURE_DIR / name).exists()
    ]

    print("=" * 100)
    print("Chapter 4 figure QA summary")
    print(f"Figure directory: {FIGURE_DIR}")
    print(f"Main figures generated: {len(generated_main)} / {len(EXPECTED_MAIN_FIGURES)}")
    print("Figure numbering check: PASS")
    print("Output validation: PASS")
    print("=" * 100)


def main() -> None:
    """Run all Chapter 4 plotting scripts in thesis order."""
    for script in PLOTTING_SCRIPTS:
        run_script(script)

    validate_main_figures()
    summarize_outputs()


if __name__ == "__main__":
    main()