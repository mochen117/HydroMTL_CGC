# ==============================================================================
# Description:
#   Analyze basin-attribute importance for CGC streamflow transfer effects.
#
# Purpose:
#   Use a Random Forest regressor as an auxiliary diagnostic tool to explore
#   which basin attributes are related to CGC streamflow NSE gain relative to
#   STL-Q. This analysis is recommended as supplementary evidence rather than
#   the main hydrological interpretation.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_with_metadata.csv
#
# Outputs:
#   - experiments/formal_ch4_training_experiments/summary/ch4_feature_importance.csv
#   - experiments/formal_ch4_training_experiments/summary/ch4_feature_importance_model_score.csv
#   - experiments/formal_ch4_training_experiments/figures/fig4_s1_feature_importance_bar.png
#   - experiments/formal_ch4_training_experiments/figures/fig4_s2_observed_vs_predicted_delta_nse.png
# ==============================================================================

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_SUMMARY_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling" / "06_summary"
CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
CH4_SUMMARY_DIR = CH4_DIR / "summary"
FIG_DIR = CH4_DIR / "figures"

INPUT_PATH = CH3_SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"
IMPORTANCE_PATH = CH4_SUMMARY_DIR / "ch4_feature_importance.csv"
SCORE_PATH = CH4_SUMMARY_DIR / "ch4_feature_importance_model_score.csv"

TARGET_COL = "Delta_NSE_CGC_minus_STLQ"

FEATURE_ALIASES: Dict[str, List[str]] = {
    "aridity": ["aridity_index", "aridity"],
    "snow_fraction": ["snow_fraction", "frac_snow"],
    "p_mean": ["p_mean"],
    "pet_mean": ["pet_mean"],
    "p_seasonality": ["p_seasonality"],
    "area_gages2": ["area_gages2"],
    "elev_mean": ["elev_mean"],
    "slope_mean": ["slope_mean"],
    "frac_forest": ["frac_forest"],
    "lai_max": ["lai_max"],
    "lai_diff": ["lai_diff"],
    "soil_porosity": ["soil_porosity"],
    "soil_conductivity": ["soil_conductivity"],
    "max_water_content": ["max_water_content"],
    "sand_frac": ["sand_frac"],
    "clay_frac": ["clay_frac"],
}

FEATURE_LABELS: Dict[str, str] = {
    "aridity": "Aridity index",
    "snow_fraction": "Snow fraction",
    "p_mean": "Mean precipitation",
    "pet_mean": "Mean PET",
    "p_seasonality": "Precipitation seasonality",
    "area_gages2": "Drainage area",
    "elev_mean": "Mean elevation",
    "slope_mean": "Mean slope",
    "frac_forest": "Forest fraction",
    "lai_max": "Maximum LAI",
    "lai_diff": "LAI seasonality",
    "soil_porosity": "Soil porosity",
    "soil_conductivity": "Soil conductivity",
    "max_water_content": "Maximum water content",
    "sand_frac": "Sand fraction",
    "clay_frac": "Clay fraction",
}

RANDOM_STATE = 42
TEST_SIZE = 0.25

CH4_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def resolve_features(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """Resolve canonical feature names to available source columns."""
    features = []
    source_map = {}

    for canonical_name, candidates in FEATURE_ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                features.append(canonical_name)
                source_map[canonical_name] = candidate
                break

    return features, source_map


def build_dataset(
    df: pd.DataFrame,
    features: List[str],
    source_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and target vector."""
    x = pd.DataFrame(
        {
            feature: pd.to_numeric(df[source_map[feature]], errors="coerce")
            for feature in features
        }
    )

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    valid = y.notna()
    return x.loc[valid].copy(), y.loc[valid].copy()


def build_model() -> Pipeline:
    """Build a Random Forest pipeline with median imputation."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def compute_feature_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    source_map: Dict[str, str],
) -> pd.DataFrame:
    """Compute impurity-based and permutation feature importance."""
    rf = model.named_steps["model"]

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="r2",
    )

    out = pd.DataFrame(
        {
            "feature": features,
            "source_column": [source_map[f] for f in features],
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "random_forest_importance": rf.feature_importances_,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )

    return out.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)


def save_model_score(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    """Save model diagnostic scores."""
    score = pd.DataFrame(
        [
            {
                "split": "train",
                "n_samples": int(len(y_train)),
                "r2": float(r2_score(y_train, y_train_pred)),
                "mae": float(mean_absolute_error(y_train, y_train_pred)),
            },
            {
                "split": "test",
                "n_samples": int(len(y_test)),
                "r2": float(r2_score(y_test, y_test_pred)),
                "mae": float(mean_absolute_error(y_test, y_test_pred)),
            },
        ]
    )

    score.to_csv(SCORE_PATH, index=False)
    print(f"Saved: {SCORE_PATH}")


def plot_feature_importance(importance: pd.DataFrame) -> None:
    """Plot permutation feature importance as supplementary evidence."""
    if importance.empty:
        print("[Skip] Empty feature importance table.")
        return

    plot_df = importance.sort_values("permutation_importance_mean").tail(12)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    bars = ax.barh(
        plot_df["label"],
        plot_df["permutation_importance_mean"],
        xerr=plot_df["permutation_importance_std"],
    )

    ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Permutation importance based on test R²")
    ax.set_ylabel("Basin attribute")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=8,
        )

    output = FIG_DIR / "fig4_s1_feature_importance_bar.png"
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_observed_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Plot observed versus predicted transfer effects."""
    fig, ax = plt.subplots(figsize=(5.8, 5.5))

    y_obs = y_test.clip(-0.5, 0.5)
    y_hat = pd.Series(y_pred, index=y_test.index).clip(-0.5, 0.5)

    ax.scatter(y_obs, y_hat, s=20, alpha=0.7)

    min_val = min(y_obs.min(), y_hat.min())
    max_val = max(y_obs.max(), y_hat.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.0)

    ax.set_xlabel("Observed Delta NSE")
    ax.set_ylabel("Predicted Delta NSE")
    ax.set_title("Observed vs predicted CGC transfer effect")
    ax.grid(True, linestyle="--", alpha=0.3)

    output = FIG_DIR / "fig4_s2_observed_vs_predicted_delta_nse.png"
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def main() -> None:
    """Run auxiliary feature importance analysis."""
    require_file(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str, "huc_02": str})

    if TARGET_COL not in df.columns:
        raise ValueError(f"Input table must contain '{TARGET_COL}'.")

    features, source_map = resolve_features(df)
    if len(features) < 3:
        raise ValueError("Not enough basin attributes for feature importance analysis.")

    x, y = build_dataset(df, features, source_map)

    if len(y) < 50:
        raise ValueError("Not enough valid basins for robust feature importance analysis.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    save_model_score(y_train, y_test, y_train_pred, y_test_pred)

    importance = compute_feature_importance(model, x_test, y_test, features, source_map)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    print(f"Saved: {IMPORTANCE_PATH}")

    plot_feature_importance(importance)
    plot_observed_vs_predicted(y_test, y_test_pred)


if __name__ == "__main__":
    main()