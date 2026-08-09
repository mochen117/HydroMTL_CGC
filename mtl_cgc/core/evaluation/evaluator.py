# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Hydrological spatial evaluation framework.
# Rectifies temporal index offsets and implements Wilcoxon paired validations.
# Employs strict NaN-masking, Expert CV, Gini indexing and Holm-Bonferroni.
# Calculates strict Wilcoxon effect sizes using returned scipy Z-statistics.
# ==============================================================================

import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any

class HydroEvaluator:
    """Reconstructs batch tensors back into geographic matrices to compute indices."""
    def __init__(self, config: Any, basin_ids: List[str], scaler: Any):
        self.config = config
        self.basin_ids = basin_ids
        self.scaler = scaler
        self.task_names = [str(t['name']).lower() for t in config.data.targets]
        self.target_configs = {
            str(target['name']).lower(): dict(target)
            for target in config.data.targets
        }
        self.target_metrics = [m.lower() for m in config.evaluation_protocol.metrics]

    def process_and_evaluate(self, collected_data: Dict[str, Any], period_dates: List[str]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Optional[xr.Dataset]]:
        """Reconstructs continuous physical timeseries without index shift modifications."""
        num_basins = len(self.basin_ids)
        start_date = pd.to_datetime(period_dates[0])
        end_date = pd.to_datetime(period_dates[1])

        if start_date > end_date:
            raise ValueError(
                "Invalid evaluation period: "
                f"start_date={start_date.date()} is after "
                f"end_date={end_date.date()}."
            )

        time_index = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        num_days = len(time_index)
        
        reconstructed_preds = {t: np.full((num_basins, num_days), np.nan) for t in self.task_names}
        reconstructed_obs = {t: np.full((num_basins, num_days), np.nan) for t in self.task_names}
        reconstructed_gates = {}
        
        if not collected_data['basin_idx']:
            return {}, {}, None

        b_idx_arr = np.concatenate(collected_data['basin_idx']).flatten()
        t_idx_arr = np.concatenate(collected_data['time_idx']).flatten()
        stat_num_arr = np.concatenate(collected_data['stat_num'], axis=0)
        
        t_idx_valid = t_idx_arr
        valid_mask = (b_idx_arr >= 0) & (b_idx_arr < num_basins) & (t_idx_valid >= 0) & (t_idx_valid < num_days)
        
        b_idx_valid = b_idx_arr[valid_mask]
        t_idx_valid = t_idx_valid[valid_mask]
        stat_num_valid = stat_num_arr[valid_mask]

        for task in self.task_names:
            if collected_data['preds'][task]:
                raw_p = np.concatenate(collected_data['preds'][task]).flatten()[valid_mask]
                raw_o = np.concatenate(collected_data['targets'][task]).flatten()[valid_mask]
                
                phys_p = self.scaler.inverse_transform_target_safe(task, raw_p, stat_num_valid)
                phys_o = self.scaler.inverse_transform_target_safe(task, raw_o, stat_num_valid)

                # Convert native target values to the configured
                # publication unit before metrics and export.
                phys_p = self._apply_target_output_scale(
                    task,
                    phys_p,
                )
                phys_o = self._apply_target_output_scale(
                    task,
                    phys_o,
                )
                
                reconstructed_preds[task][b_idx_valid, t_idx_valid] = phys_p
                reconstructed_obs[task][b_idx_valid, t_idx_valid] = phys_o
            
        for g_name, g_list in collected_data['gates'].items():
            if g_list:
                g_arr = np.concatenate(g_list, axis=0)[valid_mask]
                num_experts = g_arr.shape[-1]
                rec_g = np.full((num_basins, num_days, num_experts), np.nan)
                rec_g[b_idx_valid, t_idx_valid, :] = g_arr
                reconstructed_gates[g_name] = rec_g

        global_metrics = {}
        # Corrected: Align basin metrics structure with explicit basin_id strings to avoid spatial mismatching
        per_basin_metrics = {b_id: {} for b_id in self.basin_ids}
        
        for task in self.task_names:
            task_metric_collections = {m: [] for m in self.target_metrics}
            
            for b in range(num_basins):
                p_b = reconstructed_preds[task][b]
                o_b = reconstructed_obs[task][b]
                b_id = self.basin_ids[b]
                
                valid_t = ~np.isnan(p_b) & ~np.isnan(o_b)
                if valid_t.sum() < 10: 
                    continue
                
                p_eval = torch.from_numpy(p_b[valid_t])
                o_eval = torch.from_numpy(o_b[valid_t])
                
                res = self._compute_local_metrics({task: p_eval}, {task: o_eval}, self.target_metrics)
                
                for m in self.target_metrics:
                    val = res.get(f"{task}_{m}")
                    if val is not None and not np.isnan(val):
                        task_metric_collections[m].append(val)
                        per_basin_metrics[b_id][f"{task}_{m}"] = val
            
            for m in self.target_metrics:
                arr = np.array(task_metric_collections[m])
                if len(arr) > 0:
                    global_metrics[f"{task}_{m}_mean"] = float(np.mean(arr))
                    global_metrics[f"{task}_{m}_median"] = float(np.median(arr))
                    global_metrics[f"{task}_{m}_25th"] = float(np.percentile(arr, 25))
                    global_metrics[f"{task}_{m}_75th"] = float(np.percentile(arr, 75))
                else:
                    global_metrics[f"{task}_{m}_mean"] = float('nan')

        ds_vars = {}
        for task in self.task_names:
            ds_vars[f"{task}_sim"] = (["basin", "time"], reconstructed_preds[task])
            ds_vars[f"{task}_obs"] = (["basin", "time"], reconstructed_obs[task])
        for g_name, g_mat in reconstructed_gates.items():
            ds_vars[g_name] = (["basin", "time", f"expert_{g_name}"], g_mat)
            
        ds_export = xr.Dataset(
            data_vars=ds_vars,
            coords={"basin": self.basin_ids, "time": time_index},
            attrs={
                "description": "Spatiotemporal aligned multi-task prediction exports",
                "unit_note": (
                    "streamflow is restored to m3/s; "
                    "evapotranspiration is restored to mm/day."
                ),
            },
        )

        self._assign_variable_units(ds_export)

        self._assign_target_metadata(ds_export)

        return global_metrics, per_basin_metrics, ds_export

    def _assign_variable_units(self, ds_export: xr.Dataset) -> None:
        """Attach publication-ready physical units to exported variables."""
        for task in self.task_names:
            sim_name = f"{task}_sim"
            obs_name = f"{task}_obs"

            if "streamflow" in task:
                units = "m3 s-1"
                long_name = "streamflow"
            elif "evapo" in task:
                units = "mm day-1"
                long_name = "evapotranspiration"
            else:
                units = "native units"
                long_name = task

            for var_name, suffix in [(sim_name, "simulation"), (obs_name, "observation")]:
                if var_name in ds_export:
                    ds_export[var_name].attrs["units"] = units
                    ds_export[var_name].attrs["long_name"] = f"{long_name} {suffix}"

    def _apply_target_output_scale(
        self,
        task: str,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Convert inverse-transformed native values to configured output units.

        For Chapter 4 SSM, the processed source values are percent-like and
        unit_scale=0.01 converts them to volumetric fraction in m3 m-3.
        """
        target_config = self.target_configs.get(
            str(task).lower(),
            {},
        )

        scale = float(
            target_config.get("unit_scale", 1.0)
        )

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"Invalid unit_scale for target '{task}': {scale}."
            )

        # Upcast before unit conversion. Model outputs may be float16;
        # multiplying float16 values by 0.01 would introduce avoidable
        # quantization errors and slightly alter dimensionless metrics.
        return np.asarray(values, dtype=np.float64) * scale


    def _assign_target_metadata(
        self,
        dataset: xr.Dataset,
    ) -> None:
        """Attach target-specific physical units to exported variables."""
        unit_notes = []

        for task in self.task_names:
            target_config = self.target_configs.get(
                task,
                {},
            )

            scale = float(
                target_config.get("unit_scale", 1.0)
            )

            configured_unit = (
                target_config.get("output_unit")
                or target_config.get("units")
            )

            if configured_unit is not None:
                output_unit = str(configured_unit)
            elif "streamflow" in task:
                output_unit = "m3 s-1"
            elif "evapo" in task:
                output_unit = "mm day-1"
            elif (
                "ssm" in task
                or "soil_moisture" in task
            ):
                output_unit = "m3 m-3"
            else:
                output_unit = "native units"

            long_name = str(
                target_config.get(
                    "long_name",
                    task,
                )
            )

            source_unit = str(
                target_config.get(
                    "source_unit",
                    "native units",
                )
            )

            for suffix, role in (
                ("sim", "simulation"),
                ("obs", "observation"),
            ):
                variable_name = f"{task}_{suffix}"

                if variable_name not in dataset:
                    continue

                dataset[variable_name].attrs.update(
                    {
                        "units": output_unit,
                        "long_name": (
                            f"{long_name} {role}"
                        ),
                        "unit_scale_applied": scale,
                        "source_units": source_unit,
                    }
                )

            unit_notes.append(
                f"{task}: {output_unit} "
                f"(unit_scale={scale:g})"
            )

        dataset.attrs["unit_note"] = "; ".join(
            unit_notes
        )


    def _compute_local_metrics(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], metrics: List[str]) -> Dict[str, float]:
        """Calculates specific evaluation statistics safely converting CUDA tensors."""
        out_dict = {}
        for task in preds.keys():
            p = preds[task].detach().cpu().numpy().flatten()
            t = targets[task].detach().cpu().numpy().flatten()
            
            mask = ~np.isnan(p) & ~np.isnan(t) & ~np.isinf(p) & ~np.isinf(t)
            p, t = p[mask], t[mask]
            
            if len(p) < 2:
                continue
                
            for m in metrics:
                if m == "nse":
                    num = np.sum((t - p) ** 2)
                    den = np.sum((t - np.mean(t)) ** 2)
                    out_dict[f"{task}_nse"] = float(1.0 - num / den) if den > 1e-12 else float("nan")

                elif m == "kge":
                    std_p, std_t = np.std(p), np.std(t)
                    mean_p, mean_t = np.mean(p), np.mean(t)

                    if std_p < 1e-8 or std_t < 1e-8 or abs(mean_t) < 1e-8:
                        out_dict[f"{task}_kge"] = float("nan")
                    else:
                        r = np.corrcoef(p, t)[0, 1]
                        alpha = std_p / std_t
                        beta = mean_p / mean_t
                        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
                        out_dict[f"{task}_kge"] = float(kge)

                elif m == "rmse":
                    out_dict[f"{task}_rmse"] = float(np.sqrt(np.mean((p - t) ** 2)))

                elif m == "mae":
                    out_dict[f"{task}_mae"] = float(np.mean(np.abs(p - t)))

                elif m == "bias":
                    denom = np.sum(t)
                    relative_bias = (
                        float(np.sum(p - t) / denom)
                        if abs(denom) > 1e-8
                        else float("nan")
                    )
                    # Backward-compatible key retained; it represents relative bias.
                    out_dict[f"{task}_bias"] = relative_bias
                    out_dict[f"{task}_relative_bias"] = relative_bias

                elif m == "corr":
                    if np.std(p) < 1e-8 or np.std(t) < 1e-8:
                        out_dict[f"{task}_corr"] = float("nan")
                    else:
                        out_dict[f"{task}_corr"] = float(np.corrcoef(p, t)[0, 1])
        
        return out_dict


class ClimateSpecializationAnalyzer:
    """Classifies catchments based on climate dryness and snow ratios."""
    def __init__(self, aridity_dict: Dict[str, float], snow_fraction_dict: Dict[str, float]):
        self.aridity = aridity_dict
        self.snow = snow_fraction_dict

    def analyze_expert_utilization(self, basin_ids: List[str], gate_weights: np.ndarray, expert_labels: List[str]) -> pd.DataFrame:
        """Determines routing activation matrices stratified by climate regimes."""
        records = []
        for idx, b_id in enumerate(basin_ids):
            ai = self.aridity.get(b_id, np.nan)
            sf = self.snow.get(b_id, np.nan)
            
            ai_class = "Humid" if ai < 1.0 else ("Arid" if ai >= 2.0 else "Semi-Arid")
            sf_class = "Snowy" if sf >= 0.3 else "Non-Snowy"
            
            # Corrected: Protect averages mapping to prevent empty slice nanmean runtime warnings
            subset = gate_weights[idx]
            if np.isnan(subset).all():
                mean_gates = np.full(subset.shape[-1], np.nan)
            else:
                mean_gates = np.nanmean(subset, axis=0)
                
            rec = {"basin_id": b_id, "climate_group": ai_class, "snow_group": sf_class}
            for exp_idx, label in enumerate(expert_labels):
                rec[label] = mean_gates[exp_idx]
            records.append(rec)
            
        df = pd.DataFrame(records)
        return df.groupby(["climate_group", "snow_group"])[expert_labels].mean()


def compute_wilcoxon_paired_test(metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Applies Wilcoxon signed-rank significance check with non-parametric effect size calculation (r = Z / sqrt(N)).
    Enforces rigorous NaN/Inf cleaning filters to prevent mathematical runtime warnings.
    """
    shared_basins = list(set(metrics_a.keys()).intersection(set(metrics_b.keys())))
    if not shared_basins:
        raise ValueError("Matched comparison datasets are missing coordinates.")
        
    scores_a = np.array([metrics_a[b] for b in shared_basins], dtype=float)
    scores_b = np.array([metrics_b[b] for b in shared_basins], dtype=float)
    
    # Corrected: Enforce robust filtration of non-finite elements to protect scipy wilcoxon execution
    valid_mask = np.isfinite(scores_a) & np.isfinite(scores_b)
    scores_a = scores_a[valid_mask]
    scores_b = scores_b[valid_mask]
    
    n_samples = len(scores_a)
    if n_samples < 5:
        raise ValueError(f"Insufficient paired non-NaN samples ({n_samples} < 5) to compute a valid Wilcoxon signed-rank test.")
        
    # Corrected: Use SciPy asymptotic Z-statistic extraction instead of raw hand-approximated zero-ties
    res = stats.wilcoxon(scores_a, scores_b, alternative='two-sided', method='approx')
    stat = res.statistic
    p_val = res.pvalue
    z_stat = getattr(res, 'zstatistic', np.nan)
    
    # Corrected: Wilcoxon ties drop adjustment for exact non-parametric effect size computation
    non_zero_diff = np.sum(scores_a != scores_b)
    
    if not np.isnan(z_stat) and non_zero_diff > 0:
        effect_size = abs(z_stat) / np.sqrt(non_zero_diff)
    elif non_zero_diff > 0:
        # Fallback approximation for older SciPy distributions using effective non-zero tie bounds
        std_w = np.sqrt(non_zero_diff * (non_zero_diff + 1) * (2 * non_zero_diff + 1) / 24.0)
        mean_w = non_zero_diff * (non_zero_diff + 1) / 4.0
        z_approx = (stat - mean_w) / std_w
        effect_size = abs(z_approx) / np.sqrt(non_zero_diff)
    else:
        effect_size = 0.0
    
    return float(stat), float(p_val), float(effect_size)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down correction on multiple pairwise p-values.
    Guarantees family-wise error rate control under rigorous multi-baseline comparisons.
    """
    m = len(p_values)
    if m == 0:
        return []
        
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected_p = [0.0] * m
    
    for i, (orig_idx, p) in enumerate(indexed_p):
        corrected_val = p * (m - i)
        corrected_p[orig_idx] = float(min(1.0, max(corrected_val, corrected_p[indexed_p[i-1][0]] if i > 0 else 0.0)))
        
    return corrected_p