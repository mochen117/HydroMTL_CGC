# Chapter 3 Reproducibility Guide

## 1. Purpose

This document freezes the reproducible workflow for Chapter 3 of the
HydroMTL_CGC project.

Formal Chapter 3 models:

- STL-Q
- STL-ET
- Hard-MTL
- MMoE
- CGC

Formal paper results must be based on the independent TEST period.
Validation results are retained only for model development and
checkpoint selection.

---

## 2. Formal experimental protocol

### Time periods

Training:

```text
2001-10-01 -> 2011-09-30
```

Validation:

```text
2011-10-01 -> 2016-09-30
```

Independent test:

```text
2016-10-01 -> 2021-09-30
```

Strict role separation:

```text
TRAIN
  -> optimize model parameters

VALIDATION
  -> checkpoint selection

best_model.pth
  -> frozen checkpoint

TEST
  -> final independent evaluation
  -> paper statistics and figures
```

The test period must not be used for model selection or
hyperparameter tuning.

---

## 3. Formal model settings

Common settings:

```text
sequence_length = 365
hidden_dim      = 64
batch_size      = 64
learning_rate   = 0.001
seed            = 42
```

Multi-task loss weights:

```text
streamflow       = 1.0
evapotranspiration = 0.1
```

CGC:

```text
shared_experts    = 4
task_experts      = [4, 4]
expert_hidden_dim = 256
temperature       = 1.0
```

---

## 4. Checkpoint policy

Formal independent-test evaluation uses:

```text
best_model.pth ONLY
```

No fallback to `final_model.pth` is allowed.

Checkpoint-selection metrics:

```text
STL-Q     -> streamflow_nse_median
STL-ET    -> evapotranspiration_nse_median
Hard-MTL  -> streamflow_nse_median
MMoE      -> streamflow_nse_median
CGC       -> streamflow_nse_median
```

The STL-ET protocol bug was corrected so that its best checkpoint is
selected using evapotranspiration NSE instead of streamflow NSE.

---

## 5. External data requirements

Git stores code and protocol, but large hydrological datasets,
checkpoints, NetCDF outputs, and some spatial data are not expected
to be stored in Git.

Prepared CAMELS-US data are expected under:

```text
output_592_basins/
```

This directory should include basin NetCDF files and:

```text
output_592_basins/basin_metadata.csv
```

Current metadata consistency:

```text
available metadata basins = 671
Chapter 3 test basins     = 592
matched basins            = 592
missing metadata          = 0
```

If the prepared data are stored elsewhere, link the whole directory:

```bash
ln -s /path/to/output_592_basins \
      ~/code/HydroMTL_CGC/output_592_basins
```

A directory-level link is preferred over individual file links.

Spatial plots additionally require the CAMELS basin geometry and map
data expected by the spatial plotting scripts.

---

## 6. Formal experiment directories

```text
experiments/formal_ch3_modeling/
├── 01_stl_q/ch3_stl_q_seed42/
├── 02_stl_et/ch3_stl_et_seed42/
├── 03_hard_mtl/ch3_hard_mtl_seed42/
├── 04_mmoe_mtl/ch3_mmoe_mtl_seed42/
└── 05_cgc_mtl/ch3_cgc_mtl_seed42/
```

Each formal run should retain:

```text
best_model.pth
test_per_basin_metrics.csv
test_predictions_and_weights.nc
test_summary.csv
```

Formal test audit manifest:

```text
experiments/formal_ch3_modeling/
ch3_independent_test_manifest.json
```

The manifest records checkpoint paths, SHA256 hashes, test period,
and evaluation protocol.

---

## 7. Formal Chapter 3 data chain

```text
prepared CAMELS-US data
        |
        v
run_ch3_models.py
        |
        | train + validation
        v
best_model.pth
        |
        v
run_ch3_independent_test.py
        |
        | independent TEST
        v
test_per_basin_metrics.csv
test_predictions_and_weights.nc
test_summary.csv
        |
        v
summarize_ch3_test_results.py
        |
        v
06_summary/test/
├── ch3_test_performance_summary.csv
├── ch3_per_basin_all_models.csv
├── ch3_test_transfer_long.csv
└── ch3_test_transfer_summary.csv
        |
        +-------------------------------+
        |                               |
        v                               v
analyze_transfer.py             audit_ch3_results.py
        |                               |
        v                               v
transfer statistics             consistency audit
        |
        +-------------------------------+
        |                               |
        v                               v
diagnose_ch3_nse_outliers.py    sensitivity_analysis_ch3.py
        |
        v
merge_basin_metadata.py
        |
        v
ch3_per_basin_with_metadata.csv
        |
        v
plot_ch3_spatial_maps.py
        |
        v
ch3_spatial_basin_metrics.gpkg
+ spatial figures
        |
        v
plot_ch3_publication_adaptive_focus_v5.py
        |
        v
formal Chapter 3 paper figures
```

---

## 8. Validation and TEST must remain separate

Historical validation summaries remain under:

```text
experiments/formal_ch3_modeling/06_summary/
```

Historical validation summarizer:

```text
scripts/ch3/summarize_ch3_results.py
```

It is intentionally allowed to read:

```text
validation_per_basin_metrics.csv
```

Formal paper results must be derived from:

```text
experiments/formal_ch3_modeling/06_summary/test/
```

The formal master result table is:

```text
06_summary/test/ch3_per_basin_all_models.csv
```

Publication figures must not use the old validation performance table.

---

## 9. Delta NSE definitions

Primary STL-referenced differences:

```text
Delta_NSE_HardMTL_minus_STLQ
Delta_NSE_MMoE_minus_STLQ
Delta_NSE_CGC_minus_STLQ

Delta_NSE_HardMTL_ET_minus_STLET
Delta_NSE_MMoE_ET_minus_STLET
Delta_NSE_CGC_ET_minus_STLET
```

Additional CGC-versus-MTL differences:

```text
Delta_NSE_CGC_minus_HardMTL
Delta_NSE_CGC_minus_MMoE

Delta_NSE_CGC_ET_minus_HardMTL
Delta_NSE_CGC_ET_minus_MMoE
```

All Delta NSE values are basin-paired.

Example:

```text
Delta NSE(CGC vs STL-Q)
=
NSE(CGC streamflow) - NSE(STL-Q streamflow)
```

Positive values indicate improvement relative to the reference model.

Negative values indicate performance degradation.

Here, positive/negative transfer refers to multi-task information
sharing, not cross-domain transfer learning.

---

## 10. Gate diagnostics

Gate-utilization tables are training-stage diagnostics rather than
independent-test performance metrics.

They remain under:

```text
experiments/formal_ch3_modeling/06_summary/
```

Expected files:

```text
ch3_gate_utilization_long.csv
ch3_gate_utilization_summary.csv
```

Publication plotting therefore combines:

```text
TEST performance
+
training-stage gate diagnostics
```

---

## 11. Frozen independent-test benchmark

Median performance across 592 basins:

| Model | Q NSE | Q KGE | ET NSE | ET KGE |
|---|---:|---:|---:|---:|
| STL-Q | 0.654839 | 0.585136 | - | - |
| STL-ET | - | - | 0.659925 | 0.572079 |
| Hard-MTL | 0.662667 | 0.630851 | 0.664840 | 0.558844 |
| MMoE | 0.651334 | 0.620810 | 0.662421 | 0.569202 |
| CGC | 0.684106 | 0.639220 | 0.669271 | 0.570497 |

Basin-paired transfer statistics:

| Task | Model | Median Delta NSE | Positive rate | Negative rate |
|---|---|---:|---:|---:|
| Q | Hard-MTL | +0.007725 | 0.537162 | 0.462838 |
| Q | MMoE | -0.009213 | 0.447635 | 0.552365 |
| Q | CGC | +0.010644 | 0.572635 | 0.427365 |
| ET | Hard-MTL | +0.002670 | 0.581081 | 0.418919 |
| ET | MMoE | -0.009536 | 0.287162 | 0.712838 |
| ET | CGC | +0.001528 | 0.538851 | 0.461149 |

CGC-versus-MTL paired median differences:

```text
Q:
CGC - Hard-MTL = +0.006347
CGC - MMoE     = +0.023779

ET:
CGC - Hard-MTL = -0.000337
CGC - MMoE     = +0.012601
```

---

## 12. Interpretation constraints

The strongest CGC advantage is observed for streamflow.

For Q, CGC has:

```text
highest median NSE
highest positive-transfer rate
lowest negative-transfer rate among evaluated MTL models
```

MMoE shows an overall negative-transfer tendency relative to the
corresponding STL baselines.

For ET, CGC has the highest overall median NSE, but the basin-paired
difference between CGC and Hard-MTL is approximately zero.

Therefore Chapter 3 should not claim that CGC universally
outperforms Hard-MTL for every hydrological variable.

Because extreme negative NSE values occur in some basins, arithmetic
mean NSE should not be the primary overall statistic.

Preferred statistics are:

```text
median
interquartile range
CDF
paired Delta NSE
positive/negative transfer rates
```

---

## 13. Reproduction command

Activate the environment:

```bash
conda activate MTL_CGC
cd ~/code/HydroMTL_CGC
```

Run the complete analysis chain with:

```bash
bash scripts/ch3/reproduce_ch3_analysis.sh
```

The script does not retrain the five models.

If independent-test outputs already exist, the formal test launcher
audits them instead of rerunning inference unless explicitly forced.

---

## 14. Final audit criteria

A valid reproduction should satisfy:

```text
formal models              = 5
independent-test basins    = 592
missing required columns   = 0
Delta consistency issues   = 0
metadata matches           = 592 / 592
checkpoint policy          = best_model.pth ONLY
test period                = 2016-10-01 -> 2021-09-30
```

Formal paper plots must read:

```text
06_summary/test/ch3_per_basin_all_models.csv
```

and not the historical validation-based master table.

---

## 15. Frozen Git version

The final reproducible Chapter 3 independent-test protocol is frozen
with:

```text
ch3_independent_test_v1
```

This tag should be created only after this document and the automated
reproduction script are committed to `develop`.
