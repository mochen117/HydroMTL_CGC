# HydroMTL_CGC — Chapter 4 Experiment B (PUB)

## Native runtime bootstrap (v3.2)

HydroMTL_CGC may run on servers where the system ``libstdc++.so.6`` is older
than the C++ runtime required by the active conda environment.  Chapter 4B now
uses a process-local bootstrap before importing compiled packages such as
``torch``, ``xarray`` and ``netCDF4``.  The bootstrap prepends
``$CONDA_PREFIX/lib`` to ``LD_LIBRARY_PATH`` and preloads the conda
``libstdc++.so.6`` with ``RTLD_GLOBAL``.

This is an execution-environment safeguard only. It does **not** modify the
conda environment on disk and does **not** replace the frozen Chapter 3 /
Chapter 4A core files.

Run this before any PUB smoke experiment:

```bash
python scripts/ch4_qssm_pub/check_runtime_environment.py
```

The command must report ``Native runtime check: PASS``.  It also opens one real
``output_592_basins/gage_*.nc`` file when available, so a successful result
checks both Python imports and the actual NetCDF backend.

## English overview

This patch adds **Chapter 4 Experiment B** to the existing `HydroMTL_CGC`
project. It does not create a new project and it does not replace the frozen
Chapter 4A core implementation.

The experiment evaluates whether remotely observed surface soil moisture (SSM)
can provide auxiliary supervision for streamflow prediction in target basins
where streamflow observations are completely withheld during training.

The formal comparison contains three core scenarios:

1. `stl_q`: STL-Q baseline trained with source-basin streamflow only;
2. `hps_target_ssm`: Hard parameter sharing trained with source Q+SSM and
   target-basin SSM, while target-basin Q is fully masked;
3. `cgc_target_ssm`: CGC with exactly the same supervision policy as Hard-MTL.

The PUB period is `2015-04-01` to `2021-09-30` for both training target dates
and target-basin Q evaluation dates. This is a **spatial** generalization
experiment. Leakage is controlled by withholding target-basin streamflow labels,
not by separating train and test time periods.

Normalization statistics are fitted from **source basins only**. Target-basin
SSM is an auxiliary supervised output target; it is not a dynamic model input.
The formal evaluation target is target-basin streamflow.

### Methodological reference

The supervision logic follows the spatial PUB principle used in the reference
study by Ouyang et al.: observed-basin Q supervision is combined with auxiliary
SSM supervision in basins without Q labels. This project uses its own fixed
5-fold spatial cross-validation implementation and does not claim that the
reference paper specified the same number of folds or the same numeric random
seeds.

---

# 中文实施说明

## 1. 目录规范

本补丁采用如下目录职责划分：

```text
HydroMTL_CGC/
├── mtl_cgc/
│   ├── core/                         # 已冻结的通用模型/训练核心，不覆盖
│   ├── data/                         # 已冻结的通用数据模块，不覆盖
│   ├── protocols/
│   │   └── ch4_qssm_pub/             # Chapter 4B 专用协议实现代码
│   └── configs/
│       └── ch4_qssm_pub/
│           ├── templates/            # 协议模板
│           ├── smoke/                # 运行时生成，1-epoch 验证配置
│           └── formal/               # 运行时生成，正式配置
├── scripts/
│   └── ch4_qssm_pub/                 # 构建、审计、训练、汇总入口
├── tests/
│   └── ch4_qssm_pub/                 # Chapter 4B 单元测试
├── docs/
│   └── ch4_qssm_pub/                 # 安装后文档
└── experiments/
    └── ch4_qssm_pub/                 # 只保存运行产物，不保存 Python 源代码
        ├── protocol/                  # basin folds / manifests
        ├── runs/                      # train/test/checkpoints/metrics
        ├── manifests/                 # runner 状态记录和锁文件
        ├── ensemble/                  # 多 seed 逐日预测 ensemble
        └── summary/                   # 最终统计汇总
```

核心原则：

```text
mtl_cgc/        = Python 实现
scripts/        = CLI / 审计 / 运行入口
configs/        = YAML / JSON 配置
tests/          = 测试
experiments/    = 实验结果
```

`experiments/` 下不再放 `data_adapter.py`、`protocol.py`、
`config_factory.py` 等源码。

## 2. Git 分支

当前项目仍然是：

```text
~/code/HydroMTL_CGC
```

建议第二个实验只使用分支：

```text
ch4_qssm_pub
```

安装前确认：

```bash
cd ~/code/HydroMTL_CGC
conda activate MTL_CGC

git branch --show-current
git status --short
git log -1 --oneline
```

应处于 `ch4_qssm_pub`。首次安装建议工作区为空；若只是用 v3.2 覆盖此前尚未提交的 Chapter 4B overlay，安装器仅允许 Ch4B 自身路径处于 dirty 状态，并会拒绝覆盖任何其他未提交文件。

## 3. 安装补丁

在补丁解压目录执行：

```bash
bash INSTALL_CH4B_PUB.sh ~/code/HydroMTL_CGC
```

安装脚本：

- 不覆盖 `main.py`；
- 不覆盖 `mtl_cgc/data/data_loaders.py`；
- 不覆盖 `mtl_cgc/data/data_sets.py`；
- 不覆盖 `mtl_cgc/core/training/trainer.py`；
- 不创建 `mtl_cgc/experiments/`；
- 将文档安装到 `docs/ch4_qssm_pub/`。

安装后检查：

```bash
cd ~/code/HydroMTL_CGC
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

git status --short
python -m pytest -q tests/ch4_qssm_pub
python scripts/ch4_qssm_pub/check_runtime_environment.py
```

`check_runtime_environment.py` 必须得到 `Native runtime check: PASS` 后才进入真实 NetCDF 数据审计或 smoke training。

随后建议再跑一次全项目测试：

```bash
python -m pytest -q
```

## 4. 先审计冻结参考结果和接口兼容性

因为当前采用同一 Git 项目目录切换分支，Chapter 3 / Chapter 4A 的
`experiments/` 结果和 `output_592_basins/` 会继续保留，不需要建立 worktree
符号链接。

直接执行：

```bash
python scripts/ch4_qssm_pub/audit_reference_artifacts.py
python scripts/ch4_qssm_pub/inspect_pub_compatibility.py
```

两项都必须通过后再继续。

## 5. 构建固定 5-fold 空间 PUB 划分

```bash
python scripts/ch4_qssm_pub/build_pub_folds.py
python scripts/ch4_qssm_pub/audit_pub_folds.py
```

默认产物：

```text
experiments/ch4_qssm_pub/protocol/
├── eligible_basins.txt
├── pub_fold_assignments.csv
├── pub_fold_manifest.json
├── fold01/
│   ├── source_basins.txt
│   └── target_basins.txt
└── ...
```

每个流域在 5 个 folds 中恰好一次作为 target basin。

## 6. 先生成独立 smoke 配置

不要直接修改 formal YAML。使用独立 `smoke` profile：

```bash
python scripts/ch4_qssm_pub/generate_pub_configs.py \
  --profile smoke \
  --seeds 42 \
  --subset core \
  --folds 1
```

`smoke` profile 默认 1 epoch，配置写入：

```text
mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/fold01/
```

运行目录使用独立实验名，例如：

```text
experiments/ch4_qssm_pub/runs/
ch4b_pub_smoke_f01_hps_target_ssm_seed42/
```

因此 smoke 结果不会覆盖 formal 结果。

配置审计：

```bash
python scripts/ch4_qssm_pub/audit_pub_configs.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/\
ch4b_pub_smoke_seed42_manifest.json
```

## 7. 运行时数据语义审计是正式训练前硬门槛

Hard-MTL：

```bash
python scripts/ch4_qssm_pub/audit_pub_data_semantics.py \
  --config \
  mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/fold01/\
ch4b_pub_smoke_f01_hps_target_ssm_seed42.yaml
```

CGC：

```bash
python scripts/ch4_qssm_pub/audit_pub_data_semantics.py \
  --config \
  mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/fold01/\
ch4b_pub_smoke_f01_cgc_target_ssm_seed42.yaml
```

必须确认：

```text
Source Q        : available
Source SSM      : observed values retained
Target Q        : fully masked during training
Target SSM      : observed values retained for assisted MTL
Training basins : source + target for Hard/CGC assisted scenarios
Scaler basins   : source only
Evaluation      : target only
```

## 8. Smoke dry-run 和 1-epoch 运行

先打印命令：

```bash
python -u scripts/ch4_qssm_pub/run_pub_protocol.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/\
ch4b_pub_smoke_seed42_manifest.json \
  --mode train \
  --subset core \
  --folds 1 \
  --device auto \
  --dry-run
```

确认后执行 1 epoch：

```bash
python -u scripts/ch4_qssm_pub/run_pub_protocol.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/\
ch4b_pub_smoke_seed42_manifest.json \
  --mode train \
  --subset core \
  --folds 1 \
  --device auto
```

然后测试并严格审计输出：

```bash
python -u scripts/ch4_qssm_pub/run_pub_protocol.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/\
ch4b_pub_smoke_seed42_manifest.json \
  --mode test \
  --subset core \
  --folds 1 \
  --device auto

python scripts/ch4_qssm_pub/audit_pub_outputs.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/smoke/seed42/\
ch4b_pub_smoke_seed42_manifest.json \
  --strict
```

## 9. 生成 seed42 正式配置

Smoke 全部通过后再生成 formal：

```bash
python scripts/ch4_qssm_pub/generate_pub_configs.py \
  --profile formal \
  --seeds 42 \
  --subset core \
  --folds 1 2 3 4 5
```

`formal` profile 默认 100 epochs。

正式 manifest：

```text
mtl_cgc/configs/ch4_qssm_pub/formal/seed42/
ch4b_pub_formal_seed42_manifest.json
```

再执行：

```bash
python scripts/ch4_qssm_pub/audit_pub_configs.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed42/\
ch4b_pub_formal_seed42_manifest.json
```

## 10. seed42 × 5 folds 正式训练和测试

训练：

```bash
python -u scripts/ch4_qssm_pub/run_pub_protocol.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed42/\
ch4b_pub_formal_seed42_manifest.json \
  --mode train \
  --subset core \
  --device auto
```

状态：

```bash
python scripts/ch4_qssm_pub/check_pub_training_status.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed42/\
ch4b_pub_formal_seed42_manifest.json \
  --show-processes
```

测试：

```bash
python -u scripts/ch4_qssm_pub/run_pub_protocol.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed42/\
ch4b_pub_formal_seed42_manifest.json \
  --mode test \
  --subset core \
  --device auto

python scripts/ch4_qssm_pub/audit_pub_outputs.py \
  --manifest \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed42/\
ch4b_pub_formal_seed42_manifest.json \
  --strict
```

## 11. 多 seed ensemble

先固定研究中实际采用的 seeds，再生成对应 formal 配置。例如：

```bash
python scripts/ch4_qssm_pub/generate_pub_configs.py \
  --profile formal \
  --seeds 42 123 2024 3407 7777 \
  --subset core \
  --folds 1 2 3 4 5
```

这些数值属于本研究预先固定的随机初始化，不应表述为参考论文公开的
原始 seed 值。

最终必须先平均**物理域逐日 Q 预测**，再重新计算 NSE/KGE：

```bash
python scripts/ch4_qssm_pub/ensemble_pub_predictions.py \
  --manifests \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed42/ch4b_pub_formal_seed42_manifest.json \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed123/ch4b_pub_formal_seed123_manifest.json \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed2024/ch4b_pub_formal_seed2024_manifest.json \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed3407/ch4b_pub_formal_seed3407_manifest.json \
  mtl_cgc/configs/ch4_qssm_pub/formal/seed7777/ch4b_pub_formal_seed7777_manifest.json \
  --require-seeds 5
```

输出写入：

```text
experiments/ch4_qssm_pub/ensemble/
```

## 12. 结果汇总

```bash
python scripts/ch4_qssm_pub/summarize_pub_results.py
python scripts/ch4_qssm_pub/build_cross_experiment_summary.py
```

最终结果目录：

```text
experiments/ch4_qssm_pub/summary/
├── ch4b_pub_ensemble_per_basin_metrics.csv
├── ch4b_pub_effects_with_ch3_metadata.csv
├── ch4b_pub_model_effect_summary.csv
├── ch4b_pub_hydroclimate_group_summary.csv
├── ch3_ch4a_ch4b_cross_experiment_per_basin.csv
├── ch4_cross_experiment_directionality_summary.csv
└── ch4_cross_experiment_hydroclimate_summary.csv
```

## 13. 科学解释边界

正式 Chapter 4B 只评价 target-basin Q。Target SSM 在同一时期参与训练，
因此不能把该 SSM 输出当作独立的 out-of-sample test result。

最终主比较为：

```text
Hard-MTL-PUB - STL-Q-PUB
CGC-PUB      - STL-Q-PUB
CGC-PUB      - Hard-MTL-PUB
```

并与 Chapter 4A 的 Q→SSM 时间资料受限实验对照，用于讨论辅助作用的
**方向性**和**资料依赖性**。
