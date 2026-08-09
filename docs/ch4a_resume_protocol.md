# HydroMTL Chapter 4 Resume Patch

Replace these files in the project root:

- `main.py`
- `mtl_cgc/core/training/callbacks.py`
- `scripts/ch4_qssm/run_ch4a_q_to_ssm_protocol.py`

No change is required in `trainer.py` or the formal YAML files.

## Added behavior

- `main.py --resume`
- `main.py --resume_checkpoint PATH`
- model/optimizer/scheduler/AMP scaler/RNG restoration
- atomic `last_model.pth` written after every completed epoch
- atomic `training_history.csv` written after every completed epoch
- history and gradient-diagnostic continuation
- config/architecture/target/seed compatibility checks
- runner defaults to skip completed train/test runs
- runner defaults to resume partial runs from `last_model.pth`
- fine-tuning resumes from its own partial checkpoint instead of reapplying Q-pretraining initialization

## Compile

```bash
python -m py_compile \
  main.py \
  mtl_cgc/core/training/callbacks.py \
  mtl_cgc/core/training/trainer.py \
  scripts/ch4_qssm/run_ch4a_q_to_ssm_protocol.py
```

## Runner options

```bash
python scripts/ch4_qssm/run_ch4a_q_to_ssm_protocol.py --help
```

Expected options include:

- `--skip-completed / --no-skip-completed`
- `--resume-partial / --no-resume-partial`

## Main options

```bash
python main.py --help
```

Expected options include:

- `--resume`
- `--resume_checkpoint`

`--init_checkpoint` and resume options are mutually exclusive.
