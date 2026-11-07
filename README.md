# IndTokenSwin — How to Run

Root: `C:/sand_box/exp_script`. Commands assume Windows + Git Bash.
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the folders and files.

## Requirements

- Python 3.13.9, PyTorch 2.11.0 (CUDA 13.0), timm 1.0.26, installed at
  `C:/ProgramData/anaconda3/python.exe`.
- Datasets under `E:/data` (paths are in `scripts/core/datasets.py`).
- Pretrained weights in `pretrained_weights/` and `pretrained_ckpts/`.
- One A800 40 GB GPU.

```bash
export PYTHONPATH="C:/sand_box/exp_script/scripts/core:C:/sand_box/exp_script/scripts/train:$PYTHONPATH"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8
```

## Run one experiment

```bash
export OUTPUT_ROOT="C:/sand_box/exp_script/rerun/main"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 \
  SEED=112458 RUN_SUFFIX=0 \
  "C:/ProgramData/anaconda3/python.exe" \
  "C:/sand_box/exp_script/scripts/train/train_expB1_swinv2.py"
```

| Variable | Meaning |
|---|---|
| `SEED` | training seed; also sets the train/val split |
| `RUN_SUFFIX` | run index 0-4, written as `__vN` in the output folder |
| `OUTPUT_ROOT` | where run folders are written |
| `INIT_FROM` | set it for Path B (a G checkpoint); leave unset for Path A |

`TEST_SEED = 42` is hard-coded in all 20 scripts. Seeds by run index:
112458, 142956, 212855, 653153, 940839.

## Run all experiments

```bash
bash scripts/shell/run_by_seed.sh          # Path A: 80 runs
bash scripts/shell/run_pathB.sh --check    # Path B: check INIT_FROM only, no training
bash scripts/shell/run_pathB.sh            # Path B: 30 runs
```

Both launchers set the environment themselves. Path A writes to `rerun/main`,
Path B to `rerun/G_continue`.

## Orchestrated re-run

```bash
"C:/ProgramData/anaconda3/python.exe" scripts/shell/rerun_all.py plan
"C:/ProgramData/anaconda3/python.exe" scripts/shell/rerun_all.py run --limit 5
"C:/ProgramData/anaconda3/python.exe" scripts/shell/rerun_all.py harvest
"C:/ProgramData/anaconda3/python.exe" scripts/shell/rerun_all.py verify
"C:/ProgramData/anaconda3/python.exe" scripts/shell/rerun_all.py swap
```

`all` chains run -> harvest -> swap. A full sweep is about 220 GPU-hours, so
start it only on explicit instruction.

## Regenerate the test split files

```bash
cd indtokenswin_release
KMP_DUPLICATE_LIB_OK=TRUE "C:/ProgramData/anaconda3/python.exe" export_splits.py
```

## Output

```
$OUTPUT_ROOT/<TASK_NAME>_<YYYYMMDD>__vN/
├── checkpoints/       best_model_phase1.pth, best_model_phase2.pth, final_model.pth
├── logs/
├── test_results.txt
├── predictions.csv
└── confusion_matrix.png / .txt
```
