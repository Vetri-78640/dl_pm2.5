# dl_pm2.5

PM2.5 forecasting with deep learning - AML/DL Project 19, Team T024, Section A, Phase 2.

A controlled backbone study for 16-hour PM2.5 forecasting over India: five deep networks, each trained inside one fixed pipeline, so the network is the only difference. Data: WRF-Chem fields on a 140 x 124 grid from the ANRF AISEHack 2026 Phase 2 Theme 2 Kaggle competition; 10 input hours, 16 forecast hours.

## Phases

| Phase | Approach | Code | Status |
|:--|:--|:--|:--|
| 1 | Gradient boosting on the UCI Beijing PM2.5 data | [aml_dl_pm2.5 at tag `phase1`](https://github.com/Vetri-78640/aml_dl_pm2.5/tree/phase1) | Done |
| 2 | Deep learning: controlled backbone study | this repo | In progress |

## Question

Is the AFNO V4 network better than the team's other architectures once every model gets the same pipeline? During the AFNO iterations the pipeline changed too (leak-free split, official masks, hour fix, physics channels), so the leaderboard scores of the model families mix architecture with pipeline. This study trains each network inside the fixed AFNO V4 pipeline. The protocol and the decision rule were fixed on 2026-11-07, before any training.

## Results

Validation, the same 214 windows, pooled over every value (µg/m³). Full tables: [`reports/results.md`](reports/results.md).

| Model (val) | MAE | RMSE | R² | Competition score |
|:--|--:|--:|--:|--:|
| Persistence | 10.44 | 26.46 | 0.767 | 0.8515 |
| Seasonal naive (same hour, 24 h earlier) | 11.53 | 26.80 | 0.761 | 0.7907 |
| Ridge regression (no neural network) | 8.51 | 21.48 | 0.847 | 0.8637 |
| AFNO V4, 5-seed mean | 6.99 | 19.45 | 0.874 | 0.9145 |
| ResGRU-UNet, 5-seed mean | 6.32 | 18.04 | 0.892 | 0.9193 |

- ResGRU-UNet beats AFNO V4 on MAE, RMSE and R² in all 25 seed pairs.
- Private leaderboard (post-deadline, so it would rank and does not count), 5 seeds + calibration: ResGRU-UNet 0.8914, AFNO V4 0.8848.
- MAE, RMSE and R² exist only on validation: test labels are hidden. The leaderboard has no R²; its correlation term is the Pearson r on episode pixels.
- Calibration raises the competition score but worsens MAE, RMSE and R², so the table above uses the uncalibrated means.

## Protocol

| Item | Setting |
|:--|:--|
| Inputs | V4's 20 channels: 9 met + wind speed + ventilation + advection + PM tendency + hour sin/cos + diurnal anomaly + 4 static |
| Split | tail 18% of each month is validation, 12 h gap before it |
| Episode masks, stats | loaded from the V4 run output (`imvetri0/afno-v4`) |
| Validation | the same 214 windows, exact competition metric, equal weights |
| Output | log1p(last frame) + predicted change, final layer zero-initialized |
| Training | V4 phase A: composite loss, horizon weights 0.7 -> 1.3, episode weight 6, AdamW 2.5e-4, one-cycle, 16 epochs, effective batch 4, EMA 0.999, AMP, grad clip 1.0 |
| Selection | best epoch by validation score |
| Seed | 42, one run per model |
| Post-processing | none |
| Hardware | Kaggle T4, one GPU used |

Reference, not retrained: AFNO V4 seed 42 phase A best, val 0.9110, private 0.8762 without calibration. Persistence: val 0.8515.

## Decision rule

1. Primary measure: the val score of the best epoch, on the 214 identical windows.
2. AFNO V4 is called better than model X only if both hold:
   - its val score is higher by more than 0.003
   - its private score is also higher
3. A val gap of 0.003 or less is a tie.
4. If X beats AFNO V4 by more than 0.003 on val and also on private, X is reported as better.
5. If val and private disagree, the result is inconclusive.
6. Known bias: the recipe was tuned for AFNO V4. A narrow AFNO win is weak evidence; a wide one is strong.

## Models and changes

| Model | Source | Original params | Study params (20 ch) | Changes |
|:--|:--|--:|--:|:--|
| ConvLSTM | `PM25ForecastNet`, convlstm-v2 kernel | 19,063,792 | 19,001,840 | input width; season FiLM removed (no season input); output = last + delta, zero-init; duplicated discarded `enc1`/`enc2` call dropped |
| FMG-WACLU | `FMGWACLUNet`, notebook 13 | 3,451,057 | 3,454,645 | input width only |
| ResGRU-UNet | `ResGRUUNet`, notebook 30 | 7,096,513 | 7,122,113 | input width only |
| patch AFNO | `AFNO_PM25`, notebook 35 | 5,422,080 | 5,708,800 | input width; output = last + delta, zero-init; spectral bug fixed (below) |
| AFNO V4 | `AFNOv4`, [`reference/afno_v4/afno_v4.py`](reference/afno_v4/afno_v4.py) | 6,980,033 | not retrained | used only for the pipeline check |

Patch AFNO bug: notebook 35 calls `torch.fft.irfft2(xf.float(), ...)`. `.float()` on a complex tensor keeps only its real part, so the imaginary half of every filtered spectrum was dropped. The original Kaggle run log shows the warning "Casting complex values to real discards the imaginary part" at that line. The port passes the complex spectrum intact.

Not carried over from any original: its own loss, schedule, epochs, batch, fine-tune phases, ensembles, TTA and blends.

## Repository layout

```
src/                  notebook sources (percent format)
  study.py            shared pipeline (data, 20 channels, split, loss, training) and all 5 networks
  ablations.py        the four single-change variants (H1 to H4)
  final_round.py      final round: seed ensembles, calibration, checks
  analysis.py         5-seed analysis: ensembles, block bootstrap, breakdowns, figures
  regression.py       MAE, RMSE and R² for every compared model
  baselines.py        persistence and seasonal naive on the val windows
  ridge.py            ridge regression baseline on the val windows and the test set
  maps.py             per-pixel error maps for the paper
  local.py            setup shared by the local notebooks (loads the as-ran code, points it at data/)
notebooks/            01 to 06, run locally on CPU with outputs (below)
notebooks/kaggle/     the 15 Kaggle notebooks with kernel-metadata.json, exactly as they ran, and each run's log (run_log.txt)
configs/study.yaml    the study settings; tests/test_config.py checks them against the code
results/              run records from Kaggle: summaries, train logs, metrics JSON, maps.npz, test_scores.csv
data/                 competition data, not committed (see data/README.md)
scripts/              notebook builders, the ablation verdict, Kaggle helpers (tools/)
tests/                config test, local_check.py, the CPU dry runs (one per stage except maps), pytest wrapper
reference/afno_v4/    the AFNO V4 training script (seeds 42 and 1337), unchanged
reference/originals/  the original notebooks of the four ported networks, for local_check.py
reports/              results.md, paper_draft.md, figures/, paper/ (report.tex, report.pdf, figures/)
```

## Notebooks

Each runs on a laptop CPU from the downloaded data and `results/`. No network is retrained.

| Notebook | Owner | Contents |
|:--|:--|:--|
| `01_data_and_eda` | Ishan Goyal | files, grid, PM2.5 by month and place, daily cycle, autocorrelation, transport distance, what each finding changed |
| `02_preprocessing` | Ishan Goyal | physics channels, normalization stats and episode masks recomputed and matched to V4's, leak-free split, hour estimation, the 20-channel input |
| `03_models` | Vetriselvan R | the 9 networks and variants: parameter counts, forward pass on a real window, port check, sensitivity by distance |
| `04_training` | Hrishi Dongre | recipe, loss, regularization, training curves of all 15 logged runs, the generalization gap |
| `05_evaluation` | Vedant Madne | metric, persistence / seasonal naive / ridge recomputed from raw data (match Kaggle exactly), regression metrics, calibration trade-off |
| `06_analysis_and_ablations` | Pampana Sree Adithya | decision rules applied in code, round one, final round, 5 seeds, where the winner wins, maps, ablation verdicts |

## Running

```
pip install -e ".[dev]"            # or: pip install -r requirements.txt
pytest                             # config consistency (seconds)
pytest -m dryrun                   # every dry run and the port check (CPU, about 35 min)
python tests/local_check.py reference/originals
python tests/dry_run.py
uvx --python 3.12 --from kaggle==2.2.5 --with six kaggle kernels push -p notebooks/kaggle/<model> --accelerator NvidiaTeslaT4
```

- `local_check.py` rebuilds each original network from its notebook in `reference/originals/`, checks the logged parameter counts, checks that port and original give the same output for the same weights, and runs forward, backward and autocast for the 20-channel ports.
- The dry runs execute each stage end to end on small synthetic data on CPU. `dry_run.py` loads `src/study.py`; the others load their stage's built source from `notebooks/kaggle/`.
- The builders (`scripts/build_*.py`) write into `notebooks/kaggle/` and would overwrite the notebooks that ran. Run them only in a copy of the repo. Rebuilding the four round-one notebooks from the current `study.py` gives files that differ in one way: the round-one notebooks hard-code seed 42, while `study.py` now reads a top-level `SEED` (42 by default) for the later stages.
- Kaggle CLI 2.2.5 is needed to set the accelerator; it needs Python 3.11 or later, hence `uvx`. Kaggle runs at most 2 batch GPU sessions at a time.
- The competition data is not included. Each Kaggle notebook expects it attached on Kaggle, together with the V4 run output `imvetri0/afno-v4`. For the local notebooks, download it as described in `data/README.md`.

## Stages

| Stage | Sources | Builder and check |
|:--|:--|:--|
| Round one (5 networks, seed 42) | `study.py` | `build_notebooks.py`, `dry_run.py` |
| Final round (ResGRU-UNet vs AFNO V4, 2 seeds + calibration) | `study.py` (seed 1337), `final_round.py` | `build_final.py`, `dry_run_final.py` |
| Seeds 1 to 3 for both finalists | `study.py` | `build_seeds.py`, `dry_run_seeds.py` |
| 5-seed analysis | `analysis.py` | `build_analysis.py`, `dry_run_analysis.py` |
| Ablations H1 to H4 | `ablations.py`; verdict with `scripts/ablation_verdict.py` | `build_ablations.py`, `dry_run_ablations.py` |
| Regression metrics | `regression.py` | `build_regression.py`, `dry_run_regression.py` |
| Baselines | `baselines.py` | `build_baselines.py`, `dry_run_baselines.py` |
| Ridge regression baseline (CPU) | `ridge.py` | `build_ridge.py`, `dry_run_ridge.py` |
| Error maps for the paper | `maps.py` | `build_maps.py` (no dry run) |
| Persistence on the test set | `notebooks/kaggle/persistence/` (CPU) | - |

Ablation variants, each run once on seed 42: `afnov4_allskip`, `afnov4_pad`, `afnov4_nosparse`, `resgru_norec`. Verdict: `python scripts/ablation_verdict.py <analysis.json> <dir with summary_<variant>.json> [out.json]`.

## Safety in each training notebook

- Fails fast if there is no usable CUDA GPU, if the competition data or the V4 output is missing, or if the V4 seed-42 checkpoint re-evaluated in the notebook is more than 0.002 from its logged val score.
- The micro-batch is picked by a memory probe (4, 2 or 1). Gradients are accumulated to an effective batch of 4, with the same optimizer steps per epoch.
- Every epoch saves a last checkpoint, a best checkpoint when val improves, and a `train_log.csv` row.
- A time budget of 10.5 h. If it is hit, the status is `incomplete`; attach the notebook's own output and run again to resume.
- Outputs: `preds.npy` (asserted shape and finite) and `summary.json`.

## Team

| Role | Member | Work |
|:--|:--|:--|
| A - Data and pre-processing | Ishan Goyal | data analysis, the 20 input channels, normalization, test-hour estimation, time-based split, episode masks |
| B - Model architectures | Vetriselvan R | AFNO V4 and ResGRU-UNet; ConvLSTM, FMG-WACLU and patch AFNO ported to one interface; the patch AFNO bug fix; ablation variants |
| C - Training pipeline | Hrishi Dongre | loss, optimizer and schedule, EMA, mixed precision, checkpoint and resume, seed runs, Kaggle notebooks |
| D - Evaluation and post-processing | Vedant Madne | competition metric, calibration, seed ensembles, baselines, MAE / RMSE / R², submissions |
| E - Experiments, analysis and documentation | Pampana Sree Adithya | study design and decision rules, 5-seed statistics, ablation analysis, figures, reports |
