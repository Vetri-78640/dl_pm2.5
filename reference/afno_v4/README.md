# model - AFNO V4 (final)

Files here are unchanged copies of the AFNO V4 submission code. The notebook's code cells are identical to the submitted Kaggle version (submission 01). [Certain]

| file | purpose |
|---|---|
| `afno_v4.py` | full pipeline in percent format (source of truth) |
| `build_notebook.py` | regenerates the notebook from `afno_v4.py` |
| `Code4CleanAir_Model4_AFNOv4.ipynb` | Kaggle notebook (GPU, internet off, Run All) |

## model card

- Task: 16 h PM2.5 forecast from 10 h of inputs, 140 x 124 grid (about 25 km), hourly. WRF-Chem data.
- Architecture: U-Net encoder, multiscale AFNO (1 block at 35x31, 6 at 18x16), SimVP translator 10 -> 16, lead-time embedding, 2 post-translator AFNO blocks, U-Net decoder, persistence residual. 6.98M parameters.
- Inputs (20 channels):
  - observed (9): PM2.5, q2, t2, u10, v10, pblh, psfc, swdown, rain
  - engineered (5): wind speed, ventilation, advection, PM tendency, diurnal anomaly
  - time (2): hour sin, hour cos (estimated at test)
  - static (4): lat, lon, PM2.5 climatology, STL episode sigma
- Training: 2016, 4 months; leak-free split (tail 18% per month, 12 h gap); official STL episode masks; composite loss with horizon weights; EMA; seeds 42 and 1337; per-horizon calibration.
- Results:
  - val 0.9110 / 0.9114 (equal-weight official-style score)
  - private 0.8833, public 0.8959

  Submitted after the deadline. It would rank 2nd on the private board.
- Limits:
  - The val months are tails of the train months, so the year shift is not measured.
  - No ablation of the individual V4 changes.
  - Trained on simulation, not observations.

[Certain, details in `reports/results.md`]

## published copies

- Kaggle model: https://www.kaggle.com/models/imvetri0/afno-v4-pm25 (public; version 2 has the weights)
- Hugging Face: https://huggingface.co/imvetri/afno-v4-pm25
- GitHub: https://github.com/Vetri-78640/afno-v4-pm25
- Model card notebook: https://www.kaggle.com/code/imvetri0/afno-v4-model-card-and-architecture

All four were checked on 2026-11-07. [Certain]

## run

1. Import the notebook into the competition on Kaggle and attach the competition data.
2. Use a GPU (T4 or P100 at the time of the run). Internet off.
3. Run All. Masks take about 20 min (cached). Each seed takes about 1.5 h in phase A (16 x about 332 s), plus phase B.
4. Submit `/kaggle/working/preds.npy`.

To resume, attach the previous output and set `C.PREV`.
