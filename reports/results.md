# Results

Every number comes from the Kaggle API (2026-11-07 and 2026-11-08) or a run log.

## Leaderboard

| board | rank | score | source |
|---|---|---|---|
| private (final) | 8 | 0.8648 | Kaggle private leaderboard (top 20), 2026-11-07 |
| public | 7 of 36 teams | 0.8834 | Kaggle public leaderboard, 2026-11-07 |
| after presentations | top 4 | - | user-reported, not on Kaggle |

Private top 3 for reference: VibeCoders 0.8893, Binary Bombers 0.8795, Ctrl+alt+Defeat 0.8794.

## Best entry per approach (private)

| approach | entry | private | public | on time |
|---|---|---|---|---|
| upload, model unknown | 40 af1 | 0.8262 | 0.8505 | yes |
| patch AFNO | 35 | 0.8218 | 0.8487 | yes |
| AFNO + month CV ensemble | 33 | 0.7955 | 0.8138 | yes |
| ConvLSTM | 31 | 0.7594 | 0.7557 | yes |
| ResGRU-UNet base | 30 | 0.8506 | 0.8713 | yes |
| ResGRU + PQR q=0.85 blend | 27 (selected) | 0.8627 | 0.8825 | yes |
| ResGRU + PQR q=0.90 blend | 18 (selected, final) | 0.8648 | 0.8834 | yes |
| physics multipliers | 22 | 0.8625 | 0.8772 | yes |
| FMG-WACLU | 13 | 0.8465 | 0.8647 | yes |
| AFNO-SimVP alone, temporal holdout | 11 | 0.8437 | 0.8657 | no |
| AFNO-SimVP + ResGRU blend | 04 | 0.8561 | 0.8772 | no |
| AFNO V3 | 02 | 0.8609 | 0.8768 | no |
| AFNO V4 | 01 | 0.8833 | 0.8959 | no |

[Certain]

## V4 decomposition (post-deadline ablation submissions, 2026-11-07)

| stage | private | public |
|---|---|---|
| V3 | 0.8609 | 0.8768 |
| V4 seed 42, no calibration | 0.8762 | 0.8878 |
| V4 seed 1337, no calibration | 0.8749 | 0.8882 |
| V4 2-seed mean, no calibration | 0.8779 | 0.8899 |
| V4 final (calibrated) | 0.8833 | 0.8959 |

Private gain over V3, +0.0224: model changes +0.0147, ensemble +0.0024, calibration +0.0054. [Certain - post-deadline ablation submissions, 2026-11-07]

## Controlled backbone study (post-deadline submissions, 2026-11-07 and 2026-11-08)

Same pipeline, data, split, loss and schedule. Round one: one seed, no calibration; the final round and the 5-seed extension add seeds and calibration as stated. [Certain]

| model | params | val | private | public |
|---|---|---|---|---|
| ResGRU-UNet | 7.12M | 0.9158 | 0.8798 | 0.8924 |
| AFNO V4 seed 42 (reference) | 6.98M | 0.9110 | 0.8762 | 0.8878 |
| FMG-WACLU | 3.45M | 0.9099 | 0.8710 | 0.8862 |
| patch AFNO (spectral bug fixed) | 5.71M | 0.9031 | 0.8530 | 0.8680 |
| ConvLSTM | 19.0M | 0.8945 | 0.8452 | 0.8537 |

Verdicts at matched settings: ResGRU-UNet better than AFNO V4, FMG-WACLU tie, patch AFNO and ConvLSTM worse. Final round (V4's full recipe for both, pre-registered): ResGRU-UNet 2 seeds val 0.9175 vs V4 0.9131; private 0.8817 vs 0.8779 without calibration and 0.8898 vs 0.8833 with it. Verdict: ResGRU-UNet better (final-round submissions, 2026-11-08). 5 seeds per model (rule fixed before the 6 new runs, 2026-11-08; seeds 42 and 1337 already known): val ensembles 0.91933 vs 0.91454, gap +0.00479 [95% block bootstrap +0.00383, +0.00578]; every ResGRU-UNet seed beats every AFNO V4 seed on val and private (25 of 25); private with calibration 0.8914 vs 0.8848. Verdict again ResGRU-UNet better; 0.8914 is the best post-deadline private score (5-seed submissions, 2026-11-08). [Certain - round-one submissions, 2026-11-07]

## Regression metrics on val (MAE, RMSE, R²; added 2026-11-08)

Same 214 val windows, ug/m3, pooled over every value. Test labels are hidden, so there are none for the test set. Computed by `src/regression.py` and `src/baselines.py` (notebooks `notebooks/kaggle/regression/` and `notebooks/kaggle/baselines/`).

| model (val) | MAE | RMSE | R² |
|---|---|---|---|
| persistence | 10.44 | 26.46 | 0.767 |
| seasonal naive (same hour, 24 h earlier; val only) | 11.53 | 26.80 | 0.761 |
| ridge regression (private 0.7951) | 8.51 | 21.48 | 0.847 |
| round one, seed 42: ResGRU-UNet / AFNO V4 / FMG-WACLU / patch AFNO / ConvLSTM | 6.57 / 7.31 / 7.14 / 6.85 / 7.05 | 18.57 / 20.17 / 19.32 / 19.02 / 18.93 | 0.885 / 0.865 / 0.876 / 0.880 / 0.881 |
| 2-seed mean: ResGRU-UNet / AFNO V4 | 6.40 / 7.15 | 18.21 / 19.73 | 0.890 / 0.871 |
| 5 single seeds, mean: ResGRU-UNet / AFNO V4 | 6.58 / 7.22 | 18.59 / 19.92 | 0.885 / 0.868 |
| 5-seed mean: ResGRU-UNet / AFNO V4 | 6.32 / 6.99 | 18.04 / 19.45 | 0.892 / 0.874 |
| 5-seed mean + calibration (leave-one-month-out): ResGRU-UNet / AFNO V4 | 7.12 / 7.76 | 19.71 / 20.90 | 0.871 / 0.855 |

Calibration raises the competition score but worsens MAE, RMSE and R². [Certain]

## Validation vs private (comparable runs only)

| run | val (equal-weight official-style) | private | gap |
|---|---|---|---|
| V3 | 0.9030 | 0.8609 | 0.042 |
| V4 seed 42 / 1337 | 0.9110 / 0.9114 | 0.8833 | about 0.028 |
| V4 seed 42 alone (same setting as V3) | 0.9110 | 0.8762 | 0.035 |

The 0.028 row pairs single-seed val with the 2-seed + calibration private; compare V3 with the matched row. [Certain]

## V4 validation components (seed 42, best epoch)

| term | raw | normalized |
|---|---|---|
| global SMAPE | 0.2124 | 0.8938 |
| episode SMAPE | 0.2479 | 0.8760 |
| episode corr | 0.9264 | 0.9632 |
| score (mean of 3) | - | 0.9110 |

[Certain, from the V4 log]
