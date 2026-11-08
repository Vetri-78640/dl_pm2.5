# Forecasting PM2.5 episodes over India: what moved the score, and why

Draft for a course report in research-paper form. Evidence tags: [Certain] means checked against code, logs or the Kaggle API; [Likely] means a strong inference. Post-deadline results say "would rank".

## abstract

We study 16-hour PM2.5 forecasting on a 140 x 124 grid (25 km) over India from 10 hours of WRF-Chem fields, in the ANRF AISEHack 2026 Kaggle competition. The metric combines global SMAPE, episode SMAPE and episode correlation, where episodes are robust-STL residual outliers. We report 41 submissions across six model families: patch AFNO, ConvLSTM, ResGRU-UNet, FMG-WACLU, AFNO-SimVP, and a rebuilt AFNO V3/V4.

The team's in-competition best was 0.8648 private (rank 8). A post-deadline AFNO V4 reached 0.8833 private, which would rank 2nd. On the same validation windows, V4 beats persistence by 0.060 (0.9110 vs 0.8515).

Three analyses explain most of what we saw:
1. SMAPE equals 2 |tanh(u/2)| in the log ratio u. So the median is optimal for global SMAPE, and the common "SMAPE punishes under-prediction" argument is wrong in its usual form.
2. Episode pixels are selected on the truth: the truth sits +0.41 in log above persistence on them. This makes upward-shifted forecasts score better on the episode terms, by an amount that grows with lead time. It explains the quantile-loss chain, the multipliers that worked or failed, and the shape of V4's per-horizon calibration.
3. The validation curve's epoch-4 peak is a trade-off between global and episode SMAPE, not ordinary overfitting.

We also show that the hackathon AFNO block was almost a per-token linear map. Its plateau was therefore not evidence against AFNO.

## 1. introduction

Short-range PM2.5 forecasts matter most during episodes, the rare hours when concentrations jump far above their usual diurnal pattern. The competition metric rewards both: a global SMAPE over all pixels and two terms restricted to episode pixels.

Our questions:
- Which modelling choices moved the private score?
- Why did the reasons the team gave at the time sometimes not match the mechanism?
- What does a correct validation look like for this task?

The record is unusual: every attempt is archived with code and logs, so failures can be diagnosed from evidence rather than memory.

Contributions:
1. A complete, verified account of 41 submissions and the reasoning behind each transition.
2. Three analytical results with scripted checks: SMAPE geometry, episode selection bias, and spectral-block linearity.
3. A measured persistence baseline on the final validation split, which the earlier record lacked.
4. A failure analysis of each family, with log excerpts.

## 2. task, data and metric

- Inputs: 10 h of PM2.5, 8 meteorological fields (q2, t2, u10, v10, swdown, pblh, psfc, rain) and 7 emission fields on a 140 x 124 grid. Output: 16 h of PM2.5. Training: April, July, October and December 2016 (715 + 3 x 739 = 2,932 hours). Test: 218 windows. [Certain]
- PM2.5 is heavy-tailed: train mean 33.62, median 16.12, maximum 2849.7 ug/m3. 7.4% of pixels exceed 100, and 0.006% exceed 1000. [Certain]
- Metric per forecast hour: SMAPE over the grid, SMAPE over episode pixels, and Pearson correlation over episode pixels. Each is normalized to [0, 1] and combined with undisclosed weights. [Certain]
- Episodes: robust STL (period 24) per grid point over the month, residual > 3 sigma and value > 1, about 1.7 to 2.2% of pixels per month [13]. [Certain]

## 3. related work

- Spectral neural operators: FNO parameterizes kernels on truncated Fourier modes [2]. AFNO replaces the per-mode weights with a shared block-diagonal MLP and soft-thresholding, and keeps all modes at quasi-linear cost [1]. FourCastNet uses AFNO for global weather at 0.25 degrees [3]. AdaptFNO proposes dynamic modes and multiscale operators, with qualitative results only [4].
- Spatiotemporal recurrence: ConvLSTM [6], with its location-invariance criticized by TrajGRU; ConvGRU [8].
- Video prediction without recurrence: SimVP [5].
- Architecture and training: U-Net [9], residual learning [10], AdamW, one-cycle schedules, GroupNorm, mixup [18]-[21], iterate averaging [16], deep ensembles [15], spectral bias [17].
- Losses and metrics: regression quantiles [11]; the asymmetry of SMAPE [12].
- Validation for dependent data: blocked cross-validation [14].

## 4. methods

### 4.1 model families

| family | entries | core idea | params |
|---|---|---|---|
| patch AFNO (AFNO_PM25) | 35, 34, 33 | 4x4 patches, 8 spectral blocks, linear unpatchify | 5.42M to 5.59M |
| ConvLSTM | convlstm-v2 -> upload 31 | 3x3 ConvLSTM encoder at full resolution, U-Net, FNO bottleneck | 19.06M |
| ResGRU-UNet | 30 -> 29 -> 28 -> 27 -> 18 / 15, 26 | U-Net with full-resolution skips, ConvGRU over lead time, log-space persistence residual | 7.10M |
| FMG-WACLU | 13 | future-met ConvLSTM gating + ConvLSTM history + U-Net/ConvGRU decoder | - |
| AFNO-SimVP | 12 -> 03 | U-Net encoder, AFNO bottleneck, SimVP translator, residual | 2.68M to 5.31M |
| AFNO V3 / V4 | 02, 01 | per-frame encoder, reference AFNO2D at 18x16 (+35x31 in V4), time-only Incep1D translator, 2 post-translator AFNO blocks, last+mean skip fusion, residual, EMA | 6.68M / 6.98M |

[Certain]

### 4.2 the final model (V4)

- Inputs: 20 channels. PM2.5 and 8 meteorological fields, wind speed, ventilation log1p(pblh x ws), advection -(u dC/dx + v dC/dy) of log PM, window-local PM tendency, hour sin/cos (test start hour estimated from swdown and t2 templates, 97.7% within 1 h), diurnal climatology anomaly, and 4 static channels.
- Network: AFNOv4 (`reference/afno_v4/afno_v4.py:583`). Output = log1p(last frame) + delta, with a zero-initialized head.
- Loss: Huber in log1p + global SMAPE + episode SMAPE + (1 - episode corr), with horizon weights 0.7 -> 1.3.
- Training: two phases, AdamW, one-cycle LR, EMA, 2 seeds.
- Calibration: one scale per horizon in [0.90, 1.10], fitted on val to maximize the two SMAPE terms.

[Certain]

### 4.3 validation designs

random_split over overlapping windows (hackathon); leave-one-month-out (33, ConvLSTM); per-month tail block (12 to 03); tail block + 12 h gap with the official STL masks and the exact formula (V3, V4). [Certain]

## 5. results

| stage | best entry | private | public |
|---|---|---|---|
| patch AFNO | 35 | 0.8218 | 0.8487 |
| ConvLSTM | 31 | 0.7594 | 0.7557 |
| ResGRU base | 30 | 0.8506 | 0.8713 |
| ResGRU + quantile chain (final) | 18 | 0.8648 | 0.8834 |
| FMG-WACLU | 13 | 0.8465 | 0.8647 |
| AFNO-SimVP (post-deadline) | 04 | 0.8561 | 0.8772 |
| AFNO V3 (post-deadline) | 02 | 0.8609 | 0.8768 |
| AFNO V4 (post-deadline) | 01 | 0.8833 | 0.8959 |

[Certain - Kaggle API] The team's final private score is 0.8648, rank 8. V4's 0.8833 would rank 2nd (1st: 0.8893).

Baseline on the V4 validation windows (same masks and formula): persistence 0.8515, V3 0.9030, V4 0.9110. [Certain]

## 6. analysis

### 6.1 SMAPE is symmetric in log ratio

SMAPE(p, y) = 2 |tanh(u/2)| with u = log(p/y), checked to 5.6e-16. It is bounded and concave in |u|. Its asymmetry exists only in additive error [12]. For a symmetric unimodal log predictive distribution, the median minimizes expected global SMAPE (a layer-cake argument). Simulation over lognormal sigma 0.1 to 1.0 gives q* = 0.500 every time. On the real val windows, the global-only best multiplier for persistence is 1.00 at every horizon up to h+14. [Certain]

### 6.2 episode selection bias

The episode mask is a two-sided STL residual threshold on the truth. On episode pixels the truth exceeds persistence by +0.406 in log on average, against -0.012 over all pixels.

So:
- global SMAPE is minimized near multiplier 1
- episode SMAPE is minimized above 1
- episode correlation does not change with scale (measured constant at 0.8344 across the multiplier sweep)

With equal weights, the best multiplier for persistence is 1.10. Per horizon it rises from 1.00 at h+1 to 1.30 at h+16, as uncertainty grows (sd of the log change from 0.10 to 0.50). V4's independently fitted scales rise the same way, from 1.00 to 1.08. [Certain the numbers]

This one mechanism accounts for:
- the steady gains of the quantile chain (q 0.75 -> 0.85 -> 0.90: 0.8594 -> 0.8627 -> 0.8648)
- the failure of a global x1.12 multiplier (0.8569) and the neutrality of a spike-only boost (0.8625)
- the damage from log-space shrinkage (0.8400 -> 0.7437)
- V4's calibration design

[Likely the attribution; the scores are Certain]

### 6.3 the epoch-4 peak is a trade-off

In V3 and both V4 seeds, global SMAPE keeps improving until epoch 8 to 11, while episode SMAPE is best at epoch 3 and degrades afterwards. In V4 seed 42, episode SMAPE goes from 0.2434 to 0.3012 by epoch 16. The composite peaks at epoch 4. Generic overfitting would worsen both terms. The observed split is what 6.2 predicts as the model converges toward the conditional median. A spectral-bias explanation [17] would predict the opposite order. [Certain the curves; Likely the explanation]

### 6.4 cost and reach

At our grid sizes, attention would cost 210x an AFNO block at full resolution, but only 3.5x at the 18x16 bottleneck where V4 runs most of its blocks. AFNO's advantage for us is inductive bias, not compute. PM2.5 moves about 9.8 cells in 16 h at the mean wind (11.5 at the RMS wind). Every family's receptive field covers this, so reach does not separate the models. [Certain the numbers; Likely the reading]

### 6.5 the hackathon AFNO was nearly linear

The AFNO_PM25 block applies one complex weight per head, shared by all frequencies, with no frequency-domain MLP. Without its weak softshrink, the block is a per-token linear map plus a fixed transform along one axis (verified to 1e-15). The reference AFNO MLP, used in V3/V4, is nonlinear per mode. The 0.82 plateau of 35 and 33 therefore says little about AFNO itself. Together with the missing residual and skips, it explains the switch from patch AFNO to ResGRU-UNet better than the team's spectral-bias account. [Certain the structure; Likely the reading]

### 6.6 failure analysis

| entry | stated idea | what the evidence shows |
|---|---|---|
| ConvLSTM (31) | recurrent local transport | own val 0.8264 on DEC_16 vs 0.8473 for persistence on the same windows (masks differ). The checkpoint heap kept epochs 1, 2 and 19 for the ensemble, plus geographic flip TTA and a season-id mismatch |
| 34 | diurnal time features | test had no `time.npy`; the model saw (sin, cos) = (0, 0) at test, an impossible hour |
| 33 | honest month holdout + ensemble | outputs capped at 558, against a train maximum of 2849.7 |
| 10/09/08/06 | fix a 20% over-prediction | the "actual = 34.5" was the test-input mean (34.51), not a target; the log-space shrink cuts peaks hardest |
| 26 | architecture diversity | the geometric mean cuts peaks where members disagree; the AFNO member was the near-linear block |
| 03 | flip augmentation | mirrors geography and the static lat/lon channels without negating u10 |

[Certain the evidence; Likely the causal reading]

### 6.7 validation dependence

The lag-1 h autocorrelation of log PM2.5 is 0.978, and it is 0.535 at V4's 12 h gap. Train holds about 111 independent 26 h windows and V4's val about 20. Val-to-private gaps: 0.113 (12, own mask and formula), 0.042 (V3), 0.028 (V4). Differences of about 0.001 between val runs (seed to seed, phase A vs B) cannot be told apart from noise. Per-window scores were not saved, so no standard error can be computed [14]. [Certain the numbers; Likely the noise judgement]

## 7. threats to validity

- The metric weights are undisclosed. Equal weights are assumed in V3/V4 and in our analysis. [Certain]
- No ablations: the share of each V4 change in its +0.0224 private gain over V3 is unknown. [Certain]
- Persistence is the stand-in forecaster for the selection-bias measurements. The sign of the effect does not depend on the forecaster, but its size does. [Certain]
- Older local scores use non-official masks and formulas and are not comparable to V3/V4 or to the leaderboard. [Certain]
- About 20 independent val blocks limit what val can resolve. The public split is small and noisy. [Likely]
- The test windows come from 2017 according to the team's notes. This is not confirmed by the organizers. [Likely]
- V4 was submitted 66 days after the deadline and did not count. [Certain]

## 8. reproducibility

- V4 code: `reference/afno_v4/afno_v4.py`. Weights: Kaggle model `imvetri0/afno-v4-pm25`, Hugging Face `imvetri/afno-v4-pm25`, GitHub `Vetri-78640/afno-v4-pm25`.
- The analysis scripts behind section 6 are not part of this repository.
- Every score comes from the Kaggle API, pulled on 2026-11-07.

## 9. conclusion

The largest measured gains came from three sources:
1. Making the forecast a correction to persistence, with full-resolution skips.
2. Validating on the exact metric with a buffered temporal split.
3. Adjusting the forecast scale by horizon.

The analysis points to one structural fact behind several of the team's choices: the episode terms are computed on truth-selected pixels, which rewards upward-biased forecasts by an amount that grows with lead time. A model designed around this from the start would fit a per-horizon scale or quantile on honest validation instead of tuning multipliers by leaderboard feedback.

## references

1. Guibas, Mardani, Li, Tao, Anandkumar, Catanzaro. Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. ICLR 2022. arXiv:2111.13587.
2. Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. ICLR 2021. arXiv:2010.08895.
3. Pathak et al. FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators. arXiv:2202.11214, 2022.
4. Dang, Nguyen, Nguyen, Hy. AdaptFNO: Adaptive Fourier Neural Operator with Dynamic Spectral Modes and Multiscale Learning for Climate Modeling. NeurIPS 2025 Workshop ML4PS.
5. Gao, Tan, Wu, Li. SimVP: Simpler yet Better Video Prediction. CVPR 2022, pp. 3170-3180. DOI 10.1109/CVPR52688.2022.00317.
6. Shi, Chen, Wang, Yeung, Wong, Woo. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. NIPS 2015, pp. 802-810.
7. Shi, Gao, Lausen, Wang, Yeung, Wong, Woo. Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model. NeurIPS 2017. arXiv:1706.03458.
8. Ballas, Yao, Pal, Courville. Delving Deeper into Convolutional Networks for Learning Video Representations. ICLR 2016. arXiv:1511.06432.
9. Ronneberger, Fischer, Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015, LNCS 9351, pp. 234-241. DOI 10.1007/978-3-319-24574-4_28.
10. He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016, pp. 770-778. DOI 10.1109/CVPR.2016.90.
11. Koenker, Bassett. Regression Quantiles. Econometrica 46(1), 1978, pp. 33-50. DOI 10.2307/1913643.
12. Goodwin, Lawton. On the asymmetry of the symmetric MAPE. International Journal of Forecasting 15(4), 1999, pp. 405-408. DOI 10.1016/S0169-2070(99)00007-2.
13. Cleveland, Cleveland, McRae, Terpenning. STL: A Seasonal-Trend Decomposition Procedure Based on Loess. Journal of Official Statistics 6(1), 1990, pp. 3-73.
14. Roberts et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8), 2017, pp. 913-929. DOI 10.1111/ecog.02881.
15. Lakshminarayanan, Pritzel, Blundell. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017, pp. 6402-6413.
16. Polyak, Juditsky. Acceleration of Stochastic Approximation by Averaging. SIAM J. Control and Optimization 30(4), 1992, pp. 838-855. DOI 10.1137/0330046.
17. Rahaman et al. On the Spectral Bias of Neural Networks. ICML 2019, PMLR 97:5301-5310.
18. Loshchilov, Hutter. Decoupled Weight Decay Regularization. ICLR 2019.
19. Smith, Topin. Super-convergence: very fast training of neural networks using large learning rates. Proc. SPIE 11006, 2019. DOI 10.1117/12.2520589.
20. Wu, He. Group Normalization. ECCV 2018, pp. 3-19.
21. Zhang, Cisse, Dauphin, Lopez-Paz. mixup: Beyond Empirical Risk Minimization. ICLR 2018.
