# Data

Nothing in this folder is committed except this file and the empty-folder markers.

## Source

WRF-Chem simulation fields released for the ANRF AISEHack 2026 challenge (Phase 2, Theme 2, IIT Delhi) on Kaggle:
`anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd`. Accept the rules on the competition page first.

## Download

```bash
cd data/raw
kaggle competitions download -c anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd -p .
unzip -q *.zip && rm *.zip
```

The notebooks expect `data/raw/<MONTH>/<feature>.npy`, `data/raw/lat_long.npy` and `data/raw/test_in/<feature>.npy`
(the Kaggle folder `aisehack-theme-2/raw/` and `aisehack-theme-2/test_in/`). Set `paths.data` in
`configs/study.yaml` if your copy lives elsewhere.

## Episode masks and normalization stats (optional)

The study reused the episode masks and statistics computed by the AFNO V4 run. Computing the masks
(robust STL on every cell) takes about 14 minutes per month, so notebook 02 loads them if present:

```bash
kaggle kernels output imvetri0/afno-v4 -p data/interim/v4 --file-pattern '(ep_mask_.*\.npz|stats\.npz)$'
```

## Layout

| Folder | Holds |
|:--|:--|
| `raw/` | the competition files, unchanged |
| `interim/v4/` | V4's episode masks (`ep_mask_<MONTH>.npz`) and `stats.npz` |
| `processed/` | anything the notebooks write |
