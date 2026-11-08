# Original notebooks

The four networks the study ported, as their original Kaggle notebooks, kept so the port check can run:

```bash
python tests/local_check.py reference/originals
```

| Network | Notebook |
|:--|:--|
| ResGRU-UNet | `kaggle_archive/30_Res-1_-_v1/notebook.ipynb` |
| FMG-WACLU | `kaggle_archive/13_notebookce92d325af_-_Version_2/notebook.ipynb` |
| ConvLSTM | `kaggle_api/kernels/phase2_unsubmitted/convlstm-v2/convlstm-v2.ipynb` |
| patch AFNO | `kaggle_archive/35_Adaptive_Fourier_Neural_Operator_-_50_epoch_gpu/notebook.ipynb` |

The check rebuilds each original network from its notebook, confirms its parameter count, and confirms
that the study's port gives exactly the same output for the same weights. AFNO V4's original script is in
`reference/afno_v4/`.
