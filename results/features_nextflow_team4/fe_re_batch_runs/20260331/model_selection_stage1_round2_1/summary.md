# Model selection stage 1 round2.1 (Residual only)

ResidualMLP micro-search (4 configs) around Round2 best for score recovery while keeping overfitting control.

- Best config: `{"dropout": 0.2, "grad_clip": null, "hidden_dim": 128, "lr": 0.0001, "num_layers": 3, "pathway_norm": "pathway_zscore", "patience": 5, "weight_decay": 0.001}`
- Round2.1 best: Spearman=0.470301, std=0.010706, RMSE=2.075779, MAE=1.550819, gap=-0.081948
- vs Stage1 best: dSpearman=-0.005851, dRMSE=+0.009734, dStd=-0.000497, dGap=-0.202853
- vs Round2 best: dSpearman=-0.000513, dRMSE=+0.001480, dStd=-0.000228, dGap=+0.009504

## DL final criteria check
- Spearman close to/exceeds XGBoost Round2 (0.473565): NO
- fold std stable: YES
- train-val gap not excessive: YES
