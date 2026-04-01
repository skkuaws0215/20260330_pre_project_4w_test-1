# Model selection stage 1 round2

Narrow round2 tuning before SageMaker baseline freeze. Goal: reduce overfitting and improve generalization with fixed feature set and same 5-fold split.

## Best configs
- XGBoost: `{"colsample_bytree": 0.7, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "n_estimators": 800, "reg_alpha": 1.0, "reg_lambda": 10, "subsample": 0.8}`
- BlockWiseMLP: `{"block_hidden_dim": 64, "dropout": 0.2, "fusion_hidden_dim": 256, "lr": 0.0003, "pathway_norm": "pathway_zscore", "patience": 5, "weight_decay": 0.0001}`
- ResidualMLP: `{"dropout": 0.2, "grad_clip": null, "hidden_dim": 128, "lr": 0.0001, "num_layers": 3, "pathway_norm": "pathway_zscore", "patience": 5, "weight_decay": 0.001}`

## Metrics (round2 best)
- XGBoost: Spearman=0.473565, RMSE=2.065700, std=0.009188, gap=0.326012, MAE=1.534361
- BlockWiseMLP: Spearman=0.469690, RMSE=2.076859, std=0.009156, gap=-0.045076, MAE=1.549167
- ResidualMLP: Spearman=0.470814, RMSE=2.074299, std=0.010933, gap=-0.091452, MAE=1.554629

## Stage1 -> Round2 delta
- XGBoost: dSpearman=+0.000448, dRMSE=-0.001337, dStd=-0.000657
- BlockWiseMLP: dSpearman=-0.003713, dRMSE=+0.005806, dStd=+0.000776, dGap=-0.018122
- ResidualMLP: dSpearman=-0.005338, dRMSE=+0.008254, dStd=-0.000269, dGap=-0.212357

## Checks requested
- BlockWise stability kept/improved: NO
- Residual overfitting reduced: YES
- XGBoost generalization improved (Spearman up or RMSE down): YES
