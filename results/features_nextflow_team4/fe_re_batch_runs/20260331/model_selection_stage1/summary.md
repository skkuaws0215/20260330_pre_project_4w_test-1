# Model selection stage 1

Same 5-fold CV (`KFold`, `shuffle=True`, `random_state=42`) for all models; equal config count per family; shared DL training budget (epochs, batch, early stopping).

## XGBoost (best configuration)

- Spearman mean ± std: **0.473117** ± 0.009845
- RMSE mean ± std: 2.067037 ± 0.043921
- Params: `{"colsample_bytree": 0.7, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "n_estimators": 800, "reg_alpha": 1.0, "reg_lambda": 1, "subsample": 0.85}`

## BlockWiseMLP (best configuration)

- Spearman mean ± std: **0.473404** ± 0.008381
- RMSE mean ± std: 2.071053 ± 0.043261
- Mean val−train MSE gap (stability / overfitting hint): -0.026954
- Config: `{"block_hidden_dim": 64, "dropout": 0.2, "fusion_hidden_dim": 128, "lr": 0.0005, "variant": "A", "weight_decay": 1e-05}`

## ResidualMLP (best configuration)

- Spearman mean ± std: **0.476152** ± 0.011203
- RMSE mean ± std: 2.066046 ± 0.048176
- Mean val−train MSE gap: 0.120905
- Config: `{"dropout": 0.2, "hidden_dim": 256, "lr": 0.0005, "num_layers": 3, "variant": "A", "weight_decay": 1e-05}`

## Selected DL model (DL-only comparison)

**ResidualMLP**

Primary: higher Spearman mean (0.476152 vs 0.473404). Secondary RMSE means: Residual=2.066046, BlockWise=2.071053. Stability (std): BlockWise=0.008381, Residual=0.011203. Generalization (mean val−train MSE): BlockWise=-0.026954, Residual=0.120905.
