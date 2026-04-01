# Round2 comparison

## DL representative re-check
- BlockWiseMLP: Spearman=0.469690, std=0.009156, gap=-0.045076, RMSE=2.076859, MAE=1.549167
- ResidualMLP: Spearman=0.470814, std=0.010933, gap=-0.091452, RMSE=2.074299, MAE=1.554629
- DL recommendation for SageMaker baseline: **ResidualMLP**

## ML baseline re-check
- XGBoost(best): Spearman=0.473565, std=0.009188, gap=0.326012, RMSE=2.065700, MAE=1.534361

## SageMaker baseline candidates
- ML: XGBoost (round2 best config_id=1)
- DL: ResidualMLP (round2 best config_id=7)
