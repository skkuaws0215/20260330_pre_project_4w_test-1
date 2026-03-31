# ML 4-model Analysis Template (Step 1)

This folder is a fill-in template for the "1-step detailed analysis" added to the dashboard.

## Files

- `model_metrics_template.csv`
  - Per model and per dataset variant metrics table.
  - Fill this first from each job's `metrics.json` and logs.
- `dataset_impact_template.csv`
  - Comparison table by dataset change axis (A vs B, C old vs C new, etc.).
  - Fill this after `model_metrics_template.csv`.

## Minimum fields to fill (required)

- `valid_rmse`, `valid_mae`, `valid_r2`, `valid_spearman`
- `valid_ndcg_at_20`, `valid_hit_at_20` (ranking quality)
- `train_seconds`
- `n_features_numeric`
- `smiles_hash_enabled`, `smiles_hash_dim`
- `notes` for exceptions (e.g., RF stopped)

## How to use with dashboard section 3-1

1. Fill `model_metrics_template.csv`.
2. Fill `dataset_impact_template.csv`.
3. Move summarized rows into dashboard section:
   - "A. 모델별 상세 분석 체크포인트"
   - "B. 데이터셋이 결과에 미친 영향"
4. Mark unresolved values explicitly as `TBD` instead of leaving ambiguous text.

## DoD (same as dashboard intent)

- All completed models have RMSE/MAE/R2/Spearman and runtime.
- Dataset impact rows include at least one numeric conclusion per comparison axis.
- RF has explicit status: keep stopped or rerun with lighter config.
