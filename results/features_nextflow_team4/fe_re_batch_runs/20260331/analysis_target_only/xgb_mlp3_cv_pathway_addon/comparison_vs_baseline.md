# CV comparison: pathway block present (50 × `pathway__*`) vs prior run

**Setup:** Identical 5-fold CV as `run_xgb_mlp3_cv_local.py` (`shuffle=True`, `random_state=42`), LINCS columns excluded, labels inner-joined to features (`n=14497` rows).

| Model | Metric | Baseline (pathway block = 0) `dl_family/xgb_mlp3_cv_summary.json` | With Hallmark pathway addon `xgb_mlp3_cv_pathway_addon` |
|-------|--------|---------------------------------------------------------------------|---------------------------------------------------------|
| **XGBoost_tuned** | Spearman_mean | 0.430751 | **0.472840** |
| | RMSE_mean | 2.133085 | **2.070177** |
| **BlockWiseMLP** | Spearman_mean | 0.433991 | **0.471990** |
| | RMSE_mean | 2.130050 | **2.070041** |
| **ResidualMLP** | Spearman_mean | **0.436680** | 0.471264 |
| | RMSE_mean | **2.128796** | 2.070897 |

**Block grouping (pathway run):** `pathway_features=50`, `chem_features=2058`, `target_features=10` (same chem/target split as baseline; 50 Hallmark means added to pathway block).

**Interpretation:** With real pathway scores, **BlockWiseMLP** tracks **XGBoost** very closely (RMSE/Spearman almost tied). **ResidualMLP** remains competitive but no longer leads on Spearman in this fold setup; all three DL/tree models gain from the added pathway signal versus the baseline row counts.

**Data artifact:** `final_pathway_addon/pair_features_newfe_v2.parquet` — original 20260331 pair matrix + `pathway__*` merge on `sample_id` only.
