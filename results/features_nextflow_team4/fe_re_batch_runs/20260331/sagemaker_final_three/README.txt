SageMaker final three representatives (ML: XGBoost, DL: ResidualMLP, Graph: GCN)
==================================================================================

Purpose
  Reproducible final training only — not exploration. Local CV remains the
  selection rationale. Same data policy and metric definitions as local runs.
  Graph uses GCN selected under drug-group CV; hyperparameters baseline A
  (see graph_baseline_round1/gcn_tuning_summary.json).

Layout
  final_model_comparison.csv   — one row per representative. Columns include
                               local_validation_type, local_* means,
                               sagemaker_validation_type, sagemaker_evaluation_note
                               (GCN: row holdout ≠ drug-group CV mean — read note),
                               job name, cloud metrics, URIs.
  final_model_summary.json    — principles, output paths, local CV sources.
  artifacts/xgb/              — copy or sync model bundle + metrics.json here.
  artifacts/residualmlp/
  artifacts/gcn/
  logs/                         — CloudWatch 링크 또는 저장한 로그 조각.

Regenerate table + summary from local JSON sources
  python3 ml/pilot_sagemaker/aggregate_final_representative_outputs.py

After jobs: merge sidecar metrics into CSV
  python3 ml/pilot_sagemaker/aggregate_final_representative_outputs.py --collect

Sidecar metrics.json (per artifact folder)
  Optional top-level keys for --collect:
    sagemaker_training_job, sagemaker_status, artifact_uri, training_logs_uri
    final_eval: { "rmse", "mae", "spearman", "ndcg20", "hit20" }

SageMaker submit (AWS)
  XGBoost:     ml/pilot_sagemaker/submit_final_xgb_sagemaker.py
               (full_train off, test_size 0.1 — valid metrics in metrics.json;
                same tuned xgb_* hyperparams as local summary)
  ResidualMLP: ml/pilot_sagemaker/submit_final_residual_mlp_sagemaker.py
  GCN:         ml/pilot_sagemaker/submit_final_gcn_sagemaker.py
               (bundles graph_baseline_data, run_graph_gnn_cv, proximity, disease genes)

Sync model.tar.gz into this tree
  python3 ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py \
    --family xgb|residualmlp|gcn --model-tar /path/to/model.tar.gz

Local reproduction (no AWS)
  See README "군별 대표 3종" — run train_tabular.py, train_residual_mlp_final.py,
  train_gcn_final.py with SM_MODEL_DIR pointing at artifacts/* and the same
  default parquet paths as SageMaker staging.
