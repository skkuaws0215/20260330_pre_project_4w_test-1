SageMaker final three representatives (ML: XGBoost, DL: ResidualMLP, Graph: GCN)
==================================================================================

Purpose
  Reproducible final training only — not exploration. Local CV remains the
  selection rationale. Same data policy and metric definitions as local runs.
  Graph uses GCN selected under drug-group CV; hyperparameters baseline A
  (see graph_baseline_round1/gcn_tuning_summary.json).

Layout
  final_model_comparison.csv   — one row per representative; local CV means +
                               SageMaker job / eval / URIs (fill after jobs).
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

XGBoost SageMaker submit (existing)
  ml/pilot_sagemaker/submit_final_xgb_sagemaker.py

ResidualMLP / GCN
  Add dedicated Training Job scripts aligned with local entry points; then
  place metrics.json + artifacts under the subdirs above.
