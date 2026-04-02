"""
Build final_model_comparison.csv + final_model_summary.json for the three
representative models (XGBoost, ResidualMLP, GCN) after SageMaker final training.

- Default: write template rows with **local CV selection baselines** and TBD columns
  for SageMaker job names, cloud metrics, artifact URIs, logs.
- --collect: read artifacts/xgb|residualmlp|gcn/metrics.json (optional sidecar fields)
  and merge into the comparison table.

Principles (see final_model_summary.json): no extra tuning; same data policy as local;
Graph uses group-CV-selected GCN with baseline hyperparams (config A).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parents[1]

DEFAULT_OUT = (
    _REPO
    / "results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three"
)
DEFAULT_RESIDUAL_SUMMARY = (
    _REPO
    / "results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only"
    / "residual_mlp_cv/residual_mlp_cv_summary.json"
)
DEFAULT_GRAPH_GROUPCV = (
    _REPO
    / "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1"
    / "graph_family_groupcv_summary.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final 3-model SageMaker comparison CSV + summary JSON.")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--residual-summary", type=str, default=str(DEFAULT_RESIDUAL_SUMMARY))
    p.add_argument("--graph-groupcv-summary", type=str, default=str(DEFAULT_GRAPH_GROUPCV))
    p.add_argument(
        "--collect",
        action="store_true",
        help="Merge metrics from out_dir/artifacts/<family>/metrics.json into comparison CSV.",
    )
    return p.parse_args()


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _cv_block_to_flat(block: dict[str, Any]) -> dict[str, float]:
    return {
        "RMSE_mean": float(block["RMSE_mean"]),
        "MAE_mean": float(block["MAE_mean"]),
        "Spearman_mean": float(block["Spearman_mean"]),
        "NDCG20_mean": float(block["NDCG@20_mean"]),
        "Hit20_mean": float(block["Hit@20_mean"]),
    }


def _graph_gcn_local(graph_summary: dict[str, Any]) -> dict[str, float]:
    g = graph_summary["metrics_mean_5fold_by_model"]["GCN"]
    return {
        "RMSE_mean": float(g["RMSE"]),
        "MAE_mean": float(g["MAE"]),
        "Spearman_mean": float(g["Spearman"]),
        "NDCG20_mean": float(g["NDCG@20"]),
        "Hit20_mean": float(g["Hit@20"]),
    }


def build_template_rows(residual_path: Path, graph_path: Path) -> list[dict[str, Any]]:
    rs = _load_json(residual_path)
    gs = _load_json(graph_path)
    xgb = _cv_block_to_flat(rs["xgb_tuned_cv_summary"])
    mlp = _cv_block_to_flat(rs["residual_mlp_cv_summary"])
    gcn = _graph_gcn_local(gs)

    def row(
        family: str,
        model: str,
        artifact_dir: str,
        local_val: str,
        m: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "model_family": family,
            "representative_model": model,
            "local_selection_validation": local_val,
            "local_RMSE_mean": m["RMSE_mean"],
            "local_MAE_mean": m["MAE_mean"],
            "local_Spearman_mean": m["Spearman_mean"],
            "local_NDCG20_mean": m["NDCG20_mean"],
            "local_Hit20_mean": m["Hit20_mean"],
            "sagemaker_training_job": "",
            "sagemaker_status": "TBD",
            "sagemaker_RMSE": "",
            "sagemaker_MAE": "",
            "sagemaker_Spearman": "",
            "sagemaker_NDCG20": "",
            "sagemaker_Hit20": "",
            "artifact_uri": "",
            "training_logs_uri": "",
            "metrics_json_relative_path": f"artifacts/{artifact_dir}/metrics.json",
        }

    return [
        row("ML", "XGBoost", "xgb", "5-fold row KFold (same as residual_mlp_cv_summary)", xgb),
        row("DL", "ResidualMLP", "residualmlp", "5-fold row KFold (same as residual_mlp_cv_summary)", mlp),
        row(
            "Graph",
            "GCN",
            "gcn",
            "5-fold GroupKFold by canonical_drug_id (graph_family_groupcv)",
            gcn,
        ),
    ]


def _read_sidecar_metrics(out_dir: Path, family_key: str) -> dict[str, Any] | None:
    p = out_dir / "artifacts" / family_key / "metrics.json"
    if not p.is_file():
        return None
    return _load_json(p)


def apply_collect(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df = df.copy()
    fam_to_key = {"ML": "xgb", "DL": "residualmlp", "Graph": "gcn"}
    for i, r in df.iterrows():
        fam = str(r["model_family"])
        key = fam_to_key.get(fam)
        if not key:
            continue
        raw = _read_sidecar_metrics(out_dir, key)
        if not raw:
            continue
        # Optional top-level SageMaker / reporting fields
        if raw.get("sagemaker_training_job"):
            df.at[i, "sagemaker_training_job"] = raw["sagemaker_training_job"]
        if raw.get("sagemaker_status"):
            df.at[i, "sagemaker_status"] = raw["sagemaker_status"]
        if raw.get("artifact_uri"):
            df.at[i, "artifact_uri"] = raw["artifact_uri"]
        if raw.get("training_logs_uri"):
            df.at[i, "training_logs_uri"] = raw["training_logs_uri"]

        ev = raw.get("final_eval") or raw.get("sagemaker_eval") or raw.get("valid")
        if ev is None and isinstance(raw.get("train"), dict):
            ev = raw["train"]
        if isinstance(ev, dict):
            if "rmse" in ev:
                df.at[i, "sagemaker_RMSE"] = ev["rmse"]
            if "mae" in ev:
                df.at[i, "sagemaker_MAE"] = ev["mae"]
            if "spearman" in ev:
                df.at[i, "sagemaker_Spearman"] = ev["spearman"]
            if "ndcg20" in ev:
                df.at[i, "sagemaker_NDCG20"] = ev["ndcg20"]
            elif "NDCG@20" in ev:
                df.at[i, "sagemaker_NDCG20"] = ev["NDCG@20"]
            if "hit20" in ev:
                df.at[i, "sagemaker_Hit20"] = ev["hit20"]
            elif "Hit@20" in ev:
                df.at[i, "sagemaker_Hit20"] = ev["Hit@20"]
    return df


def build_summary_json(
    out_dir: Path,
    comparison_rel: str,
    residual_rel: str,
    graph_rel: str,
) -> dict[str, Any]:
    repo_rel = str(out_dir.resolve().relative_to(_REPO.resolve()))
    return {
        "phase": "sagemaker_final_three_representatives",
        "purpose": "Reproducible final training for selected models only; not exploration.",
        "representatives": {
            "ML": "XGBoost",
            "DL": "ResidualMLP",
            "Graph": "GCN (group CV selection; hyperparams baseline A per gcn_tuning_summary.json)",
        },
        "execution_principles": [
            "Local CV results remain the rationale for model selection; SageMaker runs do not re-tune for new champions.",
            "SageMaker runs only the three representatives; no additional model search in this phase.",
            "Same data joins, keys, and metric definitions as local experiments; document any URI changes.",
            "Graph: GCN with drug-group CV policy; use hidden_dim=64, lr=1e-3, weight_decay=1e-5 unless team policy changes.",
        ],
        "outputs": {
            "comparison_csv": comparison_rel,
            "summary_json": f"{repo_rel}/final_model_summary.json",
            "artifact_subdirs": [
                f"{repo_rel}/artifacts/xgb",
                f"{repo_rel}/artifacts/residualmlp",
                f"{repo_rel}/artifacts/gcn",
            ],
            "logs_dir": f"{repo_rel}/logs",
        },
        "local_cv_sources": {
            "ml_dl": residual_rel,
            "graph_groupcv": graph_rel,
            "gcn_hyperparams": "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1/gcn_tuning_summary.json",
        },
        "metrics_sidecar_schema": {
            "path_pattern": "artifacts/{xgb|residualmlp|gcn}/metrics.json",
            "optional_keys": [
                "sagemaker_training_job",
                "sagemaker_status",
                "artifact_uri",
                "training_logs_uri",
                "final_eval: {rmse, mae, spearman, ndcg20, hit20}",
            ],
            "note": "train_tabular.py writes nested train/valid; for final reporting add final_eval or copy eval block to top level.",
        },
        "submit_scripts": {
            "xgb": "ml/pilot_sagemaker/submit_final_xgb_sagemaker.py → train_tabular.py (xgboost, full_train on)",
            "residualmlp": "ml/pilot_sagemaker/submit_final_residual_mlp_sagemaker.py → train_residual_mlp_final.py",
            "gcn": "ml/pilot_sagemaker/submit_final_gcn_sagemaker.py → train_gcn_final.py (baseline A)",
        },
        "sync_artifacts": "ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py --family {xgb|residualmlp|gcn} --model-tar /path/to/model.tar.gz",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("artifacts/xgb", "artifacts/residualmlp", "artifacts/gcn", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    residual_path = Path(args.residual_summary)
    graph_path = Path(args.graph_groupcv_summary)
    rows = build_template_rows(residual_path, graph_path)
    df = pd.DataFrame(rows)

    if args.collect:
        df = apply_collect(df, out_dir)

    cmp_path = out_dir / "final_model_comparison.csv"
    df.to_csv(cmp_path, index=False)

    sum_rel_residual = str(residual_path.resolve().relative_to(_REPO.resolve()))
    sum_rel_graph = str(graph_path.resolve().relative_to(_REPO.resolve()))
    cmp_rel = str(cmp_path.resolve().relative_to(_REPO.resolve()))
    summary = build_summary_json(out_dir, cmp_rel, sum_rel_residual, sum_rel_graph)
    sum_path = out_dir / "final_model_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {cmp_path}")
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
