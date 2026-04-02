"""Merge Network_Proximity + GraphSAGE + GCN partial CSVs into one comparison table and summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Merge graph baseline partial metrics into one comparison table.")
    p.add_argument(
        "--out-dir",
        default=str(repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1"),
    )
    p.add_argument(
        "--preset",
        choices=["round1", "groupcv"],
        default="round1",
        help="round1: row-KFold artifacts; groupcv: drug-group CV filenames.",
    )
    p.add_argument(
        "--ml-dl-summary-json",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/residual_mlp_cv/residual_mlp_cv_summary.json"
        ),
        help="If present, write ml_dl_graph_family_mean*.csv (XGBoost + ResidualMLP + graph top model).",
    )
    return p.parse_args()


def mean_row(df: pd.DataFrame, model: str) -> dict | None:
    sub = df[(df["model"] == model) & (df["fold"] == "mean")]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {
        "RMSE": float(r["RMSE"]),
        "MAE": float(r["MAE"]),
        "Spearman": float(r["Spearman"]),
        "NDCG@20": float(r["NDCG@20"]),
        "Hit@20": float(r["Hit@20"]),
    }


def spearman_std_row(df: pd.DataFrame, model: str) -> float | None:
    sub = df[(df["model"] == model) & (df["fold"] == "spearman_std")]
    if sub.empty:
        return None
    return float(sub.iloc[0]["Spearman"])


def _row_ml_dl(
    family: str,
    model: str,
    rm_m: float | None,
    rm_s: float | None,
    ma_m: float | None,
    ma_s: float | None,
    sp_m: float | None,
    sp_s: float | None,
    nd_m: float | None,
    nd_s: float | None,
    hi_m: float | None,
    hi_s: float | None,
) -> dict:
    return {
        "family": family,
        "model": model,
        "RMSE_mean": rm_m,
        "RMSE_std": rm_s,
        "MAE_mean": ma_m,
        "MAE_std": ma_s,
        "Spearman_mean": sp_m,
        "Spearman_std": sp_s,
        "NDCG@20_mean": nd_m,
        "NDCG@20_std": nd_s,
        "Hit@20_mean": hi_m,
        "Hit@20_std": hi_s,
    }


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if args.preset == "groupcv":
        comparison_out = out_dir / "graph_family_groupcv_comparison.csv"
        summary_out = out_dir / "graph_family_groupcv_summary.json"
        net_path = comparison_out
        sage_path = out_dir / "graph_gnn_sage_groupcv_partial.csv"
        gcn_path = out_dir / "graph_gnn_gcn_groupcv_partial.csv"
        ml_dl_csv_name = "ml_dl_graph_family_mean_groupcv.csv"
    else:
        comparison_out = out_dir / "graph_family_comparison.csv"
        summary_out = out_dir / "graph_family_summary.json"
        net_path = comparison_out
        sage_path = out_dir / "graph_gnn_sage_partial.csv"
        gcn_path = out_dir / "graph_gnn_gcn_partial.csv"
        ml_dl_csv_name = "ml_dl_graph_family_mean.csv"

    parts: list[pd.DataFrame] = []
    if net_path.is_file():
        net = pd.read_csv(net_path)
        net = net[net["model"] == "Network_Proximity"]
        if not net.empty:
            parts.append(net)
    if sage_path.is_file():
        parts.append(pd.read_csv(sage_path))
    if gcn_path.is_file():
        parts.append(pd.read_csv(gcn_path))
    if not parts:
        raise SystemExit(f"No inputs found under {out_dir} (preset={args.preset})")

    full = pd.concat(parts, ignore_index=True)
    full.to_csv(comparison_out, index=False)

    models = ["Network_Proximity", "GraphSAGE", "GCN"]
    metrics_by_model: dict[str, dict] = {}
    std_by_model: dict[str, float] = {}
    for m in models:
        mr = mean_row(full, m)
        if mr:
            metrics_by_model[m] = mr
        ss = spearman_std_row(full, m)
        if ss is not None:
            std_by_model[m] = ss

    winner = None
    best_sp = float("-inf")
    for m, met in metrics_by_model.items():
        sp = met.get("Spearman", float("nan"))
        if pd.notna(sp) and sp > best_sp:
            best_sp = sp
            winner = m
    if winner is None and metrics_by_model:
        winner = max(metrics_by_model.keys(), key=lambda k: metrics_by_model[k].get("Spearman", float("-inf")))

    summary: dict[str, Any] = {
        "preset": args.preset,
        "graph_family_comparison_csv": _repo_relative(repo_root, comparison_out),
        "metrics_mean_5fold_by_model": metrics_by_model,
        "spearman_std_across_folds_by_model": std_by_model,
        "recommended_graph_representative": winner,
        "selection_rule": "Highest Spearman mean (fold mean row); tie-break manually if needed.",
        "notes": "Network_Proximity is rule-based and non-sample-specific (drug-level z-score broadcast to pairs). GNN rows use drug node embedding + scaled pair features per row.",
    }

    if args.preset == "round1":
        summary["temporary_graph_representative_candidate"] = "GraphSAGE"
        summary["temporary_candidate_caveat"] = (
            "Row-based KFold can reuse the same canonical_drug_id in train and valid; GNN drug-node embeddings may "
            "see optimistic bias. Confirm with drug-group CV (graph_family_groupcv_*)."
        )

    if args.preset == "groupcv":
        summary["cv_type"] = "GroupKFold by canonical_drug_id (no drug in both train and valid)"
        summary["temporary_graph_representative_candidate"] = "GraphSAGE"
        summary["temporary_candidate_basis"] = (
            "Round1 row-based KFold (shared drug-node embeddings across train/val rows); subject to optimistic bias."
        )
        summary["representative_finalization_policy"] = (
            "Confirm Graph family representative only if group-CV results remain strong; compare to Round1 row-CV."
        )
        summary["gnn_transductive_caveat"] = (
            "Drug-group splits remove same-drug label leakage across folds; full-graph message passing can still "
            "backpropagate into all node embeddings during training (residual transductive coupling)."
        )

    ml_dl_path = Path(args.ml_dl_summary_json)
    cmp_rows: list[dict] = []
    if ml_dl_path.is_file():
        ml_dl = json.loads(ml_dl_path.read_text(encoding="utf-8"))
        xgb = ml_dl.get("xgb_tuned_cv_summary") or {}
        rml = ml_dl.get("residual_mlp_cv_summary") or {}
        cmp_rows.append(
            _row_ml_dl(
                "ML",
                "XGBoost_tuned_cv",
                xgb.get("RMSE_mean"),
                xgb.get("RMSE_std"),
                xgb.get("MAE_mean"),
                xgb.get("MAE_std"),
                xgb.get("Spearman_mean"),
                xgb.get("Spearman_std"),
                xgb.get("NDCG@20_mean"),
                xgb.get("NDCG@20_std"),
                xgb.get("Hit@20_mean"),
                xgb.get("Hit@20_std"),
            )
        )
        cmp_rows.append(
            _row_ml_dl(
                "DL",
                "ResidualMLP_cv",
                rml.get("RMSE_mean"),
                rml.get("RMSE_std"),
                rml.get("MAE_mean"),
                rml.get("MAE_std"),
                rml.get("Spearman_mean"),
                rml.get("Spearman_std"),
                rml.get("NDCG@20_mean"),
                rml.get("NDCG@20_std"),
                rml.get("Hit@20_mean"),
                rml.get("Hit@20_std"),
            )
        )
    if winner and winner in metrics_by_model:
        gm = metrics_by_model[winner]
        gs = std_by_model.get(winner)
        cmp_rows.append(
            _row_ml_dl(
                "Graph",
                winner,
                gm.get("RMSE"),
                None,
                gm.get("MAE"),
                None,
                gm.get("Spearman"),
                gs,
                gm.get("NDCG@20"),
                None,
                gm.get("Hit@20"),
                None,
            )
        )
    if cmp_rows:
        cmp_csv = out_dir / ml_dl_csv_name
        pd.DataFrame(cmp_rows).to_csv(cmp_csv, index=False)
        summary["ml_dl_graph_family_mean_csv"] = _repo_relative(repo_root, cmp_csv)
        summary["ml_dl_summary_source"] = (
            _repo_relative(repo_root, ml_dl_path) if ml_dl_path.is_file() else None
        )

    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(comparison_out), "summary": str(summary_out), "representative": winner}, indent=2))


if __name__ == "__main__":
    main()
