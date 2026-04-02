from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_baseline_data import load_merged_pair_frame
from run_graph_gnn_cv import adj_to_tensors, train_one_fold
from run_network_proximity_baseline import build_adjacency, load_disease_genes, try_load_ppi
from graph_baseline_data import load_drug_targets_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_s3", required=True)
    p.add_argument("--features_s3", required=True)
    p.add_argument("--drug_target_s3", required=True)
    p.add_argument("--cv_fold_json", default="cv_fold_indices_drug_group.json")
    p.add_argument("--disease_genes_path", default="data/graph_baseline/disease_genes_common_v1.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    return p.parse_args()


def _summ(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in ["RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=0))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_meta = json.loads((Path(__file__).resolve().parent / args.cv_fold_json).read_text(encoding="utf-8"))

    sample_id_col = "sample_id"
    drug_id_col = "canonical_drug_id"
    target_col = "label_regression"
    target_gene_col = "target_gene_symbol"

    df, feat_cols = load_merged_pair_frame(args.labels_s3, args.features_s3, sample_id_col, drug_id_col, target_col)
    disease_set = set(load_disease_genes(Path(args.disease_genes_path)))
    drug_targets = load_drug_targets_dict(args.drug_target_s3, drug_id_col, target_gene_col)
    ppi = try_load_ppi("")
    adj, _, _ = build_adjacency(drug_targets, disease_set, ppi)
    for d in df[drug_id_col].astype(str).str.strip().unique():
        dn = f"D:{d}"
        if dn not in adj:
            adj[dn] = []
    nodes = sorted(adj.keys())
    id2i = {nid: i for i, nid in enumerate(nodes)}
    device = torch.device("cpu")
    a_hat, s_mat = adj_to_tensors(adj, nodes, device)

    dids = df[drug_id_col].astype(str).str.strip()
    drug_node_idx = np.array([id2i[f"D:{d}"] for d in dids], dtype=np.int64)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    ns = argparse.Namespace(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )

    rows = []
    for fold_info in cv_meta["folds"]:
        tr = np.array(fold_info["train_indices"], dtype=int)
        va = np.array(fold_info["valid_indices"], dtype=int)
        _, m = train_one_fold(
            "gcn",
            df,
            feat_cols,
            y,
            drug_node_idx,
            X,
            tr,
            va,
            a_hat,
            s_mat,
            sample_id_col,
            ns,
            device,
        )
        rows.append({"fold": int(fold_info["fold"]), **m})
    fold_df = pd.DataFrame(rows).round(6)
    fold_df.to_csv(out_dir / "gcn_groupcv_fold_metrics.csv", index=False)
    summary = _summ(fold_df)
    (out_dir / "gcn_groupcv_summary.json").write_text(
        json.dumps(
            {
                "model": "gcn",
                "validation_type": "5-fold GroupKFold(by canonical_drug_id)",
                "rows_total": int(len(df)),
                "fold_metrics_file": "gcn_groupcv_fold_metrics.csv",
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
