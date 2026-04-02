"""
SageMaker (or local) final training: GCN baseline A (hidden 64, lr 1e-3, wd 1e-5),
same graph build and pair features as run_graph_gnn_cv.py.

Single random train/val split on rows (default test_size=0.1) for early stopping
and reported metrics — not K-fold (final run, not CV).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from graph_baseline_data import load_drug_targets_dict, load_merged_pair_frame
from run_graph_gnn_cv import adj_to_tensors, train_one_fold
from run_network_proximity_baseline import build_adjacency, load_disease_genes, try_load_ppi


def _pip_install_requirements() -> None:
    req = Path(__file__).resolve().parent / "requirements.txt"
    if req.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_s3", type=str, required=True)
    p.add_argument("--features_s3", type=str, required=True)
    p.add_argument("--drug_target_s3", type=str, required=True)
    p.add_argument("--disease_genes_path", type=str, default="data/graph_baseline/disease_genes_common_v1.txt")
    p.add_argument("--ppi_edges_s3", type=str, default="")
    p.add_argument("--sample_id_col", type=str, default="sample_id")
    p.add_argument("--drug_id_col", type=str, default="canonical_drug_id")
    p.add_argument("--target_col", type=str, default="label_regression")
    p.add_argument("--target_gene_col", type=str, default="target_gene_symbol")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    _pip_install_requirements()
    args = parse_args()
    out_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feat_cols = load_merged_pair_frame(
        args.labels_s3,
        args.features_s3,
        args.sample_id_col,
        args.drug_id_col,
        args.target_col,
    )
    disease_set = set(load_disease_genes(Path(args.disease_genes_path)))
    drug_targets = load_drug_targets_dict(args.drug_target_s3, args.drug_id_col, args.target_gene_col)
    ppi = try_load_ppi(args.ppi_edges_s3.strip())
    adj, _, _ = build_adjacency(drug_targets, disease_set, ppi)
    for d in df[args.drug_id_col].astype(str).str.strip().unique():
        dn = f"D:{d}"
        if dn not in adj:
            adj[dn] = []
    nodes = sorted(adj.keys())
    id2i = {nid: i for i, nid in enumerate(nodes)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_hat, s_mat = adj_to_tensors(adj, nodes, device)

    dids = df[args.drug_id_col].astype(str).str.strip()
    drug_node_idx = np.array([id2i[f"D:{d}"] for d in dids], dtype=np.int64)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)

    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    ns = argparse.Namespace(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    ck_path = out_dir / "gcn_checkpoint.pt"
    _pred_va, m_row = train_one_fold(
        "gcn",
        df,
        feat_cols,
        y,
        drug_node_idx,
        X,
        tr_idx,
        va_idx,
        a_hat,
        s_mat,
        args.sample_id_col,
        ns,
        device,
        save_checkpoint_path=ck_path,
        checkpoint_extras={
            "nodes": nodes,
            "feat_cols": feat_cols,
            "sample_id_col": args.sample_id_col,
            "drug_id_col": args.drug_id_col,
            "target_col": args.target_col,
            "inputs": {
                "labels": args.labels_s3,
                "features": args.features_s3,
                "drug_target": args.drug_target_s3,
                "disease_genes_path": args.disease_genes_path,
            },
        },
    )

    job = os.environ.get("TRAINING_JOB_NAME", "")
    metrics = {
        "model": "gcn",
        "baseline_tag": "A (hidden 64, lr 1e-3, weight_decay 1e-5)",
        "evaluation_note": (
            "Metrics are on a single random row split (test_size) for early stopping and reporting; "
            "they are not drug-group CV means. Model selection and local comparison use graph_family_groupcv."
        ),
        "rows_total": int(len(df)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "sagemaker_training_job": job,
        "sagemaker_status": "Completed" if job else "local",
        "final_eval": {
            "rmse": m_row["RMSE"],
            "mae": m_row["MAE"],
            "spearman": m_row["Spearman"],
            "ndcg20": m_row["NDCG@20"],
            "hit20": m_row["Hit@20"],
        },
        "validation_metrics": m_row,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("[METRICS]", json.dumps(metrics["final_eval"], ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
