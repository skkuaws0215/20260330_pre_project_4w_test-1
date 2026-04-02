"""
GraphSAGE or GCN baseline on drug–gene bipartite graph + tabular pair features.
Same merged pair table, same cv_fold_indices.json as ML/DL and Network Proximity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.preprocessing import StandardScaler

from graph_baseline_data import load_cv_meta, load_drug_targets_dict, load_merged_pair_frame
from run_network_proximity_baseline import build_adjacency, load_disease_genes, try_load_ppi


def _resolve_out_path(out_dir: Path, name: str | None, default_name: str) -> Path:
    if not name:
        return out_dir / default_name
    p = Path(name)
    return p if p.is_absolute() else out_dir / p


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="GraphSAGE or GCN 5-fold CV on drug-gene graph + pair features.")
    p.add_argument("--model", choices=["sage", "gcn"], required=True)
    p.add_argument(
        "--labels-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet"
        ),
    )
    p.add_argument(
        "--features-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet"
        ),
    )
    p.add_argument(
        "--drug-target-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/drug_target_map_20260331.parquet"
        ),
    )
    p.add_argument(
        "--disease-genes-path",
        default=str(repo / "data/graph_baseline/disease_genes_common_v1.txt"),
    )
    p.add_argument("--ppi-edges-uri", default="")
    p.add_argument(
        "--cv-fold-json",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/model_selection_stage1/cv_fold_indices.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1"),
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--target-gene-col", default="target_gene_symbol")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--partial-csv",
        default=None,
        help="GNN partial metrics CSV (default: graph_gnn_{sage|gcn}_partial.csv in out-dir).",
    )
    return p.parse_args()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(s) if s is not None and pd.notna(s) else float("nan")


def rank_metrics(df_valid: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tmp = df_valid[[sample_id_col]].copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    ndcgs, hits = [], []
    for _, g in tmp.groupby(sample_id_col):
        if len(g) < 2:
            continue
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        m = float(np.min(yt))
        if m < 0:
            yt = yt - m
        k = min(20, len(g))
        ndcgs.append(float(ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=k)))
        top_t = set(np.argsort(-yt)[:k].tolist())
        top_p = set(np.argsort(-yp)[:k].tolist())
        hits.append(1.0 if top_t.intersection(top_p) else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else float("nan"), float(np.mean(hits)) if hits else float("nan")


def fold_metrics(df_va: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ndcg, hit = rank_metrics(df_va, sample_id_col, y_true, pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "Spearman": safe_spearman(y_true, pred),
        "NDCG@20": ndcg,
        "Hit@20": hit,
    }


def cont_binary_indices(X_tr: np.ndarray) -> tuple[list[int], list[int]]:
    cont_idx, bin_idx = [], []
    for j in range(X_tr.shape[1]):
        vals = np.unique(X_tr[:, j])
        if set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
            bin_idx.append(j)
        else:
            cont_idx.append(j)
    return cont_idx, bin_idx


def scale_continuous(X_tr: np.ndarray, X_va: np.ndarray, cont_idx: list[int]) -> None:
    if not cont_idx:
        return
    sc = StandardScaler()
    X_tr[:, cont_idx] = sc.fit_transform(X_tr[:, cont_idx])
    X_va[:, cont_idx] = sc.transform(X_va[:, cont_idx])


def adj_to_tensors(adj: dict[str, list[str]], nodes: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return A_hat (GCN) and S (row-normalized neighbor aggregation, no self) both N x N dense."""
    n = len(nodes)
    id2i = {nid: i for i, nid in enumerate(nodes)}
    A = np.zeros((n, n), dtype=np.float32)
    for u, vs in adj.items():
        iu = id2i[u]
        for v in vs:
            A[iu, id2i[v]] = 1.0
    A = np.maximum(A, A.T)
    deg = A.sum(axis=1, keepdims=True).clip(min=1e-6)
    S = A / deg
    I = np.eye(n, dtype=np.float32)
    A_sym = A + I
    d = np.power(A_sym.sum(axis=1), -0.5)
    d = np.where(np.isfinite(d), d, 0.0)
    A_hat = (d[:, None] * A_sym * d[None, :]).astype(np.float32)
    return torch.tensor(A_hat, device=device), torch.tensor(S, device=device)


class GCNStack(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.layers.append(nn.Linear(d, hidden))
            d = hidden
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.layers):
            h = a_hat @ h
            h = lin(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class GraphSAGEStack(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int = 2) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.lins = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.lins.append(nn.Linear(2 * d, hidden))
            d = hidden
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.lins):
            neigh = s @ h
            h = torch.cat([h, neigh], dim=-1)
            h = lin(h)
            if i < len(self.lins) - 1:
                h = F.relu(h)
        return h


def train_one_fold(
    model_name: str,
    df: pd.DataFrame,
    feat_cols: list[str],
    y: np.ndarray,
    drug_node_idx: np.ndarray,
    X: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    a_hat: torch.Tensor,
    s_mat: torch.Tensor,
    sample_id_col: str,
    args: argparse.Namespace,
    device: torch.device,
    save_checkpoint_path: Path | None = None,
    checkpoint_extras: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(args.seed)
    X_tr, X_va = X[tr_idx].copy(), X[va_idx].copy()
    cont_idx, _ = cont_binary_indices(X_tr)
    scale_continuous(X_tr, X_va, cont_idx)
    y_tr, y_va = y[tr_idx], y[va_idx]
    d_tr = torch.tensor(drug_node_idx[tr_idx], dtype=torch.long, device=device)
    d_va = torch.tensor(drug_node_idx[va_idx], dtype=torch.long, device=device)
    x_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    x_va = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    n = a_hat.shape[0]
    in_dim = args.hidden_dim
    h0 = nn.Parameter(torch.randn(n, in_dim, device=device) * 0.1)
    if model_name == "gcn":
        gnn = GCNStack(in_dim, args.hidden_dim, n_layers=2).to(device)
    else:
        gnn = GraphSAGEStack(in_dim, args.hidden_dim, n_layers=2).to(device)
    head = nn.Linear(args.hidden_dim + X.shape[1], 1).to(device)
    params = [h0] + list(gnn.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    bad = 0

    for _ in range(args.epochs):
        gnn.train()
        head.train()
        opt.zero_grad()
        mat = a_hat if model_name == "gcn" else s_mat
        h = gnn(h0, mat)
        pred = head(torch.cat([h[d_tr], x_tr], dim=-1)).squeeze(-1)
        loss = F.mse_loss(pred, y_tr_t)
        loss.backward()
        opt.step()

        gnn.eval()
        head.eval()
        with torch.no_grad():
            h = gnn(h0, mat)
            p_va = head(torch.cat([h[d_va], x_va], dim=-1)).squeeze(-1)
            vloss = F.mse_loss(p_va, y_va_t).item()
        if vloss < best_val:
            best_val = vloss
            bad = 0
            best_state = (
                h0.detach().clone(),
                {k: v.cpu().clone() for k, v in gnn.state_dict().items()},
                {k: v.cpu().clone() for k, v in head.state_dict().items()},
            )
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        h0.data.copy_(best_state[0].to(device))
        gnn.load_state_dict({k: v.to(device) for k, v in best_state[1].items()})
        head.load_state_dict({k: v.to(device) for k, v in best_state[2].items()})
    gnn.eval()
    head.eval()
    with torch.no_grad():
        mat = a_hat if model_name == "gcn" else s_mat
        h = gnn(h0, mat)
        p_va = head(torch.cat([h[d_va], x_va], dim=-1)).squeeze(-1).cpu().numpy()
    df_va = df.iloc[va_idx].reset_index(drop=True)
    m = fold_metrics(df_va, sample_id_col, y_va, p_va)
    if save_checkpoint_path is not None:
        save_checkpoint_path = Path(save_checkpoint_path)
        save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_name": model_name,
            "h0": h0.detach().cpu(),
            "gnn_state_dict": {k: v.cpu() for k, v in gnn.state_dict().items()},
            "head_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
            "hidden_dim": args.hidden_dim,
            "pair_feat_dim": int(X.shape[1]),
            "n_graph_nodes": int(a_hat.shape[0]),
        }
        if checkpoint_extras:
            payload.update(checkpoint_extras)
        torch.save(payload, save_checkpoint_path)
    return p_va, m


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feat_cols = load_merged_pair_frame(
        args.labels_uri,
        args.features_uri,
        args.sample_id_col,
        args.drug_id_col,
        args.target_col,
    )
    cv_meta = load_cv_meta(Path(args.cv_fold_json), len(df))

    disease_set = set(load_disease_genes(Path(args.disease_genes_path)))
    drug_targets = load_drug_targets_dict(args.drug_target_uri, args.drug_id_col, args.target_gene_col)
    ppi = try_load_ppi(args.ppi_edges_uri.strip())
    adj, _, _ = build_adjacency(drug_targets, disease_set, ppi)
    for d in df[args.drug_id_col].astype(str).str.strip().unique():
        dn = f"D:{d}"
        if dn not in adj:
            adj[dn] = []
    nodes = sorted(adj.keys())
    id2i = {nid: i for i, nid in enumerate(nodes)}
    a_hat, s_mat = adj_to_tensors(adj, nodes, device)

    dids = df[args.drug_id_col].astype(str).str.strip()
    drug_node_idx = np.array([id2i[f"D:{d}"] for d in dids], dtype=np.int64)

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)

    model_label = "GraphSAGE" if args.model == "sage" else "GCN"
    fold_rows: list[dict[str, Any]] = []
    spearmans: list[float] = []
    for fold_info in cv_meta["folds"]:
        tr = np.array(fold_info["train_indices"], dtype=int)
        va = np.array(fold_info["valid_indices"], dtype=int)
        _, m = train_one_fold(
            args.model,
            df,
            feat_cols,
            y,
            drug_node_idx,
            X,
            tr,
            va,
            a_hat,
            s_mat,
            args.sample_id_col,
            args,
            device,
        )
        spearmans.append(m["Spearman"])
        fold_rows.append({"model": model_label, "fold": fold_info["fold"], **m})

    mean_row = {
        "model": model_label,
        "fold": "mean",
        "RMSE": float(np.mean([r["RMSE"] for r in fold_rows])),
        "MAE": float(np.mean([r["MAE"] for r in fold_rows])),
        "Spearman": float(np.nanmean([r["Spearman"] for r in fold_rows])),
        "NDCG@20": float(np.nanmean([r["NDCG@20"] for r in fold_rows])),
        "Hit@20": float(np.nanmean([r["Hit@20"] for r in fold_rows])),
    }
    std_row = {
        "model": model_label,
        "fold": "spearman_std",
        "RMSE": float("nan"),
        "MAE": float("nan"),
        "Spearman": float(np.nanstd(spearmans, ddof=0)),
        "NDCG@20": float("nan"),
        "Hit@20": float("nan"),
    }
    part = pd.DataFrame(fold_rows + [mean_row, std_row])
    default_partial = f"graph_gnn_{args.model}_partial.csv"
    part_path = _resolve_out_path(out_dir, args.partial_csv, default_partial)
    part.to_csv(part_path, index=False)

    print(json.dumps({"model": model_label, "wrote_partial": str(part_path), "spearman_mean": mean_row["Spearman"]}, indent=2))


if __name__ == "__main__":
    main()
