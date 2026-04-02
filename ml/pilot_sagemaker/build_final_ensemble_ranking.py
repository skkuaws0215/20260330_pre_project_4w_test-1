from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_graph_gnn_cv import GCNStack, adj_to_tensors
from run_network_proximity_baseline import build_adjacency, load_disease_genes, try_load_ppi

KEY_COLS = ["sample_id", "canonical_drug_id"]


def _feature_frame_for_model_cols(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Columns expected by a checkpoint, in order; missing names (e.g. omitted LINCS) -> 0.0."""
    out: dict[str, pd.Series] = {}
    for c in cols:
        if c in feats.columns:
            out[c] = feats[c]
        else:
            out[c] = pd.Series(0.0, index=feats.index, dtype=np.float64)
    return pd.DataFrame(out, index=feats.index)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU())
        self.b1 = ResidualBlock(256)
        self.b2 = ResidualBlock(256)
        self.b3 = ResidualBlock(256)
        self.out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        return self.out(h).squeeze(-1)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    base = repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331"
    sm = base / "sagemaker_final_three"
    p = argparse.ArgumentParser(description="Build local weighted ensemble ranking from final 3 model artifacts.")
    p.add_argument("--features-uri", default=str(base / "final_pathway_addon/pair_features_newfe_v2.parquet"))
    p.add_argument("--drug-target-uri", default=str(base / "input_refs/drug_target_map_20260331.parquet"))
    p.add_argument("--disease-genes-path", default=str(repo / "data/graph_baseline/disease_genes_common_v1.txt"))
    p.add_argument("--ppi-edges-uri", default="")
    p.add_argument("--xgb-artifact", default=str(sm / "artifacts/xgb/artifact.joblib"))
    p.add_argument("--mlp-checkpoint", default=str(sm / "artifacts/residualmlp/checkpoint.pt"))
    p.add_argument("--gcn-checkpoint", default=str(sm / "artifacts/gcn/gcn_checkpoint.pt"))
    p.add_argument("--out-csv", default=str(sm / "final_ensemble_ranking.csv"))
    return p.parse_args()


def _predict_xgb(feats: pd.DataFrame, artifact_path: Path) -> pd.DataFrame:
    bundle = joblib.load(artifact_path)
    model = bundle["model"]
    cols = bundle["feature_columns"]
    X = _feature_frame_for_model_cols(feats, cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pred = model.predict(X)
    return feats[KEY_COLS].assign(pred_xgb=pred.astype(float))


def _predict_residual_mlp(feats: pd.DataFrame, ckpt_path: Path) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cols = ckpt["feat_cols"]
    cont_idx = list(ckpt.get("cont_idx", []))
    X = (
        _feature_frame_for_model_cols(feats, cols)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if cont_idx:
        mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        X[:, cont_idx] = (X[:, cont_idx] - mean) / np.clip(scale, 1e-12, None)
    model = ResidualMLP(X.shape[1])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy().astype(np.float64)
    return feats[KEY_COLS].assign(pred_residualmlp=pred)


def _predict_gcn(
    feats: pd.DataFrame,
    ckpt_path: Path,
    drug_target_uri: str,
    disease_genes_path: str,
    ppi_edges_uri: str,
) -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    nodes: list[str] = list(ckpt["nodes"])
    id2i = {nid: i for i, nid in enumerate(nodes)}

    dt = pd.read_parquet(drug_target_uri)[["canonical_drug_id", "target_gene_symbol"]].dropna()
    dt["canonical_drug_id"] = dt["canonical_drug_id"].astype(str).str.strip()
    dt["target_gene_symbol"] = dt["target_gene_symbol"].astype(str).str.strip().str.upper()
    drug_targets: dict[str, set[str]] = {}
    for did, g in dt.groupby("canonical_drug_id"):
        drug_targets[str(did)] = set(g["target_gene_symbol"].tolist())

    disease_set = set(load_disease_genes(Path(disease_genes_path)))
    ppi = try_load_ppi(ppi_edges_uri.strip())
    adj, _, _ = build_adjacency(drug_targets, disease_set, ppi)
    for d in feats["canonical_drug_id"].astype(str).str.strip().unique():
        dn = f"D:{d}"
        if dn not in adj:
            adj[dn] = []
    known = set(nodes)
    safe_adj: dict[str, list[str]] = {}
    for u in nodes:
        vs = adj.get(u, [])
        safe_adj[u] = [v for v in vs if v in known]
    a_hat, _s = adj_to_tensors(safe_adj, nodes, torch.device("cpu"))

    feat_cols = ckpt["feat_cols"]
    X = (
        _feature_frame_for_model_cols(feats, feat_cols)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    dids = feats["canonical_drug_id"].astype(str).str.strip()
    drug_idx = np.array([id2i[f"D:{d}"] for d in dids], dtype=np.int64)

    h0 = ckpt["h0"].to(torch.float32)
    hidden_dim = int(ckpt["hidden_dim"])
    pair_feat_dim = int(ckpt["pair_feat_dim"])
    gnn = GCNStack(hidden_dim, hidden_dim, n_layers=2)
    gnn.load_state_dict(ckpt["gnn_state_dict"])
    head = nn.Linear(hidden_dim + pair_feat_dim, 1)
    head.load_state_dict(ckpt["head_state_dict"])
    gnn.eval()
    head.eval()
    with torch.no_grad():
        h = gnn(h0, a_hat)
        x = torch.tensor(X, dtype=torch.float32)
        d = torch.tensor(drug_idx, dtype=torch.long)
        pred = head(torch.cat([h[d], x], dim=-1)).squeeze(-1).numpy().astype(np.float64)
    return feats[KEY_COLS].assign(pred_gcn=pred)


def main() -> None:
    args = parse_args()
    feats = pd.read_parquet(args.features_uri).sort_values(KEY_COLS).reset_index(drop=True)

    xgb_df = _predict_xgb(feats, Path(args.xgb_artifact))
    mlp_df = _predict_residual_mlp(feats, Path(args.mlp_checkpoint))
    gcn_df = _predict_gcn(
        feats=feats,
        ckpt_path=Path(args.gcn_checkpoint),
        drug_target_uri=args.drug_target_uri,
        disease_genes_path=args.disease_genes_path,
        ppi_edges_uri=args.ppi_edges_uri,
    )

    merged = xgb_df.merge(mlp_df, on=KEY_COLS, how="inner").merge(gcn_df, on=KEY_COLS, how="inner")
    merged["ensemble_score"] = (
        0.5 * merged["pred_xgb"] + 0.3 * merged["pred_residualmlp"] + 0.2 * merged["pred_gcn"]
    )
    merged = merged.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1, dtype=np.int64)
    merged["is_top100"] = merged["rank"] <= 100

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(merged)} rows)")
    print("Top100 preview:")
    print(merged.loc[merged["rank"] <= 5, KEY_COLS + ["ensemble_score", "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
