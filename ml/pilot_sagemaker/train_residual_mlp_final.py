"""
SageMaker (or local) final training: single ResidualMLP, same data prep as
run_residual_mlp_cv_local.py (LINCS excluded, binary vs continuous scaling).

Writes to SM_MODEL_DIR (or /opt/ml/model): checkpoint.pt, metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _pip_install_requirements() -> None:
    req = Path(__file__).resolve().parent / "requirements.txt"
    if req.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


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
    return float(np.mean(ndcgs)), float(np.mean(hits))


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


def fold_metric_row(df_valid: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    sp = safe_spearman(y_true, pred)
    ndcg20, hit20 = rank_metrics(df_valid, sample_id_col, y_true, pred)
    return {"RMSE": rmse, "MAE": mae, "Spearman": sp, "NDCG@20": ndcg20, "Hit@20": hit20}


def cont_binary_indices(X_tr: np.ndarray) -> tuple[list[int], list[int]]:
    cont_idx, bin_idx = [], []
    for j in range(X_tr.shape[1]):
        vals = np.unique(X_tr[:, j])
        if set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
            bin_idx.append(j)
        else:
            cont_idx.append(j)
    return cont_idx, bin_idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features_s3", type=str, required=True)
    p.add_argument("--labels_s3", type=str, required=True)
    p.add_argument("--sample_id_col", type=str, default="sample_id")
    p.add_argument("--drug_id_col", type=str, default="canonical_drug_id")
    p.add_argument("--target_col", type=str, default="label_regression")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    _pip_install_requirements()
    args = parse_args()
    out_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels_s3)
    feats = pd.read_parquet(args.features_s3)
    key = [args.sample_id_col, args.drug_id_col]
    df = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + [args.target_col])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    df = df[key + [args.target_col] + feat_cols].copy().sort_values(key).reset_index(drop=True)

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)

    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    X_tr = X[tr_idx].copy()
    X_va = X[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    cont_idx, _bin_idx = cont_binary_indices(X_tr)
    scaler = None
    if cont_idx:
        scaler = StandardScaler()
        X_tr[:, cont_idx] = scaler.fit_transform(X_tr[:, cont_idx])
        X_va[:, cont_idx] = scaler.transform(X_va[:, cont_idx])

    torch.manual_seed(args.seed)
    model = ResidualMLP(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    xva = torch.from_numpy(X_va)
    yva = torch.from_numpy(y_va)

    best_state = None
    best_val = float("inf")
    no_imp = 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xva), yva).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = model(xva).numpy().astype(np.float32)

    m_row = fold_metric_row(df.iloc[va_idx], args.sample_id_col, y_va, pred_va)
    job = os.environ.get("TRAINING_JOB_NAME", "")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cont_idx": cont_idx,
            "scaler_mean": scaler.mean_.tolist() if scaler is not None else [],
            "scaler_scale": scaler.scale_.tolist() if scaler is not None else [],
            "feat_cols": feat_cols,
            "train_config": {
                "epochs_cap": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "patience": args.patience,
                "test_size": args.test_size,
                "seed": args.seed,
            },
        },
        out_dir / "checkpoint.pt",
    )

    metrics = {
        "model": "residual_mlp",
        "validation_type": f"Single row-level holdout (test_size={args.test_size})",
        "evaluation_note": (
            "Single random row split (test_size), same preprocessing as run_residual_mlp_cv_local per fold; "
            "not identical to 5-fold CV mean in residual_mlp_cv_summary."
        ),
        "rows_total": int(len(df)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "n_features": int(len(feat_cols)),
        "inputs": {"features": args.features_s3, "labels": args.labels_s3},
        "sagemaker_training_job": job,
        "sagemaker_status": "Completed" if job else "local",
        "final_eval": {
            "rmse": m_row["RMSE"],
            "mae": m_row["MAE"],
            "spearman": m_row["Spearman"],
            "ndcg20": m_row["NDCG@20"],
            "hit20": m_row["Hit@20"],
        },
        "validation_fold_metrics_row": m_row,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[METRICS]", json.dumps(metrics["final_eval"], ensure_ascii=False))


if __name__ == "__main__":
    main()
