from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["xgb", "residualmlp"], required=True)
    p.add_argument("--labels_s3", required=True)
    p.add_argument("--features_s3", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_splits", type=int, default=5)
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


def _summ(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in ["RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=0))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path("/opt/ml/model")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(args.labels_s3)
    feats = pd.read_parquet(args.features_s3)
    key = ["sample_id", "canonical_drug_id"]
    df = labels[key + ["label_regression"]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + ["label_regression"])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    df = df[key + ["label_regression"] + feat_cols].sort_values(key).reset_index(drop=True)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df["label_regression"].to_numpy(dtype=np.float32)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X[tr_idx].copy(), X[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]
        if args.model == "xgb":
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=args.seed + fold,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va).astype(np.float32)
        else:
            cont_idx = []
            for j in range(X_tr.shape[1]):
                vals = np.unique(X_tr[:, j])
                if not set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
                    cont_idx.append(j)
            if cont_idx:
                sc = StandardScaler()
                X_tr[:, cont_idx] = sc.fit_transform(X_tr[:, cont_idx])
                X_va[:, cont_idx] = sc.transform(X_va[:, cont_idx])
            torch.manual_seed(args.seed + fold)
            model = ResidualMLP(X_tr.shape[1])
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
            xva = torch.from_numpy(X_va)
            yva = torch.from_numpy(y_va)
            best, bad, best_state = float("inf"), 0, None
            for _ in range(45):
                model.train()
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = float(loss_fn(model(xva), yva).item())
                if vl < best:
                    best = vl
                    bad = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 8:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred = model(xva).numpy().astype(np.float32)
        ndcg20, hit20 = rank_metrics(df.iloc[va_idx], "sample_id", y_va, pred)
        rows.append(
            {
                "fold": fold,
                "RMSE": float(np.sqrt(mean_squared_error(y_va, pred))),
                "MAE": float(mean_absolute_error(y_va, pred)),
                "Spearman": safe_spearman(y_va, pred),
                "NDCG@20": ndcg20,
                "Hit@20": hit20,
            }
        )
    fold_df = pd.DataFrame(rows).round(6)
    fold_csv = f"{args.model}_cv_fold_metrics.csv"
    fold_df.to_csv(out_dir / fold_csv, index=False)
    (out_dir / f"{args.model}_cv_metrics.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "validation_type": "5-fold row KFold",
                "rows_total": int(len(df)),
                "fold_metrics_file": fold_csv,
                "summary": _summ(fold_df),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
