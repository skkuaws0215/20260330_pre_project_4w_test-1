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
    p = argparse.ArgumentParser(description="ResidualMLP 5-fold CV and XGBoost tuned CV comparison.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument(
        "--xgb-best-params-json",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/xgb_cv_tuning/xgb_best_params.json",
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


def train_residual_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = ResidualMLP(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    xva = torch.from_numpy(X_va)
    yva = torch.from_numpy(y_va)

    best_state = None
    best_val = float("inf")
    no_imp = 0
    for _ in range(epochs):
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
        if no_imp >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xva).numpy()
    return pred.astype(np.float32)


def fold_metric_row(df_valid: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    sp = safe_spearman(y_true, pred)
    ndcg20, hit20 = rank_metrics(df_valid, sample_id_col, y_true, pred)
    return {"RMSE": rmse, "MAE": mae, "Spearman": sp, "NDCG@20": ndcg20, "Hit@20": hit20}


def summarize(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for m in ["RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]:
        out[f"{m}_mean"] = float(df[m].mean())
        out[f"{m}_std"] = float(df[m].std(ddof=0))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    df = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + [args.target_col])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    df = df[key + [args.target_col] + feat_cols].copy().sort_values(key).reset_index(drop=True)

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # ResidualMLP CV
    residual_rows: list[dict[str, float | int]] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr = X[tr_idx].copy()
        X_va = X[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        binary_idx, cont_idx = [], []
        for j in range(X_tr.shape[1]):
            vals = np.unique(X_tr[:, j])
            if set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
                binary_idx.append(j)
            else:
                cont_idx.append(j)
        if cont_idx:
            scaler = StandardScaler()
            X_tr[:, cont_idx] = scaler.fit_transform(X_tr[:, cont_idx])
            X_va[:, cont_idx] = scaler.transform(X_va[:, cont_idx])

        pred = train_residual_fold(
            X_tr,
            y_tr,
            X_va,
            y_va,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed + fold,
        )
        m = fold_metric_row(df.iloc[va_idx], args.sample_id_col, y_va, pred)
        residual_rows.append({"fold": fold, **m})

    residual_fold_df = pd.DataFrame(residual_rows).round(6)
    residual_fold_df.to_csv(out_dir / "residual_mlp_cv_fold_metrics.csv", index=False)
    residual_summary = summarize(residual_fold_df)

    # XGBoost tuned CV (same folds for direct comparison)
    import xgboost as xgb

    best = json.loads(Path(args.xgb_best_params_json).read_text(encoding="utf-8"))
    xgb_params = best["best_params"]
    xgb_rows: list[dict[str, float | int]] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
            tree_method="hist",
            **xgb_params,
        )
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[va_idx]).astype(np.float32)
        m = fold_metric_row(df.iloc[va_idx], args.sample_id_col, y[va_idx], pred)
        xgb_rows.append({"fold": fold, **m})
    xgb_fold_df = pd.DataFrame(xgb_rows).round(6)
    xgb_summary = summarize(xgb_fold_df)

    comp = pd.DataFrame(
        [
            {"model": "XGBoost_tuned_cv", **xgb_summary},
            {"model": "ResidualMLP_cv", **residual_summary},
        ]
    ).round(6)
    comp.to_csv(out_dir / "xgb_vs_residual_mlp_cv_comparison.csv", index=False)

    summary = {
        "dataset": "pair_features_newfe_v2 (LINCS excluded)",
        "rows": int(len(df)),
        "feature_cols_used": int(len(feat_cols)),
        "cv_config": {"n_splits": args.n_splits, "shuffle": True, "random_state": args.seed},
        "residual_mlp_train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
        },
        "xgb_tuned_params": xgb_params,
        "residual_mlp_cv_summary": {k: round(float(v), 6) for k, v in residual_summary.items()},
        "xgb_tuned_cv_summary": {k: round(float(v), 6) for k, v in xgb_summary.items()},
        "selection_note": "BlockWiseMLP is deprioritized due to incomplete block grouping; ResidualMLP stability check is prioritized.",
    }
    (out_dir / "residual_mlp_cv_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
