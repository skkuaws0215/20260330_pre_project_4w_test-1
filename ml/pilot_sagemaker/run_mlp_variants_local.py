from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run block-wise and residual MLP baselines.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--xgb-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/model_dataset_metrics.csv",
    )
    p.add_argument(
        "--flat-mlp-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/mlp_baseline/mlp_metrics.csv",
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
        min_yt = float(np.min(yt))
        if min_yt < 0:
            yt = yt - min_yt
        k = min(20, len(g))
        ndcgs.append(float(ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=k)))
        top_t = set(np.argsort(-yt)[:k].tolist())
        top_p = set(np.argsort(-yp)[:k].tolist())
        hits.append(1.0 if top_t.intersection(top_p) else 0.0)
    return float(np.mean(ndcgs)), float(np.mean(hits))


class BlockWiseMLP(nn.Module):
    def __init__(self, in_path: int, in_chem: int, in_target: int):
        super().__init__()
        ep, ec, et = 32, 128, 32
        self.in_path, self.in_chem, self.in_target = in_path, in_chem, in_target
        self.path_enc = nn.Sequential(nn.Linear(max(1, in_path), 64), nn.ReLU(), nn.Linear(64, ep), nn.ReLU())
        self.chem_enc = nn.Sequential(nn.Linear(max(1, in_chem), 256), nn.ReLU(), nn.Linear(256, ec), nn.ReLU())
        self.target_enc = nn.Sequential(nn.Linear(max(1, in_target), 64), nn.ReLU(), nn.Linear(64, et), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(ep + ec + et, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x_path, x_chem, x_target):
        p = self.path_enc(x_path if self.in_path > 0 else torch.zeros((x_path.shape[0], 1), device=x_path.device))
        c = self.chem_enc(x_chem if self.in_chem > 0 else torch.zeros((x_chem.shape[0], 1), device=x_chem.device))
        t = self.target_enc(x_target if self.in_target > 0 else torch.zeros((x_target.shape[0], 1), device=x_target.device))
        h = torch.cat([p, c, t], dim=1)
        return self.head(h).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(x + out)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU())
        self.block1 = ResidualBlock(256)
        self.block2 = ResidualBlock(256)
        self.block3 = ResidualBlock(256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.out(h).squeeze(-1)


def train_torch_model(model: nn.Module, train_loader, valid_data, epochs: int, lr: float, patience: int, mode: str) -> tuple[np.ndarray, list[dict]]:
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    no_improve = 0
    curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad()
            if mode == "block":
                xb_path, xb_chem, xb_target, yb = [t.to(device) for t in batch]
                pred = model(xb_path, xb_chem, xb_target)
            else:
                xb, yb = [t.to(device) for t in batch]
                pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            if mode == "block":
                xv_path, xv_chem, xv_target, yv = [t.to(device) for t in valid_data]
                pv = model(xv_path, xv_chem, xv_target)
            else:
                xv, yv = [t.to(device) for t in valid_data]
                pv = model(xv)
            val_loss = float(loss_fn(pv, yv).item())

        curve.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "valid_loss": val_loss, "model": mode})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if mode == "block":
            xv_path, xv_chem, xv_target, _ = [t.to(device) for t in valid_data]
            pred = model(xv_path, xv_chem, xv_target).cpu().numpy()
        else:
            xv, _ = [t.to(device) for t in valid_data]
            pred = model(xv).cpu().numpy()
    return pred.astype(np.float32), curve


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    merged = labels[key + [args.target_col]].merge(feats, on=key, how="inner").sort_values(key).reset_index(drop=True)
    feat_cols = [c for c in merged.columns if c not in set(key + [args.target_col])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    X_df = merged[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = merged[args.target_col].to_numpy(dtype=np.float32)

    binary_cols, continuous_cols = [], []
    for c in X_df.columns:
        vals = pd.unique(X_df[c].astype(np.float32))
        vals = vals[~pd.isna(vals)]
        uniq = set(np.round(vals, 6).tolist())
        if uniq.issubset({0.0, 1.0}):
            binary_cols.append(c)
        else:
            continuous_cols.append(c)

    idx = np.arange(len(merged))
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    X_all = X_df.to_numpy(dtype=np.float32)
    X_train = X_all[tr_idx].copy()
    X_valid = X_all[va_idx].copy()
    y_train = y[tr_idx]
    y_valid = y[va_idx]

    if continuous_cols:
        cont_idx = [feat_cols.index(c) for c in continuous_cols]
        scaler = StandardScaler()
        X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
        X_valid[:, cont_idx] = scaler.transform(X_valid[:, cont_idx])

    pathway_cols = [c for c in feat_cols if ("pathway" in c.lower() and not c.lower().startswith("target_"))]
    target_cols = [c for c in feat_cols if c.lower().startswith("target_")]
    chem_cols = [c for c in feat_cols if c not in set(pathway_cols + target_cols)]
    path_idx = [feat_cols.index(c) for c in pathway_cols]
    chem_idx = [feat_cols.index(c) for c in chem_cols]
    target_idx = [feat_cols.index(c) for c in target_cols]

    # Block-wise MLP
    xtr_p = torch.from_numpy(X_train[:, path_idx] if path_idx else np.zeros((len(X_train), 0), dtype=np.float32))
    xtr_c = torch.from_numpy(X_train[:, chem_idx] if chem_idx else np.zeros((len(X_train), 0), dtype=np.float32))
    xtr_t = torch.from_numpy(X_train[:, target_idx] if target_idx else np.zeros((len(X_train), 0), dtype=np.float32))
    ytr = torch.from_numpy(y_train.astype(np.float32))
    xv_p = torch.from_numpy(X_valid[:, path_idx] if path_idx else np.zeros((len(X_valid), 0), dtype=np.float32))
    xv_c = torch.from_numpy(X_valid[:, chem_idx] if chem_idx else np.zeros((len(X_valid), 0), dtype=np.float32))
    xv_t = torch.from_numpy(X_valid[:, target_idx] if target_idx else np.zeros((len(X_valid), 0), dtype=np.float32))
    yv = torch.from_numpy(y_valid.astype(np.float32))

    ds_block = torch.utils.data.TensorDataset(xtr_p, xtr_c, xtr_t, ytr)
    loader_block = torch.utils.data.DataLoader(ds_block, batch_size=args.batch_size, shuffle=True)
    model_block = BlockWiseMLP(len(path_idx), len(chem_idx), len(target_idx))
    pred_block, curve_block = train_torch_model(
        model_block, loader_block, (xv_p, xv_c, xv_t, yv), args.epochs, args.lr, args.patience, mode="block"
    )

    # Residual MLP
    xtr = torch.from_numpy(X_train.astype(np.float32))
    xv = torch.from_numpy(X_valid.astype(np.float32))
    ds_res = torch.utils.data.TensorDataset(xtr, ytr)
    loader_res = torch.utils.data.DataLoader(ds_res, batch_size=args.batch_size, shuffle=True)
    model_res = ResidualMLP(X_train.shape[1])
    pred_res, curve_res = train_torch_model(
        model_res, loader_res, (xv, yv), args.epochs, args.lr, args.patience, mode="residual"
    )

    rows = []
    for name, pred in [("BlockWiseMLP", pred_block), ("ResidualMLP", pred_res)]:
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        mae = float(mean_absolute_error(y_valid, pred))
        spearman = safe_spearman(y_valid, pred)
        ndcg20, hit20 = rank_metrics(merged.iloc[va_idx], args.sample_id_col, y_valid, pred)
        rows.append(
            {
                "model": name,
                "RMSE": rmse,
                "MAE": mae,
                "Spearman": spearman,
                "NDCG@20": ndcg20,
                "Hit@20": hit20,
            }
        )
    metrics_df = pd.DataFrame(rows).round(6)
    metrics_df.to_csv(out_dir / "mlp_variants_metrics.csv", index=False)

    xgb = pd.read_csv(args.xgb_csv)
    xgb_row = xgb[(xgb["model"] == "XGBoost") & (xgb["dataset"] == "newfe_v2")]
    flat = pd.read_csv(args.flat_mlp_csv).iloc[0]
    comp = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "RMSE": float(xgb_row.iloc[0]["RMSE"]),
                "MAE": float(xgb_row.iloc[0]["MAE"]),
                "Spearman": float(xgb_row.iloc[0]["Spearman"]),
                "NDCG@20": float(xgb_row.iloc[0]["NDCG@20"]),
                "Hit@20": float(xgb_row.iloc[0]["Hit@20"]),
            },
            {
                "model": "FlatMLP",
                "RMSE": float(flat["RMSE"]),
                "MAE": float(flat["MAE"]),
                "Spearman": float(flat["Spearman"]),
                "NDCG@20": float(flat["NDCG@20"]),
                "Hit@20": float(flat["Hit@20"]),
            },
        ]
    )
    comp = pd.concat([comp, metrics_df], ignore_index=True).round(6)
    comp.to_csv(out_dir / "xgb_mlp_variants_comparison.csv", index=False)

    curves = pd.DataFrame(curve_block + curve_res).round(8)
    curves.to_csv(out_dir / "mlp_variants_learning_curve.csv", index=False)

    best_idx = comp["Spearman"].astype(float).idxmax()
    best_model = str(comp.iloc[best_idx]["model"])
    summary = {
        "dataset": "pair_features_newfe_v2.parquet (LINCS excluded)",
        "rows_total": int(len(merged)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "feature_cols_total": int(len(feat_cols)),
        "blocks": {
            "pathway_cols": int(len(path_idx)),
            "chem_cols": int(len(chem_idx)),
            "target_cols": int(len(target_idx)),
        },
        "split": {"test_size": args.test_size, "seed": args.seed},
        "train_config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience},
        "best_by_spearman": best_model,
        "outputs": {
            "variants_metrics": str(out_dir / "mlp_variants_metrics.csv"),
            "comparison": str(out_dir / "xgb_mlp_variants_comparison.csv"),
            "learning_curve": str(out_dir / "mlp_variants_learning_curve.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
