from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local VAE baseline on newfe_v2 target-only features.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent-dims", default="32,64")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--pred-weight", type=float, default=1.0)
    p.add_argument(
        "--xgb-mlp-comparison-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/mlp_baseline/xgb_vs_mlp_comparison.csv",
    )
    return p.parse_args()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    if s is None or pd.isna(s):
        return float("nan")
    return float(s)


def rank_metrics_by_sample(df_valid: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tmp = df_valid[[sample_id_col]].copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    ndcgs: list[float] = []
    hits: list[float] = []
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
        top_true = set(np.argsort(-yt)[:k].tolist())
        top_pred = set(np.argsort(-yp)[:k].tolist())
        hits.append(1.0 if top_true.intersection(top_pred) else 0.0)
    return (float(np.mean(ndcgs)) if ndcgs else float("nan"), float(np.mean(hits)) if hits else float("nan"))


def build_data(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str], np.ndarray, np.ndarray]:
    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    merged = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    merged = merged.sort_values(key).reset_index(drop=True)

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

    scaler = StandardScaler()
    if continuous_cols:
        cont_idx = [feat_cols.index(c) for c in continuous_cols]
        X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
        X_valid[:, cont_idx] = scaler.transform(X_valid[:, cont_idx])

    y_train = y[tr_idx]
    y_valid = y[va_idx]
    return merged, X_train, X_valid, y_train, y_valid, feat_cols, binary_cols, continuous_cols, tr_idx, va_idx


def train_eval_vae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    pred_weight: float,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = torch.device("cpu")
    in_dim = X_train.shape[1]
    h1, h2 = 512, 256

    class VAEReg(nn.Module):
        def __init__(self, input_dim: int, latent: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.mu = nn.Linear(h2, latent)
            self.logvar = nn.Linear(h2, latent)
            self.dec1 = nn.Linear(latent, h2)
            self.dec2 = nn.Linear(h2, h1)
            self.out_recon = nn.Linear(h1, input_dim)
            self.reg1 = nn.Linear(latent, 64)
            self.reg2 = nn.Linear(64, 1)

        def encode(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.mu(h), self.logvar(h)

        def reparam(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = F.relu(self.dec1(z))
            h = F.relu(self.dec2(h))
            return self.out_recon(h)

        def regress(self, z):
            h = F.relu(self.reg1(z))
            return self.reg2(h).squeeze(-1)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            recon = self.decode(z)
            pred = self.regress(z)
            return recon, pred, mu, logvar

    model = VAEReg(in_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    curve_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        sum_total = sum_recon = sum_kl = sum_pred = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            recon, pred, mu, logvar = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = F.mse_loss(pred, yb)
            total = recon_loss + beta * kl_loss + pred_weight * pred_loss
            opt.zero_grad()
            total.backward()
            opt.step()

            bs = xb.size(0)
            n += bs
            sum_total += float(total.item()) * bs
            sum_recon += float(recon_loss.item()) * bs
            sum_kl += float(kl_loss.item()) * bs
            sum_pred += float(pred_loss.item()) * bs

        curve_rows.append(
            {
                "epoch": epoch,
                "total_loss": sum_total / max(1, n),
                "recon_loss": sum_recon / max(1, n),
                "kl_loss": sum_kl / max(1, n),
                "pred_loss": sum_pred / max(1, n),
                "latent_dim": latent_dim,
            }
        )

    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(X_valid.astype(np.float32)).to(device)
        _, pred, _, _ = model(xv)
        pred_np = pred.cpu().numpy().astype(np.float32)
    return pred_np, pd.DataFrame(curve_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        merged,
        X_train,
        X_valid,
        y_train,
        y_valid,
        feat_cols,
        binary_cols,
        continuous_cols,
        tr_idx,
        va_idx,
    ) = build_data(args)

    latent_dims = [int(x.strip()) for x in args.latent_dims.split(",") if x.strip()]
    metric_rows = []
    curve_all = []
    for ld in latent_dims:
        pred, curve = train_eval_vae(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            latent_dim=ld,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta=args.beta,
            pred_weight=args.pred_weight,
            seed=args.seed,
        )
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        mae = float(mean_absolute_error(y_valid, pred))
        spear = safe_spearman(y_valid, pred)
        ndcg20, hit20 = rank_metrics_by_sample(merged.iloc[va_idx], args.sample_id_col, y_valid, pred)
        metric_rows.append(
            {
                "model": "VAE",
                "latent_dim": ld,
                "RMSE": rmse,
                "MAE": mae,
                "Spearman": spear,
                "NDCG@20": ndcg20,
                "Hit@20": hit20,
            }
        )
        curve_all.append(curve)

    vae_metrics = pd.DataFrame(metric_rows).sort_values("Spearman", ascending=False).reset_index(drop=True).round(6)
    vae_metrics.to_csv(out_dir / "vae_metrics.csv", index=False)

    best = vae_metrics.iloc[0]
    best_latent = int(best["latent_dim"])

    xgb_mlp = pd.read_csv(args.xgb_mlp_comparison_csv)
    merged_comp = pd.concat(
        [
            xgb_mlp,
            pd.DataFrame(
                [
                    {
                        "model": "VAE",
                        "dataset": "newfe_v2_target_only",
                        "RMSE": float(best["RMSE"]),
                        "MAE": float(best["MAE"]),
                        "Spearman": float(best["Spearman"]),
                        "NDCG@20": float(best["NDCG@20"]),
                        "Hit@20": float(best["Hit@20"]),
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).round(6)
    merged_comp.to_csv(out_dir / "xgb_mlp_vae_comparison.csv", index=False)

    latent_effect = vae_metrics[["latent_dim", "RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]].copy()
    latent_effect.to_csv(out_dir / "vae_latent_effect.csv", index=False)

    curve_df = pd.concat(curve_all, ignore_index=True).round(8)
    curve_df.to_csv(out_dir / "vae_learning_curve.csv", index=False)

    summary = {
        "dataset": "pair_features_newfe_v2.parquet (LINCS excluded)",
        "rows_total": int(len(merged)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "feature_cols_total": int(len(feat_cols)),
        "feature_cols_continuous_scaled": int(len(continuous_cols)),
        "feature_cols_binary_passthrough": int(len(binary_cols)),
        "split": {"test_size": args.test_size, "seed": args.seed},
        "vae_config": {
            "latent_dims": latent_dims,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "beta": args.beta,
            "pred_weight": args.pred_weight,
            "loss": "reconstruction + beta*KL + pred_weight*prediction",
        },
        "best_latent_dim": best_latent,
        "best_metrics": {
            "RMSE": float(best["RMSE"]),
            "MAE": float(best["MAE"]),
            "Spearman": float(best["Spearman"]),
            "NDCG@20": float(best["NDCG@20"]),
            "Hit@20": float(best["Hit@20"]),
        },
        "outputs": {
            "metric_table": str(out_dir / "vae_metrics.csv"),
            "comparison_table": str(out_dir / "xgb_mlp_vae_comparison.csv"),
            "latent_effect": str(out_dir / "vae_latent_effect.csv"),
            "learning_curve": str(out_dir / "vae_learning_curve.csv"),
        },
    }
    (out_dir / "vae_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
