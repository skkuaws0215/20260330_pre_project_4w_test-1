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
    p = argparse.ArgumentParser(description="Run local TabNet baseline on newfe_v2 target-only features.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-d", type=int, default=16)
    p.add_argument("--n-a", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--gamma", type=float, default=1.3)
    p.add_argument("--lambda-sparse", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--comparison-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/vae_baseline/xgb_mlp_vae_comparison.csv",
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
        m = float(np.min(yt))
        if m < 0:
            yt = yt - m
        k = min(20, len(g))
        ndcgs.append(float(ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=k)))
        top_true = set(np.argsort(-yt)[:k].tolist())
        top_pred = set(np.argsort(-yp)[:k].tolist())
        hits.append(1.0 if top_true.intersection(top_pred) else 0.0)
    return (float(np.mean(ndcgs)) if ndcgs else float("nan"), float(np.mean(hits)) if hits else float("nan"))


def main() -> None:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    y_train = y[tr_idx].reshape(-1, 1)
    y_valid = y[va_idx].reshape(-1, 1)

    scaler = StandardScaler()
    if continuous_cols:
        cont_idx = [feat_cols.index(c) for c in continuous_cols]
        X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
        X_valid[:, cont_idx] = scaler.transform(X_valid[:, cont_idx])

    model = TabNetRegressor(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        lambda_sparse=args.lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": args.lr},
        seed=args.seed,
        verbose=0,
    )
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=min(128, args.batch_size),
        drop_last=False,
    )

    pred = model.predict(X_valid).reshape(-1)
    yv = y_valid.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(yv, pred)))
    mae = float(mean_absolute_error(yv, pred))
    spearman = safe_spearman(yv, pred)
    ndcg20, hit20 = rank_metrics_by_sample(merged.iloc[va_idx], args.sample_id_col, yv, pred)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "TabNet",
                "dataset": "newfe_v2_target_only",
                "RMSE": rmse,
                "MAE": mae,
                "Spearman": spearman,
                "NDCG@20": ndcg20,
                "Hit@20": hit20,
            }
        ]
    ).round(6)
    metrics_df.to_csv(out_dir / "tabnet_metrics.csv", index=False)

    prev = pd.read_csv(args.comparison_csv)
    merged_comp = pd.concat([prev, metrics_df], ignore_index=True).round(6)
    merged_comp.to_csv(out_dir / "xgb_mlp_vae_tabnet_comparison.csv", index=False)

    hist = model.history
    hist_data = getattr(hist, "history", {})
    loss_arr = hist_data.get("loss", [])
    valid_rmse_arr = hist_data.get("valid_rmse", [np.nan] * len(loss_arr))
    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(loss_arr) + 1),
            "train_loss": loss_arr,
            "valid_rmse": valid_rmse_arr,
        }
    ).round(8)
    hist_df.to_csv(out_dir / "tabnet_learning_curve.csv", index=False)

    summary = {
        "dataset": "pair_features_newfe_v2.parquet (LINCS excluded)",
        "rows_total": int(len(merged)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "feature_cols_total": int(len(feat_cols)),
        "feature_cols_continuous_scaled": int(len(continuous_cols)),
        "feature_cols_binary_passthrough": int(len(binary_cols)),
        "split": {"test_size": args.test_size, "seed": args.seed},
        "tabnet_config": {
            "n_d": args.n_d,
            "n_a": args.n_a,
            "n_steps": args.n_steps,
            "gamma": args.gamma,
            "lambda_sparse": args.lambda_sparse,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "early_stopping": True,
        },
        "metrics": metrics_df.iloc[0].to_dict(),
        "outputs": {
            "metric_table": str(out_dir / "tabnet_metrics.csv"),
            "comparison_table": str(out_dir / "xgb_mlp_vae_tabnet_comparison.csv"),
            "learning_curve": str(out_dir / "tabnet_learning_curve.csv"),
        },
    }
    (out_dir / "tabnet_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
