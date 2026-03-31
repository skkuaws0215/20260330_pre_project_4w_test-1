from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local MLP baseline on newfe_v2 target-only features.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-layers", default="512,256,64")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--learning-rate-init", type=float, default=1e-3)
    p.add_argument(
        "--xgb-reference-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/model_dataset_metrics.csv",
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

    # Binary columns remain unchanged; continuous columns are standardized.
    binary_cols = []
    continuous_cols = []
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

    hidden = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=args.learning_rate_init,
        batch_size=256,
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    mae = float(mean_absolute_error(y_valid, pred))
    spearman = safe_spearman(y_valid, pred)
    ndcg20, hit20 = rank_metrics_by_sample(merged.iloc[va_idx], args.sample_id_col, y_valid, pred)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "MLP",
                "dataset": "newfe_v2_target_only",
                "RMSE": rmse,
                "MAE": mae,
                "Spearman": spearman,
                "NDCG@20": ndcg20,
                "Hit@20": hit20,
            }
        ]
    ).round(6)
    metrics_df.to_csv(out_dir / "mlp_metrics.csv", index=False)

    xgb_ref = pd.read_csv(args.xgb_reference_csv)
    xgb_row = xgb_ref[(xgb_ref["model"] == "XGBoost") & (xgb_ref["dataset"] == "newfe_v2")].copy()
    if xgb_row.empty:
        raise ValueError("Could not find XGBoost/newfe_v2 row from reference csv.")
    xgb_row = xgb_row.rename(columns={"dataset": "dataset_ref"})
    comparison = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "dataset": "newfe_v2_target_only",
                "RMSE": float(xgb_row.iloc[0]["RMSE"]),
                "MAE": float(xgb_row.iloc[0]["MAE"]),
                "Spearman": float(xgb_row.iloc[0]["Spearman"]),
                "NDCG@20": float(xgb_row.iloc[0]["NDCG@20"]),
                "Hit@20": float(xgb_row.iloc[0]["Hit@20"]),
            },
            {
                "model": "MLP",
                "dataset": "newfe_v2_target_only",
                "RMSE": rmse,
                "MAE": mae,
                "Spearman": spearman,
                "NDCG@20": ndcg20,
                "Hit@20": hit20,
            },
        ]
    ).round(6)
    comparison.to_csv(out_dir / "xgb_vs_mlp_comparison.csv", index=False)

    loss_curve = pd.DataFrame({"epoch": np.arange(1, len(model.loss_curve_) + 1), "loss": model.loss_curve_}).round(8)
    loss_curve.to_csv(out_dir / "mlp_learning_curve.csv", index=False)

    summary = {
        "dataset": "pair_features_newfe_v2.parquet (LINCS excluded)",
        "rows_total": int(len(merged)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "feature_cols_total": int(len(feat_cols)),
        "feature_cols_continuous_scaled": int(len(continuous_cols)),
        "feature_cols_binary_passthrough": int(len(binary_cols)),
        "split": {"test_size": args.test_size, "seed": args.seed},
        "mlp_config": {
            "hidden_layer_sizes": list(hidden),
            "max_iter": args.max_iter,
            "learning_rate_init": args.learning_rate_init,
            "early_stopping": True,
        },
        "metrics": metrics_df.iloc[0].to_dict(),
        "outputs": {
            "metric_table": str(out_dir / "mlp_metrics.csv"),
            "comparison_table": str(out_dir / "xgb_vs_mlp_comparison.csv"),
            "learning_curve": str(out_dir / "mlp_learning_curve.csv"),
        },
    }
    (out_dir / "mlp_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
