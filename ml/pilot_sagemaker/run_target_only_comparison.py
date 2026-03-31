from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline/newfe/newfe_v2 target-only comparison.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--baseline-features-uri", required=True)
    p.add_argument("--newfe-features-uri", required=True)
    p.add_argument("--newfe-v2-features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    if s is None or pd.isna(s):
        return float("nan")
    return float(s)


def _metrics_with_rank(
    df_valid: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, sample_id_col: str, k: int = 20
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    spear = safe_spearman(y_true, y_pred)

    tmp = df_valid[[sample_id_col]].copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    ndcgs: list[float] = []
    hits: list[float] = []
    for _sid, g in tmp.groupby(sample_id_col):
        if len(g) < 2:
            continue
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        # sklearn ndcg_score requires non-negative relevance.
        min_yt = float(np.min(yt))
        if min_yt < 0:
            yt = yt - min_yt
        kk = min(k, len(g))
        ndcgs.append(float(ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=kk)))

        top_true_idx = set(np.argsort(-yt)[:kk].tolist())
        top_pred_idx = set(np.argsort(-yp)[:kk].tolist())
        hits.append(1.0 if len(top_true_idx.intersection(top_pred_idx)) > 0 else 0.0)

    ndcg20 = float(np.mean(ndcgs)) if ndcgs else float("nan")
    hit20 = float(np.mean(hits)) if hits else float("nan")
    return {"RMSE": rmse, "MAE": mae, "Spearman": spear, "NDCG@20": ndcg20, "Hit@20": hit20}


def _prepare_dataset(
    labels: pd.DataFrame,
    features_uri: str,
    key_cols: list[str],
    target_col: str,
    common_keys: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    feat = pd.read_parquet(features_uri)
    merged = labels[key_cols + [target_col]].merge(feat, on=key_cols, how="inner")
    feat_cols = [c for c in merged.columns if c not in set(key_cols + [target_col])]
    # Requested condition: LINCS feature excluded.
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    d = merged[key_cols + [target_col] + feat_cols].copy()

    if common_keys is not None:
        idx = list(zip(d[key_cols[0]].astype(str), d[key_cols[1]].astype(str)))
        keep_mask = pd.Series(idx).isin(common_keys).to_numpy()
        d = d.loc[keep_mask].copy()

    d[key_cols[0]] = d[key_cols[0]].astype(str)
    d[key_cols[1]] = d[key_cols[1]].astype(str)
    d = d.sort_values(key_cols).reset_index(drop=True)
    return d, feat_cols


def _train_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray) -> np.ndarray:
    if model_name == "LightGBM":
        import lightgbm as lgb

        m = lgb.LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(X_train, y_train)
        return m.predict(X_valid)
    if model_name == "XGBoost":
        import xgboost as xgb

        m = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X_train, y_train)
        return m.predict(X_valid)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_valid)
    m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=8000, random_state=42)
    m.fit(Xtr, y_train)
    return m.predict(Xva)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels_uri)
    key = [args.sample_id_col, args.drug_id_col]
    for c in key + [args.target_col]:
        if c not in labels.columns:
            raise ValueError(f"labels missing column: {c}")

    d_base, _ = _prepare_dataset(labels, args.baseline_features_uri, key, args.target_col)
    d_newfe, _ = _prepare_dataset(labels, args.newfe_features_uri, key, args.target_col)
    d_newfe_v2, _ = _prepare_dataset(labels, args.newfe_v2_features_uri, key, args.target_col)

    keys_base = set(zip(d_base[key[0]].astype(str), d_base[key[1]].astype(str)))
    keys_newfe = set(zip(d_newfe[key[0]].astype(str), d_newfe[key[1]].astype(str)))
    keys_newfe_v2 = set(zip(d_newfe_v2[key[0]].astype(str), d_newfe_v2[key[1]].astype(str)))
    common = keys_base.intersection(keys_newfe).intersection(keys_newfe_v2)
    if not common:
        raise ValueError("No common rows across baseline/newfe/newfe_v2 after key join.")

    d_base, base_cols = _prepare_dataset(labels, args.baseline_features_uri, key, args.target_col, common)
    d_newfe, newfe_cols = _prepare_dataset(labels, args.newfe_features_uri, key, args.target_col, common)
    d_newfe_v2, newfe_v2_cols = _prepare_dataset(labels, args.newfe_v2_features_uri, key, args.target_col, common)

    n = len(d_base)
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)

    datasets = {
        "baseline": (d_base, base_cols),
        "newfe": (d_newfe, newfe_cols),
        "newfe_v2": (d_newfe_v2, newfe_v2_cols),
    }
    models = ["LightGBM", "XGBoost", "ElasticNet"]

    rows: list[dict[str, object]] = []
    for model in models:
        for ds_name, (d, cols) in datasets.items():
            print(f"[RUN] model={model} dataset={ds_name} rows={len(d)} cols={len(cols)}", flush=True)
            X = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            y = d[args.target_col].to_numpy(dtype=np.float32)
            pred = _train_predict(model, X[tr_idx], y[tr_idx], X[va_idx])
            m = _metrics_with_rank(d.iloc[va_idx], y[va_idx], pred, args.sample_id_col, k=20)
            rows.append(
                {
                    "model": model,
                    "dataset": ds_name,
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                    "Spearman": m["Spearman"],
                    "NDCG@20": m["NDCG@20"],
                    "Hit@20": m["Hit@20"],
                }
            )

    metrics_df = pd.DataFrame(rows)
    for c in ["RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]:
        metrics_df[c] = metrics_df[c].astype(float).round(6)
    metrics_df.to_csv(out_dir / "model_dataset_metrics.csv", index=False)

    delta_rows: list[dict[str, object]] = []
    for model in models:
        mdf = metrics_df[metrics_df["model"] == model].set_index("dataset")
        delta_rows.append(
            {
                "model": model,
                "delta_newfe_spear": round(float(mdf.loc["newfe", "Spearman"] - mdf.loc["baseline", "Spearman"]), 6),
                "delta_target_spear": round(float(mdf.loc["newfe_v2", "Spearman"] - mdf.loc["newfe", "Spearman"]), 6),
                "delta_target_ndcg": round(float(mdf.loc["newfe_v2", "NDCG@20"] - mdf.loc["newfe", "NDCG@20"]), 6),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "model_delta_summary.csv", index=False)

    summary = {
        "rows_common_across_3_datasets": int(n),
        "split": {"test_size": args.test_size, "seed": args.seed},
        "labels_uri": args.labels_uri,
        "baseline_features_uri": args.baseline_features_uri,
        "newfe_features_uri": args.newfe_features_uri,
        "newfe_v2_features_uri": args.newfe_v2_features_uri,
        "outputs": {
            "metrics_csv": str(out_dir / "model_dataset_metrics.csv"),
            "delta_csv": str(out_dir / "model_delta_summary.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
