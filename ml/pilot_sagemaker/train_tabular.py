"""
SageMaker PyTorch training entry (Python runtime only; no torch model training).
Reads B-set parquet from S3, trains a tabular regressor, writes metrics + model artifact under /opt/ml/model/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _pip_install_requirements() -> None:
    req = Path(__file__).resolve().parent / "requirements.txt"
    if req.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


def _safe_spearman(y_true, y_pred):
    import pandas as pd

    if len(y_true) < 2:
        return float("nan")
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(s) if s is not None and s == s else float("nan")


def main() -> None:
    _pip_install_requirements()

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from scipy import sparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features_s3", type=str, required=True)
    parser.add_argument("--labels_s3", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["lightgbm", "xgboost", "rf", "elasticnet"])
    parser.add_argument("--sample_id_col", type=str, default="sample_id")
    parser.add_argument("--drug_id_col", type=str, default="canonical_drug_id")
    parser.add_argument("--target_col", type=str, default="label_regression")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_smiles", type=str, default="auto", choices=["off", "auto", "on"])
    parser.add_argument("--smiles_col", type=str, default="drug__smiles")
    parser.add_argument("--smiles_n_features", type=int, default=2048)
    args, _unknown = parser.parse_known_args()

    features = pd.read_parquet(args.features_s3)
    labels = pd.read_parquet(args.labels_s3)
    key = [args.sample_id_col, args.drug_id_col]
    for c in key + [args.target_col]:
        if c not in labels.columns:
            raise ValueError(f"labels missing column: {c}")
    for c in key:
        if c not in features.columns:
            raise ValueError(f"features missing column: {c}")

    merged = labels[key + [args.target_col]].merge(features, on=key, how="inner")
    feat_cols = [c for c in merged.columns if c not in set(key + [args.target_col])]
    X = merged[feat_cols]
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X_num = X[numeric_cols].fillna(0.0)

    use_smiles = False
    if args.use_smiles == "on":
        if args.smiles_col not in X.columns:
            raise ValueError(f"use_smiles=on but missing smiles column: {args.smiles_col}")
        use_smiles = True
    elif args.use_smiles == "auto":
        use_smiles = args.smiles_col in X.columns

    X_num_csr = sparse.csr_matrix(X_num.to_numpy(dtype=np.float32))
    if use_smiles:
        smiles_raw = X[args.smiles_col].fillna("").astype(str)
        hv = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=args.smiles_n_features,
            alternate_sign=False,
            norm=None,
            lowercase=False,
        )
        X_smiles = hv.transform(smiles_raw)
        X_all = sparse.hstack([X_num_csr, X_smiles], format="csr")
    else:
        X_all = X_num_csr

    y = merged[args.target_col].to_numpy(dtype=np.float64)

    idx = np.arange(len(merged))
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    X_train, X_valid = X_all[tr_idx], X_all[va_idx]
    y_train, y_valid = y[tr_idx], y[va_idx]

    Xt_tr, Xt_va = X_train, X_valid
    scaler = None

    if args.model == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(Xt_tr, y_train)
    elif args.model == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(Xt_tr, y_train)
    elif args.model == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(Xt_tr, y_train)
    else:
        scaler = StandardScaler(with_mean=False)
        Xt_tr = scaler.fit_transform(X_train)
        Xt_va = scaler.transform(X_valid)
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=8000, random_state=args.seed)
        model.fit(Xt_tr, y_train)

    pred_tr = model.predict(Xt_tr)
    pred_va = model.predict(Xt_va)

    def _metrics(yt, yp):
        return {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "spearman": _safe_spearman(yt, yp),
        }

    metrics = {
        "model": args.model,
        "rows_total": int(len(merged)),
        "n_features_numeric": int(X_num.shape[1]),
        "n_features_total": int(X_all.shape[1]),
        "smiles": {
            "enabled": bool(use_smiles),
            "column": args.smiles_col,
            "n_features_hashed": int(args.smiles_n_features if use_smiles else 0),
        },
        "train": _metrics(y_train, pred_tr),
        "valid": _metrics(y_valid, pred_va),
        "split": {"test_size": args.test_size, "seed": args.seed},
        "inputs": {"features_s3": args.features_s3, "labels_s3": args.labels_s3},
    }

    out_model = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out_model.mkdir(parents=True, exist_ok=True)
    (out_model / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    import joblib

    joblib.dump(
        {"model": model, "scaler": scaler, "feature_columns": numeric_cols, "args": vars(args)},
        out_model / "artifact.joblib",
    )

    print("[METRICS]", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
