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
    parser.add_argument("--full_train", type=str, default="off", choices=["off", "on"])
    parser.add_argument("--exclude_lincs", type=str, default="off", choices=["off", "on"])
    parser.add_argument("--use_smiles", type=str, default="auto", choices=["off", "auto", "on"])
    parser.add_argument("--smiles_col", type=str, default="drug__smiles")
    parser.add_argument("--smiles_n_features", type=int, default=2048)
    parser.add_argument("--xgb_max_depth", type=int, default=8)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_n_estimators", type=int, default=400)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
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
    if args.exclude_lincs == "on":
        feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
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

    do_full_train = args.full_train == "on"
    idx = np.arange(len(merged))
    if do_full_train:
        tr_idx = idx
        va_idx = np.array([], dtype=int)
    else:
        tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    X_train = X_all[tr_idx]
    y_train = y[tr_idx]
    X_valid = X_all[va_idx] if len(va_idx) else None
    y_valid = y[va_idx] if len(va_idx) else None

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
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
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
    pred_va = model.predict(Xt_va) if Xt_va is not None and len(va_idx) else None

    def _metrics(yt, yp):
        return {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "spearman": _safe_spearman(yt, yp),
        }

    metrics: dict = {
        "model": args.model,
        "rows_total": int(len(merged)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "n_features_numeric": int(X_num.shape[1]),
        "n_features_total": int(X_all.shape[1]),
        "smiles": {
            "enabled": bool(use_smiles),
            "column": args.smiles_col,
            "n_features_hashed": int(args.smiles_n_features if use_smiles else 0),
        },
        "train": _metrics(y_train, pred_tr),
        "split": {"test_size": args.test_size, "seed": args.seed, "full_train": bool(do_full_train)},
        "exclude_lincs": bool(args.exclude_lincs == "on"),
        "inputs": {"features_s3": args.features_s3, "labels_s3": args.labels_s3},
    }
    if pred_va is not None and y_valid is not None and len(y_valid):
        metrics["valid"] = _metrics(y_valid, pred_va)
    else:
        metrics["valid"] = None

    metrics["sagemaker_training_job"] = os.environ.get("TRAINING_JOB_NAME", "")
    metrics["sagemaker_status"] = "Completed" if os.environ.get("TRAINING_JOB_NAME") else "local"
    if not do_full_train:
        metrics["validation_type"] = f"Single row-level holdout (test_size={args.test_size}; valid metrics)"
        metrics["evaluation_note"] = (
            "Single holdout split (test_size); not the 5-fold CV mean in residual_mlp_cv_summary / xgb_tuned_cv_summary."
        )
    else:
        metrics["validation_type"] = "Full train on all rows (train metrics only if valid absent)"
        metrics["evaluation_note"] = (
            "full_train on all rows; valid metrics absent. Train metrics only unless a separate eval split is added."
        )

    feature_importance_rows = []
    if args.model == "xgboost":
        booster = model.get_booster()
        gain_imp = booster.get_score(importance_type="gain")
        for i, cname in enumerate(numeric_cols):
            key_name = f"f{i}"
            feature_importance_rows.append(
                {
                    "feature": cname,
                    "gain_importance": float(gain_imp.get(key_name, 0.0)),
                }
            )
    elif hasattr(model, "feature_importances_"):
        for cname, val in zip(numeric_cols, model.feature_importances_):
            feature_importance_rows.append({"feature": cname, "gain_importance": float(val)})
    fi_df = pd.DataFrame(feature_importance_rows).sort_values("gain_importance", ascending=False)

    out_model = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out_model.mkdir(parents=True, exist_ok=True)
    (out_model / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    fi_df.to_csv(out_model / "feature_importance.csv", index=False)
    experiment_meta = {
        "model": args.model,
        "training_mode": "full_train" if do_full_train else "train_valid_split",
        "fixed_parameters": {
            "xgb_max_depth": args.xgb_max_depth,
            "xgb_learning_rate": args.xgb_learning_rate,
            "xgb_n_estimators": args.xgb_n_estimators,
            "xgb_subsample": args.xgb_subsample,
            "xgb_colsample_bytree": args.xgb_colsample_bytree,
            "seed": args.seed,
            "exclude_lincs": args.exclude_lincs,
        },
        "artifacts": {
            "metrics": "metrics.json",
            "feature_importance": "feature_importance.csv",
            "model": "artifact.joblib",
        },
    }
    (out_model / "experiment_metadata.json").write_text(
        json.dumps(experiment_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    import joblib

    joblib.dump(
        {"model": model, "scaler": scaler, "feature_columns": numeric_cols, "args": vars(args)},
        out_model / "artifact.joblib",
    )

    print("[METRICS]", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
