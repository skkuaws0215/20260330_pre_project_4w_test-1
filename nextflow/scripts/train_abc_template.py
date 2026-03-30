from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def _read_parquet(uri: str) -> pd.DataFrame:
    return pd.read_parquet(uri)


def _write_json(obj: dict[str, Any], path: str) -> None:
    content = json.dumps(obj, ensure_ascii=False, indent=2)
    if path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        subprocess.run(["aws", "s3", "cp", tmp_path, path], check=True)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    if path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        df.to_parquet(tmp_path, index=False)
        subprocess.run(["aws", "s3", "cp", tmp_path, path], check=True)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _join_path(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B/C common training template (regression baseline).")
    p.add_argument("--features-uri", required=True)
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--experiment-tag", required=True, help="e.g., A_miss30, B_miss70_no_smiles, C_miss70_smiles_fp")
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["rf"], default="rf")
    return p.parse_args()


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(s) if s is not None else float("nan")


def main() -> None:
    args = parse_args()
    features = _read_parquet(args.features_uri)
    labels = _read_parquet(args.labels_uri)

    key_cols = [args.sample_id_col, args.drug_id_col]
    for c in key_cols:
        if c not in features.columns:
            raise ValueError(f"features missing key column: {c}")
        if c not in labels.columns:
            raise ValueError(f"labels missing key column: {c}")
    if args.target_col not in labels.columns:
        raise ValueError(f"labels missing target column: {args.target_col}")

    merged = labels[key_cols + [args.target_col]].merge(features, on=key_cols, how="inner")

    # Keep only numeric feature columns for a robust baseline.
    feature_cols = [c for c in merged.columns if c not in set(key_cols + [args.target_col])]
    X = merged[feature_cols]
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric_cols].fillna(0.0)
    y = merged[args.target_col].to_numpy()

    meta_cols = merged[key_cols].copy()
    idx = np.arange(len(merged))
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)

    X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_valid = y[tr_idx], y[va_idx]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)

    metrics = {
        "experiment_tag": args.experiment_tag,
        "model": args.model,
        "rows_total": int(len(merged)),
        "n_features_numeric": int(X.shape[1]),
        "train": {
            "rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "spearman": _safe_spearman(y_train, pred_train),
        },
        "valid": {
            "rmse": float(np.sqrt(mean_squared_error(y_valid, pred_valid))),
            "mae": float(mean_absolute_error(y_valid, pred_valid)),
            "spearman": _safe_spearman(y_valid, pred_valid),
        },
        "data": {
            "features_uri": args.features_uri,
            "labels_uri": args.labels_uri,
            "target_col": args.target_col,
        },
        "split": {
            "test_size": args.test_size,
            "seed": args.seed,
        },
    }

    pred_df = pd.concat(
        [
            meta_cols.iloc[tr_idx].assign(
                split="train",
                y_true=y_train,
                y_pred=pred_train,
                experiment_tag=args.experiment_tag,
            ),
            meta_cols.iloc[va_idx].assign(
                split="valid",
                y_true=y_valid,
                y_pred=pred_valid,
                experiment_tag=args.experiment_tag,
            ),
        ],
        ignore_index=True,
    )

    out_pred = _join_path(args.output_prefix, f"{args.experiment_tag}_predictions.parquet")
    out_metrics = _join_path(args.output_prefix, f"{args.experiment_tag}_metrics.json")
    _write_parquet(pred_df, out_pred)
    _write_json(metrics, out_metrics)

    print(json.dumps({"metrics_uri": out_metrics, "predictions_uri": out_pred, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
