from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SHAP top-20 summary for local XGBoost model.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--best-params-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    import xgboost as xgb
    import shap

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]

    merged = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in merged.columns if c not in set(key + [args.target_col])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]

    X_df = merged[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = merged[args.target_col].to_numpy(dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)

    best_payload = json.loads(Path(args.best_params_json).read_text(encoding="utf-8"))
    best = best_payload["best_params"]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        n_estimators=int(best["n_estimators"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
    )
    model.fit(X, y)

    n = min(int(args.max_samples), X.shape[0])
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X_sub = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        shap_arr = np.asarray(shap_values[0], dtype=np.float64)
    else:
        shap_arr = np.asarray(shap_values, dtype=np.float64)

    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    df = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    top20 = df.head(20).copy()
    top20["rank"] = np.arange(1, len(top20) + 1)
    top20 = top20[["rank", "feature", "mean_abs_shap"]]
    top20["mean_abs_shap"] = top20["mean_abs_shap"].astype(float).round(8)
    top20.to_csv(out_dir / "xgb_shap_top20.csv", index=False)

    summary = {
        "model": "XGBoost",
        "dataset": "newfe_v2 target-only (LINCS excluded)",
        "rows_total": int(X.shape[0]),
        "feature_cols_used": int(len(feat_cols)),
        "shap_sample_size": int(n),
        "best_params": best,
        "top1_feature": str(top20.iloc[0]["feature"]) if len(top20) else "",
        "top1_mean_abs_shap": float(top20.iloc[0]["mean_abs_shap"]) if len(top20) else 0.0,
        "outputs": {"top20_csv": str(out_dir / "xgb_shap_top20.csv")},
    }
    (out_dir / "xgb_shap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
