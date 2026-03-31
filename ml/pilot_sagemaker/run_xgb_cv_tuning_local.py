from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import KFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local XGBoost CV + simple tuning for newfe_v2 target-only.")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-trials", type=int, default=20)
    return p.parse_args()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    if s is None or pd.isna(s):
        return float("nan")
    return float(s)


def ndcg_at_20_by_sample(df_fold: pd.DataFrame, sample_id_col: str, y_col: str, p_col: str) -> float:
    vals: list[float] = []
    for _, g in df_fold.groupby(sample_id_col):
        if len(g) < 2:
            continue
        yt = g[y_col].to_numpy(dtype=float)
        yp = g[p_col].to_numpy(dtype=float)
        m = float(np.min(yt))
        if m < 0:
            yt = yt - m
        k = min(20, len(g))
        vals.append(float(ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=k)))
    return float(np.mean(vals)) if vals else float("nan")


def eval_fold_metrics(df_valid: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, sample_id_col: str) -> dict[str, float]:
    work = df_valid[[sample_id_col]].copy()
    work["y_true"] = y_true
    work["y_pred"] = y_pred
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": safe_spearman(y_true, y_pred),
        "NDCG@20": ndcg_at_20_by_sample(work, sample_id_col, "y_true", "y_pred"),
    }


def build_dataset(labels_uri: str, features_uri: str, sample_id_col: str, drug_id_col: str, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    labels = pd.read_parquet(labels_uri)
    feats = pd.read_parquet(features_uri)
    key = [sample_id_col, drug_id_col]

    for c in key + [target_col]:
        if c not in labels.columns:
            raise ValueError(f"labels missing column: {c}")
    for c in key:
        if c not in feats.columns:
            raise ValueError(f"features missing column: {c}")

    merged = labels[key + [target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in merged.columns if c not in set(key + [target_col])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    merged = merged[key + [target_col] + feat_cols].copy()
    merged[sample_id_col] = merged[sample_id_col].astype(str)
    merged[drug_id_col] = merged[drug_id_col].astype(str)
    merged = merged.sort_values(key).reset_index(drop=True)
    return merged, feat_cols


def run_cv_once(
    df: pd.DataFrame,
    feat_cols: list[str],
    sample_id_col: str,
    target_col: str,
    seed: int,
    n_splits: int,
    params: dict[str, float | int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    import xgboost as xgb

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows: list[dict[str, float | int]] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[va_idx])
        m = eval_fold_metrics(df.iloc[va_idx], y[va_idx], pred, sample_id_col)
        rows.append({"fold": fold, **m})

    fold_df = pd.DataFrame(rows)
    summary = {
        "RMSE_mean": float(fold_df["RMSE"].mean()),
        "RMSE_std": float(fold_df["RMSE"].std(ddof=0)),
        "MAE_mean": float(fold_df["MAE"].mean()),
        "MAE_std": float(fold_df["MAE"].std(ddof=0)),
        "Spearman_mean": float(fold_df["Spearman"].mean()),
        "Spearman_std": float(fold_df["Spearman"].std(ddof=0)),
        "NDCG@20_mean": float(fold_df["NDCG@20"].mean()),
        "NDCG@20_std": float(fold_df["NDCG@20"].std(ddof=0)),
    }
    return fold_df, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feat_cols = build_dataset(
        labels_uri=args.labels_uri,
        features_uri=args.features_uri,
        sample_id_col=args.sample_id_col,
        drug_id_col=args.drug_id_col,
        target_col=args.target_col,
    )

    base_params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    fold_df, cv_summary = run_cv_once(
        df=df,
        feat_cols=feat_cols,
        sample_id_col=args.sample_id_col,
        target_col=args.target_col,
        seed=args.seed,
        n_splits=args.n_splits,
        params=base_params,
    )
    fold_df = fold_df.round(6)
    fold_df.to_csv(out_dir / "xgb_cv_fold_metrics.csv", index=False)

    grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    all_combos = list(
        itertools.product(
            grid["max_depth"],
            grid["learning_rate"],
            grid["n_estimators"],
            grid["subsample"],
            grid["colsample_bytree"],
        )
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_combos)
    combos = all_combos[: min(args.max_trials, len(all_combos))]

    tuning_rows: list[dict[str, float | int]] = []
    for i, (max_depth, learning_rate, n_estimators, subsample, colsample_bytree) in enumerate(combos, start=1):
        params = {
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "n_estimators": int(n_estimators),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
        }
        _, s = run_cv_once(
            df=df,
            feat_cols=feat_cols,
            sample_id_col=args.sample_id_col,
            target_col=args.target_col,
            seed=args.seed,
            n_splits=args.n_splits,
            params=params,
        )
        tuning_rows.append(
            {
                "trial": i,
                **params,
                "cv_spearman_mean": s["Spearman_mean"],
                "cv_spearman_std": s["Spearman_std"],
                "cv_ndcg20_mean": s["NDCG@20_mean"],
                "cv_ndcg20_std": s["NDCG@20_std"],
                "cv_rmse_mean": s["RMSE_mean"],
                "cv_mae_mean": s["MAE_mean"],
            }
        )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df = tuning_df.sort_values(["cv_spearman_mean", "cv_ndcg20_mean"], ascending=[False, False]).reset_index(drop=True)
    tuning_df = tuning_df.round(6)
    tuning_df.to_csv(out_dir / "xgb_tuning_results.csv", index=False)

    best = tuning_df.iloc[0].to_dict()
    best_params = {
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
    }
    (out_dir / "xgb_best_params.json").write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_cv_spearman_mean": float(best["cv_spearman_mean"]),
                "best_cv_spearman_std": float(best["cv_spearman_std"]),
                "best_cv_ndcg20_mean": float(best["cv_ndcg20_mean"]),
                "best_cv_ndcg20_std": float(best["cv_ndcg20_std"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    baseline_ref = {
        "source": "analysis_target_only/model_dataset_metrics.csv",
        "xgboost_baseline_spearman_holdout": 0.263278,
        "xgboost_newfe_v2_spearman_holdout": 0.445384,
    }
    summary = {
        "dataset": "newfe_v2 target-only (LINCS excluded)",
        "rows": int(len(df)),
        "feature_cols_used": int(len(feat_cols)),
        "cv_config": {
            "n_splits": int(args.n_splits),
            "shuffle": True,
            "random_state": int(args.seed),
        },
        "step1_cv_base_params": base_params,
        "step1_cv_summary": {k: round(float(v), 6) for k, v in cv_summary.items()},
        "step2_trials": int(len(combos)),
        "step2_best_params": best_params,
        "step2_best_cv": {
            "spearman_mean": round(float(best["cv_spearman_mean"]), 6),
            "spearman_std": round(float(best["cv_spearman_std"]), 6),
            "ndcg20_mean": round(float(best["cv_ndcg20_mean"]), 6),
            "ndcg20_std": round(float(best["cv_ndcg20_std"]), 6),
        },
        "baseline_reference": baseline_ref,
        "improvement_note": "Primary selection uses CV Spearman, secondary uses CV NDCG@20. Baseline comparison uses prior holdout snapshot.",
    }
    (out_dir / "xgb_cv_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
