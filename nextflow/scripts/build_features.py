from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_parquet(uri: str) -> pd.DataFrame:
    return pd.read_parquet(uri)


def _ensure_drug_id(df: pd.DataFrame, drug_id_col: str, fallback_col: str) -> pd.DataFrame:
    out = df.copy()
    if drug_id_col not in out.columns and fallback_col in out.columns:
        out[drug_id_col] = out[fallback_col]
    return out


def _drop_high_missing(df: pd.DataFrame, threshold: float, protected_cols: set[str]) -> tuple[pd.DataFrame, list[str]]:
    miss = df.isna().mean()
    drop_cols = [c for c in df.columns if c not in protected_cols and miss.get(c, 0.0) > threshold]
    return df.drop(columns=drop_cols, errors="ignore"), drop_cols


def _impute(df: pd.DataFrame, protected_cols: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    impute_map: dict[str, str] = {}
    for col in out.columns:
        if col in protected_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            med = out[col].median()
            out[col] = out[col].fillna(med)
            impute_map[col] = "median"
        else:
            out[col] = out[col].fillna("UNK")
            impute_map[col] = "UNK"
    return out, impute_map


def _variance_filter(df: pd.DataFrame, threshold: float, protected_cols: set[str]) -> tuple[pd.DataFrame, list[str]]:
    num_cols = [c for c in df.columns if c not in protected_cols and pd.api.types.is_numeric_dtype(df[c])]
    var = df[num_cols].var(numeric_only=True)
    drop_cols = [c for c in num_cols if var.get(c, 0.0) <= threshold]
    return df.drop(columns=drop_cols, errors="ignore"), drop_cols


def _zscore_df(df: pd.DataFrame, protected_cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if c not in protected_cols and pd.api.types.is_numeric_dtype(out[c])]
    for col in num_cols:
        std = out[col].std()
        if std and std > 0:
            out[col] = (out[col] - out[col].mean()) / std
        else:
            out[col] = 0.0
    return out


def _prefix_except(df: pd.DataFrame, prefix: str, keep_cols: set[str]) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        if col in keep_cols:
            continue
        renamed[col] = f"{prefix}{col}"
    return df.rename(columns=renamed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FE tables for team4 Nextflow pipeline.")
    p.add_argument("--sample-feature-uri", required=True)
    p.add_argument("--drug-feature-uri", required=True)
    p.add_argument("--label-uri", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--drug-fallback-col", default="inchikey")
    p.add_argument("--regression-label-col", default="ic50")
    p.add_argument("--binary-label-col", default="binary_label")
    p.add_argument("--missing-threshold", type=float, default=0.7)
    p.add_argument("--variance-threshold", type=float, default=0.0)
    p.add_argument("--leakage-cols", default="")
    p.add_argument("--normalization-branch", choices=["tree", "dl", "both"], default="both")
    p.add_argument("--binary-from-quantile", type=float, default=0.3)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-features", required=True)
    p.add_argument("--out-labels", required=True)
    p.add_argument("--out-manifest", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    leak_cols = {c.strip() for c in args.leakage_cols.split(",") if c.strip()}

    sample_df = _read_parquet(args.sample_feature_uri)
    drug_df = _read_parquet(args.drug_feature_uri)
    label_df = _read_parquet(args.label_uri)

    drug_df = _ensure_drug_id(drug_df, args.drug_id_col, args.drug_fallback_col)
    label_df = _ensure_drug_id(label_df, args.drug_id_col, args.drug_fallback_col)

    required = {
        "sample_df": [args.sample_id_col],
        "drug_df": [args.drug_id_col],
        "label_df": [args.sample_id_col, args.drug_id_col, args.regression_label_col],
    }
    name_to_df = {"sample_df": sample_df, "drug_df": drug_df, "label_df": label_df}
    for name, cols in required.items():
        missing = [c for c in cols if c not in name_to_df[name].columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    key_cols = {args.sample_id_col, args.drug_id_col}
    sample_df = _prefix_except(sample_df, "sample__", key_cols)
    drug_df = _prefix_except(drug_df, "drug__", key_cols)

    merged = (
        label_df[[c for c in label_df.columns if c in key_cols or c in {args.regression_label_col, args.binary_label_col}]]
        .merge(sample_df, on=args.sample_id_col, how="inner")
        .merge(drug_df, on=args.drug_id_col, how="left")
    )

    protected_cols = set(key_cols) | {args.regression_label_col, args.binary_label_col} | leak_cols
    merged = merged.drop(columns=list(leak_cols), errors="ignore")

    if args.binary_label_col not in merged.columns:
        q = merged[args.regression_label_col].quantile(args.binary_from_quantile)
        merged[args.binary_label_col] = (merged[args.regression_label_col] <= q).astype(int)

    merged, dropped_missing = _drop_high_missing(merged, args.missing_threshold, protected_cols)
    merged, impute_map = _impute(merged, protected_cols)
    merged, dropped_variance = _variance_filter(merged, args.variance_threshold, protected_cols)

    labels = merged[[args.sample_id_col, args.drug_id_col, args.regression_label_col, args.binary_label_col]].copy()
    labels = labels.rename(
        columns={
            args.regression_label_col: "label_regression",
            args.binary_label_col: "label_binary",
        }
    )
    labels["label_main"] = labels["label_regression"]
    labels["label_aux"] = labels["label_binary"]
    labels["label_main_type"] = "regression"
    labels["label_aux_type"] = "binary"

    feature_cols = [c for c in merged.columns if c not in {args.regression_label_col, args.binary_label_col}]
    features_tree = merged[feature_cols].copy()

    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    features_tree.to_parquet(args.out_features, index=False)
    labels.to_parquet(args.out_labels, index=False)

    produced_dl = False
    if args.normalization_branch in {"dl", "both"}:
        features_dl = _zscore_df(features_tree, set(key_cols))
        features_dl.to_parquet("features_dl.parquet", index=False)
        produced_dl = True

    manifest = {
        "run_id": args.run_id,
        "task_contract": {
            "target_unit": "sample-drug pair",
            "main_label": "regression",
            "aux_label": "binary",
        },
        "input_uris": {
            "sample_feature_uri": args.sample_feature_uri,
            "drug_feature_uri": args.drug_feature_uri,
            "label_uri": args.label_uri,
        },
        "keys": {
            "sample_id_col": args.sample_id_col,
            "drug_id_col": args.drug_id_col,
            "drug_fallback_col": args.drug_fallback_col,
        },
        "filters": {
            "missing_threshold": args.missing_threshold,
            "variance_threshold": args.variance_threshold,
            "dropped_high_missing_columns": dropped_missing,
            "dropped_low_variance_columns": dropped_variance,
            "imputation": impute_map,
            "leakage_columns_removed": sorted(list(leak_cols)),
        },
        "row_counts": {
            "features_rows": int(features_tree.shape[0]),
            "features_cols": int(features_tree.shape[1]),
            "labels_rows": int(labels.shape[0]),
        },
        "outputs": {
            "features": args.out_features,
            "labels": args.out_labels,
            "features_dl": "features_dl.parquet" if produced_dl else None,
        },
    }
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
