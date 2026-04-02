"""Shared loaders for graph baselines: same merged pair table + CV as ML/DL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_merged_pair_frame(
    labels_uri: str,
    features_uri: str,
    sample_id_col: str,
    drug_id_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    labels = _read_parquet_uri(labels_uri)
    feats = _read_parquet_uri(features_uri)
    key = [sample_id_col, drug_id_col]
    df = labels[key + [target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + [target_col])]
    out = df[key + [target_col] + feat_cols].copy().sort_values(key).reset_index(drop=True)
    return out, feat_cols


def load_cv_meta(cv_fold_json: Path, n_rows: int) -> dict[str, Any]:
    meta = json.loads(cv_fold_json.read_text(encoding="utf-8"))
    if int(meta["n_rows"]) != n_rows:
        raise ValueError(
            f"cv_fold_indices n_rows={meta['n_rows']} != merged df rows={n_rows}. Check labels/features alignment."
        )
    return meta


def _read_parquet_uri(uri: str) -> pd.DataFrame:
    if uri.startswith("s3://"):
        try:
            import fsspec  # noqa: F401
            import s3fs  # noqa: F401 — s3:// backend for fsspec
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Reading s3:// parquet requires `pip install s3fs` (and valid AWS credentials). "
                "Or pass a local file path via --drug-target-uri / --labels-uri / --features-uri."
            ) from e
        # PyArrow sometimes resolves s3:// as a local path unless opened via fsspec.
        with fsspec.open(uri, "rb") as f:
            return pd.read_parquet(f)
    return pd.read_parquet(uri)


def load_drug_targets_dict(
    drug_target_uri: str,
    drug_id_col: str,
    target_gene_col: str,
) -> dict[str, set[str]]:
    dt = _read_parquet_uri(drug_target_uri)
    if drug_id_col not in dt.columns or target_gene_col not in dt.columns:
        raise ValueError(f"drug-target parquet needs {drug_id_col} and {target_gene_col}")
    dt = dt[[drug_id_col, target_gene_col]].dropna()
    dt[drug_id_col] = dt[drug_id_col].astype(str).str.strip()
    dt[target_gene_col] = dt[target_gene_col].astype(str).str.strip().str.upper()
    drug_targets: dict[str, set[str]] = {}
    for did, grp in dt.groupby(drug_id_col):
        drug_targets[str(did)] = {str(x).upper() for x in grp[target_gene_col] if str(x).strip()}
    return drug_targets
