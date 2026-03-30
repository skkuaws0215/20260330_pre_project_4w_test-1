from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare B experiment input: keep miss70 features but drop raw string SMILES columns."
    )
    p.add_argument("--features-uri", required=True, help="Input features parquet URI (e.g., miss70 features.parquet)")
    p.add_argument("--labels-uri", required=True, help="Input labels parquet URI")
    p.add_argument("--output-prefix", required=True, help="Output folder (local path or s3://...)")
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument(
        "--drop-cols",
        default="drug__smiles,drug__canonical_smiles_raw",
        help="Comma-separated columns to drop for B baseline",
    )
    return p.parse_args()


def _join_path(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


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

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    existing_drop_cols = [c for c in drop_cols if c in features.columns]
    features_b = features.drop(columns=existing_drop_cols, errors="ignore")

    out_features = _join_path(args.output_prefix, "features_b.parquet")
    out_labels = _join_path(args.output_prefix, "labels.parquet")
    out_meta = _join_path(args.output_prefix, "b_input_manifest.json")

    _write_parquet(features_b, out_features)
    _write_parquet(labels, out_labels)

    manifest = {
        "purpose": "B baseline input (miss70 minus raw SMILES text columns)",
        "input": {
            "features_uri": args.features_uri,
            "labels_uri": args.labels_uri,
        },
        "output": {
            "features_uri": out_features,
            "labels_uri": out_labels,
            "manifest_uri": out_meta,
        },
        "keys": {
            "sample_id_col": args.sample_id_col,
            "drug_id_col": args.drug_id_col,
        },
        "drop_cols_requested": drop_cols,
        "drop_cols_applied": existing_drop_cols,
        "row_counts": {
            "features_rows": int(features_b.shape[0]),
            "features_cols": int(features_b.shape[1]),
            "labels_rows": int(labels.shape[0]),
            "labels_cols": int(labels.shape[1]),
        },
    }
    _write_json(manifest, out_meta)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
