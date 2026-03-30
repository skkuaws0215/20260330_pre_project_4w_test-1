from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build bridge input tables for Nextflow FE (sample_features, drug_features, labels)."
    )
    p.add_argument(
        "--label-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/gdsc/21_ic50.parquet",
        help="GDSC IC50 parquet URI.",
    )
    p.add_argument(
        "--drug-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_glue/feature/model_input/elasticnet/elasticnet_drug_dataset.parquet",
        help="Drug-level feature table URI with canonical_smiles (recommended: elasticnet_drug_dataset).",
    )
    p.add_argument(
        "--sample-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/depmap/57_crispr.parquet",
        help="Sample-level source parquet URI (DepMap CRISPR).",
    )
    p.add_argument(
        "--output-prefix",
        required=True,
        help="Output directory/prefix (local path or s3://...). Example: s3://.../results/features_nextflow_team4/input/20260330_v1",
    )
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--binary-quantile",
        type=float,
        default=0.3,
        help="If binary_label is absent, generate with ic50 <= quantile as 1.",
    )
    return p.parse_args()


def _read_parquet(uri: str) -> pd.DataFrame:
    return pd.read_parquet(uri)


def _safe_mkdir(output_prefix: str) -> None:
    if output_prefix.startswith("s3://"):
        return
    Path(output_prefix).mkdir(parents=True, exist_ok=True)


def _write_json(obj: dict[str, Any], path: str) -> None:
    content = json.dumps(obj, ensure_ascii=False, indent=2)
    if path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        subprocess.run(["aws", "s3", "cp", tmp_path, path], check=True)
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _join_path(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def build_labels(label_df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = ["cell_line_name", "DRUG_ID", "ln_IC50"]
    missing = [c for c in required if c not in label_df.columns]
    if missing:
        raise ValueError(f"label source missing required columns: {missing}")

    mapping_table = label_df[["cell_line_name", "DRUG_ID"]].copy()
    mapping_table = mapping_table.rename(
        columns={"cell_line_name": "sample_id_raw", "DRUG_ID": "canonical_drug_id_raw"}
    )
    mapping_table["sample_id"] = mapping_table["sample_id_raw"].astype(str).str.strip()
    mapping_table["canonical_drug_id"] = mapping_table["canonical_drug_id_raw"].astype(str).str.strip()
    mapping_table["mapping_rule"] = "sample_id=cell_line_name;canonical_drug_id=DRUG_ID"

    labels = label_df.copy()
    labels = labels.rename(
        columns={
            "cell_line_name": "sample_id",
            "DRUG_ID": "canonical_drug_id",
            "ln_IC50": "ic50",
            "label": "binary_label",
        }
    )
    labels["sample_id"] = labels["sample_id"].astype(str).str.strip()
    labels["canonical_drug_id"] = labels["canonical_drug_id"].astype(str).str.strip()
    labels["ic50"] = pd.to_numeric(labels["ic50"], errors="coerce")

    has_binary = "binary_label" in labels.columns
    if not has_binary:
        labels["binary_label"] = pd.NA

    # Duplicate (sample,drug) pairs are aggregated to median to get deterministic labels.
    pair_before = int(labels.shape[0])
    labels = (
        labels.groupby(["sample_id", "canonical_drug_id"], as_index=False)
        .agg({"ic50": "median", "binary_label": "max"})
        .reset_index(drop=True)
    )
    pair_after = int(labels.shape[0])

    if not has_binary:
        thr = float(labels["ic50"].quantile(q))
        labels["binary_label"] = (labels["ic50"] <= thr).astype(int)
    else:
        labels["binary_label"] = labels["binary_label"].fillna((labels["ic50"] <= labels["ic50"].quantile(q)).astype(int))
        labels["binary_label"] = labels["binary_label"].astype(int)

    qc = {
        "labels_pair_rows_before_groupby": pair_before,
        "labels_pair_rows_after_groupby": pair_after,
        "labels_unique_samples": int(labels["sample_id"].nunique()),
        "labels_unique_drugs": int(labels["canonical_drug_id"].nunique()),
        "labels_ic50_missing_rows": int(labels["ic50"].isna().sum()),
    }
    return labels, mapping_table, qc


def _norm_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
        .str.strip()
    )


def build_sample_features(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["cell_line_name", "gene_name", "dependency"]
    missing = [c for c in required if c not in sample_df.columns]
    if missing:
        raise ValueError(f"sample source missing required columns: {missing}")

    wide = sample_df.pivot_table(
        index="cell_line_name",
        columns="gene_name",
        values="dependency",
        aggfunc="mean",
    )
    wide.columns = [f"crispr__{str(c)}" for c in wide.columns]
    wide = wide.reset_index().rename(columns={"cell_line_name": "sample_id"})
    wide["sample_id"] = wide["sample_id"].astype(str).str.strip()
    wide = wide.fillna(wide.median(numeric_only=True))

    qc = {
        "sample_features_rows": int(wide.shape[0]),
        "sample_features_cols": int(wide.shape[1]),
    }
    return wide, qc


def build_drug_features(drug_df: pd.DataFrame, label_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["drug_name_norm", "canonical_smiles"]
    missing = [c for c in required if c not in drug_df.columns]
    if missing:
        raise ValueError(f"drug source missing required columns: {missing}")

    if "DRUG_ID" not in label_df.columns or "drug_name" not in label_df.columns:
        raise ValueError("label source must include DRUG_ID and drug_name for drug mapping.")

    label_drug = label_df[["DRUG_ID", "drug_name"]].dropna().copy()
    label_drug["canonical_drug_id"] = label_drug["DRUG_ID"].astype(int).astype(str)
    label_drug["drug_name_norm"] = _norm_name(label_drug["drug_name"])
    label_drug = label_drug.drop_duplicates(subset=["canonical_drug_id", "drug_name_norm"])

    src = drug_df.copy()
    src["drug_name_norm"] = _norm_name(src["drug_name_norm"])
    src["canonical_smiles_raw"] = src["canonical_smiles"]
    src["smiles"] = src["canonical_smiles"].astype(str).str.strip()
    src.loc[src["smiles"].isin(["", "nan", "None"]), "smiles"] = pd.NA

    merged = label_drug.merge(
        src[["drug_name_norm", "smiles", "canonical_smiles_raw"]].drop_duplicates("drug_name_norm"),
        on="drug_name_norm",
        how="left",
    )

    grouped = (
        merged.groupby("canonical_drug_id", as_index=False)
        .agg(
            smiles=("smiles", "first"),
            canonical_smiles_raw=("canonical_smiles_raw", "first"),
            drug_name_norm=("drug_name_norm", "first"),
        )
        .reset_index(drop=True)
    )
    grouped["has_smiles"] = (~grouped["smiles"].isna()).astype(int)

    qc = {
        "drug_features_rows": int(grouped.shape[0]),
        "drug_features_cols": int(grouped.shape[1]),
        "missing_smiles_rows": int(grouped["smiles"].isna().sum()),
        "missing_smiles_rate": float(grouped["smiles"].isna().mean()),
    }
    return grouped, qc


def main() -> None:
    args = parse_args()
    _safe_mkdir(args.output_prefix)

    label_src = _read_parquet(args.label_uri)
    sample_src = _read_parquet(args.sample_uri)
    drug_src = _read_parquet(args.drug_uri)

    labels, mapping_table, labels_qc = build_labels(label_src, args.binary_quantile)
    sample_features, sample_qc = build_sample_features(sample_src)
    drug_features, drug_qc = build_drug_features(drug_src, label_src)

    labels_key = labels[["sample_id", "canonical_drug_id"]].drop_duplicates()
    sample_key = sample_features[["sample_id"]].drop_duplicates()
    drug_key = drug_features[["canonical_drug_id"]].drop_duplicates()

    label_with_sample = labels_key.merge(sample_key, on="sample_id", how="left", indicator=True)
    label_with_drug = labels_key.merge(drug_key, on="canonical_drug_id", how="left", indicator=True)

    unmatched_samples = int((label_with_sample["_merge"] == "left_only").sum())
    unmatched_drugs = int((label_with_drug["_merge"] == "left_only").sum())
    join_rate_samples = float(1.0 - unmatched_samples / max(len(labels_key), 1))
    join_rate_drugs = float(1.0 - unmatched_drugs / max(len(labels_key), 1))

    out_sample = _join_path(args.output_prefix, "sample_features.parquet")
    out_drug = _join_path(args.output_prefix, "drug_features.parquet")
    out_labels = _join_path(args.output_prefix, "labels.parquet")
    out_mapping = _join_path(args.output_prefix, "mapping_table.parquet")
    out_qc = _join_path(args.output_prefix, "join_qc_report.json")
    out_manifest = _join_path(args.output_prefix, "bridge_manifest.json")

    sample_features.to_parquet(out_sample, index=False)
    drug_features.to_parquet(out_drug, index=False)
    labels.to_parquet(out_labels, index=False)
    mapping_table.to_parquet(out_mapping, index=False)

    qc_report = {
        "run_id": args.run_id,
        "input_uris": {
            "label_uri": args.label_uri,
            "sample_uri": args.sample_uri,
            "drug_uri": args.drug_uri,
        },
        "label_to_feature_join_qc": {
            "label_pair_rows": int(labels_key.shape[0]),
            "unmatched_samples": unmatched_samples,
            "unmatched_drugs": unmatched_drugs,
            "join_rate_samples": join_rate_samples,
            "join_rate_drugs": join_rate_drugs,
        },
        "labels_qc": labels_qc,
        "sample_qc": sample_qc,
        "drug_qc": drug_qc,
    }
    _write_json(qc_report, out_qc)

    manifest = {
        "run_id": args.run_id,
        "purpose": "bridge preprocessing for Nextflow FE input contract",
        "output_files": {
            "sample_features": out_sample,
            "drug_features": out_drug,
            "labels": out_labels,
            "mapping_table": out_mapping,
            "join_qc_report": out_qc,
        },
        "notes": {
            "labels_mapping": "cell_line_name->sample_id, DRUG_ID->canonical_drug_id, ln_IC50->ic50",
            "smiles_policy": "smiles is included; has_smiles indicates availability for downstream ADMET/descriptor steps",
        },
    }
    _write_json(manifest, out_manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
