#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def norm_id(x: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper().strip())


def read_parquet_s3(uri: str, local_path: Path) -> pd.DataFrame:
    _run(["aws", "s3", "cp", uri, str(local_path)])
    return pd.read_parquet(local_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit bridge mapping for true METABRIC external validation.")
    ap.add_argument(
        "--metabric-matrix-s3",
        default="s3://drug-discovery-joe-raw-data-team4/8/metabric/54_filtered.parquet",
    )
    ap.add_argument(
        "--internal-features-s3",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    ap.add_argument(
        "--internal-labels-s3",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet",
    )
    ap.add_argument(
        "--ml-ready-meta-s3",
        default="s3://drug-discovery-joe-raw-data-team4/ml_ready/metadata.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/mapping_audit",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/mapping_audit/",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as td:
        tdp = Path(td)
        met = read_parquet_s3(args.metabric_matrix_s3, tdp / "metabric.parquet")
        feat = read_parquet_s3(args.internal_features_s3, tdp / "features.parquet")
        lab = read_parquet_s3(args.internal_labels_s3, tdp / "labels.parquet")
        meta = read_parquet_s3(args.ml_ready_meta_s3, tdp / "ml_ready_metadata.parquet")

    # Internal IDs
    internal_samples = feat["sample_id"].astype(str).str.strip().unique().tolist()
    internal_samples_set = set(internal_samples)
    internal_drugs = lab["canonical_drug_id"].astype(str).str.strip().unique().tolist()

    # METABRIC sample-like ids are matrix columns (MB-xxxx)
    met_cols = [str(c) for c in met.columns]
    met_sample_cols = [c for c in met_cols if c.startswith("MB-")]
    met_sample_set = set(met_sample_cols)

    # Direct sample overlap
    direct_overlap = sorted(list(internal_samples_set & met_sample_set))

    # Normalized overlap
    internal_norm = {norm_id(s): s for s in internal_samples}
    met_norm = {norm_id(s): s for s in met_sample_cols}
    norm_overlap_keys = sorted(list(set(internal_norm) & set(met_norm)))
    norm_overlap_pairs = [
        {"internal_sample_id": internal_norm[k], "metabric_sample_col": met_norm[k]} for k in norm_overlap_keys
    ]

    # Cross-check with ml_ready metadata cell_line_name (if useful as bridge)
    meta_cols = set(meta.columns.astype(str))
    has_cellline = "cell_line_name" in meta_cols
    cell_line_overlap = []
    if has_cellline:
        cell_lines = sorted(set(meta["cell_line_name"].astype(str).str.strip()))
        cell_line_overlap = sorted(list(internal_samples_set & set(cell_lines)))

    report = {
        "inputs": {
            "metabric_matrix_s3": args.metabric_matrix_s3,
            "internal_features_s3": args.internal_features_s3,
            "internal_labels_s3": args.internal_labels_s3,
            "ml_ready_meta_s3": args.ml_ready_meta_s3,
        },
        "counts": {
            "internal_sample_id_nunique": int(len(internal_samples_set)),
            "internal_canonical_drug_id_nunique": int(len(set(internal_drugs))),
            "metabric_sample_column_n": int(len(met_sample_cols)),
            "direct_sample_overlap_n": int(len(direct_overlap)),
            "normalized_sample_overlap_n": int(len(norm_overlap_pairs)),
            "cell_line_overlap_with_internal_samples_n": int(len(cell_line_overlap)),
        },
        "examples": {
            "internal_samples_head": internal_samples[:15],
            "metabric_sample_cols_head": met_sample_cols[:15],
            "direct_sample_overlap_head": direct_overlap[:15],
            "normalized_overlap_head": norm_overlap_pairs[:15],
            "cell_line_overlap_head": cell_line_overlap[:15],
        },
        "critical_gap": {
            "metabric_has_canonical_drug_id_axis": False,
            "metabric_has_pair_level_label_regression": False,
            "reason": "Current METABRIC matrix is sample x gene expression; it does not contain canonical_drug_id pair axis used by current 3-model pipeline.",
        },
        "verdict": "BLOCKED_NO_SAMPLE_BRIDGE_AND_NO_DRUG_AXIS",
        "next_actions": [
            "Define explicit bridge map: MB-* sample IDs -> project sample_id space (if biologically intended).",
            "Construct external pair table with canonical_drug_id axis and label_regression target.",
            "Regenerate pair_features_newfe_v2-compatible external feature table before model inference.",
        ],
    }

    (out_dir / "mapping_bridge_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    pd.DataFrame(
        [
            {"metric": "internal_sample_id_nunique", "value": report["counts"]["internal_sample_id_nunique"]},
            {"metric": "internal_canonical_drug_id_nunique", "value": report["counts"]["internal_canonical_drug_id_nunique"]},
            {"metric": "metabric_sample_column_n", "value": report["counts"]["metabric_sample_column_n"]},
            {"metric": "direct_sample_overlap_n", "value": report["counts"]["direct_sample_overlap_n"]},
            {"metric": "normalized_sample_overlap_n", "value": report["counts"]["normalized_sample_overlap_n"]},
            {"metric": "cell_line_overlap_with_internal_samples_n", "value": report["counts"]["cell_line_overlap_with_internal_samples_n"]},
        ]
    ).to_csv(out_dir / "mapping_bridge_audit_summary.csv", index=False)

    _run(["aws", "s3", "cp", str(out_dir), args.upload_prefix, "--recursive"])
    print(f"[done] wrote: {out_dir}")
    print(f"[done] uploaded to: {args.upload_prefix}")
    print(f"[verdict] {report['verdict']}")


if __name__ == "__main__":
    main()

