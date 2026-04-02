#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create manual bridge template: MB sample -> project sample_id")
    ap.add_argument(
        "--metabric-matrix-s3",
        default="s3://drug-discovery-joe-raw-data-team4/8/metabric/54_filtered.parquet",
    )
    ap.add_argument(
        "--internal-features-s3",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_template",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_template/",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as td:
        tdp = Path(td)
        met_p = tdp / "met.parquet"
        feat_p = tdp / "feat.parquet"
        _run(["aws", "s3", "cp", args.metabric_matrix_s3, str(met_p)])
        _run(["aws", "s3", "cp", args.internal_features_s3, str(feat_p)])
        met = pd.read_parquet(met_p)
        feat = pd.read_parquet(feat_p)

    met_samples = sorted([str(c) for c in met.columns if str(c).startswith("MB-")])
    internal_samples = sorted(feat["sample_id"].astype(str).str.strip().unique())

    bridge = pd.DataFrame(
        {
            "metabric_sample_id": met_samples,
            "sample_id": "",
            "mapping_confidence": "",
            "source_note": "",
            "is_active": False,
        }
    )
    bridge.to_csv(out_dir / "metabric_sample_bridge_template.csv", index=False)

    pd.DataFrame({"sample_id": internal_samples}).to_csv(
        out_dir / "internal_sample_id_reference.csv", index=False
    )

    (out_dir / "README_bridge_template.txt").write_text(
        "Fill metabric_sample_bridge_template.csv with MB-* -> sample_id mappings.\n"
        "Columns:\n"
        "- metabric_sample_id: MB-* sample ID from METABRIC matrix\n"
        "- sample_id: project sample_id to map to\n"
        "- mapping_confidence: high/medium/low\n"
        "- source_note: mapping evidence\n"
        "- is_active: True only for approved mappings\n"
    )

    _run(["aws", "s3", "cp", str(out_dir), args.upload_prefix, "--recursive"])
    print(f"[done] template written: {out_dir}")
    print(f"[done] uploaded: {args.upload_prefix}")
    print(f"[counts] metabric_samples={len(met_samples)} internal_samples={len(internal_samples)}")


if __name__ == "__main__":
    main()

