#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate manual METABRIC MB-* to sample_id bridge")
    ap.add_argument(
        "--format",
        choices=("template", "finalized"),
        default="template",
        help="template: metabric_sample_id + sample_id + is_active; finalized: metabric_sample_id + sample_id (all active)",
    )
    ap.add_argument(
        "--bridge-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_template/metabric_sample_bridge_template.csv",
    )
    ap.add_argument(
        "--internal-features-s3",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_validation",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_validation/",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_csv = Path(args.bridge_csv)
    if not bridge_csv.is_absolute():
        bridge_csv = (Path.cwd() / bridge_csv).resolve()

    bridge = pd.read_csv(bridge_csv)
    bridge["metabric_sample_id"] = bridge["metabric_sample_id"].astype(str).str.strip()
    if args.format == "finalized":
        bridge["sample_id"] = bridge["sample_id"].fillna("").astype(str).str.strip()
        bridge["is_active"] = True
    else:
        bridge["sample_id"] = bridge["sample_id"].fillna("").astype(str).str.strip()
        bridge["is_active"] = bridge["is_active"].fillna(False).astype(bool)

    with TemporaryDirectory() as td:
        tdp = Path(td)
        feat_p = tdp / "feat.parquet"
        _run(["aws", "s3", "cp", args.internal_features_s3, str(feat_p)])
        feat = pd.read_parquet(feat_p)
    valid_internal = set(feat["sample_id"].astype(str).str.strip().unique())

    active = bridge[bridge["is_active"] & bridge["sample_id"].ne("")].copy()
    invalid_sample_rows = active[~active["sample_id"].isin(valid_internal)].copy()
    dup_met = active["metabric_sample_id"].duplicated(keep=False).sum()
    dup_internal = active["sample_id"].duplicated(keep=False).sum()

    report = {
        "bridge_format": args.format,
        "bridge_csv": str(bridge_csv),
        "internal_sample_id_nunique": int(len(valid_internal)),
        "metabric_rows_total": int(len(bridge)),
        "active_mapping_rows": int(len(active)),
        "coverage_ratio_active": float(len(active) / len(bridge)) if len(bridge) else 0.0,
        "invalid_sample_id_rows": int(len(invalid_sample_rows)),
        "duplicate_metabric_sample_rows": int(dup_met),
        "duplicate_internal_sample_rows": int(dup_internal),
        "status": "PASS" if (len(invalid_sample_rows) == 0 and dup_met == 0 and dup_internal == 0) else "FAIL",
    }

    (out_dir / "bridge_validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pd.DataFrame([report]).to_csv(out_dir / "bridge_validation_report.csv", index=False)
    invalid_sample_rows.to_csv(out_dir / "bridge_invalid_rows.csv", index=False)

    _run(["aws", "s3", "cp", str(out_dir), args.upload_prefix, "--recursive"])
    print(f"[done] validation written: {out_dir}")
    print(f"[done] uploaded: {args.upload_prefix}")
    print(f"[status] {report['status']} coverage={report['coverage_ratio_active']:.4f}")


if __name__ == "__main__":
    main()

