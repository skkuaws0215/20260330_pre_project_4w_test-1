#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REQUIRED_COLS = ["sample_id", "canonical_drug_id", "label_regression"]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def inspect_parquet(path: Path) -> dict:
    df = pd.read_parquet(path)
    cols = list(df.columns)
    info = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns_head": cols[:40],
        "required_cols_present": {k: (k in cols) for k in REQUIRED_COLS},
        "n_required_present": int(sum((k in cols) for k in REQUIRED_COLS)),
        "index_name": str(df.index.name),
        "index_dtype": str(df.index.dtype),
    }
    mb_cols = [c for c in cols if isinstance(c, str) and c.startswith("MB-")]
    info["mb_like_column_count"] = int(len(mb_cols))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare METABRIC true-validation staging and schema audit.")
    ap.add_argument(
        "--metabric-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/8/metabric/",
        help="Source METABRIC S3 prefix",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep",
        help="Local output directory for audit and staging metadata",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/",
        help="Target S3 prefix under results/features_nextflow_team4/",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = ["08_raw.parquet", "54_filtered.parquet"]
    report = {
        "source_prefix": args.metabric_prefix,
        "target_prefix": args.upload_prefix,
        "required_for_true_validation": REQUIRED_COLS,
        "files": {},
        "verdict": "",
        "next_actions": [],
    }

    with TemporaryDirectory() as td:
        tdp = Path(td)
        for fn in files:
            src = args.metabric_prefix.rstrip("/") + "/" + fn
            dst = tdp / fn
            _run(["aws", "s3", "cp", src, str(dst)])
            file_info = inspect_parquet(dst)
            report["files"][fn] = file_info

    all_missing = []
    for fn, info in report["files"].items():
        miss = [k for k, ok in info["required_cols_present"].items() if not ok]
        all_missing.extend(f"{fn}:{m}" for m in miss)

    if all_missing:
        report["verdict"] = "BLOCKED_FOR_TRUE_EXTERNAL_VALIDATION"
        report["next_actions"] = [
            "Build mapping from METABRIC sample IDs (MB-*) to project sample_id space.",
            "Create canonical_drug_id-linked external pair table.",
            "Attach label_regression compatible with current training target definition.",
            "Regenerate external pair feature table with same schema as pair_features_newfe_v2.parquet.",
        ]
    else:
        report["verdict"] = "READY_FOR_TRUE_EXTERNAL_VALIDATION"

    (out_dir / "schema_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    summary = pd.DataFrame(
        [
            {
                "file": fn,
                "n_rows": info["shape"][0],
                "n_cols": info["shape"][1],
                "mb_like_column_count": info["mb_like_column_count"],
                "n_required_present": info["n_required_present"],
            }
            for fn, info in report["files"].items()
        ]
    )
    summary.to_csv(out_dir / "schema_audit_summary.csv", index=False)

    _run(["aws", "s3", "cp", str(out_dir), args.upload_prefix, "--recursive"])
    print(f"[done] wrote: {out_dir}")
    print(f"[done] uploaded to: {args.upload_prefix}")
    print(f"[verdict] {report['verdict']}")


if __name__ == "__main__":
    main()

