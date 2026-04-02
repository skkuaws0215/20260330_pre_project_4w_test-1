#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CWD = Path.cwd()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd or REPO_ROOT))


def _resolve_under_cwd(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (CWD / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orchestrate METABRIC true validation prep after finalized sample bridge exists."
    )
    ap.add_argument(
        "--finalized-bridge-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_finalized/finalized_metabric_sample_bridge.csv",
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
        "--metabric-matrix-s3",
        default="s3://drug-discovery-joe-raw-data-team4/8/metabric/54_filtered.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/pipeline",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/pipeline/",
    )
    ap.add_argument(
        "--run-schema-audit",
        action="store_true",
        help="Also run prepare_metabric_true_validation_inputs.py (schema audit upload)",
    )
    args = ap.parse_args()

    out_dir = _resolve_under_cwd(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_path = _resolve_under_cwd(Path(args.finalized_bridge_csv))
    if not bridge_path.exists():
        manifest = {
            "bridge_status": "MISSING_FILE",
            "finalized_bridge_csv": str(bridge_path),
            "message": "Run finalize_metabric_sample_bridge.py after filling manual_bridge_candidates.csv",
        }
        man_path = out_dir / "metabric_true_validation_pipeline_manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        _run(["aws", "s3", "cp", str(man_path.resolve()), args.upload_prefix])
        print(f"[exit] {manifest['bridge_status']}: {bridge_path}")
        return

    bridge = pd.read_csv(bridge_path)
    if bridge.empty or "metabric_sample_id" not in bridge.columns or "sample_id" not in bridge.columns:
        manifest = {
            "bridge_status": "EMPTY_OR_INVALID",
            "finalized_bridge_csv": str(bridge_path.resolve()),
            "message": "Bridge CSV empty or missing metabric_sample_id/sample_id. Fix finalize step or candidates.",
        }
        man_path = out_dir / "metabric_true_validation_pipeline_manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        _run(["aws", "s3", "cp", str(man_path.resolve()), args.upload_prefix])
        print(f"[exit] {manifest['bridge_status']}")
        return

    bridge["metabric_sample_id"] = bridge["metabric_sample_id"].astype(str).str.strip()
    bridge["sample_id"] = bridge["sample_id"].astype(str).str.strip()
    dup_mb = bridge["metabric_sample_id"].duplicated(keep=False)
    dup_int = bridge["sample_id"].duplicated(keep=False)

    with TemporaryDirectory() as td:
        tdp = Path(td)
        feat_p = tdp / "feat.parquet"
        _run(["aws", "s3", "cp", args.internal_features_s3, str(feat_p)])
        feat = pd.read_parquet(feat_p)
    valid_internal = set(feat["sample_id"].astype(str).str.strip().unique())
    invalid = bridge[~bridge["sample_id"].isin(valid_internal)]

    bridge_ok = not dup_mb.any() and not dup_int.any() and len(invalid) == 0 and len(bridge) > 0

    manifest = {
        "pipeline_version": "1",
        "bridge_status": "READY" if bridge_ok else "INVALID_BRIDGE",
        "finalized_bridge_csv": str(bridge_path.resolve()),
        "counts": {
            "n_bridge_rows": int(len(bridge)),
            "duplicate_metabric_rows": int(dup_mb.sum()),
            "duplicate_internal_rows": int(dup_int.sum()),
            "invalid_internal_sample_rows": int(len(invalid)),
        },
        "inputs": {
            "internal_features_s3": args.internal_features_s3,
            "internal_labels_s3": args.internal_labels_s3,
            "metabric_matrix_s3": args.metabric_matrix_s3,
        },
        "true_validation_blockers": [] if bridge_ok else [],
        "next_steps": [],
    }

    if not bridge_ok:
        manifest["true_validation_blockers"].extend(
            [
                "duplicate_metabric_sample_id" if dup_mb.any() else None,
                "duplicate_internal_sample_id" if dup_int.any() else None,
                "invalid_internal_sample_id" if len(invalid) else None,
            ]
        )
        manifest["true_validation_blockers"] = [x for x in manifest["true_validation_blockers"] if x]

    if bridge_ok:
        manifest["next_steps"] = [
            {
                "step": 1,
                "name": "build_external_pair_table",
                "description": "Build METABRIC pair rows: keys (sample_id after bridge, canonical_drug_id), label_regression, and feature columns aligned to train schema (transform-only from train FE).",
            },
            {
                "step": 2,
                "name": "run_model_inference",
                "description": "Load SageMaker artifacts; predict on external pair table; write predictions parquet.",
            },
            {
                "step": 3,
                "name": "aggregate_metrics",
                "description": "Compute RMSE/MAE/Spearman, rank consistency vs internal holdout; write metabric_true_validation_summary.csv.",
            },
        ]
        manifest["recommended_commands"] = [
            "python3 ml/pilot_sagemaker/prepare_metabric_true_validation_inputs.py",
            "python3 ml/pilot_sagemaker/validate_metabric_sample_bridge.py --format finalized --bridge-csv results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_finalized/finalized_metabric_sample_bridge.csv",
        ]

    man_path = out_dir / "metabric_true_validation_pipeline_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    _run(["aws", "s3", "cp", str(man_path.resolve()), args.upload_prefix])

    if args.run_schema_audit:
        prep = SCRIPT_DIR / "prepare_metabric_true_validation_inputs.py"
        if prep.exists():
            _run(["python3", str(prep)], cwd=REPO_ROOT)

    print(f"[done] manifest: {out_dir / 'metabric_true_validation_pipeline_manifest.json'}")
    print(f"[bridge_status] {manifest['bridge_status']}")


if __name__ == "__main__":
    main()
