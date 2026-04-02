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


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "t")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract finalized MB-* -> sample_id bridge from manual_bridge_candidates.csv"
    )
    ap.add_argument(
        "--candidates-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_candidates/manual_bridge_candidates.csv",
    )
    ap.add_argument(
        "--internal-features-s3",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_finalized",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_finalized/",
    )
    args = ap.parse_args()

    cand_path = Path(args.candidates_csv)
    if not cand_path.is_absolute():
        cand_path = (Path.cwd() / cand_path).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(cand_path)
    if "manual_selected" not in cand.columns or "manual_selected_sample_id" not in cand.columns:
        raise SystemExit("candidates CSV must contain manual_selected and manual_selected_sample_id")

    cand["manual_selected"] = cand["manual_selected"].map(_as_bool)
    cand["manual_selected_sample_id"] = cand["manual_selected_sample_id"].fillna("").astype(str).str.strip()
    cand["metabric_sample_id"] = cand["metabric_sample_id"].astype(str).str.strip()

    n_unique_mb_in_file = int(cand["metabric_sample_id"].nunique())

    selected = cand[cand["manual_selected"]].copy()
    selected = selected[selected["manual_selected_sample_id"].ne("")]

    with TemporaryDirectory() as td:
        tdp = Path(td)
        feat_p = tdp / "feat.parquet"
        _run(["aws", "s3", "cp", args.internal_features_s3, str(feat_p)])
        feat = pd.read_parquet(feat_p)
    valid_internal = set(feat["sample_id"].astype(str).str.strip().unique())

    invalid = selected[~selected["manual_selected_sample_id"].isin(valid_internal)]
    dup_mb = selected["metabric_sample_id"].duplicated(keep=False)
    dup_int = selected["manual_selected_sample_id"].duplicated(keep=False)

    has_dup_mb = bool(dup_mb.any())
    has_dup_int = bool(dup_int.any())
    has_invalid = len(invalid) > 0
    ok = not has_dup_mb and not has_dup_int and not has_invalid and len(selected) > 0

    bridge_path = out_dir / "finalized_metabric_sample_bridge.csv"
    if ok:
        finalized = selected[["metabric_sample_id", "manual_selected_sample_id", "manual_note"]].copy()
        finalized = finalized.rename(columns={"manual_selected_sample_id": "sample_id"})
        finalized["is_active"] = True
        finalized.to_csv(bridge_path, index=False)
        n_mappings = int(len(finalized))
    else:
        pd.DataFrame(
            columns=["metabric_sample_id", "sample_id", "manual_note", "is_active"]
        ).to_csv(bridge_path, index=False)
        n_mappings = 0

    coverage_ratio = float(n_mappings / n_unique_mb_in_file) if n_unique_mb_in_file else 0.0

    summary = {
        "input_candidates_csv": str(cand_path),
        "rules": {
            "use_rows_where": "manual_selected == True",
            "final_internal_sample_id_column": "manual_selected_sample_id -> sample_id in output CSV",
        },
        "validation": {
            "n_rows_manual_selected_true": int(len(selected)),
            "n_unique_metabric_in_candidates_file": n_unique_mb_in_file,
            "n_finalized_active_mappings": n_mappings,
            "coverage_ratio_vs_unique_mb_in_candidates": round(coverage_ratio, 6),
            "duplicate_metabric_sample_id": has_dup_mb,
            "duplicate_metabric_sample_id_count": int(dup_mb.sum()),
            "duplicate_internal_sample_id": has_dup_int,
            "duplicate_internal_sample_id_count": int(dup_int.sum()),
            "invalid_internal_sample_id_rows": int(len(invalid)),
            "duplicate_metabric_ids": sorted(
                selected.loc[dup_mb, "metabric_sample_id"].drop_duplicates().tolist()
            ),
            "duplicate_internal_ids": sorted(
                selected.loc[dup_int, "manual_selected_sample_id"].drop_duplicates().tolist()
            ),
            "invalid_internal_sample_details": invalid[
                ["metabric_sample_id", "manual_selected_sample_id"]
            ].to_dict("records")[:50],
        },
        "status": "PASS"
        if ok
        else ("EMPTY" if len(selected) == 0 else "FAIL"),
        "output_csv": str(bridge_path.resolve()),
        "next_step": "python3 ml/pilot_sagemaker/run_metabric_true_validation_pipeline.py --finalized-bridge-csv <path>",
    }

    (out_dir / "finalized_metabric_sample_bridge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    _run(["aws", "s3", "cp", str(out_dir.resolve()), args.upload_prefix, "--recursive"])
    print(f"[done] {bridge_path}")
    print(f"[done] summary status={summary['status']} mappings={n_mappings} coverage={coverage_ratio:.4f}")
    print(f"[upload] {args.upload_prefix}")


if __name__ == "__main__":
    main()
