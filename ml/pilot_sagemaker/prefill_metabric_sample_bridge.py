#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper().strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-prefill METABRIC MB-* -> sample_id bridge candidates")
    ap.add_argument(
        "--bridge-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_template/metabric_sample_bridge_template.csv",
    )
    ap.add_argument(
        "--internal-features-s3",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    ap.add_argument(
        "--ml-ready-meta-s3",
        default="s3://drug-discovery-joe-raw-data-team4/ml_ready/metadata.parquet",
    )
    ap.add_argument(
        "--output-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_template/metabric_sample_bridge_template_prefilled.csv",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_template/",
    )
    args = ap.parse_args()

    bridge = pd.read_csv(args.bridge_csv)
    bridge["metabric_sample_id"] = bridge["metabric_sample_id"].astype(str).str.strip()
    if "sample_id" not in bridge.columns:
        bridge["sample_id"] = ""
    if "mapping_confidence" not in bridge.columns:
        bridge["mapping_confidence"] = ""
    if "source_note" not in bridge.columns:
        bridge["source_note"] = ""
    if "is_active" not in bridge.columns:
        bridge["is_active"] = False

    with TemporaryDirectory() as td:
        tdp = Path(td)
        feat_p = tdp / "feat.parquet"
        meta_p = tdp / "meta.parquet"
        _run(["aws", "s3", "cp", args.internal_features_s3, str(feat_p)])
        _run(["aws", "s3", "cp", args.ml_ready_meta_s3, str(meta_p)])
        feat = pd.read_parquet(feat_p)
        meta = pd.read_parquet(meta_p)

    internal_samples = sorted(set(feat["sample_id"].astype(str).str.strip()))
    norm_to_internal = {}
    for s in internal_samples:
        k = _norm(s)
        norm_to_internal.setdefault(k, []).append(s)

    # Optional helper: known cell lines in ml_ready metadata
    meta_cell_lines = set()
    if "cell_line_name" in meta.columns:
        meta_cell_lines = set(meta["cell_line_name"].astype(str).str.strip())

    n_prefilled = 0
    for i, row in bridge.iterrows():
        if str(row.get("sample_id", "")).strip():
            continue
        mb = str(row["metabric_sample_id"]).strip()
        k = _norm(mb)
        cands = norm_to_internal.get(k, [])
        if len(cands) == 1:
            bridge.at[i, "sample_id"] = cands[0]
            bridge.at[i, "mapping_confidence"] = "high"
            bridge.at[i, "source_note"] = "auto_exact_normalized_match"
            bridge.at[i, "is_active"] = True
            n_prefilled += 1
        elif len(cands) > 1:
            bridge.at[i, "mapping_confidence"] = "low"
            bridge.at[i, "source_note"] = f"ambiguous_auto_match:{'|'.join(cands)}"
            bridge.at[i, "is_active"] = False
        else:
            # No direct bridge; keep inactive and annotate if internal cell-line catalog exists.
            if mb in meta_cell_lines:
                bridge.at[i, "mapping_confidence"] = "medium"
                bridge.at[i, "source_note"] = "present_in_ml_ready_cell_line_catalog_only"
            else:
                bridge.at[i, "source_note"] = "no_auto_match"

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(out, index=False)
    _run(["aws", "s3", "cp", str(out), args.upload_prefix])
    print(f"[done] wrote {out}")
    print(f"[prefilled] {n_prefilled}")


if __name__ == "__main__":
    main()

