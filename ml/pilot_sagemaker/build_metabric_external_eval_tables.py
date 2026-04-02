#!/usr/bin/env python3
"""
Build METABRIC external evaluation tables from a finalized MB-* -> sample_id bridge.

Uses the same (sample_id, canonical_drug_id) rows and feature values as the
internal train FE parquet — projection is by bridge: only pairs whose sample_id
is bridged are exported. No feature selection, scaling, or encoding refit.

Outputs (under --output-dir):
  - metabric_external_pair_features.parquet
  - metabric_external_labels.parquet
  - metabric_external_eval_pairs.parquet
  - metabric_external_eval_build_report.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CWD = Path.cwd()

KEY_COLS = ["sample_id", "canonical_drug_id"]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (CWD / p).resolve()


def _load_parquet(uri: str, tdp: Path, name: str) -> pd.DataFrame:
    uri = uri.strip()
    if uri.startswith("s3://"):
        dst = tdp / f"{name}.parquet"
        _run(["aws", "s3", "cp", uri, str(dst)])
        return pd.read_parquet(dst)
    return pd.read_parquet(Path(uri))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build METABRIC external pair features/labels/eval table (train schema, transform-only)."
    )
    ap.add_argument(
        "--finalized-bridge-csv",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_finalized/finalized_metabric_sample_bridge.csv",
    )
    ap.add_argument(
        "--internal-features-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    ap.add_argument(
        "--internal-labels-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/external_eval_tables",
    )
    ap.add_argument(
        "--upload-prefix",
        default="",
        help="If set (s3://.../), upload outputs with aws s3 cp --recursive",
    )
    ap.add_argument(
        "--formats",
        default="parquet",
        help="parquet, csv, or both (comma-separated)",
    )
    args = ap.parse_args()

    out_dir = _resolve_path(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_path = _resolve_path(Path(args.finalized_bridge_csv))
    formats = {x.strip().lower() for x in args.formats.split(",") if x.strip()}
    if not formats.issubset({"parquet", "csv"}):
        raise SystemExit("--formats must be parquet, csv, and/or both")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "finalized_bridge_csv": str(bridge_path),
            "internal_features_uri": args.internal_features_uri,
            "internal_labels_uri": args.internal_labels_uri,
            "output_dir": str(out_dir),
            "transform_only_policy": (
                "Feature matrix rows are copied verbatim from internal train FE for each "
                "(sample_id, canonical_drug_id) after filtering labels to bridged sample_id. "
                "No scaler/encoder fit, no feature selection, no re-ingest from METABRIC matrix."
            ),
            "outputs": {},
        },
        "validation": {},
        "summary": {},
    }

    if not bridge_path.is_file():
        report["validation"]["bridge_status"] = "MISSING_FILE"
        report["summary"]["n_pairs"] = 0
        out_report = out_dir / "metabric_external_eval_build_report.json"
        out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[exit] missing bridge: {bridge_path}")
        return

    bridge = pd.read_csv(bridge_path)
    req = {"metabric_sample_id", "sample_id"}.issubset(bridge.columns)
    if bridge.empty or not req:
        report["validation"]["bridge_status"] = "EMPTY_OR_INVALID_COLUMNS"
        report["summary"]["n_pairs"] = 0
        (out_dir / "metabric_external_eval_build_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[exit] empty or invalid bridge")
        return

    bridge = bridge.copy()
    bridge["metabric_sample_id"] = bridge["metabric_sample_id"].astype(str).str.strip()
    bridge["sample_id"] = bridge["sample_id"].astype(str).str.strip()
    bridge = bridge.drop_duplicates(subset=["metabric_sample_id", "sample_id"], keep="first")
    allowed_samples = set(bridge["sample_id"])

    use_tmp = args.internal_features_uri.startswith("s3://") or args.internal_labels_uri.startswith(
        "s3://"
    )

    ctx = TemporaryDirectory() if use_tmp else None
    tdp = Path(ctx.__enter__()) if ctx else REPO_ROOT

    try:
        feat = _load_parquet(args.internal_features_uri, tdp, "internal_feat")
        lab = _load_parquet(args.internal_labels_uri, tdp, "internal_lab")
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    train_cols = list(feat.columns)
    train_feat_order = [c for c in train_cols if c not in KEY_COLS]

    for c in KEY_COLS:
        if c not in feat.columns:
            raise SystemExit(f"features missing key column {c}")
        if c not in lab.columns:
            raise SystemExit(f"labels missing key column {c}")

    lab = lab.copy()
    lab["sample_id"] = lab["sample_id"].astype(str).str.strip()
    lab["canonical_drug_id"] = lab["canonical_drug_id"].astype(str).str.strip()
    lab_f = lab[lab["sample_id"].isin(allowed_samples)].copy()

    feat = feat.copy()
    feat["sample_id"] = feat["sample_id"].astype(str).str.strip()
    feat["canonical_drug_id"] = feat["canonical_drug_id"].astype(str).str.strip()

    merged = lab_f.merge(feat, on=KEY_COLS, how="inner", validate="one_to_one")

    label_cols = [c for c in lab.columns if c not in KEY_COLS]
    labels_out = merged[KEY_COLS + label_cols].drop_duplicates(KEY_COLS).sort_values(KEY_COLS)

    feat_out = merged[KEY_COLS + train_feat_order].sort_values(KEY_COLS)

    bridge_dedup = bridge.sort_values("sample_id").drop_duplicates("sample_id", keep="first")
    base_pairs = merged[KEY_COLS + label_cols + train_feat_order]
    eval_pairs = base_pairs.merge(
        bridge_dedup, on="sample_id", how="left", validate="many_to_one"
    )
    bridge_lead = ["metabric_sample_id"] + [
        c
        for c in bridge_dedup.columns
        if c not in ("sample_id", "metabric_sample_id")
    ]
    front = [c for c in bridge_lead if c in eval_pairs.columns] + KEY_COLS + label_cols
    rest = [c for c in eval_pairs.columns if c not in front]
    eval_pairs = eval_pairs[front + rest].sort_values(KEY_COLS)

    ext_feat_names = [c for c in feat_out.columns if c not in KEY_COLS]
    missing_cols = [c for c in train_feat_order if c not in feat_out.columns]
    extra_cols = [c for c in ext_feat_names if c not in train_feat_order]
    order_ok = ext_feat_names == train_feat_order

    lab_keys_miss = lab_f.merge(feat[KEY_COLS], on=KEY_COLS, how="left", indicator=True)
    n_missing_feat_row = int((lab_keys_miss["_merge"] == "left_only").sum())

    report["validation"].update(
        {
            "bridge_status": "OK",
            "feature_column_order_matches_train": bool(order_ok),
            "n_train_feature_columns": int(len(train_feat_order)),
            "n_external_feature_columns": int(len(ext_feat_names)),
            "missing_feature_columns_vs_train": missing_cols,
            "extra_feature_columns_vs_train": extra_cols,
            "label_pairs_without_feature_row": int(n_missing_feat_row),
            "transform_only_compliance": (
                "PASS: rows taken from train FE via inner join on keys; "
                "no sklearn/pytorch preprocessor refit in this script."
            ),
        }
    )
    report["summary"] = {
        "n_bridge_rows": int(len(bridge)),
        "n_bridged_unique_sample_id": int(len(allowed_samples)),
        "n_label_pairs_for_bridged_samples": int(len(lab_f)),
        "n_eval_pairs_written": int(len(feat_out)),
        "n_unique_canonical_drug_id": int(feat_out["canonical_drug_id"].nunique()),
        "note": (
            "Labels are internal labels_B_graph rows for bridged samples (not METABRIC-native response)."
            " Features are internal pair_features_newfe_v2 values for the same keys."
        ),
    }

    def _write_df(df: pd.DataFrame, stem: str) -> None:
        paths = {}
        if "parquet" in formats:
            pp = out_dir / f"{stem}.parquet"
            df.to_parquet(pp, index=False)
            paths["parquet"] = str(pp)
        if "csv" in formats:
            cp = out_dir / f"{stem}.csv"
            df.to_csv(cp, index=False)
            paths["csv"] = str(cp)
        report["manifest"]["outputs"][stem] = paths

    _write_df(feat_out, "metabric_external_pair_features")
    _write_df(labels_out, "metabric_external_labels")
    _write_df(eval_pairs, "metabric_external_eval_pairs")

    rep_path = out_dir / "metabric_external_eval_build_report.json"
    report["manifest"]["outputs"]["metabric_external_eval_build_report"] = {"json": str(rep_path)}
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.upload_prefix.strip():
        pref = args.upload_prefix.strip().rstrip("/") + "/"
        _run(["aws", "s3", "cp", str(out_dir), pref, "--recursive"])

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
