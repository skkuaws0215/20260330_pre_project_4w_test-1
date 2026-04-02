#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).upper(), str(b).upper()).ratio()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build manual bridge candidates for MB-* -> sample_id mapping")
    ap.add_argument(
        "--ml-ready-meta-s3",
        default="s3://drug-discovery-joe-raw-data-team4/ml_ready/metadata.parquet",
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
        "--metabric-clinical-tsv",
        default="/Users/skku_aws2_14/Downloads/brca_metabric_clinical_data.tsv",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_candidates",
    )
    ap.add_argument(
        "--upload-prefix",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/bridge_candidates/",
    )
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as td:
        tdp = Path(td)
        meta_p = tdp / "meta.parquet"
        feat_p = tdp / "feat.parquet"
        lab_p = tdp / "lab.parquet"
        _run(["aws", "s3", "cp", args.ml_ready_meta_s3, str(meta_p)])
        _run(["aws", "s3", "cp", args.internal_features_s3, str(feat_p)])
        _run(["aws", "s3", "cp", args.internal_labels_s3, str(lab_p)])
        meta = pd.read_parquet(meta_p)
        feat = pd.read_parquet(feat_p)
        lab = pd.read_parquet(lab_p)

    clin = pd.read_csv(args.metabric_clinical_tsv, sep="\t")

    # Internal sample-level hints
    feat_samples = feat["sample_id"].astype(str).str.strip()
    feat_sample_count = feat_samples.value_counts().rename_axis("sample_id").reset_index(name="internal_n_feature_rows")

    lab_s = lab.copy()
    lab_s["sample_id"] = lab_s["sample_id"].astype(str).str.strip()
    sample_stats = (
        lab_s.groupby("sample_id")
        .agg(
            internal_n_label_rows=("label_regression", "size"),
            internal_label_mean=("label_regression", "mean"),
            internal_label_std=("label_regression", "std"),
        )
        .reset_index()
    )
    sample_stats["internal_label_std"] = sample_stats["internal_label_std"].fillna(0.0)

    # Metadata hints
    meta_s = meta.copy()
    meta_s["cell_line_name"] = meta_s["cell_line_name"].astype(str).str.strip()
    meta_hint = (
        meta_s.groupby("cell_line_name")
        .agg(
            internal_meta_drug_coverage_n=("drug_id", "nunique"),
            internal_meta_pam50_nonnull_n=("pam50_subtype", lambda s: int(s.notna().sum())),
            internal_meta_pam50_example=("pam50_subtype", lambda s: next((str(v) for v in s if pd.notna(v)), "")),
        )
        .reset_index()
        .rename(columns={"cell_line_name": "sample_id"})
    )

    internal_ref = (
        pd.DataFrame({"sample_id": sorted(set(feat_samples.tolist()))})
        .merge(feat_sample_count, on="sample_id", how="left")
        .merge(sample_stats, on="sample_id", how="left")
        .merge(meta_hint, on="sample_id", how="left")
    )
    internal_ref["internal_meta_pam50_example"] = internal_ref["internal_meta_pam50_example"].fillna("")
    internal_ref["internal_meta_pam50_nonnull_n"] = internal_ref["internal_meta_pam50_nonnull_n"].fillna(0).astype(int)
    internal_ref["internal_meta_drug_coverage_n"] = internal_ref["internal_meta_drug_coverage_n"].fillna(0).astype(int)

    # Clinical subset
    keep_cols = [
        "Patient ID",
        "Sample ID",
        "Pam50 + Claudin-low subtype",
        "ER Status",
        "PR Status",
        "HER2 Status",
        "3-Gene classifier subtype",
        "Overall Survival (Months)",
        "Overall Survival Status",
        "Tumor Stage",
    ]
    c = clin[keep_cols].copy()
    c.columns = [
        "metabric_patient_id",
        "metabric_sample_id",
        "metabric_pam50_subtype",
        "metabric_er_status",
        "metabric_pr_status",
        "metabric_her2_status",
        "metabric_3gene_subtype",
        "metabric_overall_survival_months",
        "metabric_overall_survival_status",
        "metabric_tumor_stage",
    ]
    c["metabric_sample_id"] = c["metabric_sample_id"].astype(str).str.strip()

    rows = []
    for _, r in c.iterrows():
        mb = r["metabric_sample_id"]
        scored = []
        for _, ir in internal_ref.iterrows():
            sid = ir["sample_id"]
            sim = _sim(mb, sid)
            # metadata pam50 is mostly null; keep weight low but included.
            subtype_match = (
                1.0
                if (
                    str(r["metabric_pam50_subtype"]).strip().lower()
                    and str(ir["internal_meta_pam50_example"]).strip().lower()
                    and str(r["metabric_pam50_subtype"]).strip().lower()
                    == str(ir["internal_meta_pam50_example"]).strip().lower()
                )
                else 0.0
            )
            score = 0.85 * sim + 0.15 * subtype_match
            scored.append((score, sim, subtype_match, ir))
        scored.sort(key=lambda t: t[0], reverse=True)
        for rank, (score, sim, subtype_match, ir) in enumerate(scored[: args.top_k], start=1):
            rows.append(
                {
                    "metabric_sample_id": mb,
                    "metabric_patient_id": r["metabric_patient_id"],
                    "internal_sample_id_candidate": ir["sample_id"],
                    "candidate_rank": rank,
                    "candidate_score": round(float(score), 6),
                    "id_string_similarity": round(float(sim), 6),
                    "subtype_match_flag": int(subtype_match),
                    "metabric_pam50_subtype": r["metabric_pam50_subtype"],
                    "internal_meta_pam50_subtype": ir["internal_meta_pam50_example"],
                    "metabric_er_status": r["metabric_er_status"],
                    "metabric_pr_status": r["metabric_pr_status"],
                    "metabric_her2_status": r["metabric_her2_status"],
                    "metabric_3gene_subtype": r["metabric_3gene_subtype"],
                    "metabric_overall_survival_months": r["metabric_overall_survival_months"],
                    "metabric_overall_survival_status": r["metabric_overall_survival_status"],
                    "metabric_tumor_stage": r["metabric_tumor_stage"],
                    "internal_n_feature_rows": ir["internal_n_feature_rows"],
                    "internal_n_label_rows": ir["internal_n_label_rows"],
                    "internal_label_mean": ir["internal_label_mean"],
                    "internal_label_std": ir["internal_label_std"],
                    "internal_meta_drug_coverage_n": ir["internal_meta_drug_coverage_n"],
                    "internal_meta_pam50_nonnull_n": ir["internal_meta_pam50_nonnull_n"],
                    "manual_selected_sample_id": "",
                    "manual_selected": False,
                    "manual_note": "",
                }
            )

    cand = pd.DataFrame(rows)
    cand.to_csv(out_dir / "manual_bridge_candidates.csv", index=False)

    summary = {
        "input_sources": {
            "ml_ready_metadata": args.ml_ready_meta_s3,
            "metabric_clinical_tsv": args.metabric_clinical_tsv,
            "internal_features": args.internal_features_s3,
            "internal_labels": args.internal_labels_s3,
        },
        "proposal_logic": [
            "Candidates generated per MB sample against all internal sample_id values.",
            "candidate_score = 0.85 * id_string_similarity + 0.15 * subtype_match_flag.",
            "id_string_similarity uses SequenceMatcher over MB-* ID and sample_id string.",
            "subtype_match_flag compares METABRIC Pam50 subtype vs metadata pam50 subtype (if available).",
            "Top-K candidates kept per MB sample (default K=5).",
            "manual_selected_* columns are for final human curation.",
        ],
        "limitations": [
            "Current metadata pam50_subtype is mostly null, subtype evidence is weak.",
            "No direct canonical_drug_id axis in METABRIC matrix; this is sample bridge aid only.",
            "Clinical receptor/survival fields are one-sided (METABRIC side) for reviewer context.",
        ],
        "output_file": str(out_dir / "manual_bridge_candidates.csv"),
    }
    (out_dir / "manual_bridge_candidates_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    _run(["aws", "s3", "cp", str(out_dir), args.upload_prefix, "--recursive"])
    print(f"[done] wrote: {out_dir / 'manual_bridge_candidates.csv'}")
    print(f"[done] wrote: {out_dir / 'manual_bridge_candidates_summary.json'}")
    print(f"[done] uploaded to: {args.upload_prefix}")


if __name__ == "__main__":
    main()

