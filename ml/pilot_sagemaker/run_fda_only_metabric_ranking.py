#!/usr/bin/env python3
"""
End-to-end: FDA-approved drug universe → METABRIC-native pair features (same FE as chemical pool)
→ frozen XGB + ResidualMLP + GCN ensemble ranking.

Writes under --run-dir (default …/fda_only_universe):
  - fda_only_pairs.parquet
  - metabric_native_pair_features_no_lincs.parquet (from build_metabric_native_pair_features.py)
  - fda_only_ranking.csv
  - fda_top30_shortlist.csv
  - fda_only_run_meta.json

Prerequisites: METABRIC matrix + gene order CSV reachable; train schema parquet; SageMaker artifacts;
optional train drug_target / LINCS parquets for targets and (if not --omit-lincs) LINCS block.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_shortlist_topk(ranking_csv: Path, top_k: int, out_csv: Path) -> None:
    df = pd.read_csv(ranking_csv)
    need = ["sample_id", "canonical_drug_id", "rank", "ensemble_score"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"Ranking CSV missing {c!r}")
    filt = df.sort_values("rank", ascending=True, kind="mergesort")
    per_drug = (
        filt.groupby("canonical_drug_id", sort=False)
        .head(1)
        .sort_values("rank", ascending=True, kind="mergesort")
        .head(int(top_k))
        .reset_index(drop=True)
    )
    per_drug.insert(0, "fda_drug_rank", range(1, len(per_drug) + 1))
    per_drug = per_drug.rename(columns={"rank": "best_pair_rank"})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_drug.to_csv(out_csv, index=False)


def main() -> None:
    prep = (
        REPO_ROOT
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
    )
    default_run = prep / "fda_only_universe"
    default_ranking = (
        prep
        / "native_pair_features_run"
        / "metabric_native_ensemble_ranking_no_lincs.csv"
    )
    default_gene_csv = prep / "native_inputs" / "metabric_matrix_gene_order.csv"
    sm = REPO_ROOT / "results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=str, default=str(default_run), help="Universe assets + outputs.")
    ap.add_argument("--skip-build-universe", action="store_true", help="Reuse existing fda_approved_drug_table.parquet.")
    ap.add_argument(
        "--max-universe-drugs",
        type=int,
        default=0,
        help="Forward to build_fda_approved_universe_assets.py (0 = no cap).",
    )
    ap.add_argument(
        "--universe-sources",
        default="drugsfda",
        help="Forwarded to build_fda_approved_universe_assets.py --sources (e.g. drugsfda+drugbank with DrugBank XML/JSON).",
    )
    ap.add_argument("--drugbank-xml", default="", help="Forwarded when building universe.")
    ap.add_argument("--drugbank-approved-ids-json", default="", help="Forwarded when building universe.")
    ap.add_argument(
        "--sample-source-ranking-csv",
        type=str,
        default=str(default_ranking),
        help="Used to list MB-* sample_id values for cross-join (unique).",
    )
    ap.add_argument(
        "--metabric-matrix-uri",
        default="s3://drug-discovery-joe-raw-data-team4/8/metabric/54_filtered.parquet",
    )
    ap.add_argument(
        "--metabric-gene-order-csv",
        type=str,
        default=str(default_gene_csv),
    )
    ap.add_argument(
        "--reference-sample-expression-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v2/sample_features.parquet",
    )
    ap.add_argument(
        "--train-schema-parquet-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    ap.add_argument(
        "--train-drug-target-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/drug_target_map_20260331.parquet",
    )
    ap.add_argument(
        "--train-lincs-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/lincs_drug_signature_proxy_20260331.parquet",
    )
    ap.add_argument(
        "--pathway-gmt",
        default=str(REPO_ROOT / "nextflow/refs/h.all.v7.5.symbols.gmt"),
    )
    ap.add_argument("--omit-lincs-features", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--build-lincs-stub-in-universe", action="store_true")
    ap.add_argument("--top-k-shortlist", type=int, default=30)
    ap.add_argument("--xgb-artifact", default=str(sm / "artifacts/xgb/artifact.joblib"))
    ap.add_argument("--mlp-checkpoint", default=str(sm / "artifacts/residualmlp/checkpoint.pt"))
    ap.add_argument("--gcn-checkpoint", default=str(sm / "artifacts/gcn/gcn_checkpoint.pt"))
    ap.add_argument(
        "--disease-genes-path",
        default=str(REPO_ROOT / "data/graph_baseline/disease_genes_common_v1.txt"),
    )
    ap.add_argument("--ppi-edges-uri", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if not args.skip_build_universe:
        ucmd = [
            py,
            str(SCRIPT_DIR / "build_fda_approved_universe_assets.py"),
            "--output-dir",
            str(run_dir),
            "--sources",
            args.universe_sources,
            "--train-drug-target-uri",
            args.train_drug_target_uri,
            "--train-lincs-uri",
            args.train_lincs_uri,
        ]
        if args.max_universe_drugs and args.max_universe_drugs > 0:
            ucmd.extend(["--max-universe-drugs", str(args.max_universe_drugs)])
        if args.drugbank_xml:
            ucmd.extend(["--drugbank-xml", args.drugbank_xml])
        if args.drugbank_approved_ids_json:
            ucmd.extend(["--drugbank-approved-ids-json", args.drugbank_approved_ids_json])
        if args.build_lincs_stub_in_universe:
            ucmd.append("--build-lincs-stub")
        _run(ucmd)

    drug_tbl = run_dir / "fda_approved_drug_table.parquet"
    if not drug_tbl.is_file():
        raise SystemExit(f"Missing {drug_tbl}; run without --skip-build-universe.")

    samples = pd.read_csv(Path(args.sample_source_ranking_csv))["sample_id"].drop_duplicates()
    samples = samples.sort_values().reset_index(drop=True)
    drugs = pd.read_parquet(drug_tbl)["canonical_drug_id"].drop_duplicates()
    pairs = samples.to_frame(name="sample_id").assign(_k=1).merge(
        drugs.to_frame(name="canonical_drug_id").assign(_k=1),
        on="_k",
    ).drop(columns="_k")
    pairs_path = run_dir / "fda_only_pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    feat_py = SCRIPT_DIR / "build_metabric_native_pair_features.py"
    fcmd = [
        py,
        str(feat_py),
        "--metabric-matrix-uri",
        args.metabric_matrix_uri,
        "--metabric-gene-order-csv",
        args.metabric_gene_order_csv,
        "--reference-sample-expression-uri",
        args.reference_sample_expression_uri,
        "--train-schema-parquet-uri",
        args.train_schema_parquet_uri,
        "--pairs-parquet-uri",
        str(pairs_path),
        "--drug-uri",
        str(drug_tbl),
        "--drug-target-uri",
        str(run_dir / "fda_approved_drug_targets.parquet"),
        "--pathway-gmt",
        args.pathway_gmt,
        "--output-dir",
        str(run_dir),
    ]
    if args.omit_lincs_features:
        fcmd.append("--omit-lincs-features")
    else:
        fcmd.extend(
            [
                "--lincs-drug-signature-uri",
                str(run_dir / "fda_approved_lincs_extended.parquet")
                if (run_dir / "fda_approved_lincs_extended.parquet").is_file()
                else args.train_lincs_uri,
            ]
        )
    _run(fcmd)

    feat_parquet = run_dir / (
        "metabric_native_pair_features_no_lincs.parquet"
        if args.omit_lincs_features
        else "metabric_native_pair_features.parquet"
    )
    if not feat_parquet.is_file():
        raise SystemExit(f"Expected features at {feat_parquet}")

    ranking_csv = run_dir / "fda_only_ranking.csv"
    ens_py = SCRIPT_DIR / "build_final_ensemble_ranking.py"
    _run(
        [
            py,
            str(ens_py),
            "--features-uri",
            str(feat_parquet),
            "--drug-target-uri",
            args.train_drug_target_uri,
            "--disease-genes-path",
            args.disease_genes_path,
            "--ppi-edges-uri",
            args.ppi_edges_uri,
            "--xgb-artifact",
            args.xgb_artifact,
            "--mlp-checkpoint",
            args.mlp_checkpoint,
            "--gcn-checkpoint",
            args.gcn_checkpoint,
            "--out-csv",
            str(ranking_csv),
        ]
    )

    short_csv = run_dir / "fda_top30_shortlist.csv"
    build_shortlist_topk(ranking_csv, args.top_k_shortlist, short_csv)

    meta = {
        "run_dir": str(run_dir),
        "n_samples": int(len(samples)),
        "n_drugs_universe": int(drugs.nunique()),
        "n_pairs": int(len(pairs)),
        "omit_lincs_features": bool(args.omit_lincs_features),
        "outputs": {
            "pairs": str(pairs_path),
            "pair_features": str(feat_parquet),
            "fda_only_ranking": str(ranking_csv),
            "fda_top30_shortlist": str(short_csv),
        },
    }
    (run_dir / "fda_only_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
