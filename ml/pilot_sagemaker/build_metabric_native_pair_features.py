#!/usr/bin/env python3
"""
METABRIC-native pair features: transpose expression to sample_expression layout,
run the same FE logic as build_pair_features_newfe_v2, align columns to train
pair_features_newfe_v2 for XGB / ResidualMLP / GCN inference (transform-only on model side).

Does not refit model scalers here; export raw features matching train schema.
Gene-level cohort z-score (optional) is for FE threshold semantics only, not sklearn.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "nextflow" / "scripts" / "build_pair_features_newfe_v2.py").is_file():
            return p
    raise SystemExit(
        "Could not locate repo root (missing nextflow/scripts/build_pair_features_newfe_v2.py)."
    )


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
NEXTFLOW_SCRIPTS = REPO_ROOT / "nextflow" / "scripts"
if str(NEXTFLOW_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(NEXTFLOW_SCRIPTS))

import build_pair_features_newfe_v2 as fe  # noqa: E402

LOGGER = logging.getLogger("build_metabric_native_pair_features")

KEY_COLS = ["sample_id", "canonical_drug_id"]


def is_lincs_feature_column(name: str) -> bool:
    """Train FE uses names like lincs_cosine; exclude any lincs_* / lincs__* feature column."""
    n = str(name).lower()
    return n.startswith("lincs_") or n.startswith("lincs__")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _resolve(p: Path) -> Path:
    cwd = Path.cwd()
    return p if p.is_absolute() else (cwd / p).resolve()


def _ensure_local(uri: str, tdp: Path, fname: str) -> Path:
    uri = uri.strip()
    if uri.startswith("s3://"):
        dst = tdp / fname
        _run(["aws", "s3", "cp", uri, str(dst)])
        return dst
    return Path(uri)


def _parquet_column_names(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(str(path)).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path, engine="pyarrow").columns)


def load_gene_symbols_for_matrix(matrix: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    n = int(matrix.shape[0])
    if args.use_matrix_index_as_gene_symbols:
        syms = matrix.index.astype(str).str.strip().str.upper().tolist()
        if len(syms) != n:
            raise SystemExit("Index length mismatch matrix rows")
        return syms
    if args.metabric_gene_order_csv:
        p = _resolve(Path(args.metabric_gene_order_csv))
        if not p.is_file():
            raise SystemExit(f"Missing gene order file: {p}")
        gdf = pd.read_csv(p)
        col = args.gene_order_column
        if col not in gdf.columns:
            raise SystemExit(f"gene CSV missing column {col!r}")
        syms = gdf[col].astype(str).str.strip().str.upper().tolist()
        if len(syms) != n:
            raise SystemExit(
                f"gene_order rows ({len(syms)}) must equal METABRIC matrix rows ({n})"
            )
        return syms
    raise SystemExit(
        "Provide --metabric-gene-order-csv (recommended) or --use-matrix-index-as-gene-symbols"
    )


def metabric_matrix_to_sample_expression(
    matrix: pd.DataFrame,
    gene_symbols: Sequence[str],
    reference_columns: list[str],
    sample_id_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    matrix: genes x samples (rows = genes, columns include MB-* sample ids).
    reference_columns: full list of columns from TCGA sample_expression (including sample_id).
    Output: one row per MB-* column, numeric columns aligned to reference gene columns.
    """
    mb_cols = [c for c in matrix.columns if str(c).startswith("MB-")]
    if not mb_cols:
        raise SystemExit("No MB-* columns found in METABRIC matrix (expected samples as columns).")

    sym_upper = [str(g).strip().upper() for g in gene_symbols]
    if len(sym_upper) != len(matrix):
        raise SystemExit("gene_symbols length must match matrix row count")

    sym_to_row: dict[str, int] = {}
    dup = 0
    for i, s in enumerate(sym_upper):
        if s not in sym_to_row:
            sym_to_row[s] = i
        else:
            dup += 1
    if dup:
        LOGGER.warning("Duplicate gene symbols in order: %d rows skipped (first wins)", dup)

    ref_gene_cols = [c for c in reference_columns if c != sample_id_col]
    values = matrix[mb_cols].to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    n_matched_tokens = 0
    for c in ref_gene_cols:
        tok = str(c).split("__")[-1].strip().upper()
        if tok in sym_to_row:
            n_matched_tokens += 1
    for j, sid in enumerate(mb_cols):
        rec: dict[str, Any] = {sample_id_col: str(sid).strip()}
        col_vec = values[:, j]
        for c in ref_gene_cols:
            tok = str(c).split("__")[-1].strip().upper()
            ri = sym_to_row.get(tok)
            rec[c] = float(col_vec[ri]) if ri is not None else np.nan
        rows.append(rec)
    stats = {
        "n_reference_gene_columns": len(ref_gene_cols),
        "n_reference_columns_with_matrix_gene_match": int(n_matched_tokens),
        "n_metabric_matrix_rows": len(sym_upper),
        "n_distinct_gene_symbols_in_order": len(sym_to_row),
    }
    return pd.DataFrame(rows), stats


def apply_cohort_zscore_per_gene(sample_expr_df: pd.DataFrame, sample_id_col: str) -> pd.DataFrame:
    out = sample_expr_df.copy()
    num_cols = [
        c
        for c in out.columns
        if c != sample_id_col and pd.api.types.is_numeric_dtype(out[c])
    ]
    for c in num_cols:
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        m = float(s.mean())
        std = float(s.std(ddof=0))
        if std == 0.0 or np.isnan(std):
            out[c] = 0.0
        else:
            out[c] = (s - m) / std
    return out


def lincs_overlap_stats(
    sample_expr_df: pd.DataFrame,
    lincs_drug_df: pd.DataFrame,
    sample_id_col: str,
    drug_id_col: str,
) -> dict[str, Any]:
    sample_numeric = [
        c
        for c in sample_expr_df.columns
        if c != sample_id_col and pd.api.types.is_numeric_dtype(sample_expr_df[c])
    ]
    drug_numeric = [
        c
        for c in lincs_drug_df.columns
        if c != drug_id_col and pd.api.types.is_numeric_dtype(lincs_drug_df[c])
    ]
    common = sorted(set(sample_numeric).intersection(drug_numeric))
    return {
        "n_sample_numeric_cols": len(sample_numeric),
        "n_lincs_drug_numeric_cols": len(drug_numeric),
        "n_lincs_intersection_cols": len(common),
        "lincs_intersection_empty": len(common) == 0,
        "lincs_intersection_head": common[:30],
    }


def pathway_coverage_stats(sample_pathway_df: pd.DataFrame, sample_id_col: str) -> dict[str, Any]:
    pc = [c for c in sample_pathway_df.columns if str(c).startswith("pathway__")]
    return {
        "n_pathway_score_columns": len(pc),
        "pathway_column_head": pc[:15],
    }


def align_to_train_schema(df: pd.DataFrame, train_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reorder and pad/truncate columns to match train parquet exactly; numeric gaps -> 0."""
    missing = [c for c in train_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in train_columns]
    out = df.reindex(columns=train_columns)
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out[num_cols] = out[num_cols].fillna(0.0)
    stats = {
        "missing_columns_filled_zero": missing,
        "extra_columns_dropped_from_output": [c for c in extra if c not in KEY_COLS],
        "n_missing_filled": len(missing),
        "n_extra_dropped": len([c for c in extra if c not in KEY_COLS]),
    }
    return out, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metabric-matrix-uri",
        default="s3://drug-discovery-joe-raw-data-team4/8/metabric/54_filtered.parquet",
    )
    p.add_argument(
        "--metabric-gene-order-csv",
        default="",
        help="CSV with gene symbol per matrix row order (same nrows as matrix).",
    )
    p.add_argument(
        "--gene-order-column",
        default="gene_symbol",
        help="Column name in gene order CSV",
    )
    p.add_argument(
        "--use-matrix-index-as-gene-symbols",
        action="store_true",
        help="Use matrix index strings as HGNC symbols (must match row count).",
    )
    p.add_argument(
        "--reference-sample-expression-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v2/sample_features.parquet",
        help=(
            "Train TCGA wide sample_expression parquet; used for column names only "
            "(pyarrow schema). Required for LINCS / gene alignment."
        ),
    )
    p.add_argument(
        "--train-schema-parquet-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
        help="Train pair_features_newfe_v2 for exact output column order.",
    )
    p.add_argument("--pairs-parquet-uri", required=True, help="MB-* sample_id x canonical_drug_id pairs")
    p.add_argument(
        "--drug-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v2/drug_features.parquet",
    )
    p.add_argument(
        "--lincs-drug-signature-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/lincs_drug_signature_proxy_20260331.parquet",
    )
    p.add_argument(
        "--drug-target-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/drug_target_map_20260331.parquet",
    )
    p.add_argument(
        "--smiles-col",
        default="canonical_smiles",
        help="Drug table SMILES column; if missing, falls back to smiles / canonical_smiles_raw.",
    )
    p.add_argument(
        "--pathway-gmt",
        default=str(REPO_ROOT / "nextflow/refs/h.all.v7.5.symbols.gmt"),
    )
    p.add_argument("--high-z-threshold", type=float, default=1.0)
    p.add_argument("--low-z-threshold", type=float, default=-1.0)
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-nbits", type=int, default=2048)
    p.add_argument("--reverse-topk-small", type=int, default=50)
    p.add_argument("--reverse-topk-large", type=int, default=100)
    p.add_argument(
        "--skip-cohort-zscore",
        action="store_true",
        help="If set, do not apply per-gene cohort z-score across METABRIC samples before FE.",
    )
    p.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/native_pair_features",
    )
    p.add_argument(
        "--omit-lincs-features",
        action="store_true",
        help=(
            "Skip LINCS pair block in FE and align to train schema with all lincs_* columns removed. "
            "Writes metabric_native_pair_features_no_lincs.parquet and *_no_lincs.json reports."
        ),
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = _resolve(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_id_col = "sample_id"
    drug_id_col = "canonical_drug_id"
    target_gene_col = "target_gene_symbol"

    with TemporaryDirectory() as td:
        tdp = Path(td)
        mpath = _ensure_local(args.metabric_matrix_uri, tdp, "metabric_matrix.parquet")
        pairs_path = _ensure_local(args.pairs_parquet_uri, tdp, "pairs.parquet")
        drug_path = _ensure_local(args.drug_uri, tdp, "drug.parquet")
        if args.omit_lincs_features:
            lincs_path = None
        else:
            lincs_path = _ensure_local(args.lincs_drug_signature_uri, tdp, "lincs.parquet")
        dt_path = _ensure_local(args.drug_target_uri, tdp, "drug_target.parquet")
        schema_path = _ensure_local(args.train_schema_parquet_uri, tdp, "train_schema.parquet")

        matrix = pd.read_parquet(mpath)
        gene_syms = load_gene_symbols_for_matrix(matrix, args)

        if not args.reference_sample_expression_uri.strip():
            raise SystemExit(
                "--reference-sample-expression-uri is required so gene columns match TCGA/LINCS naming."
            )
        ref_path = _ensure_local(
            args.reference_sample_expression_uri.strip(), tdp, "reference_sample_expr.parquet"
        )
        ref_cols = _parquet_column_names(ref_path)
        LOGGER.info("Reference sample_expression columns: %d", len(ref_cols))

        sample_expr_df, gene_map_stats = metabric_matrix_to_sample_expression(
            matrix=matrix,
            gene_symbols=gene_syms,
            reference_columns=ref_cols,
            sample_id_col=sample_id_col,
        )
        if not args.skip_cohort_zscore:
            LOGGER.info("Applying per-gene cohort z-score across METABRIC samples")
            sample_expr_df = apply_cohort_zscore_per_gene(sample_expr_df, sample_id_col)

        pairs_df = pd.read_parquet(pairs_path)
        drug_df = pd.read_parquet(drug_path)
        drug_target_df = pd.read_parquet(dt_path)
        train_columns = _parquet_column_names(schema_path)
        train_columns_align = (
            [c for c in train_columns if not is_lincs_feature_column(c)]
            if args.omit_lincs_features
            else train_columns
        )
        train_lincs_columns = [c for c in train_columns if is_lincs_feature_column(c)]

        smiles_col = args.smiles_col
        if smiles_col not in drug_df.columns:
            for alt in ("canonical_smiles", "smiles", "canonical_smiles_raw"):
                if alt in drug_df.columns:
                    smiles_col = alt
                    LOGGER.info("Using smiles column %r (requested %r missing)", alt, args.smiles_col)
                    break

        if args.omit_lincs_features:
            lincs_drug_df = pd.DataFrame(columns=[drug_id_col])
        else:
            lincs_drug_df = pd.read_parquet(lincs_path)

        for name, df, cols in [
            ("pairs", pairs_df, [sample_id_col, drug_id_col]),
            ("sample_expression", sample_expr_df, [sample_id_col]),
            ("drug", drug_df, [drug_id_col, smiles_col]),
        ]:
            miss = [c for c in cols if c not in df.columns]
            if miss:
                raise SystemExit(f"{name} missing columns: {miss}")
        if not args.omit_lincs_features:
            miss = [c for c in [drug_id_col] if c not in lincs_drug_df.columns]
            if miss:
                raise SystemExit(f"lincs_drug_signature missing columns: {miss}")

        if target_gene_col not in drug_target_df.columns:
            raise SystemExit(f"drug_target missing {target_gene_col!r}")

        LOGGER.info("Building pair features (shared newfe_v2 logic)")
        built = fe.build_pair_features_newfe_v2_from_frames(
            pairs_df,
            sample_expr_df,
            drug_df,
            lincs_drug_df,
            drug_target_df,
            pathway_gmt=args.pathway_gmt or "",
            sample_id_col=sample_id_col,
            drug_id_col=drug_id_col,
            smiles_col=smiles_col,
            target_gene_col=target_gene_col,
            high_z_threshold=args.high_z_threshold,
            low_z_threshold=args.low_z_threshold,
            morgan_radius=args.morgan_radius,
            morgan_nbits=args.morgan_nbits,
            reverse_topk_small=args.reverse_topk_small,
            reverse_topk_large=args.reverse_topk_large,
            include_pair_lincs=not args.omit_lincs_features,
        )
        raw_v2 = built["pair_features_newfe_v2"]

        if args.omit_lincs_features:
            lincs_stats = {
                "skipped_omit_lincs_features": True,
                "note": "LINCS pair block not computed; train schema lincs_* columns excluded from output.",
            }
        else:
            lincs_stats = lincs_overlap_stats(
                sample_expr_df, lincs_drug_df, sample_id_col, drug_id_col
            )
        pathway_stats = pathway_coverage_stats(built["sample_pathway_df"], sample_id_col)

        train_pathway_n = len([c for c in train_columns_align if str(c).startswith("pathway__")])
        feat_before_align = [c for c in raw_v2.columns if c not in KEY_COLS]
        missing_vs_train = [c for c in train_columns_align if c not in raw_v2.columns]
        extra_vs_train = [c for c in raw_v2.columns if c not in train_columns_align]

        aligned, align_stats = align_to_train_schema(raw_v2, train_columns_align)
        order_ok = list(aligned.columns) == train_columns_align

        lincs_in_output = [c for c in aligned.columns if is_lincs_feature_column(c)]

        if args.omit_lincs_features:
            out_parquet = out_dir / "metabric_native_pair_features_no_lincs.parquet"
            report_suffix = "_no_lincs"
        else:
            out_parquet = out_dir / "metabric_native_pair_features.parquet"
            report_suffix = ""
        aligned.to_parquet(out_parquet, index=False)

        manifest = {
            "purpose": (
                "METABRIC-native pair_features aligned to train schema without LINCS (pathway+chem+target only)"
                if args.omit_lincs_features
                else "METABRIC-native pair_features aligned to train pair_features_newfe_v2 schema"
            ),
            "inputs": {
                "metabric_matrix_uri": args.metabric_matrix_uri,
                "metabric_gene_order_csv": args.metabric_gene_order_csv or None,
                "use_matrix_index_as_gene_symbols": bool(args.use_matrix_index_as_gene_symbols),
                "reference_sample_expression_uri": args.reference_sample_expression_uri,
                "train_schema_parquet_uri": args.train_schema_parquet_uri,
                "pairs_parquet_uri": args.pairs_parquet_uri,
                "drug_uri": args.drug_uri,
                "lincs_drug_signature_uri": None if args.omit_lincs_features else args.lincs_drug_signature_uri,
                "drug_target_uri": args.drug_target_uri,
                "pathway_gmt": args.pathway_gmt,
                "omit_lincs_features": bool(args.omit_lincs_features),
                "train_schema_lincs_columns_dropped": train_lincs_columns if args.omit_lincs_features else [],
            },
            "frozen_fe_hyperparameters": {
                "high_z_threshold": args.high_z_threshold,
                "low_z_threshold": args.low_z_threshold,
                "morgan_radius": args.morgan_radius,
                "morgan_nbits": args.morgan_nbits,
                "reverse_topk_small": args.reverse_topk_small,
                "reverse_topk_large": args.reverse_topk_large,
                "skip_cohort_zscore": bool(args.skip_cohort_zscore),
                "include_pair_lincs": not bool(args.omit_lincs_features),
            },
            "outputs": {
                "metabric_native_pair_features_parquet": str(out_parquet),
                "metabric_native_feature_validation_json": str(
                    out_dir / f"metabric_native_feature_validation{report_suffix}.json"
                ),
            },
            "row_counts": {
                "metabric_samples_in_expression": int(sample_expr_df[sample_id_col].nunique()),
                "pairs_built": int(len(built["pairs_df"])),
                "pair_feature_rows": int(len(aligned)),
            },
            "gene_mapping": gene_map_stats,
            "pathway": {
                **pathway_stats,
                "n_pathway_columns_train_schema": train_pathway_n,
            },
            "lincs": lincs_stats,
            "schema_alignment": {
                "feature_column_order_matches_train_schema_used": order_ok,
                "n_train_columns_aligned": len(train_columns_align),
                "n_train_columns_full_schema_file": len(train_columns),
                "n_feature_columns_before_align": len(feat_before_align),
                "lincs_columns_in_output": lincs_in_output,
                **align_stats,
            },
            "inference_note": (
                "build_final_ensemble_ranking.py pads any checkpoint feat_cols missing from this parquet "
                "(e.g. LINCS) with 0.0 so GCN/MLP tensors match training width; parquet itself omits those columns."
                if args.omit_lincs_features
                else (
                    "Apply XGB/ResidualMLP/GCN bundles with train-frozen feat_cols and scalers only; "
                    "do not refit sklearn encoders on METABRIC."
                )
            ),
        }
        man_path = out_dir / f"metabric_native_feature_manifest{report_suffix}.json"
        manifest["outputs"]["metabric_native_feature_manifest_json"] = str(man_path)
        man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        validation = {
            "gene_mapping": gene_map_stats,
            "omit_lincs_features": bool(args.omit_lincs_features),
            "train_full_schema_n_columns": len(train_columns),
            "train_aligned_schema_n_columns": len(train_columns_align),
            "train_schema_lincs_column_names": train_lincs_columns,
            "metabric_vs_train_aligned_columns_match": order_ok,
            "metabric_column_order_list_equals_train_no_lincs": order_ok
            if args.omit_lincs_features
            else None,
            "lincs_columns_present_in_metabric_output": lincs_in_output,
            "lincs_columns_present_count": len(lincs_in_output),
            "missing_columns_vs_raw_fe": missing_vs_train,
            "extra_columns_vs_train_aligned_before_align": extra_vs_train,
            "alignment": align_stats,
            "pathway_coverage": pathway_stats,
            "pathway_columns_in_aligned_schema": train_pathway_n,
            "lincs_overlap": lincs_stats,
            "transform_only_compliance": (
                "Model scaler/encoding: not applied in this script; use checkpoint.joblib / "
                "checkpoint.pt when running inference. Cohort z-score here is FE-only (optional)."
            ),
        }
        val_path = out_dir / f"metabric_native_feature_validation{report_suffix}.json"
        val_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("Wrote %s", out_parquet)
    print(
        json.dumps(
            {
                "output_parquet": str(out_parquet),
                "order_ok_vs_aligned_schema": order_ok,
                "omit_lincs": bool(args.omit_lincs_features),
                "lincs_columns_in_output": lincs_in_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
