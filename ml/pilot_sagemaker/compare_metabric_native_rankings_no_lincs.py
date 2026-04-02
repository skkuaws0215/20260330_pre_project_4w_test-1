#!/usr/bin/env python3
"""
Compare METABRIC-native ensemble rankings: full FE (with LINCS in parquet) vs no-LINCS FE.

Writes:
  - metabric_native_ranking_comparison_no_lincs.csv (pair-level + per-sample overlap columns)
  - metabric_native_ranking_comparison_summary.json (aggregates + policy note)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["sample_id", "canonical_drug_id"]


def _topk_drug_set(df: pd.DataFrame, sample_id: str, k: int) -> set[str]:
    sub = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    sub = sub.sort_values("rank", ascending=True).head(k)
    return set(sub["canonical_drug_id"].astype(str))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return float(len(a & b) / u) if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    run_dir = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/native_pair_features_run"
    )
    ap.add_argument(
        "--ranking-with-lincs-csv",
        default=str(run_dir / "metabric_native_ensemble_ranking.csv"),
        help="Ensemble ranking from native FE that included LINCS columns in features parquet.",
    )
    ap.add_argument(
        "--ranking-no-lincs-csv",
        default=str(run_dir / "metabric_native_ensemble_ranking_no_lincs.csv"),
        help="Ensemble ranking from native FE with --omit-lincs-features (canonical external run).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(run_dir),
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.ranking_with_lincs_csv)
    nol = pd.read_csv(args.ranking_no_lincs_csv)

    for c in KEY + ["rank", "ensemble_score"]:
        if c not in base.columns or c not in nol.columns:
            raise SystemExit(f"Both CSVs must contain column {c!r}")

    merged = base.merge(nol, on=KEY, how="inner", suffixes=("_with_lincs", "_no_lincs"))
    if len(merged) != len(base) or len(merged) != len(nol):
        raise SystemExit(
            f"Row count mismatch after merge: with_lincs={len(base)} no_lincs={len(nol)} merged={len(merged)}"
        )

    merged["rank_delta"] = merged["rank_no_lincs"] - merged["rank_with_lincs"]
    merged["ensemble_score_delta"] = merged["ensemble_score_no_lincs"] - merged["ensemble_score_with_lincs"]
    merged["rank_delta_abs"] = merged["rank_delta"].abs()

    if "pred_xgb_with_lincs" in merged.columns:
        merged["pred_xgb_delta"] = merged["pred_xgb_no_lincs"] - merged["pred_xgb_with_lincs"]
        merged["pred_residualmlp_delta"] = (
            merged["pred_residualmlp_no_lincs"] - merged["pred_residualmlp_with_lincs"]
        )
        merged["pred_gcn_delta"] = merged["pred_gcn_no_lincs"] - merged["pred_gcn_with_lincs"]

    samples = sorted(merged["sample_id"].astype(str).unique())
    per_sample: list[dict] = []
    top10_ov: list[int] = []
    top30_ov: list[int] = []
    top10_j: list[float] = []
    top30_j: list[float] = []

    overlap_map10: dict[str, int] = {}
    overlap_map30: dict[str, int] = {}
    jacc_map10: dict[str, float] = {}
    jacc_map30: dict[str, float] = {}

    for sid in samples:
        a10 = _topk_drug_set(base, sid, 10)
        b10 = _topk_drug_set(nol, sid, 10)
        a30 = _topk_drug_set(base, sid, 30)
        b30 = _topk_drug_set(nol, sid, 30)
        o10 = len(a10 & b10)
        o30 = len(a30 & b30)
        j10 = _jaccard(a10, b10)
        j30 = _jaccard(a30, b30)
        top10_ov.append(o10)
        top30_ov.append(o30)
        top10_j.append(j10)
        top30_j.append(j30)
        overlap_map10[sid] = o10
        overlap_map30[sid] = o30
        jacc_map10[sid] = j10
        jacc_map30[sid] = j30
        per_sample.append(
            {
                "sample_id": sid,
                "top10_overlap_count": o10,
                "top30_overlap_count": o30,
                "top10_jaccard": j10,
                "top30_jaccard": j30,
                "n_drugs_ranked": int((merged["sample_id"].astype(str) == sid).sum()),
            }
        )

    merged["sample_top10_overlap"] = merged["sample_id"].astype(str).map(overlap_map10)
    merged["sample_top30_overlap"] = merged["sample_id"].astype(str).map(overlap_map30)
    merged["sample_top10_jaccard"] = merged["sample_id"].astype(str).map(jacc_map10)
    merged["sample_top30_jaccard"] = merged["sample_id"].astype(str).map(jacc_map30)

    out_csv = out_dir / "metabric_native_ranking_comparison_no_lincs.csv"
    merged.to_csv(out_csv, index=False)

    summary = {
        "policy_final_external_inference_ranking": (
            "Use the no-LINCS METABRIC-native pipeline as the canonical external inference ranking: "
            "metabric_native_ensemble_ranking_no_lincs.csv (from metabric_native_pair_features_no_lincs.parquet). "
            "Training excluded LINCS from model feature sets; the with-LINCS native parquet was misaligned. "
            "Ensemble code may zero-pad missing LINCS slots for tensor width, but the authoritative ranked table "
            "for reporting is the no-LINCS run."
        ),
        "inputs": {
            "ranking_with_lincs_csv": str(Path(args.ranking_with_lincs_csv).resolve()),
            "ranking_no_lincs_csv": str(Path(args.ranking_no_lincs_csv).resolve()),
        },
        "outputs": {
            "comparison_csv": str(out_csv.resolve()),
            "comparison_summary_json": str((out_dir / "metabric_native_ranking_comparison_summary.json").resolve()),
        },
        "counts": {
            "n_pairs": int(len(merged)),
            "n_samples": int(len(samples)),
        },
        "topk_overlap_per_sample": {
            "top10_overlap_count_mean": float(np.mean(top10_ov)),
            "top10_overlap_count_std": float(np.std(top10_ov)),
            "top10_overlap_count_min": int(np.min(top10_ov)),
            "top10_overlap_count_max": int(np.max(top10_ov)),
            "top30_overlap_count_mean": float(np.mean(top30_ov)),
            "top30_overlap_count_std": float(np.std(top30_ov)),
            "top30_overlap_count_min": int(np.min(top30_ov)),
            "top30_overlap_count_max": int(np.max(top30_ov)),
            "top10_jaccard_mean": float(np.mean(top10_j)),
            "top30_jaccard_mean": float(np.mean(top30_j)),
        },
        "rank_shift": {
            "mean_abs_rank_delta": float(merged["rank_delta_abs"].mean()),
            "median_abs_rank_delta": float(merged["rank_delta_abs"].median()),
            "max_abs_rank_delta": int(merged["rank_delta_abs"].max()),
            "share_exact_same_rank": float((merged["rank_delta"] == 0).mean()),
        },
        "ensemble_score_change": {
            "mean_delta": float(merged["ensemble_score_delta"].mean()),
            "mean_abs_delta": float(merged["ensemble_score_delta"].abs().mean()),
            "max_abs_delta": float(merged["ensemble_score_delta"].abs().max()),
            "share_exact_same_score": float(
                np.isclose(merged["ensemble_score_delta"].to_numpy(), 0.0, rtol=0.0, atol=1e-9).mean()
            ),
        },
        "per_drug_rank_shift": (
            merged.groupby("canonical_drug_id", as_index=False)
            .agg(
                n_samples=("sample_id", "count"),
                mean_rank_delta=("rank_delta", "mean"),
                mean_abs_rank_delta=("rank_delta_abs", "mean"),
                max_abs_rank_delta=("rank_delta_abs", "max"),
                mean_ensemble_score_delta=("ensemble_score_delta", "mean"),
                mean_abs_ensemble_score_delta=("ensemble_score_delta", lambda s: float(np.abs(s).mean())),
            )
            .sort_values("mean_abs_rank_delta", ascending=False)
            .head(50)
            .to_dict(orient="records")
        ),
        "per_sample": per_sample,
    }

    out_json = out_dir / "metabric_native_ranking_comparison_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_keys = {"per_sample", "per_drug_rank_shift"}
    print(json.dumps({k: summary[k] for k in summary if k not in _print_keys}, indent=2, ensure_ascii=False))
    print(f"[done] {out_csv}")
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
