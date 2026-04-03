#!/usr/bin/env python3
"""
Build fda_only_ranking.csv + fda_top30_shortlist.csv when full run_fda_only_metabric_ranking
is blocked (e.g. S3): cross-join METABRIC samples from native ranking with FDA universe CIDs,
left-join observed pair scores, impute missing preds with cohort means, recompute ensemble + rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["sample_id", "canonical_drug_id"]


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    prep = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--native-ranking-csv",
        default=str(prep / "native_pair_features_run" / "metabric_native_ensemble_ranking_no_lincs.csv"),
    )
    ap.add_argument(
        "--universe-drug-parquet",
        default=str(prep / "fda_only_universe" / "fda_approved_drug_table.parquet"),
    )
    ap.add_argument("--output-dir", default=str(prep / "fda_only_universe"))
    ap.add_argument("--top-k-shortlist", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    native = pd.read_csv(args.native_ranking_csv)
    drugs = pd.read_parquet(args.universe_drug_parquet)
    if "canonical_drug_id" not in drugs.columns:
        raise SystemExit("Universe drug table missing canonical_drug_id")

    samples = native["sample_id"].drop_duplicates().sort_values()
    d_ids = drugs["canonical_drug_id"].astype(int).drop_duplicates().sort_values()
    pairs = samples.to_frame(name="sample_id").assign(_k=1).merge(
        d_ids.to_frame(name="canonical_drug_id").assign(_k=1),
        on="_k",
    ).drop(columns="_k")

    obs = native[KEY + ["pred_xgb", "pred_residualmlp", "pred_gcn", "ensemble_score"]].copy()
    obs["canonical_drug_id"] = obs["canonical_drug_id"].astype(int)
    pairs["canonical_drug_id"] = pairs["canonical_drug_id"].astype(int)

    m = pairs.merge(obs, on=KEY, how="left")
    for c in ("pred_xgb", "pred_residualmlp", "pred_gcn"):
        mean_v = float(native[c].mean())
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(mean_v)
    m["ensemble_score"] = 0.5 * m["pred_xgb"] + 0.3 * m["pred_residualmlp"] + 0.2 * m["pred_gcn"]
    m = m.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1, dtype=np.int64)
    m["is_top100"] = m["rank"] <= 100

    rank_path = out_dir / "fda_only_ranking.csv"
    m.to_csv(rank_path, index=False)

    filt = m.sort_values("rank", ascending=True, kind="mergesort")
    per_drug = (
        filt.groupby("canonical_drug_id", sort=False)
        .head(1)
        .sort_values("rank", ascending=True, kind="mergesort")
        .head(int(args.top_k_shortlist))
        .reset_index(drop=True)
    )
    per_drug.insert(0, "fda_drug_rank", range(1, len(per_drug) + 1))
    per_drug = per_drug.rename(columns={"rank": "best_pair_rank"})
    short_path = out_dir / "fda_top30_shortlist.csv"
    per_drug.to_csv(short_path, index=False)

    meta = {
        "native_ranking_csv": str(Path(args.native_ranking_csv).resolve()),
        "universe_drug_parquet": str(Path(args.universe_drug_parquet).resolve()),
        "n_samples": int(len(samples)),
        "n_universe_drugs": int(len(d_ids)),
        "n_pairs": int(len(m)),
        "shortlist_n": int(len(per_drug)),
        "note": "Missing pairs imputed with mean pred_* from native chemical-space ranking; for production use run_fda_only_metabric_ranking.py.",
        "outputs": {"fda_only_ranking": str(rank_path), "fda_top30_shortlist": str(short_path)},
    }
    (out_dir / "fda_only_ranking_backfill_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
