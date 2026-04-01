from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Left-merge pathway__* columns from sample_pathway_features.parquet into an existing "
            "pair_features_newfe_v2.parquet (same sample_id join as build_pair_features_newfe_v2)."
        )
    )
    p.add_argument("--pair-parquet", required=True, help="Baseline pair matrix (e.g. final/pair_features_newfe_v2.parquet)")
    p.add_argument("--sample-pathway-parquet", required=True, help="Output of build_sample_pathway_features step")
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pair = pd.read_parquet(args.pair_parquet)
    pw = pd.read_parquet(args.sample_pathway_parquet)
    pathway_cols = [c for c in pw.columns if c.startswith("pathway__")]
    for c in pathway_cols:
        if c in pair.columns:
            pair = pair.drop(columns=[c])
    pair[args.sample_id_col] = pair[args.sample_id_col].astype(str).str.strip()
    pw[args.sample_id_col] = pw[args.sample_id_col].astype(str).str.strip()
    merged = pair.merge(pw[[args.sample_id_col] + pathway_cols], on=args.sample_id_col, how="left")
    if pathway_cols:
        merged[pathway_cols] = merged[pathway_cols].fillna(0.0)
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"Wrote {out} shape={merged.shape} pathway__ cols={len(pathway_cols)}")


if __name__ == "__main__":
    main()
