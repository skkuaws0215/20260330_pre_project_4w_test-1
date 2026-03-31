from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


BRD_RE = re.compile(r"(BRD-[A-Z0-9]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize LINCS mapping BRD/sig_id -> canonical_drug_id.")
    p.add_argument("--lincs-uri", required=True, help="LINCS parquet with sig_id and numeric signature columns")
    p.add_argument("--brd-map-uri", required=True, help="Parquet with columns: brd_id, canonical_drug_id")
    p.add_argument(
        "--feature-cols-from-uri",
        default="",
        help="Optional parquet with canonical_drug_id + selected numeric cols. If set, only overlapping cols are read from LINCS.",
    )
    p.add_argument("--out-parquet", required=True, help="Mapped lincs signature output parquet keyed by canonical_drug_id")
    p.add_argument("--out-report", required=True, help="Mapping report json")
    return p.parse_args()


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _extract_brd(sig_id: str) -> str:
    m = BRD_RE.search(str(sig_id))
    return m.group(1) if m else ""


def main() -> None:
    args = parse_args()
    out_parquet = Path(args.out_parquet)
    out_report = Path(args.out_report)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    cols_to_read = None
    if args.feature_cols_from_uri:
        seed = pd.read_parquet(args.feature_cols_from_uri)
        seed_cols = [c for c in seed.columns if c != "canonical_drug_id"]
        lincs_schema_cols = set(pq.read_schema(args.lincs_uri).names)
        overlap = [c for c in seed_cols if c in lincs_schema_cols]
        if overlap:
            cols_to_read = ["sig_id"] + overlap
    lincs = pd.read_parquet(args.lincs_uri, columns=cols_to_read)
    if "sig_id" not in lincs.columns:
        raise ValueError("LINCS parquet must contain sig_id")
    lincs = lincs.copy()
    lincs["brd_id"] = lincs["sig_id"].map(_extract_brd)

    brd_map = _read_table(args.brd_map_uri)
    need = {"brd_id", "canonical_drug_id"}
    if not need.issubset(set(brd_map.columns)):
        raise ValueError(f"brd_map missing columns: {sorted(list(need - set(brd_map.columns)))}")
    optional_cols = ["sig_id", "pert_iname", "source", "mapping_confidence", "note"]
    keep_cols = ["brd_id", "canonical_drug_id"] + [c for c in optional_cols if c in brd_map.columns]
    brd_map = brd_map[keep_cols].copy()
    brd_map["brd_id"] = brd_map["brd_id"].astype(str).str.strip().str.upper()
    brd_map["canonical_drug_id"] = brd_map["canonical_drug_id"].astype(str).str.strip()
    brd_map = brd_map[(brd_map["brd_id"] != "") & (brd_map["canonical_drug_id"] != "")]
    brd_map = brd_map.drop_duplicates(subset=["brd_id", "canonical_drug_id"])

    merged = lincs.merge(brd_map[["brd_id", "canonical_drug_id"]], on="brd_id", how="left")
    matched = merged.dropna(subset=["canonical_drug_id"]).copy()

    num_cols = [c for c in lincs.columns if c not in {"sig_id", "brd_id"} and pd.api.types.is_numeric_dtype(lincs[c])]
    out = (
        matched.groupby("canonical_drug_id", as_index=False)[num_cols]
        .mean()
        .reset_index(drop=True)
    )
    out.to_parquet(out_parquet, index=False)

    report = {
        "lincs_rows": int(lincs.shape[0]),
        "lincs_brd_extracted_rows": int((lincs["brd_id"] != "").sum()),
        "brd_map_rows": int(brd_map.shape[0]),
        "matched_rows": int(matched.shape[0]),
        "matched_ratio": float(matched.shape[0] / max(lincs.shape[0], 1)),
        "mapped_drug_count": int(out["canonical_drug_id"].nunique()) if not out.empty else 0,
        "note": "If matched_ratio is low, enrich brd_map first (BRD -> canonical_drug_id).",
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
