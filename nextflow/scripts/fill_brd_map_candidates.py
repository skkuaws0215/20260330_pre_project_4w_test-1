from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-fill BRD -> canonical_drug_id candidate mapping using LINCS metadata and bridge drug features."
    )
    p.add_argument(
        "--template-csv",
        required=True,
        help="Input template CSV (brd_map_20260331_template.csv).",
    )
    p.add_argument(
        "--bridge-drug-uri",
        required=True,
        help="Bridge drug_features parquet (must include canonical_drug_id, drug_name_norm, and smiles/canonical_smiles_raw).",
    )
    p.add_argument(
        "--lincs-metadata-uri",
        default="",
        help=(
            "Optional LINCS metadata table (csv/parquet). "
            "Should include brd/sig/pert_iname/smiles if available."
        ),
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path, e.g. brd_map_20260331_filled_candidates.csv",
    )
    p.add_argument(
        "--out-summary-json",
        default="",
        help="Optional output JSON path for summary.",
    )
    p.add_argument(
        "--out-topk-csv",
        default="",
        help="Optional output CSV path for pert_iname similarity top-k candidates.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of similarity candidates to emit per unmatched BRD (default: 5).",
    )
    return p.parse_args()


def _read_table(path_or_uri: str) -> pd.DataFrame:
    p = path_or_uri.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path_or_uri)
    if p.endswith(".tsv") or p.endswith(".txt") or p.endswith(".txt.gz") or p.endswith(".tsv.gz"):
        return pd.read_csv(path_or_uri, sep="\t", compression="infer")
    return pd.read_parquet(path_or_uri)


def _norm_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum())


def _norm_smiles(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    # v1 baseline: exact-string normalized by whitespace only
    return str(s).strip()


def _extract_brd_from_sig(sig_id: Any) -> str:
    if sig_id is None or (isinstance(sig_id, float) and pd.isna(sig_id)):
        return ""
    m = re.search(r"(BRD-[A-Z0-9]+)", str(sig_id).upper())
    return m.group(1) if m else ""


def _resolve_col(candidates: list[str], columns: list[str]) -> str:
    colset = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in colset:
            return colset[cand.lower()]
    return ""


def _sim_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _is_blank_like(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in {"", "nan", "none", "null"}


def main() -> None:
    args = parse_args()

    template = pd.read_csv(args.template_csv)
    if "brd_id" not in template.columns:
        raise ValueError("template-csv must contain brd_id column")

    bridge = pd.read_parquet(args.bridge_drug_uri)
    need_bridge = {"canonical_drug_id", "drug_name_norm"}
    miss = sorted(list(need_bridge - set(bridge.columns)))
    if miss:
        raise ValueError(f"bridge-drug-uri missing columns: {miss}")

    # Choose bridge smiles column if available.
    bridge_smiles_col = ""
    for c in ["smiles", "canonical_smiles_raw", "canonical_smiles"]:
        if c in bridge.columns:
            bridge_smiles_col = c
            break

    # Build bridge lookup maps.
    bridge_name = (
        bridge[["canonical_drug_id", "drug_name_norm"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    bridge_name["name_key"] = bridge_name["drug_name_norm"].map(_norm_text)
    # If duplicate normalized name maps to multiple IDs, keep first for v1 baseline.
    name_to_id = (
        bridge_name[bridge_name["name_key"] != ""]
        .drop_duplicates(subset=["name_key"])
        .set_index("name_key")["canonical_drug_id"]
        .astype(str)
        .to_dict()
    )
    bridge_name_for_topk = (
        bridge_name[bridge_name["name_key"] != ""]
        .drop_duplicates(subset=["canonical_drug_id", "drug_name_norm", "name_key"])
        .copy()
    )

    smiles_to_id: dict[str, str] = {}
    if bridge_smiles_col:
        bridge_smiles = (
            bridge[["canonical_drug_id", bridge_smiles_col]]
            .dropna()
            .drop_duplicates()
            .copy()
        )
        bridge_smiles["smiles_key"] = bridge_smiles[bridge_smiles_col].map(_norm_smiles)
        smiles_to_id = (
            bridge_smiles[bridge_smiles["smiles_key"] != ""]
            .drop_duplicates(subset=["smiles_key"])
            .set_index("smiles_key")["canonical_drug_id"]
            .astype(str)
            .to_dict()
        )

    # Load optional LINCS metadata.
    lincs_meta = pd.DataFrame()
    if args.lincs_metadata_uri:
        lincs_meta = _read_table(args.lincs_metadata_uri)

    # Try to locate columns in metadata.
    meta_brd_col = ""
    meta_sig_col = ""
    meta_name_col = ""
    meta_smiles_col = ""
    if not lincs_meta.empty:
        cols = list(lincs_meta.columns)
        meta_brd_col = _resolve_col(["brd_id", "pert_id", "pertid", "broad_id"], cols)
        meta_sig_col = _resolve_col(["sig_id", "signature_id"], cols)
        meta_name_col = _resolve_col(["pert_iname", "compound_name", "drug_name", "iname"], cols)
        meta_smiles_col = _resolve_col(["smiles", "canonical_smiles"], cols)

        if not meta_brd_col and meta_sig_col:
            lincs_meta = lincs_meta.copy()
            lincs_meta["__brd_id_from_sig"] = lincs_meta[meta_sig_col].map(_extract_brd_from_sig)
            meta_brd_col = "__brd_id_from_sig"

    # Metadata keyed by brd_id (first row per brd).
    meta_by_brd: dict[str, dict[str, Any]] = {}
    if not lincs_meta.empty and meta_brd_col:
        md = lincs_meta.copy()
        md["__brd_key"] = md[meta_brd_col].astype(str).str.upper().str.strip()
        md = md[md["__brd_key"] != ""].drop_duplicates(subset=["__brd_key"])
        for _, r in md.iterrows():
            b = r["__brd_key"]
            meta_by_brd[b] = {
                "pert_iname": (str(r[meta_name_col]).strip() if meta_name_col and pd.notna(r[meta_name_col]) else ""),
                "smiles": (str(r[meta_smiles_col]).strip() if meta_smiles_col and pd.notna(r[meta_smiles_col]) else ""),
                "sig_id": (str(r[meta_sig_col]).strip() if meta_sig_col and pd.notna(r[meta_sig_col]) else ""),
            }

    # Fill candidates
    out = template.copy()
    if "canonical_drug_id" not in out.columns:
        out["canonical_drug_id"] = ""
    for c in ["sig_id", "pert_iname", "source", "mapping_confidence", "note"]:
        if c not in out.columns:
            out[c] = ""
    out["matched_by"] = "none"

    name_match = 0
    smiles_match = 0
    auto_success = 0

    for i, row in out.iterrows():
        brd = str(row["brd_id"]).upper().strip()
        if not brd:
            continue

        # Keep user-pre-filled mapping untouched.
        existing_raw = row.get("canonical_drug_id", "")
        existing = str(existing_raw).strip().lower()
        if existing not in {"", "nan", "none"}:
            out.at[i, "matched_by"] = "manual"
            out.at[i, "mapping_confidence"] = 1.0
            if not str(row.get("note", "")).strip():
                out.at[i, "note"] = "kept pre-filled canonical_drug_id"
            continue

        meta = meta_by_brd.get(brd, {})
        pert_iname = meta.get("pert_iname", "")
        smiles = meta.get("smiles", "")
        sig_id = meta.get("sig_id", "")
        if pert_iname and _is_blank_like(row.get("pert_iname", "")):
            out.at[i, "pert_iname"] = pert_iname
        if sig_id and _is_blank_like(row.get("sig_id", "")):
            out.at[i, "sig_id"] = sig_id

        # Rule 1) pert_iname -> drug_name_norm normalized exact
        cid = ""
        if pert_iname:
            cid = name_to_id.get(_norm_text(pert_iname), "")
            if cid:
                out.at[i, "canonical_drug_id"] = cid
                out.at[i, "matched_by"] = "name"
                out.at[i, "mapping_confidence"] = 1.0
                out.at[i, "source"] = "auto:name_norm_exact"
                out.at[i, "note"] = "matched by pert_iname -> drug_name_norm"
                name_match += 1
                auto_success += 1
                continue

        # Rule 2) smiles exact
        if smiles:
            cid = smiles_to_id.get(_norm_smiles(smiles), "")
            if cid:
                out.at[i, "canonical_drug_id"] = cid
                out.at[i, "matched_by"] = "smiles"
                out.at[i, "mapping_confidence"] = 0.5
                out.at[i, "source"] = "auto:smiles_exact"
                out.at[i, "note"] = "matched by smiles exact"
                smiles_match += 1
                auto_success += 1
                continue

        # Rule 3) keep blank
        out.at[i, "matched_by"] = "none"
        out.at[i, "mapping_confidence"] = 0.0
        if not str(out.at[i, "note"]).strip():
            out.at[i, "note"] = "no candidate from metadata(name/smiles)"

    total = int(out.shape[0])
    cid_norm = out["canonical_drug_id"].astype(str).str.strip().str.lower()
    unmatched = int(cid_norm.isin(["", "nan", "none"]).sum())
    summary = {
        "total_brd": total,
        "auto_mapping_success": int(auto_success),
        "name_match_count": int(name_match),
        "smiles_match_count": int(smiles_match),
        "unmatched_count": int(unmatched),
        "metadata_rows": int(len(meta_by_brd)),
        "metadata_cols_detected": {
            "brd_col": meta_brd_col,
            "sig_col": meta_sig_col,
            "pert_iname_col": meta_name_col,
            "smiles_col": meta_smiles_col,
        },
    }

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if args.out_summary_json:
        Path(args.out_summary_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(out_path))

    # Optional: emit similarity-based top-k candidates for unmatched BRD rows.
    if args.out_topk_csv:
        k = max(1, int(args.topk))
        topk_rows: list[dict[str, Any]] = []
        unmatched_rows = out[
            out["canonical_drug_id"].astype(str).str.strip().str.lower().isin(["", "nan", "none"])
        ].copy()
        for _, r in unmatched_rows.iterrows():
            brd_id = str(r.get("brd_id", "")).strip()
            pert_iname = str(r.get("pert_iname", "")).strip()
            if not pert_iname:
                m = meta_by_brd.get(brd_id.upper(), {})
                pert_iname = str(m.get("pert_iname", "")).strip()
            pert_key = _norm_text(pert_iname)
            if not pert_key:
                topk_rows.append(
                    {
                        "brd_id": brd_id,
                        "pert_iname": pert_iname,
                        "candidate_rank": 1,
                        "candidate_drug_name_norm": "",
                        "candidate_canonical_drug_id": "",
                        "similarity_score": 0.0,
                        "match_rule": "name_similarity_difflib",
                        "note": "pert_iname missing/empty after normalization",
                    }
                )
                continue

            scored: list[tuple[float, str, str]] = []
            for _, b in bridge_name_for_topk.iterrows():
                score = _sim_ratio(pert_key, str(b["name_key"]))
                scored.append((score, str(b["drug_name_norm"]), str(b["canonical_drug_id"])))
            scored.sort(key=lambda x: (-x[0], x[1]))
            for rank, (score, drug_name, cid) in enumerate(scored[:k], start=1):
                topk_rows.append(
                    {
                        "brd_id": brd_id,
                        "pert_iname": pert_iname,
                        "candidate_rank": rank,
                        "candidate_drug_name_norm": drug_name,
                        "candidate_canonical_drug_id": cid,
                        "similarity_score": round(float(score), 6),
                        "match_rule": "name_similarity_difflib",
                        "note": "normalized(lower/trim/remove hyphen underscore spaces and non-alnum)",
                    }
                )

        topk_df = pd.DataFrame(
            topk_rows,
            columns=[
                "brd_id",
                "pert_iname",
                "candidate_rank",
                "candidate_drug_name_norm",
                "candidate_canonical_drug_id",
                "similarity_score",
                "match_rule",
                "note",
            ],
        )
        topk_path = Path(args.out_topk_csv)
        topk_path.parent.mkdir(parents=True, exist_ok=True)
        topk_df.to_csv(topk_path, index=False)
        print(str(topk_path))


if __name__ == "__main__":
    main()
