#!/usr/bin/env python3
"""
Merge SwissADME (web batch) export with swissadme_input_top30_with_meta.csv for the N≈11
compounds that were submitted, then emit a ranked/summary layer alongside ensemble scores.

SwissADME does not ship a stable public API in this repo: the user downloads a table from
the SwissADME batch page and passes it via --swissadme-export.

Matching: RDKit MolToSmiles when available (SwissADME uses aromatic lowercase; inputs may use
Kekule uppercase — plain string match often fails). Else stripped string equality. SMILES column
detection prefers columns whose name contains "smiles" (not SwissADME's "Molecule" label column).

Scoring (optional heuristics, best-effort on column name substrings):
  - swissadme_penalty_gi_low: +1 if any column mentions "GI absorption" and value looks like Low
  - swissadme_penalty_lipinski: numeric Lipinski violations if a matching column exists
  - swissadme_composite_risk: sum of penalties (higher = more concern on crude read)

Outputs (default --output-dir = parent of --input-meta):
  - swissadme_web_merged_n{N}.csv
  - swissadme_web_summary_n{N}.json  (N = number of rows in meta)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def _norm_smiles(s: str) -> str:
    return str(s or "").strip()


def _canonical_smiles_key(s: str) -> str:
    raw = _norm_smiles(s)
    if not raw:
        return raw
    if _HAS_RDKIT:
        mol = Chem.MolFromSmiles(raw)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    return raw


def _detect_smiles_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "smiles" in str(c).strip().lower():
            return str(c)
    for c in df.columns:
        key = str(c).strip().lower()
        if key in {"structure", "canonical_smiles"}:
            return str(c)
    raise SystemExit(
        f"Could not detect SMILES column in SwissADME export. Columns: {list(df.columns)[:40]}..."
    )


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", encoding="utf-8")
    return pd.read_csv(path, encoding="utf-8")


def _col_lower_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in df.columns}


def _penalty_gi_absorption(row: pd.Series, lower_map: dict[str, str]) -> float:
    pen = 0.0
    for lk, orig in lower_map.items():
        if "gi absorption" not in lk:
            continue
        v = row.get(orig, np.nan)
        if pd.isna(v):
            continue
        vs = str(v).lower()
        if "low" in vs and "high" not in vs:
            pen += 1.0
    return pen


def _lipinski_violations(row: pd.Series, lower_map: dict[str, str]) -> float:
    for lk, orig in lower_map.items():
        if "violation" not in lk:
            continue
        if "lipinski" not in lk and "rule of five" not in lk and "ro5" not in lk:
            continue
        v = row.get(orig, np.nan)
        if pd.isna(v) or str(v).strip() == "":
            continue
        try:
            return float(v)
        except ValueError:
            m = re.search(r"(\d+)", str(v))
            if m:
                return float(m.group(1))
    return float("nan")


def _bbb_penalty(row: pd.Series, lower_map: dict[str, str]) -> float:
    for lk, orig in lower_map.items():
        if "bbb" not in lk or "perme" not in lk:
            continue
        v = row.get(orig, np.nan)
        if pd.isna(v):
            continue
        vs = str(v).lower()
        if "yes" in vs:
            return 0.5
    return 0.0


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_meta = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
        / "fda_only_universe"
        / "swissadme_input_top30_with_meta.csv"
    )
    default_shortlist = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
        / "fda_only_universe"
        / "fda_top30_shortlist.csv"
    )
    epilog = f"""Example: replace with the real filename SwissADME saved (not "실제파일이름" or /path/to/...).

  python3 ml/pilot_sagemaker/merge_swissadme_web_results.py \\
    --swissadme-export "$HOME/Downloads/swissadme_results.tsv"

Default --input-meta (if unchanged):

  {default_meta}
"""
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-meta", default=str(default_meta), help="swissadme_input_top30_with_meta.csv")
    ap.add_argument(
        "--swissadme-export",
        required=True,
        metavar="FILE.tsv",
        help="Real path to TSV/CSV saved from SwissADME batch (not a /path/to/... placeholder)",
    )
    ap.add_argument(
        "--shortlist-csv",
        default=str(default_shortlist),
        help="Optional fda_top30_shortlist.csv to attach ensemble_score / preds per cid",
    )
    ap.add_argument("--output-dir", default="", help="Default: parent of --input-meta")
    args = ap.parse_args()

    meta_path = Path(args.input_meta).resolve()
    exp_path = Path(args.swissadme_export).resolve()
    if not meta_path.is_file():
        raise SystemExit(f"Missing meta: {meta_path}")
    if not exp_path.is_file():
        extra = ""
        raw = str(args.swissadme_export)
        norm = raw.replace("\\", "/").lower()
        if "path/to" in norm:
            extra = (
                "\n\nNote: `/path/to/...` in docs is only a placeholder. "
                "Use the real path of the file SwissADME saved on your machine."
            )
        # Common copy-paste: Korean/English "actual filename" placeholders
        base = exp_path.name.lower()
        if "실제" in raw or "실 제" in raw or ("actualfile" in base.replace(" ", "")):
            extra += (
                "\n\nYou may have pasted the words meaning \"actual file name\" instead of renaming "
                "the argument to your real download (e.g. `swissadme_batch_20260402.tsv`)."
            )
        dl = Path.home() / "Downloads"
        if dl.is_dir():
            extra += (
                f'\n\nTo list candidate files:\n  ls -lt "{dl}"/*.tsv "{dl}"/*.csv 2>/dev/null | head -20'
            )
        raise SystemExit(f"Missing SwissADME export: {exp_path}{extra}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else meta_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(meta_path)
    if "cid" not in meta.columns or "smiles" not in meta.columns:
        raise SystemExit("input-meta must contain cid and smiles columns")
    n_meta = len(meta)
    merged_out = out_dir / f"swissadme_web_merged_n{n_meta}.csv"
    summary_out = out_dir / f"swissadme_web_summary_n{n_meta}.json"

    meta = meta.copy()
    meta["_smiles_key"] = meta["smiles"].map(_canonical_smiles_key)

    sw = _read_table(exp_path)
    sm_col = _detect_smiles_column(sw)
    sw = sw.copy()
    sw["_smiles_key"] = sw[sm_col].map(_canonical_smiles_key)

    merged = meta.merge(
        sw,
        on="_smiles_key",
        how="left",
        indicator=True,
        suffixes=("_meta", "_swiss"),
    )
    matched_mask = merged["_merge"] == "both"
    merged.drop(columns=["_smiles_key", "_merge"], inplace=True)

    # Ensemble / model context from shortlist (first row per canonical_drug_id)
    shortlist_path = Path(args.shortlist_csv).resolve()
    if shortlist_path.is_file():
        sl = pd.read_csv(shortlist_path)
        if "canonical_drug_id" in sl.columns:
            sort_col = "fda_drug_rank" if "fda_drug_rank" in sl.columns else sl.columns[0]
            sub = sl.sort_values(sort_col, kind="mergesort").drop_duplicates(
                subset=["canonical_drug_id"], keep="first"
            )
            keep = [c for c in ("canonical_drug_id", "ensemble_score", "pred_xgb", "pred_residualmlp", "pred_gcn", "best_pair_rank") if c in sub.columns]
            if keep:
                merged = merged.merge(
                    sub[keep],
                    left_on="cid",
                    right_on="canonical_drug_id",
                    how="left",
                )
                if "canonical_drug_id" in merged.columns:
                    merged.drop(columns=["canonical_drug_id"], inplace=True)

    lower_map = _col_lower_map(merged)
    penalties_gi: list[float] = []
    penalties_lip: list[float] = []
    penalties_bbb: list[float] = []
    composites: list[float] = []
    for _, row in merged.iterrows():
        lm = {k: v for k, v in lower_map.items()}
        g = _penalty_gi_absorption(row, lm)
        l = _lipinski_violations(row, lm)
        b = _bbb_penalty(row, lm)
        penalties_gi.append(g)
        penalties_lip.append(0.0 if np.isnan(l) else l)
        penalties_bbb.append(b)
        comp = g + (0.0 if np.isnan(l) else l) + b
        composites.append(comp)

    merged["swissadme_penalty_gi_low"] = penalties_gi
    merged["swissadme_penalty_lipinski"] = penalties_lip
    merged["swissadme_penalty_bbb"] = penalties_bbb
    merged["swissadme_composite_risk"] = composites

    es = pd.to_numeric(merged.get("ensemble_score", pd.Series(np.nan)), errors="coerce")
    risk = pd.to_numeric(merged["swissadme_composite_risk"], errors="coerce").fillna(0.0)
    merged["swissadme_rank_score"] = es - 0.15 * risk

    matched = int(matched_mask.sum())
    unmatched = (
        merged.loc[~matched_mask, ["cid", "drug_name", "smiles"]].to_dict(orient="records")
        if len(merged)
        else []
    )
    mean_es_matched = (
        float(es.loc[matched_mask].mean()) if matched and es.loc[matched_mask].notna().any() else None
    )
    mean_risk_matched = (
        float(pd.to_numeric(merged.loc[matched_mask, "swissadme_composite_risk"], errors="coerce").mean())
        if matched
        else None
    )

    merged = merged.sort_values(
        ["swissadme_rank_score", "cid"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    merged["swissadme_summary_rank"] = np.arange(1, len(merged) + 1)

    merged.to_csv(merged_out, index=False)

    summary = {
        "input_meta": str(meta_path),
        "swissadme_export": str(exp_path),
        "rdkit_canonical_keys": _HAS_RDKIT,
        "n_meta_rows": int(len(meta)),
        "n_export_rows": int(len(sw)),
        "smiles_column_detected": sm_col,
        "n_matched_to_export": matched,
        "unmatched_meta_rows": unmatched,
        "wrote_merged": str(merged_out),
        "mean_ensemble_score_matched": mean_es_matched,
        "mean_swissadme_composite_risk_matched": mean_risk_matched,
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(merged_out), "wrote_summary": str(summary_out), **{k: summary[k] for k in ("n_meta_rows", "n_matched_to_export")}}, indent=2))


if __name__ == "__main__":
    main()
