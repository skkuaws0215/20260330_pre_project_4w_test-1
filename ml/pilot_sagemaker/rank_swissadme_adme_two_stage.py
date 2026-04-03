#!/usr/bin/env python3
"""
SwissADME merged table → hard ADME filters → soft scoring → final ranking.

Stage 1 (this script): SwissADME-derived physicochemical / drug-likeness cut.
Stage 2 (manual / separate): ADMETlab for toxicity & metabolism — queue file emitted here.

Inputs: swissadme_web_merged_n*.csv (from merge_swissadme_web_results.py) or raw swissadme.csv
        + meta join if raw (optional --input-meta).

Outputs under --out-dir:
  - swissadme_adme_all_decisions.csv
  - swissadme_adme_hard_pass.csv          (survivors after hard filter)
  - swissadme_adme_soft_top10.csv
  - swissadme_adme_report.json
  - admetlab_stage2_queue.csv             (hard-pass candidates for ADMETlab batch)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    norm = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\s+", " ", cand.strip().lower())
        if key in norm:
            return str(norm[key])
    return None


def _to_float(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _to_int_safe(x) -> int:
    v = _to_float(x)
    if np.isnan(v):
        return -1
    return int(round(v))


def _yes_no(val) -> bool | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    return None


def hard_filter_row(row: pd.Series, cols: dict[str, str | None]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    lip_c = cols.get("lipinski")
    gi_c = cols.get("gi")
    pains_c = cols.get("pains")
    brenk_c = cols.get("brenk")

    if lip_c:
        lv = _to_int_safe(row.get(lip_c))
        if lv >= 2:
            reasons.append(f"Lipinski_violations>={lv}")
    if gi_c:
        gi = str(row.get(gi_c, "")).strip().lower()
        if gi == "low":
            reasons.append("GI_absorption_Low")
    if pains_c:
        pv = _to_int_safe(row.get(pains_c))
        if pv > 0:
            reasons.append(f"PAINS_alerts>{0}")
    if brenk_c:
        bv = _to_int_safe(row.get(brenk_c))
        if bv >= 2:
            reasons.append(f"Brenk_alerts>={bv}")

    return (len(reasons) == 0, reasons)


def soft_penalties(
    row: pd.Series,
    cols: dict[str, str | None],
    cns_target: bool,
) -> dict[str, float | int]:
    wlogp_c = cols.get("wlogp")
    tpsa_c = cols.get("tpsa")
    bio_c = cols.get("bio")
    pgp_c = cols.get("pgp")
    bbb_c = cols.get("bbb")

    w_pen = t_pen = pgp_pen = bbb_pen = 0
    w = _to_float(row[wlogp_c]) if wlogp_c else float("nan")
    if not np.isnan(w):
        if w > 5:
            w_pen = 2
        elif w < 1 or w > 4:
            w_pen = 1
        else:
            w_pen = 0

    t = _to_float(row[tpsa_c]) if tpsa_c else float("nan")
    if not np.isnan(t):
        if 20.0 <= t <= 140.0:
            t_pen = 0
        else:
            t_pen = 1

    if pgp_c and _yes_no(row.get(pgp_c)) is True:
        pgp_pen = 1

    if not cns_target and bbb_c and _yes_no(row.get(bbb_c)) is True:
        bbb_pen = 1

    bio_bonus = 0
    if bio_c:
        b = _to_float(row[bio_c])
        if not np.isnan(b) and b >= 0.55:
            bio_bonus = 1

    penalty_total = float(w_pen + t_pen + pgp_pen + bbb_pen)
    return {
        "penalty_wlogp": w_pen,
        "penalty_tpsa": t_pen,
        "penalty_pgp": pgp_pen,
        "penalty_bbb": bbb_pen,
        "penalty_total": penalty_total,
        "bioavailability_bonus": bio_bonus,
    }


def resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "lipinski": _col(df, "Lipinski #violations", "lipinski #violations"),
        "gi": _col(df, "GI absorption"),
        "pains": _col(df, "PAINS #alerts", "pains #alerts"),
        "brenk": _col(df, "Brenk #alerts", "brenk #alerts"),
        "wlogp": _col(df, "WLOGP", "wlogp"),
        "tpsa": _col(df, "TPSA", "tpsa"),
        "bio": _col(df, "Bioavailability Score", "bioavailability score"),
        "pgp": _col(df, "Pgp substrate", "P-glycoprotein substrate"),
        "bbb": _col(df, "BBB permeant", "bbb permeant"),
        "ensemble": _col(df, "ensemble_score", "ensemble score"),
        "cid": _col(df, "cid"),
        "drug_name": _col(df, "drug_name", "drug name"),
        "smiles": _col(df, "smiles"),
    }


def adme_summary(row: pd.Series, cols: dict[str, str | None]) -> str:
    parts = []
    for key, label in (
        ("gi", "GI"),
        ("lipinski", "Lipinski_v"),
        ("wlogp", "WLOGP"),
        ("tpsa", "TPSA"),
        ("bio", "F(oral)"),
        ("pgp", "P-gp"),
        ("bbb", "BBB"),
        ("pains", "PAINS"),
        ("brenk", "Brenk"),
    ):
        c = cols.get(key)
        if not c:
            continue
        v = row.get(c)
        if pd.notna(v):
            parts.append(f"{label}={v}")
    return "; ".join(parts)


def top10_pattern_summary(df10: pd.DataFrame, cols: dict[str, str | None]) -> dict:
    if df10.empty:
        return {"note": "no rows"}
    out: dict = {"n": int(len(df10))}
    for name, c in (("WLOGP", cols.get("wlogp")), ("TPSA", cols.get("tpsa")), ("Bioavailability Score", cols.get("bio"))):
        if not c or c not in df10.columns:
            continue
        s = pd.to_numeric(df10[c], errors="coerce")
        out[name] = {
            "mean": float(s.mean()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    pgp_c, bbb_c = cols.get("pgp"), cols.get("bbb")
    if pgp_c and pgp_c in df10.columns:
        out["Pgp_substrate_Yes_count"] = int(df10[pgp_c].astype(str).str.lower().eq("yes").sum())
    if bbb_c and bbb_c in df10.columns:
        out["BBB_permeant_Yes_count"] = int(df10[bbb_c].astype(str).str.lower().eq("yes").sum())
    gi_c = cols.get("gi")
    if gi_c and gi_c in df10.columns:
        out["GI_absorption_all"] = df10[gi_c].astype(str).value_counts().to_dict()
    out["common_good_traits"] = (
        "Survivors skew toward Lipinski violations < 2, GI High, PAINS=0, Brenk<2; "
        "soft-ranked favor moderate WLOGP (1–4), TPSA 20–140 when possible, "
        "oral bioavailability score ≥0.55, fewer P-gp/BBB penalties for non-CNS."
    )
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_in = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
        / "fda_only_universe"
        / "swissadme_web_merged_n15.csv"
    )
    default_out = default_in.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", default=str(default_in), help="Merged SwissADME+meta or raw Swiss export")
    ap.add_argument("--input-meta", default="", help="If raw Swiss only, join swissadme_input_top30_with_meta.csv")
    ap.add_argument("--out-dir", default=str(default_out))
    ap.add_argument(
        "--cns-target",
        action="store_true",
        help="If set, do not apply BBB Yes penalty (CNS programme)",
    )
    ap.add_argument("--top-k-soft", type=int, default=10)
    args = ap.parse_args()

    inp = Path(args.input_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    meta_path = Path(args.input_meta).resolve() if args.input_meta else None
    if meta_path and meta_path.is_file() and "ensemble_score" not in df.columns:
        meta = pd.read_csv(meta_path)
        try:
            from rdkit import Chem

            def can(s):
                m = Chem.MolFromSmiles(str(s or ""))
                return Chem.MolToSmiles(m) if m else str(s or "").strip()

            meta = meta.copy()
            meta["_k"] = meta["smiles"].map(can)
            sm_col = _detect_smiles_col(df)
            df = df.copy()
            df["_k"] = df[sm_col].map(can)
            df = df.merge(meta, on="_k", how="left", suffixes=("", "_meta"))
            df.drop(columns=["_k"], inplace=True)
        except ImportError:
            raise SystemExit("Raw Swiss export + --input-meta requires RDKit for SMILES join")

    cols = resolve_columns(df)
    if not cols.get("ensemble"):
        raise SystemExit("Missing ensemble_score column (use merged CSV or pass --input-meta with RDKit)")

    records = []
    for _, row in df.iterrows():
        ok, h_reasons = hard_filter_row(row, cols)
        soft = soft_penalties(row, cols, cns_target=args.cns_target)
        es = _to_float(row[cols["ensemble"]])
        if np.isnan(es):
            es = 0.0
        final_score = es - 0.2 * soft["penalty_total"] + 0.1 * soft["bioavailability_bonus"]

        if ok:
            decision = "HARD_PASS"
            reason = "Passed Lipinski<2, GI not Low, PAINS=0, Brenk<2"
        else:
            decision = "HARD_FAIL"
            reason = "; ".join(h_reasons)

        soft_reason_parts = []
        if soft["penalty_wlogp"]:
            soft_reason_parts.append(f"WLOGP_pen={int(soft['penalty_wlogp'])}")
        if soft["penalty_tpsa"]:
            soft_reason_parts.append("TPSA_out_of_20_140")
        if soft["penalty_pgp"]:
            soft_reason_parts.append("Pgp_substrate")
        if soft["penalty_bbb"]:
            soft_reason_parts.append("BBB_permeant_non_CNS_penalty")
        soft_reason = ", ".join(soft_reason_parts) if soft_reason_parts else "no_soft_penalty"

        records.append(
            {
                "cid": row.get(cols["cid"], np.nan),
                "drug_name": row.get(cols["drug_name"], ""),
                "decision": decision,
                "hard_fail_reasons": "; ".join(h_reasons),
                "selection_or_removal_summary": reason if not ok else f"{reason}. Soft: {soft_reason}",
                "adme_key_summary": adme_summary(row, cols),
                "ensemble_score": es,
                **{k: soft[k] for k in soft},
                "adme_final_score": final_score,
            }
        )

    dec_df = pd.DataFrame(records)
    out_all = df.reset_index(drop=True).copy()
    for c in dec_df.columns:
        out_all[c] = dec_df[c].values

    out_all.to_csv(out_dir / "swissadme_adme_all_decisions.csv", index=False)

    passed = dec_df[dec_df["decision"] == "HARD_PASS"].copy()
    passed = passed.sort_values("adme_final_score", ascending=False, kind="mergesort").reset_index(drop=True)
    passed["swissadme_soft_rank"] = np.arange(1, len(passed) + 1)

    pass_detail = out_all[out_all["decision"] == "HARD_PASS"].copy()
    pass_detail = pass_detail.sort_values("adme_final_score", ascending=False, kind="mergesort").reset_index(drop=True)
    pass_detail.to_csv(out_dir / "swissadme_adme_hard_pass.csv", index=False)

    top_k = min(args.top_k_soft, len(passed))
    top_detail = pass_detail.head(top_k).copy()
    top_detail.to_csv(out_dir / "swissadme_adme_soft_top10.csv", index=False)

    pattern = top10_pattern_summary(top_detail, cols)

    report = {
        "pipeline": {
            "stage1": "SwissADME physicochemical / drug-likeness (this script)",
            "stage2": "ADMETlab toxicity & metabolism — use admetlab_stage2_queue.csv; final list = pass both + model score",
        },
        "input_csv": str(inp),
        "n_input": int(len(df)),
        "n_hard_pass": int(len(passed)),
        "n_hard_fail": int(len(df) - len(passed)),
        "hard_fail_by_cid": dec_df[dec_df["decision"] == "HARD_FAIL"][["cid", "drug_name", "hard_fail_reasons"]].to_dict(orient="records"),
        "soft_top10": passed.head(top_k)[["cid", "drug_name", "adme_final_score", "ensemble_score", "penalty_total", "bioavailability_bonus"]].to_dict(orient="records"),
        "top10_adme_pattern": pattern,
        "parameters": {
            "cns_target_mode": bool(args.cns_target),
            "final_score_formula": "ensemble_score - 0.2 * penalty_total + 0.1 * bioavailability_bonus",
        },
    }
    (out_dir / "swissadme_adme_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(pass_detail):
        queue = pass_detail[["cid", "drug_name"]].copy()
        if "smiles" in pass_detail.columns:
            queue["smiles"] = pass_detail["smiles"].values
        queue["stage1_swissadme_pass"] = True
        queue["ensemble_score"] = pass_detail["ensemble_score"].values
        queue["adme_final_score_after_soft"] = pass_detail["adme_final_score"].values
    else:
        queue = pd.DataFrame(columns=["cid", "drug_name", "smiles", "stage1_swissadme_pass", "ensemble_score", "adme_final_score_after_soft"])
    queue.to_csv(out_dir / "admetlab_stage2_queue.csv", index=False)

    print(json.dumps({"wrote_dir": str(out_dir), "n_pass": len(passed), "n_top": top_k}, indent=2))


def _detect_smiles_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "smiles" in str(c).lower() and "canonical" in str(c).lower():
            return str(c)
    for c in df.columns:
        if str(c).strip().lower() == "smiles":
            return str(c)
    raise SystemExit("Could not find SMILES column for join")


if __name__ == "__main__":
    main()
