#!/usr/bin/env python3
"""
Stage-2 ADMET screening for compounds in admetlab_stage2_queue.csv.

Official ADMETlab 3.0 POST /api/single/admet was returning HTTP 500 (server KeyError)
as of 2026-04; this script uses ADMET-AI (TDC-trained Chemprop) as the default engine
so the pipeline is reproducible offline. Override with --import-admetlab-csv when you
paste a batch export from ADMETlab web (column names normalized case-insensitively).

Outputs (under --out-dir):
  - admetlab_stage2_results.csv
  - admetlab_stage2_summary.json
  - final_fda_top_candidates.csv   # Swiss stage-1 pass ∩ stage-2 pass, sorted by ensemble_score

Hard fail (classification probabilities in [0,1]):
  hERG, H_HT, DILI, AMES >= 0.7
  (H_HT from ClinTox when using admet-ai — see summary JSON.)

Soft penalty flags (do not remove; recorded + soft_penalty_count):
  - >=2 of CYP1A2/CYP2C19/CYP2C9/CYP2D6/CYP3A4 inhibitor models >= cyp_inhibitor_prob_cutoff (default 0.5)
  - Clearance_Microsome_AZ < clearance_lt (default 5, uL/min/mg TDC AZ task)
  - Fu_percent < fu_lt_percent (unbound = 100 - PPBR_AZ; default Fu < 5%)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CYP_VEITH_COLS = [
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
]


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _find_col(df: pd.DataFrame, *names: str) -> str | None:
    """Map desired logical names to actual column (case-insensitive, slug match)."""
    slug_map = {_norm_col(c): c for c in df.columns}
    for n in names:
        key = _norm_col(n)
        if key in slug_map:
            return str(slug_map[key])
    for n in names:
        key = _norm_col(n)
        for sk, orig in slug_map.items():
            if key in sk or sk in key:
                return str(orig)
    return None


def load_admetlab_manual_csv(path: Path) -> pd.DataFrame:
    """User export from ADMETlab: must include smiles + toxicity columns."""
    raw = pd.read_csv(path)
    smi_c = _find_col(raw, "smiles", "canonical_smiles")
    if not smi_c:
        raise SystemExit(f"Could not find SMILES column in {path}")
    out = raw.rename(columns={smi_c: "smiles"})
    return out


def run_admet_ai(queue: pd.DataFrame) -> pd.DataFrame:
    from admet_ai import ADMETModel

    model = ADMETModel()
    smiles = queue["smiles"].astype(str).tolist()
    preds = model.predict(smiles=smiles)
    if not isinstance(preds, pd.DataFrame):
        raise RuntimeError("admet_ai.predict expected DataFrame for list input")
    preds = preds.copy()
    preds.reset_index(names="smiles", inplace=True)
    merged = queue.merge(preds, on="smiles", how="left", suffixes=("", "_pred"))
    return merged


def build_results_table(
    merged: pd.DataFrame,
    *,
    h_ht_source: str,
    engine: str,
    cyp_cutoff: float,
    clearance_lt: float,
    fu_lt_percent: float,
    hard_ge: float = 0.7,
) -> pd.DataFrame:
    rows = []
    for _, r in merged.iterrows():
        herg = float(r.get("hERG", np.nan))
        if h_ht_source == "ClinTox":
            h_ht = float(r.get("ClinTox", np.nan))
        else:
            h_ht = float(r.get("H_HT", r.get("H-HT", np.nan)))
        dili = float(r.get("DILI", np.nan))
        ames = float(r.get("AMES", np.nan))

        cyp_flags = []
        for c in CYP_VEITH_COLS:
            v = float(r.get(c, np.nan))
            if not np.isnan(v) and v >= cyp_cutoff:
                cyp_flags.append(c)
        n_cyp = len(cyp_flags)

        clr = float(r.get("Clearance_Microsome_AZ", np.nan))
        ppbr = float(r.get("PPBR_AZ", np.nan))
        fu_pct = np.nan
        if not np.isnan(ppbr):
            if 0.0 <= ppbr <= 100.0:
                fu_pct = 100.0 - ppbr
            # model outliers: skip Fu rule when binding rate is out of physical range
        t_half = float(r.get("Half_Life_Obach", np.nan))

        hard_reasons = []
        if not np.isnan(herg) and herg >= hard_ge:
            hard_reasons.append(f"hERG>={hard_ge}")
        if not np.isnan(h_ht) and h_ht >= hard_ge:
            hard_reasons.append(f"H_HT>={hard_ge}")
        if not np.isnan(dili) and dili >= hard_ge:
            hard_reasons.append(f"DILI>={hard_ge}")
        if not np.isnan(ames) and ames >= hard_ge:
            hard_reasons.append(f"AMES>={hard_ge}")

        soft_reasons = []
        if n_cyp >= 2:
            soft_reasons.append(f"multi_CYP_inhibitor_count={n_cyp}")
        if not np.isnan(clr) and clr < clearance_lt:
            soft_reasons.append(f"Clearance_Microsome_AZ<{clearance_lt}")
        if not np.isnan(fu_pct) and fu_pct < fu_lt_percent:
            soft_reasons.append(f"Fu_pct<{fu_lt_percent}")
        ppbr_note = ""
        if not np.isnan(ppbr) and (ppbr < 0 or ppbr > 100):
            ppbr_note = "PPBR_AZ_out_of_0_100_range"

        rows.append(
            {
                "cid": r.get("cid"),
                "drug_name": r.get("drug_name"),
                "smiles": r.get("smiles"),
                "stage1_swissadme_pass": r.get("stage1_swissadme_pass"),
                "ensemble_score": r.get("ensemble_score"),
                "adme_final_score_after_soft": r.get("adme_final_score_after_soft"),
                "stage2_engine": engine,
                "h_ht_proxy": h_ht_source,
                "hERG": herg,
                "H_HT": h_ht,
                "DILI": dili,
                "AMES": ames,
                "CYP1A2_Veith": r.get("CYP1A2_Veith"),
                "CYP2C19_Veith": r.get("CYP2C19_Veith"),
                "CYP2C9_Veith": r.get("CYP2C9_Veith"),
                "CYP2D6_Veith": r.get("CYP2D6_Veith"),
                "CYP3A4_Veith": r.get("CYP3A4_Veith"),
                "n_cyp_inhibitor_ge_cutoff": n_cyp,
                "cyp_inhibitor_flags": ";".join(cyp_flags),
                "Clearance_Microsome_AZ": clr,
                "Half_Life_Obach_hr": t_half,
                "PPBR_AZ_percent_bound": ppbr,
                "Fu_percent_unbound": fu_pct,
                "fu_ppbr_note": ppbr_note,
                "stage2_hard_fail": len(hard_reasons) > 0,
                "stage2_hard_fail_reasons": ";".join(hard_reasons),
                "stage2_soft_penalty": len(soft_reasons) > 0,
                "stage2_soft_penalty_reasons": ";".join(soft_reasons),
                "stage2_soft_penalty_count": len(soft_reasons),
                "stage2_pass_operational": len(hard_reasons) == 0,
            }
        )
    return pd.DataFrame(rows)


def apply_manual_columns(df: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Join manual ADMETlab export on smiles."""
    m = manual.copy()
    m["smiles"] = m["smiles"].astype(str).str.strip()
    base = df.copy()
    base["smiles"] = base["smiles"].astype(str).str.strip()
    # pick toxicity columns from manual
    colmap = {}
    for logical, candidates in [
        ("hERG", ["hERG", "herg", "HERG blocker", "hERG_inhibition"]),
        ("H_HT", ["H-HT", "H_HT", "human hepatotoxicity", "hepatotoxicity"]),
        ("DILI", ["DILI", "dili"]),
        ("AMES", ["AMES", "ames"]),
        ("Clearance_Microsome_AZ", ["Clearance", "clearance", "CL", "clearance_microsome"]),
        ("Half_Life_Obach", ["T1/2", "t12", "half_life", "Half-Life", "Half life"]),
        ("PPBR_AZ", ["Fu", "fu", "PPBR", "ppbr", "fup"]),
    ]:
        c = _find_col(m, *candidates)
        if c and c != "smiles":
            colmap[logical] = c
    sub = m[["smiles"] + [colmap[k] for k in colmap if k in colmap]].copy()
    rename = {v: k for k, v in colmap.items()}
    sub.rename(columns=rename, inplace=True)
    # If manual used Fu as unbound fraction directly (0-100)
    if "PPBR_AZ" in sub.columns and sub["PPBR_AZ"].max() <= 1.5:
        sub["PPBR_AZ"] = 100.0 * (1.0 - sub["PPBR_AZ"].astype(float))
    out = base.merge(sub, on="smiles", how="left", suffixes=("", "_manual"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "results/features_nextflow_team4/fe_re_batch_runs/20260402/"
            "metabric_true_validation_prep/fda_only_universe/admetlab_stage2_queue.csv"
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--engine",
        choices=("admet-ai", "manual-csv"),
        default="admet-ai",
    )
    ap.add_argument("--import-admetlab-csv", type=Path, default=None, help="With --engine manual-csv")
    ap.add_argument("--cyp-inhibitor-prob-cutoff", type=float, default=0.5)
    ap.add_argument("--clearance-lt", type=float, default=5.0)
    ap.add_argument("--fu-lt-percent", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = args.out_dir or args.queue.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    queue = pd.read_csv(args.queue)
    for c in ("cid", "drug_name", "smiles"):
        if c not in queue.columns:
            raise SystemExit(f"queue CSV missing column: {c}")

    engine_note = args.engine
    h_ht_source = "ClinTox"

    if args.engine == "admet-ai":
        merged = run_admet_ai(queue)
    else:
        if not args.import_admetlab_csv:
            raise SystemExit("--import-admetlab-csv required for manual-csv engine")
        manual = load_admetlab_manual_csv(args.import_admetlab_csv)
        merged = apply_manual_columns(queue, manual)
        h_ht_source = "ADMETlab_export"
        engine_note = "manual-csv"

    results = build_results_table(
        merged,
        h_ht_source=h_ht_source,
        engine=engine_note,
        cyp_cutoff=args.cyp_inhibitor_prob_cutoff,
        clearance_lt=args.clearance_lt,
        fu_lt_percent=args.fu_lt_percent,
    )

    results_path = out_dir / "admetlab_stage2_results.csv"
    results.to_csv(results_path, index=False)

    passed = results[results["stage2_pass_operational"]].copy()
    passed = passed.sort_values(
        ["ensemble_score", "adme_final_score_after_soft"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    passed["final_rank_after_stage2"] = np.arange(1, len(passed) + 1)

    final_path = out_dir / "final_fda_top_candidates.csv"
    passed.to_csv(final_path, index=False)

    summary = {
        "queue": str(args.queue.resolve()),
        "out_dir": str(out_dir.resolve()),
        "engine": args.engine,
        "admetlab_api_note": (
            "ADMETlab 3.0 https://admetlab3.scbdd.com/api/single/admet returned HTTP 500 "
            "(KeyError BSEP in server stack) when tested — use web export + --engine manual-csv "
            "if you need ADMETlab-branded numbers."
        ),
        "h_ht_mapping": (
            "H_HT column uses ClinTox (TDC clinical trial toxicity probability) when engine=admet-ai; "
            "replace with true ADMETlab H-HT via --engine manual-csv."
        ),
        "hard_fail_rules": {
            "hERG_ge": 0.7,
            "H_HT_ge": 0.7,
            "DILI_ge": 0.7,
            "AMES_ge": 0.7,
        },
        "soft_penalty_rules": {
            "multi_cyp_inhibitor": f">=2 of {{{','.join(CYP_VEITH_COLS)}}} with prob>={args.cyp_inhibitor_prob_cutoff}",
            "clearance_microsome_az_lt": args.clearance_lt,
            "fu_percent_unbound_lt": args.fu_lt_percent,
            "fu_derived_from_PPBR_AZ": "Fu_percent_unbound = 100 - PPBR_AZ (TDC: plasma protein binding rate %)",
        },
        "counts": {
            "n_queue": int(len(queue)),
            "n_stage2_evaluated": int(len(results)),
            "n_stage2_hard_fail": int(results["stage2_hard_fail"].sum()),
            "n_stage2_pass": int(passed.shape[0]),
            "n_stage2_soft_penalty": int(results["stage2_soft_penalty"].sum()),
        },
        "outputs": {
            "admetlab_stage2_results.csv": str(results_path.resolve()),
            "admetlab_stage2_summary.json": str((out_dir / "admetlab_stage2_summary.json").resolve()),
            "final_fda_top_candidates.csv": str(final_path.resolve()),
        },
    }
    (out_dir / "admetlab_stage2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
