#!/usr/bin/env python3
"""
Merge multiple ADMETlab 3.0 web-export CSVs (one molecule per file is OK), dedupe by
canonical SMILES, join METABRIC/Swiss queue, apply hard cut on toxicity probabilities,
then soft-rank using METABRIC ensemble + Swiss ADME final score + ADMETlab-derived terms.

Hard filter (exclude from final ranked shortlist):
  hERG, H-HT, DILI, Ames >= 0.7 (any one triggers hard_fail)

Soft ranking (among hard-pass rows):
  final_rank_score =
    zscore(ensemble_score)
    + zscore(adme_final_score_after_soft)
    + w_admet * zscore(admet_favorable_raw)
    - cyp_demote * max(0, n_cyp_inhibitors - 1)

  admet_favorable_raw (higher better, before z-scaling within cohort):
    mean_i clip(0.7 - p_i, 0, 0.7) for p in {hERG, H-HT, DILI, Ames}
    + fu_bonus * max(0, Fu - 5) / 95
    + cl_bonus * max(0, CL_eff - 5) / max(CL_eff, 1e-6)  # CL_eff = first non-NaN of cl-int, cl-plasma

CYP inhibitor count: CYP*-inh columns with value >= cyp_inh_cutoff (default 0.5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except ImportError as e:
    raise SystemExit("RDKit required: pip install rdkit") from e


CYP_INH_COLS = [
    "CYP1A2-inh",
    "CYP2C19-inh",
    "CYP2C9-inh",
    "CYP2D6-inh",
    "CYP3A4-inh",
]


def _canon_smiles(s: str) -> str | None:
    m = Chem.MolFromSmiles(str(s).strip())
    return Chem.MolToSmiles(m) if m else None


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def load_admetlab_paths(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    frames = []
    per_file = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df.assign(_source_file=p.name))
        per_file.append({"file": p.name, "n_rows": len(df)})
    all_df = pd.concat(frames, ignore_index=True, copy=False)
    if "smiles" not in all_df.columns:
        raise SystemExit("ADMETlab CSV missing 'smiles' column")
    all_df["canonical_smiles"] = all_df["smiles"].map(_canon_smiles)
    bad = all_df["canonical_smiles"].isna().sum()
    if bad:
        raise SystemExit(f"{bad} rows: invalid SMILES")
    dup_mask = all_df.duplicated(subset=["canonical_smiles"], keep="first")
    n_dup = int(dup_mask.sum())
    merged = all_df.loc[~dup_mask].copy()
    audit = {
        "n_files": len(paths),
        "n_rows_total": int(len(all_df)),
        "n_unique_canonical": int(len(merged)),
        "n_dropped_duplicate_rows": n_dup,
        "per_file": per_file,
    }
    return merged, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--admetlab-glob",
        type=str,
        default="",
        help="Glob for ADMETlab CSVs (quoted), e.g. '/Users/me/Downloads/ADMETlab3_result*.csv'",
    )
    ap.add_argument(
        "--admetlab-files",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit CSV paths (alternative to glob)",
    )
    ap.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "results/features_nextflow_team4/fe_re_batch_runs/20260402/"
            "metabric_true_validation_prep/fda_only_universe/admetlab_stage2_queue.csv"
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--hard-cut", type=float, default=0.7)
    ap.add_argument("--cyp-inh-cutoff", type=float, default=0.5)
    ap.add_argument("--w-admet-z", type=float, default=1.0)
    ap.add_argument("--cyp-demote", type=float, default=0.15)
    ap.add_argument("--fu-bonus-scale", type=float, default=1.0, help="multiplier on (Fu-5)/95 term")
    ap.add_argument("--cl-bonus-scale", type=float, default=0.5, help="scale for CL tail above 5")
    args = ap.parse_args()

    if args.admetlab_files:
        paths = sorted(set(args.admetlab_files))
    else:
        import glob

        paths = sorted(Path(p) for p in glob.glob(args.admetlab_glob))
    if not paths:
        raise SystemExit("No ADMETlab CSV paths resolved")

    out_dir = args.out_dir or args.queue.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lab, lab_audit = load_admetlab_paths(paths)
    lab.to_csv(out_dir / "admetlab_web_merged_dedup.csv", index=False)

    q = pd.read_csv(args.queue)
    q["canonical_smiles"] = q["smiles"].map(_canon_smiles)
    if q["canonical_smiles"].isna().any():
        raise SystemExit("queue: invalid SMILES")

    merged = q.merge(lab, on="canonical_smiles", how="left", suffixes=("", "_lab"))
    missing_lab = merged["taskId"].isna() if "taskId" in merged.columns else merged["MW"].isna()
    n_miss = int(missing_lab.sum())
    if n_miss:
        raise SystemExit(f"{n_miss} queue rows did not match any ADMETlab export after dedup")

    # normalize toxicity column names
    def col(df, *names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_herg = col(merged, "hERG")
    c_hht = col(merged, "H-HT")
    c_dili = col(merged, "DILI")
    c_ames = col(merged, "Ames", "AMES")
    for cname, label in [(c_herg, "hERG"), (c_hht, "H-HT"), (c_dili, "DILI"), (c_ames, "Ames")]:
        if not cname:
            raise SystemExit(f"Missing column for {label}")

    merged["admetlab_hERG"] = pd.to_numeric(merged[c_herg], errors="coerce")
    merged["admetlab_H_HT"] = pd.to_numeric(merged[c_hht], errors="coerce")
    merged["admetlab_DILI"] = pd.to_numeric(merged[c_dili], errors="coerce")
    merged["admetlab_Ames"] = pd.to_numeric(merged[c_ames], errors="coerce")

    tox = merged[["admetlab_hERG", "admetlab_H_HT", "admetlab_DILI", "admetlab_Ames"]]
    merged["admetlab_hard_fail"] = (tox >= args.hard_cut).any(axis=1)
    merged["admetlab_hard_fail_reasons"] = tox.apply(
        lambda r: ";".join(
            n.replace("admetlab_", "")
            for n, v in zip(["hERG", "H_HT", "DILI", "Ames"], r)
            if pd.notna(v) and v >= args.hard_cut
        ),
        axis=1,
    )

    # ADMETlab PK
    merged["Fu_pct"] = pd.to_numeric(merged.get("Fu"), errors="coerce")
    merged["CL_int"] = pd.to_numeric(merged.get("cl-int"), errors="coerce")
    merged["CL_plasma"] = pd.to_numeric(merged.get("cl-plasma"), errors="coerce")
    merged["CL_eff"] = merged["CL_int"].combine_first(merged["CL_plasma"])
    merged["T_half"] = pd.to_numeric(merged.get("t0.5"), errors="coerce")

    def favorable_row(r) -> float:
        parts = []
        for p in [r.admetlab_hERG, r.admetlab_H_HT, r.admetlab_DILI, r.admetlab_Ames]:
            if pd.isna(p):
                continue
            parts.append(float(np.clip(0.7 - float(p), 0.0, 0.7)))
        base = float(np.mean(parts)) if parts else 0.0
        fu = r.Fu_pct
        if pd.notna(fu):
            base += args.fu_bonus_scale * max(0.0, float(fu) - 5.0) / 95.0
        cl = r.CL_eff
        if pd.notna(cl) and float(cl) > 0:
            base += args.cl_bonus_scale * max(0.0, float(cl) - 5.0) / float(cl)
        return base

    merged["admet_favorable_raw"] = merged.apply(favorable_row, axis=1)

    n_cyp = []
    flags = []
    for _, r in merged.iterrows():
        fs = []
        n = 0
        for c in CYP_INH_COLS:
            if c not in merged.columns:
                continue
            v = pd.to_numeric(r.get(c), errors="coerce")
            if pd.notna(v) and float(v) >= args.cyp_inh_cutoff:
                n += 1
                fs.append(c)
        n_cyp.append(n)
        flags.append(";".join(fs))
    merged["n_cyp_inhibitor_ge_cutoff"] = n_cyp
    merged["cyp_inhibitor_cols_ge_cutoff"] = flags

    z_e = _z(merged["ensemble_score"])
    z_s = _z(merged["adme_final_score_after_soft"])
    z_a = _z(merged["admet_favorable_raw"])
    cyp_pen = args.cyp_demote * np.maximum(0, merged["n_cyp_inhibitor_ge_cutoff"].values - 1)
    merged["final_rank_score"] = z_e + z_s + args.w_admet_z * z_a - cyp_pen

    merged["stage1_swissadme_pass"] = merged["stage1_swissadme_pass"].astype(str)
    hard_ok = ~merged["admetlab_hard_fail"]
    ranked = merged.loc[hard_ok].sort_values("final_rank_score", ascending=False, kind="mergesort")
    ranked = ranked.reset_index(drop=True)
    ranked["final_rank"] = np.arange(1, len(ranked) + 1)

    merged_path = out_dir / "admetlab_web_queue_merged_full.csv"
    merged.to_csv(merged_path, index=False)
    ranked_path = out_dir / "admetlab_web_final_ranked_hardpass.csv"
    ranked.to_csv(ranked_path, index=False)

    summary = {
        "admetlab_input_audit": lab_audit,
        "queue": str(args.queue.resolve()),
        "n_queue": int(len(q)),
        "hard_fail_cutoff_ge": args.hard_cut,
        "n_admetlab_hard_fail": int(merged["admetlab_hard_fail"].sum()),
        "n_hard_pass_ranked": int(len(ranked)),
        "weights": {
            "z_ensemble": 1.0,
            "z_swiss_adme_final": 1.0,
            "w_admet_z": args.w_admet_z,
            "cyp_demote_per_extra_inhibitor": args.cyp_demote,
        },
        "outputs": {
            "admetlab_web_merged_dedup.csv": str((out_dir / "admetlab_web_merged_dedup.csv").resolve()),
            "admetlab_web_queue_merged_full.csv": str(merged_path.resolve()),
            "admetlab_web_final_ranked_hardpass.csv": str(ranked_path.resolve()),
        },
    }
    (out_dir / "admetlab_web_merge_rank_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
