#!/usr/bin/env python3
"""
FDA Top30 shortlist → SMILES → direct ADMET-style assessment + proxy ADMET + soft/hard gates.

SwissADME / ADMETlab: no stable public batch API in-repo; when RDKit is available we emit
**descriptor proxies** in columns prefixed swissadme_proxy_* and admetlab_proxy_* (Lipinski,
LogP, TPSA, MW, rotatable bonds, aromatic rings). Documented to align with SwissADME-like panels.

Hard filters (conservative heuristics, review with medicinal chemistry):
  - hERG: proxy high-risk if MolLogP >= 5.0 AND TPSA <= 75 (NaN-safe: skip filter if no RDKit)
  - hepatotoxicity: RDKit SMARTS alert counts (aniline, nitroaromatic, azide, epoxide) >= 2

Soft score: normalize ensemble_score + proxy lipophilicity penalty + toxicity alert count.

Outputs (default --out-dir):
  direct_admet_results.csv
  admet_combined_summary.csv
  final_top10_fda_drugs.csv
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HTTP_UA = "evaluate_fda_shortlist_direct_admet/1.0 (research)"

# RDKit optional
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

TOX_SMARTS = [
    ("alert_nitro_aromatic", "[N+](=O)[O-]"),
    ("alert_aniline", "c[NH2]"),
    ("alert_azide", "[N-]=[N+]=[N-]"),
    ("alert_epoxide", "C1OC1"),
]


def _http_json(url: str, timeout: float = 45.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def pubchem_canonical_smiles(cid: int) -> str:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{int(cid)}/property/CanonicalSMILES/JSON"
    )
    try:
        data = _http_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ""
        raise
    props = (data.get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return ""
    row0 = props[0]
    return str(
        row0.get("CanonicalSMILES") or row0.get("ConnectivitySMILES") or row0.get("SMILES") or ""
    ).strip()


def pubchem_title(cid: int) -> str:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{int(cid)}/property/Title/JSON"
    )
    try:
        data = _http_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ""
        raise
    props = (data.get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return ""
    return str(props[0].get("Title") or "").strip()


def rdkit_row(smiles: str) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "rdkit_valid": 0,
        "swissadme_proxy_mw": np.nan,
        "swissadme_proxy_logp": np.nan,
        "swissadme_proxy_tpsa": np.nan,
        "swissadme_proxy_hbd": np.nan,
        "swissadme_proxy_hba": np.nan,
        "swissadme_proxy_rotatable_bonds": np.nan,
        "swissadme_proxy_aromatic_rings": np.nan,
        "swissadme_proxy_lipinski_violations": np.nan,
        "admetlab_proxy_heavy_atoms": np.nan,
        "structural_alert_count": np.nan,
    }
    if not _HAS_RDKIT or not smiles or not str(smiles).strip():
        return out
    m = Chem.MolFromSmiles(str(smiles).strip())
    if m is None:
        return out
    out["rdkit_valid"] = 1
    out["swissadme_proxy_mw"] = float(Descriptors.MolWt(m))
    out["swissadme_proxy_logp"] = float(Descriptors.MolLogP(m))
    out["swissadme_proxy_tpsa"] = float(Descriptors.TPSA(m))
    out["swissadme_proxy_hbd"] = int(Lipinski.NumHDonors(m))
    out["swissadme_proxy_hba"] = int(Lipinski.NumHAcceptors(m))
    out["swissadme_proxy_rotatable_bonds"] = int(Lipinski.NumRotatableBonds(m))
    out["swissadme_proxy_aromatic_rings"] = int(Lipinski.NumAromaticRings(m))
    out["swissadme_proxy_lipinski_violations"] = int(
        max(
            0,
            int(out["swissadme_proxy_hbd"] > 5)
            + int(out["swissadme_proxy_hba"] > 10)
            + int(out["swissadme_proxy_mw"] > 500)
            + int(out["swissadme_proxy_logp"] > 5),
        )
    )
    out["admetlab_proxy_heavy_atoms"] = int(m.GetNumHeavyAtoms())
    n_alerts = 0
    for _name, smarts in TOX_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat and m.HasSubstructMatch(pat):
            n_alerts += 1
    out["structural_alert_count"] = int(n_alerts)
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    prep = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
    )
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--shortlist-csv",
        default=str(prep / "fda_only_universe" / "fda_top30_shortlist.csv"),
    )
    ap.add_argument(
        "--universe-drug-parquet",
        default=str(prep / "fda_only_universe" / "fda_approved_drug_table.parquet"),
    )
    ap.add_argument(
        "--proxy-shortlist-csv",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260402/sagemaker_dual_validation/final_shortlist.csv"
        ),
        help="TCGA-path final shortlist for admet_pass / rule columns (optional).",
    )
    ap.add_argument(
        "--out-dir",
        default=str(prep / "fda_only_universe"),
        help="Writes direct_admet_results.csv, admet_combined_summary.csv, final_top10_fda_drugs.csv here.",
    )
    ap.add_argument("--pubchem-sleep", type=float, default=0.2)
    ap.add_argument("--final-top-k", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sl = pd.read_csv(args.shortlist_csv)
    if "canonical_drug_id" not in sl.columns:
        raise SystemExit("shortlist missing canonical_drug_id")
    drugs = pd.read_parquet(args.universe_drug_parquet)
    drugs["canonical_drug_id"] = drugs["canonical_drug_id"].astype(int)
    smap = drugs.set_index("canonical_drug_id")["canonical_smiles"].to_dict()

    proxy_df = pd.DataFrame()
    ppath = Path(args.proxy_shortlist_csv)
    if ppath.is_file():
        proxy_df = pd.read_csv(ppath)
        proxy_df["canonical_drug_id"] = proxy_df["canonical_drug_id"].astype(int)

    rows: list[dict] = []
    for _, r in sl.iterrows():
        cid = int(r["canonical_drug_id"])
        smi = str(smap.get(cid, "") or "").strip()
        title = pubchem_title(cid) if cid else ""
        time.sleep(args.pubchem_sleep)
        if not smi:
            smi = pubchem_canonical_smiles(cid)
            time.sleep(args.pubchem_sleep)
        desc = rdkit_row(smi)
        logp = desc.get("swissadme_proxy_logp")
        tpsa = desc.get("swissadme_proxy_tpsa")
        herg_hard = False
        if _HAS_RDKIT and desc.get("rdkit_valid") == 1 and not (np.isnan(logp) or np.isnan(tpsa)):
            herg_hard = bool(float(logp) >= 5.0 and float(tpsa) <= 75.0)
        hep_hard = False
        ac = desc.get("structural_alert_count")
        if _HAS_RDKIT and desc.get("rdkit_valid") == 1 and not np.isnan(ac):
            hep_hard = bool(int(ac) >= 2)

        proxy_admet_pass = np.nan
        if not proxy_df.empty and cid in set(proxy_df["canonical_drug_id"].tolist()):
            prow = proxy_df.loc[proxy_df["canonical_drug_id"] == cid].iloc[0]
            proxy_admet_pass = float(prow["admet_pass"]) if "admet_pass" in prow.index else np.nan

        rows.append(
            {
                "fda_drug_rank": r.get("fda_drug_rank", np.nan),
                "canonical_drug_id": cid,
                "pubchem_title": title,
                "canonical_smiles": smi,
                "ensemble_score": float(r.get("ensemble_score", np.nan)),
                "best_pair_rank": r.get("best_pair_rank", np.nan),
                "pred_xgb": r.get("pred_xgb", np.nan),
                "pred_residualmlp": r.get("pred_residualmlp", np.nan),
                "pred_gcn": r.get("pred_gcn", np.nan),
                "proxy_tcga_admet_pass": proxy_admet_pass,
                "hard_fail_herg_proxy": herg_hard,
                "hard_fail_hepatotoxicity_proxy": hep_hard,
                "hard_fail_any": bool(herg_hard or hep_hard),
                "admetlab_batch_note": "ADMETlab 3.0 web/API batch not called in this script; use scbdd portal for full DMPNN endpoints.",
                "swissadme_batch_note": "SwissADME web not scraped; swissadme_proxy_* from RDKit when installed.",
                **desc,
            }
        )

    direct = pd.DataFrame(rows)
    es = pd.to_numeric(direct["ensemble_score"], errors="coerce")
    es_n = (es - es.mean()) / (es.std(ddof=0) + 1e-9)
    logp = pd.to_numeric(direct["swissadme_proxy_logp"], errors="coerce")
    logp_pen = np.where(np.isnan(logp), 0.0, np.clip(logp - 3.0, 0, None) / 3.0)
    alert = pd.to_numeric(direct["structural_alert_count"], errors="coerce").fillna(0.0)
    direct["soft_composite_score"] = es_n - 0.4 * logp_pen - 0.25 * alert
    direct.to_csv(out_dir / "direct_admet_results.csv", index=False)

    ok = ~direct["hard_fail_any"].astype(bool)
    ranked = direct.loc[ok].sort_values("soft_composite_score", ascending=False).reset_index(drop=True)
    ranked["soft_rank_after_hard_filter"] = np.arange(1, len(ranked) + 1)

    summary = direct.merge(
        ranked[["canonical_drug_id", "soft_rank_after_hard_filter"]],
        on="canonical_drug_id",
        how="left",
    )
    summary.to_csv(out_dir / "admet_combined_summary.csv", index=False)

    top = ranked.head(int(args.final_top_k)).copy()
    top.insert(0, "final_top10_rank", range(1, len(top) + 1))
    top.to_csv(out_dir / "final_top10_fda_drugs.csv", index=False)

    meta = {
        "rdkit_available": _HAS_RDKIT,
        "n_shortlist_input": int(len(sl)),
        "n_after_hard_filter": int(len(ranked)),
        "n_final_top": int(len(top)),
        "outputs": {
            "direct_admet_results": str(out_dir / "direct_admet_results.csv"),
            "admet_combined_summary": str(out_dir / "admet_combined_summary.csv"),
            "final_top10_fda_drugs": str(out_dir / "final_top10_fda_drugs.csv"),
        },
    }
    (out_dir / "direct_admet_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
