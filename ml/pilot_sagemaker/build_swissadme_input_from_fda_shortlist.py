#!/usr/bin/env python3
"""
Build SwissADME batch input from fda_top30_shortlist.csv:

  - swissadme_input_top30.txt — one SMILES per line (deduplicated)
  - swissadme_input_top30_with_meta.csv — cid, drug_name, smiles

Uses existing SMILES column when present and non-empty; otherwise PubChem PUG REST
(property CanonicalSMILES + Title). PubChem may return SMILES under SMILES or
ConnectivitySMILES instead of CanonicalSMILES — handled like other repo scripts.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HTTP_UA = "build_swissadme_input_from_fda_shortlist/1.0 (research; +https://pubchem.ncbi.nlm.nih.gov/)"


def _http_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def pubchem_smiles_and_title(cid: int) -> tuple[str, str]:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{int(cid)}/property/CanonicalSMILES,Title/JSON"
    )
    try:
        data = _http_json(url, timeout=45.0)
    except urllib.error.HTTPError as e:
        if e.code in (404, 400):
            return "", ""
        raise
    except urllib.error.URLError:
        return "", ""
    props = (data.get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return "", ""
    row = props[0]
    smi = (
        row.get("CanonicalSMILES")
        or row.get("ConnectivitySMILES")
        or row.get("SMILES")
        or ""
    )
    title = str(row.get("Title") or "").strip()
    return str(smi).strip(), title


def pubchem_title_only(cid: int) -> str:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{int(cid)}/property/Title/JSON"
    )
    try:
        data = _http_json(url, timeout=45.0)
    except urllib.error.HTTPError as e:
        if e.code in (404, 400):
            return ""
        raise
    except urllib.error.URLError:
        return ""
    props = (data.get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return ""
    return str(props[0].get("Title") or "").strip()


def detect_cid_column(columns: list[str]) -> str:
    for c in columns:
        cl = c.lower()
        if cl == "canonical_drug_id" or cl.endswith("_cid") or cl == "cid":
            return c
    for c in columns:
        if "cid" in c.lower():
            return c
    raise SystemExit(f"No CID column found (expected canonical_drug_id or *cid*). Got: {columns}")


def detect_smiles_column(columns: list[str]) -> str | None:
    for c in columns:
        if "smiles" in c.lower():
            return c
    return None


def detect_name_column(columns: list[str]) -> str | None:
    for c in columns:
        cl = c.lower()
        if cl in ("pubchem_title", "drug_name", "title", "compound_name"):
            return c
    return None


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_in = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
        / "fda_only_universe"
        / "fda_top30_shortlist.csv"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", default=str(default_in), help="fda_top30_shortlist.csv path")
    ap.add_argument(
        "--output-dir",
        default="",
        help="Directory for outputs (default: same as input CSV parent)",
    )
    ap.add_argument("--pubchem-sleep", type=float, default=0.22, help="Delay between PubChem calls")
    ap.add_argument(
        "--failed-cids-out",
        default="swissadme_input_top30_pubchem_failed_cids.txt",
        help="Write CIDs that needed PubChem SMILES but got none (one CID per line). Empty if all ok.",
    )
    args = ap.parse_args()

    inp = Path(args.input_csv).resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing input: {inp}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "swissadme_input_top30.txt"
    out_meta = out_dir / "swissadme_input_top30_with_meta.csv"
    failed_path = out_dir / args.failed_cids_out

    df = pd.read_csv(inp)
    cid_col = detect_cid_column(list(df.columns))
    smiles_col = detect_smiles_column(list(df.columns))
    name_col = detect_name_column(list(df.columns))

    work = pd.DataFrame()
    work["cid"] = pd.to_numeric(df[cid_col], errors="coerce").astype("Int64")
    if smiles_col:
        work["smiles"] = df[smiles_col].astype(str).replace({"nan": "", "None": ""})
    else:
        work["smiles"] = ""
    if name_col:
        work["drug_name"] = df[name_col].astype(str)
    else:
        work["drug_name"] = ""

    work = work.dropna(subset=["cid"])
    work["cid"] = work["cid"].astype(int)

    rows: list[dict[str, str]] = []
    pubchem_smiles_failed: list[int] = []
    seen_cid: set[int] = set()
    for _, r in work.iterrows():
        cid = int(r["cid"])
        if cid in seen_cid:
            continue
        seen_cid.add(cid)
        smi = str(r["smiles"] or "").strip()
        dname = str(r["drug_name"] or "").strip()
        if smi.lower() in ("", "nan", "none"):
            smi = ""
        needed_pubchem_smiles = not smi
        if not smi:
            smi, t = pubchem_smiles_and_title(cid)
            time.sleep(args.pubchem_sleep)
            if not dname and t:
                dname = t
        elif not dname:
            t = pubchem_title_only(cid)
            time.sleep(args.pubchem_sleep)
            if t:
                dname = t
        smi = smi.strip()
        if not smi:
            if needed_pubchem_smiles:
                pubchem_smiles_failed.append(cid)
            continue
        rows.append({"cid": cid, "drug_name": dname or f"CID_{cid}", "smiles": smi})

    if pubchem_smiles_failed:
        failed_path.write_text(
            "\n".join(str(x) for x in sorted(set(pubchem_smiles_failed))) + "\n",
            encoding="utf-8",
        )
    else:
        failed_path.write_text("", encoding="utf-8")

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit(
            f"No SMILES resolved for any row; check CIDs and network. "
            f"Failed CIDs (PubChem SMILES): {sorted(set(pubchem_smiles_failed))}. "
            f"See also: {failed_path}"
        )

    out_df = out_df.drop_duplicates(subset=["smiles"], keep="first")
    out_df = out_df.sort_values("cid", kind="mergesort").reset_index(drop=True)

    out_txt.write_text("\n".join(out_df["smiles"].tolist()) + "\n", encoding="utf-8")
    out_df.to_csv(out_meta, index=False)

    print(
        json.dumps(
            {
                "wrote": str(out_txt),
                "wrote_meta": str(out_meta),
                "n_smiles": len(out_df),
                "n_pubchem_smiles_failed": len(set(pubchem_smiles_failed)),
                "failed_cids_file": str(failed_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
