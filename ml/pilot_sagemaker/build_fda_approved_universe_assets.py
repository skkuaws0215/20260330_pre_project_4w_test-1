#!/usr/bin/env python3
"""
Build an FDA-aligned PubChem CID universe and asset tables for METABRIC pair FE:

  - Drugs@FDA (openFDA bulk zip): active ingredient tokens → PubChem name → CID
  - DrugBank approved (optional XML or precomputed DB##### JSON): RegistryID xref → CID
  - Merge: union (default) or intersection of CID sets from enabled sources
  - Drop script default non-therapeutic CIDs (same list as build_fda_metabric_shortlist.py)
  - SMILES via PubChem PUG property (batched)
  - Target rows: left-join from train drug_target parquet (CIDs without targets omitted)
  - Optional: extend train LINCS signature table with zero rows for new CIDs (--build-lincs-stub)

Caches under --cache-dir to avoid repeated PubChem calls.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_fda_metabric_shortlist as fda_sb  # noqa: E402

HTTP_UA = "build_fda_approved_universe_assets/1.0 (research; +https://pubchem.ncbi.nlm.nih.gov/)"
NAME_SKIP_RE = re.compile(r"^[0-9.\s%-]+$")


def _http_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def pubchem_name_to_first_cid(name: str) -> int | None:
    q = urllib.parse.quote(name.strip(), safe="")
    if len(q) < 2:
        return None
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON"
    try:
        data = _http_json(url, timeout=45.0)
    except urllib.error.HTTPError as e:
        if e.code in (404, 400):
            return None
        raise
    ids = (data.get("IdentifierList") or {}).get("CID") or []
    if not ids:
        return None
    return int(ids[0])


def pubchem_registry_to_cids(registry_id: str) -> list[int]:
    rid = urllib.parse.quote(str(registry_id).strip().upper(), safe="")
    if not rid:
        return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{rid}/cids/JSON"
    try:
        data = _http_json(url, timeout=45.0)
    except urllib.error.HTTPError as e:
        if e.code in (404, 400):
            return []
        raise
    ids = (data.get("IdentifierList") or {}).get("CID") or []
    return [int(x) for x in ids]


def pubchem_fetch_smiles_batch(cids: list[int], chunk: int = 200) -> dict[int, str]:
    out: dict[int, str] = {}
    for i in range(0, len(cids), chunk):
        batch = cids[i : i + chunk]
        cstr = ",".join(str(int(c)) for c in batch)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cstr}/property/CanonicalSMILES/JSON"
        )
        data = _http_json(url, timeout=120.0)
        props = (data.get("PropertyTable") or {}).get("Properties") or []
        for row in props:
            cid = int(row["CID"])
            smi = row.get("CanonicalSMILES") or row.get("ConnectivitySMILES") or row.get("SMILES") or ""
            if smi:
                out[cid] = str(smi).strip()
        time.sleep(0.12)
    return out


def load_json_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def save_json_cache(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=0, sort_keys=True) + "\n", encoding="utf-8")


def cids_from_drugsfda_tokens(
    tokens: set[str],
    name_cache: dict[str, int | None],
    sleep_s: float,
    *,
    stop_after_n_distinct_cids: int | None = None,
) -> set[int]:
    """
    stop_after_n_distinct_cids: if set, stop resolving new names once this many distinct
    CIDs have been found (cache hits count). Speeds up --max-universe-drugs smoke runs.
    """
    cids: set[int] = set()
    for tok in sorted(tokens):
        if stop_after_n_distinct_cids is not None and len(cids) >= stop_after_n_distinct_cids:
            break
        t = tok.strip().lower()
        if len(t) < 2 or NAME_SKIP_RE.match(t):
            continue
        if t in name_cache:
            v = name_cache[t]
            if v is not None:
                cids.add(int(v))
            continue
        cid = pubchem_name_to_first_cid(t)
        name_cache[t] = cid
        if cid is not None:
            cids.add(cid)
        time.sleep(sleep_s)
    return cids


def cids_from_drugbank_approved(
    approved_db_ids: set[str],
    sleep_s: float,
) -> set[int]:
    cids: set[int] = set()
    for dbid in sorted(approved_db_ids):
        if not fda_sb.DB_ID_RE.match(dbid):
            continue
        for cid in pubchem_registry_to_cids(dbid):
            cids.add(cid)
        time.sleep(sleep_s)
    return cids


def merge_target_table(universe_cids: list[int], train_dt_path: Path, drug_id_col: str, target_col: str) -> pd.DataFrame:
    if not train_dt_path.is_file():
        raise SystemExit(f"Missing train drug_target parquet: {train_dt_path}")
    dt = pd.read_parquet(train_dt_path)
    for c in (drug_id_col, target_col):
        if c not in dt.columns:
            raise SystemExit(f"drug_target parquet missing {c!r}")
    u = {str(int(x)) for x in universe_cids}
    dt = dt.copy()
    dt[drug_id_col] = dt[drug_id_col].astype(str).str.strip()
    return dt[dt[drug_id_col].isin(u)].reset_index(drop=True)


def extend_lincs_stub(
    universe_cids: list[int],
    train_lincs_path: Path,
    drug_id_col: str,
) -> pd.DataFrame:
    if not train_lincs_path.is_file():
        raise SystemExit(f"Missing train LINCS parquet: {train_lincs_path}")
    base = pd.read_parquet(train_lincs_path)
    if drug_id_col not in base.columns:
        raise SystemExit(f"LINCS parquet missing {drug_id_col!r}")
    num_cols = [
        c
        for c in base.columns
        if c != drug_id_col and pd.api.types.is_numeric_dtype(base[c])
    ]
    u_str = {str(int(x)) for x in universe_cids}
    have = set(base[drug_id_col].astype(str).str.strip())
    missing = sorted(u_str - have, key=lambda x: int(x))
    if not missing:
        out = base[base[drug_id_col].astype(str).str.strip().isin(u_str)].copy()
        return out.reset_index(drop=True)
    stubs = []
    for mid in missing:
        row = {drug_id_col: mid, **{c: 0.0 for c in num_cols}}
        stubs.append(row)
    stub_df = pd.DataFrame(stubs)
    sub = base[base[drug_id_col].astype(str).str.strip().isin(u_str)].copy()
    out = pd.concat([sub, stub_df], ignore_index=True)
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_prep = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep"
    )
    default_out = default_prep / "fda_only_universe"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=str, default=str(default_out))
    ap.add_argument("--cache-dir", type=str, default="", help="Default: <output-dir>/pubchem_cache")
    ap.add_argument(
        "--sources",
        default="drugsfda",
        help="drugsfda, drugbank, or drugsfda+drugbank (DrugBank needs --drugbank-xml or --drugbank-approved-ids-json).",
    )
    ap.add_argument(
        "--combine",
        choices=("union", "intersect"),
        default="union",
        help="How to merge CID sets when both drugsfda and drugbank run.",
    )
    ap.add_argument(
        "--drugsfda-zip",
        default="",
        help="Local drugsfda json zip; if missing, download to cache from openFDA.",
    )
    ap.add_argument("--drugsfda-zip-url", default=fda_sb.DRUGSFDA_ZIP_DEFAULT)
    ap.add_argument("--drugbank-xml", default="")
    ap.add_argument("--drugbank-approved-ids-json", default="")
    ap.add_argument(
        "--max-universe-drugs",
        type=int,
        default=0,
        help="If >0, keep only this many CIDs (sorted ascending) after merge+exclude.",
    )
    ap.add_argument("--pubchem-sleep", type=float, default=0.22)
    ap.add_argument(
        "--no-default-exclude-cids",
        action="store_true",
        help="Do not apply DEFAULT_THERAPEUTIC_EXCLUDE_CIDS.",
    )
    ap.add_argument(
        "--train-drug-target-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/drug_target_map_20260331.parquet"
        ),
        help="Train drug→target parquet for subset copy (local path or s3:// copied manually).",
    )
    ap.add_argument(
        "--train-lincs-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/lincs_drug_signature_proxy_20260331.parquet"
        ),
    )
    ap.add_argument(
        "--build-lincs-stub",
        action="store_true",
        help="Write fda_approved_lincs_extended.parquet (train rows subset + zero stubs).",
    )
    ap.add_argument("--skip-smiles", action="store_true", help="Debug: skip SMILES property fetch.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "pubchem_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    name_cache_path = cache_dir / "pubchem_name_to_cid.json"
    smiles_cache_path = cache_dir / "pubchem_cid_to_smiles.json"

    sources = {s.strip().lower() for s in args.sources.replace("+", ",").split(",") if s.strip()}
    if "drugsfda+drugbank" in sources:
        sources = {"drugsfda", "drugbank"}

    cids_fda: set[int] | None = None
    cids_db: set[int] | None = None
    n_drugsfda_tokens = 0

    if "drugsfda" in sources:
        zpath = Path(args.drugsfda_zip) if args.drugsfda_zip else cache_dir / "drug-drugsfda-0001-of-0001.json.zip"
        if not zpath.is_file():
            zpath.write_bytes(fda_sb._http_bytes(args.drugsfda_zip_url))
        tokens = fda_sb.load_drugsfda_active_tokens(zpath, args.drugsfda_zip_url)
        n_drugsfda_tokens = len(tokens)
        raw_nc = load_json_cache(name_cache_path)
        name_cache: dict[str, int | None] = {}
        for k, v in raw_nc.items():
            if v is None:
                name_cache[str(k)] = None
            else:
                name_cache[str(k)] = int(v)
        early_stop = None
        if args.max_universe_drugs and args.max_universe_drugs > 0:
            # Resolve only enough names to likely fill the cap after exclude (heuristic buffer).
            early_stop = max(int(args.max_universe_drugs) * 4, int(args.max_universe_drugs) + 20)
        cids_fda = cids_from_drugsfda_tokens(
            tokens,
            name_cache,
            args.pubchem_sleep,
            stop_after_n_distinct_cids=early_stop,
        )
        save_json_cache(
            name_cache_path,
            {k: name_cache[k] for k in sorted(name_cache, key=lambda x: x.lower())},
        )

    if "drugbank" in sources:
        if args.drugbank_approved_ids_json:
            data = json.loads(Path(args.drugbank_approved_ids_json).read_text(encoding="utf-8"))
            approved = {str(x).strip().upper() for x in data if fda_sb.DB_ID_RE.match(str(x).strip().upper())}
            cids_db = cids_from_drugbank_approved(approved, args.pubchem_sleep)
        elif args.drugbank_xml:
            approved = fda_sb.parse_drugbank_approved_db_ids(Path(args.drugbank_xml))
            cids_db = cids_from_drugbank_approved(approved, args.pubchem_sleep)
        else:
            print(
                "DrugBank in --sources but no --drugbank-xml / --drugbank-approved-ids-json; "
                "using empty DrugBank CID set (union still uses Drugs@FDA if enabled)."
            )
            cids_db = set()

    if cids_fda is None and cids_db is None:
        raise SystemExit("No sources enabled.")

    if cids_fda is None:
        universe = set(cids_db or ())
    elif cids_db is None:
        universe = set(cids_fda)
    elif args.combine == "union":
        universe = set(cids_fda) | set(cids_db)
    else:
        universe = set(cids_fda) & set(cids_db)

    if not args.no_default_exclude_cids:
        universe -= set(fda_sb.DEFAULT_THERAPEUTIC_EXCLUDE_CIDS)

    n_after_exclude = len(universe)
    u_list = sorted(universe)
    if args.max_universe_drugs and args.max_universe_drugs > 0:
        u_list = u_list[: int(args.max_universe_drugs)]

    smiles_map: dict[int, str] = {}
    for k, v in load_json_cache(smiles_cache_path).items():
        smiles_map[int(k)] = str(v).strip()
    need_smiles = [c for c in u_list if c not in smiles_map]
    if not args.skip_smiles and need_smiles:
        fetched = pubchem_fetch_smiles_batch(need_smiles)
        smiles_map.update(fetched)
        save_json_cache(smiles_cache_path, {str(k): smiles_map[k] for k in sorted(smiles_map)})

    drug_rows = []
    for cid in u_list:
        drug_rows.append(
            {
                "canonical_drug_id": int(cid),
                "canonical_smiles": smiles_map.get(int(cid), ""),
            }
        )
    drug_df = pd.DataFrame(drug_rows)
    drug_path = out_dir / "fda_approved_drug_table.parquet"
    drug_df.to_parquet(drug_path, index=False)

    train_dt = Path(args.train_drug_target_uri)
    if not train_dt.is_file():
        dt_out = pd.DataFrame(columns=["canonical_drug_id", "target_gene_symbol"])
        print(f"WARNING: train drug_target not found at {train_dt}; writing empty targets parquet.")
        dt_out.to_parquet(out_dir / "fda_approved_drug_targets.parquet", index=False)
    else:
        dt_sub = merge_target_table(u_list, train_dt, "canonical_drug_id", "target_gene_symbol")
        dt_sub.to_parquet(out_dir / "fda_approved_drug_targets.parquet", index=False)

    if args.build_lincs_stub:
        train_li = Path(args.train_lincs_uri)
        if train_li.is_file():
            lex = extend_lincs_stub(u_list, train_li, "canonical_drug_id")
            lex.to_parquet(out_dir / "fda_approved_lincs_extended.parquet", index=False)
        else:
            print(f"WARNING: --build-lincs-stub but missing {train_li}")

    merged_pre = (
        (set(cids_fda or ()) | set(cids_db or ()))
        if args.combine == "union"
        else (set(cids_fda or ()) & set(cids_db or ()))
    )
    meta = {
        "output_dir": str(out_dir),
        "sources_requested": sorted(sources),
        "combine": args.combine,
        "n_drugsfda_tokens": int(n_drugsfda_tokens),
        "n_cids_from_drugsfda": len(cids_fda or ()),
        "n_cids_from_drugbank": len(cids_db or ()),
        "n_cids_merged_pre_exclude": len(merged_pre),
        "n_cids_after_default_exclude": int(n_after_exclude),
        "n_cids_final_written": len(u_list),
        "max_universe_drugs_cap": int(args.max_universe_drugs or 0),
        "default_exclude_applied": not bool(args.no_default_exclude_cids),
        "outputs": {
            "drug_table": str(drug_path),
            "drug_targets": str(out_dir / "fda_approved_drug_targets.parquet"),
        },
    }
    if args.build_lincs_stub and Path(args.train_lincs_uri).is_file():
        meta["outputs"]["lincs_extended"] = str(out_dir / "fda_approved_lincs_extended.parquet")
    (out_dir / "fda_approved_universe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
