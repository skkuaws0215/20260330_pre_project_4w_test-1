#!/usr/bin/env python3
"""
Filter METABRIC native ensemble ranking to FDA-aligned *therapeutic* drugs (stricter than NDC
ingredient-name matching, which admits excipients and simple chemicals).

Primary mode — DrugBank *approved* (recommended):
  Parse DrugBank full-database XML (academic download) for entries whose <groups> contains
  `approved`, collect `drugbank-id` values (e.g. DB00339). For each candidate PubChem CID,
  query PUG-REST `.../compound/cid/{cid}/xrefs/RegistryID/JSON` and keep CIDs that list a
  RegistryID matching `^DB\\d{5}$` present in the approved set.

Secondary mode — Drugs@FDA (openFDA bulk JSON):
  Build a set of **active ingredient** strings from `drug-drugsfda-*.json.zip` (approved human
  drug products). Expand combination strings (`a || b`, comma lists) into tokens. Match a CID if
  **any** PubChem synonym equals a token (case-insensitive, exact). This is narrower than the
  old NDC substring heuristic but still lists some non-drug actives; pair with DrugBank for
  stricter policy (`--combine intersect`).

Combine policies:
  `any`   — pass DrugBank-approved xref OR Drugs@FDA exact token match (default when both enabled)
  `intersect` — pass only if both match (most conservative)

Legacy:
  `--source ndc` — previous OpenFDA NDC + loose PubChem synonym match (not recommended).

Outputs (default: same directory as input ranking):
  - fda_filtered_ranking.csv
  - fda_top30_shortlist.csv
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

PUBCHEM_REGISTRY_TMPL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/RegistryID/JSON"
)
PUBCHEM_SYNONYMS_TMPL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
OPENFDA_NDC_URL = "https://api.fda.gov/drug/ndc.json"
DRUGSFDA_ZIP_DEFAULT = "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip"
OPENFDA_MAX_SKIP = 25_000
OPENFDA_PAGE = 1_000
HTTP_UA = "build_fda_metabric_shortlist/2.0 (research; +https://api.fda.gov/)"
DB_ID_RE = re.compile(r"^DB\d{5}$", re.I)

# Drugs@FDA lists many non-therapeutic actives (excipients, acids, solvents). Drop known
# offenders that still match PubChem synonyms for this METABRIC native CID pool.
DEFAULT_THERAPEUTIC_EXCLUDE_CIDS: frozenset[int] = frozenset(
    {
        1004,  # phosphoric acid
        1015,  # phosphoethanolamine
        1017,  # phthalic acid
        1018,  # picolinic acid
        1021,  # porphobilinogen
        1023,  # pyrophosphoric acid
        1030,  # propylene glycol
        1031,  # 1-propanol
        1032,  # propionic acid
        1047,  # pyrazinoic acid (metabolite)
        1049,  # pyridine
    }
)


def _http_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_bytes(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --- DrugBank XML: approved DrugBank IDs ---


def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


def parse_drugbank_approved_db_ids(xml_path: Path) -> set[str]:
    """
    Stream-parse DrugBank full database XML; return uppercase DB##### for drugs with
    <groups><group>approved</group></groups>.
    Namespace-agnostic (works with xmlns http://www.drugbank.ca or no namespace).
    """
    approved: set[str] = set()
    has_approved = False
    current_primary_id: str | None = None
    in_groups = False
    in_drugbank_id = False
    id_primary_attr = False

    for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
        tag = _local_tag(elem.tag)
        if event == "start":
            if tag == "drug":
                has_approved = False
                current_primary_id = None
            elif tag == "groups":
                in_groups = True
            elif tag == "drugbank-id":
                in_drugbank_id = True
                id_primary_attr = elem.attrib.get("primary") == "true"
        else:
            if tag == "group" and in_groups:
                if (elem.text or "").strip().lower() == "approved":
                    has_approved = True
            elif tag == "groups":
                in_groups = False
            elif tag == "drugbank-id" and in_drugbank_id:
                txt = (elem.text or "").strip().upper()
                if id_primary_attr and DB_ID_RE.match(txt):
                    current_primary_id = txt
                in_drugbank_id = False
            elif tag == "drug":
                if has_approved and current_primary_id:
                    approved.add(current_primary_id)
                elem.clear()
    return approved


# --- PubChem: DrugBank ID via RegistryID ---


def pubchem_drugbank_ids_for_cid(cid: int) -> set[str]:
    url = PUBCHEM_REGISTRY_TMPL.format(cid=int(cid))
    try:
        data = _http_json(url, timeout=45.0)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return set()
        raise
    info = (data.get("InformationList") or {}).get("Information") or []
    if not info:
        return set()
    out: set[str] = set()
    for rid in info[0].get("RegistryID") or []:
        s = str(rid).strip().upper()
        if DB_ID_RE.match(s):
            out.add(s)
    return out


def load_registry_cache(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Registry cache must be JSON object cid->list: {path}")
    return {str(k): list(v) for k, v in raw.items()}


def save_registry_cache(path: Path, m: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: m[k] for k in sorted(m, key=lambda x: int(x))}, indent=0) + "\n",
        encoding="utf-8",
    )


def cids_allowed_drugbank(
    cids: list[int],
    approved_db_ids: set[str],
    reg_cache: Path | None,
    sleep_s: float,
) -> set[int]:
    cache = load_registry_cache(reg_cache) if reg_cache else {}
    allowed: set[int] = set()
    for cid in sorted(set(cids)):
        key = str(int(cid))
        ids: list[str]
        if key in cache:
            ids = cache[key]
        else:
            s = pubchem_drugbank_ids_for_cid(int(cid))
            ids = sorted(s)
            if reg_cache is not None:
                cache[key] = ids
            time.sleep(sleep_s)
        if set(ids) & approved_db_ids:
            allowed.add(int(cid))
    if reg_cache:
        save_registry_cache(reg_cache, cache)
    return allowed


# --- Drugs@FDA: active ingredient tokens ---


def _expand_drugsfda_name(name: str) -> list[str]:
    n = name.strip().lower()
    if not n:
        return []
    parts = re.split(r"\s*\|\|\s*|\s*;\s*|,\s*", n)
    return [p.strip() for p in parts if p.strip()]


def load_drugsfda_active_tokens(zip_path: Path | None, zip_url: str) -> set[str]:
    raw: bytes
    if zip_path and zip_path.is_file():
        raw = zip_path.read_bytes()
    else:
        raw = _http_bytes(zip_url)
    tokens: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if not names:
            raise SystemExit("drugsfda zip: no json inside")
        data = json.loads(zf.read(names[0]).decode("utf-8"))
    for r in data.get("results") or []:
        for prod in r.get("products") or []:
            for ai in prod.get("active_ingredients") or []:
                nm = (ai.get("name") or "").strip()
                for t in _expand_drugsfda_name(nm):
                    tokens.add(t)
    return tokens


def pubchem_synonyms(cid: int) -> list[str]:
    url = PUBCHEM_SYNONYMS_TMPL.format(cid=int(cid))
    try:
        data = _http_json(url, timeout=45.0)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    info = (data.get("InformationList") or {}).get("Information") or []
    if not info:
        return []
    return [str(s).lower() for s in info[0].get("Synonym") or []]


def load_synonyms_cache(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Synonyms cache must be JSON object cid->list: {path}")
    return {str(k): list(v) for k, v in raw.items()}


def save_synonyms_cache(path: Path, m: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: m[k] for k in sorted(m, key=lambda x: int(x))}, indent=0) + "\n",
        encoding="utf-8",
    )


def _synonym_hits_drugsfda_token(syns: list[str], tokens: set[str], mode: str) -> bool:
    """exact | prefix | relaxed (substring among longer strings; Drugs@FDA token set only)."""
    for s in syns:
        if s in tokens:
            return True
    if mode == "prefix":
        for ing in tokens:
            if len(ing) < 4:
                continue
            for syn in syns:
                if len(syn) < 6:
                    continue
                if ing == syn or ing.startswith(syn + " ") or ing.startswith(syn + ","):
                    return True
        return False
    if mode != "relaxed":
        return False
    for ing in tokens:
        if len(ing) < 5:
            continue
        for syn in syns:
            if len(syn) < 8:
                continue
            if ing == syn:
                return True
            if len(ing) >= 8 and syn in ing:
                return True
            if len(syn) >= 8 and ing in syn:
                return True
    return False


def cids_allowed_drugsfda(
    cids: list[int],
    tokens: set[str],
    syn_cache: Path | None,
    sleep_s: float,
    match_mode: str,
) -> set[int]:
    cache = load_synonyms_cache(syn_cache) if syn_cache else {}
    allowed: set[int] = set()
    for cid in sorted(set(cids)):
        key = str(int(cid))
        sy = cache.get(key)
        if sy is None:
            sy = pubchem_synonyms(int(cid))
            if syn_cache is not None:
                cache[key] = sy
            time.sleep(sleep_s)
        if _synonym_hits_drugsfda_token(sy, tokens, match_mode):
            allowed.add(int(cid))
    if syn_cache:
        save_synonyms_cache(syn_cache, cache)
    return allowed


# --- Legacy NDC ---


def fetch_ndc_active_ingredient_names() -> set[str]:
    names: set[str] = set()
    for skip in range(0, OPENFDA_MAX_SKIP + 1, OPENFDA_PAGE):
        url = f"{OPENFDA_NDC_URL}?limit={OPENFDA_PAGE}&skip={skip}"
        data = _http_json(url)
        results = data.get("results") or []
        if not results:
            break
        for res in results:
            for ai in res.get("active_ingredients") or []:
                n = (ai.get("name") or "").strip().lower()
                if n:
                    names.add(n)
        time.sleep(0.35)
    return names


def synonyms_match_ndc(syns: list[str], ndc_names: set[str]) -> bool:
    for ing in ndc_names:
        if len(ing) < 4:
            continue
        for s in syns:
            if ing == s:
                return True
            if len(ing) >= 6 and ing in s:
                return True
            if len(s) >= 6 and s in ing:
                return True
    return False


def cids_allowed_ndc(
    cids: list[int],
    ndc_names: set[str],
    syn_cache: Path | None,
    sleep_s: float,
) -> set[int]:
    cache = load_synonyms_cache(syn_cache) if syn_cache else {}
    allowed: set[int] = set()
    for cid in sorted(set(cids)):
        key = str(int(cid))
        sy = cache.get(key)
        if sy is None:
            sy = pubchem_synonyms(int(cid))
            if syn_cache is not None:
                cache[key] = sy
            time.sleep(sleep_s)
        if sy and synonyms_match_ndc(sy, ndc_names):
            allowed.add(int(cid))
    if syn_cache:
        save_synonyms_cache(syn_cache, cache)
    return allowed


def parse_extra_cids(path: Path | None, inline: str | None) -> set[int]:
    out: set[int] = set()
    if inline:
        for part in inline.split(","):
            part = part.strip()
            if part:
                out.add(int(part))
    if path and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(int(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parents[2]
    default_run = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/native_pair_features_run"
    )
    ap.add_argument(
        "--ranking-csv",
        default=str(default_run / "metabric_native_ensemble_ranking_no_lincs.csv"),
        help="Input pair-level ensemble ranking CSV.",
    )
    ap.add_argument("--output-dir", default="", help="Directory for outputs (default: ranking parent).")
    ap.add_argument("--top-k", type=int, default=30, help="Max unique drugs in shortlist.")
    ap.add_argument(
        "--cache-dir",
        default="",
        help="Cache dir for PubChem registry/synonym JSON and optional drugsfda zip copy.",
    )
    ap.add_argument(
        "--source",
        choices=("drugbank", "drugsfda", "ndc", "drugbank+drugsfda"),
        default="drugsfda",
        help="drugsfda: openFDA bulk Drugs@FDA actives + PubChem synonyms (default; no DrugBank file needed). "
        "drugbank: DrugBank XML approved + PubChem RegistryID DB#####. "
        "drugbank+drugsfda: both (see --combine). ndc: legacy NDC loose match.",
    )
    ap.add_argument(
        "--combine",
        choices=("any", "intersect"),
        default="any",
        help="When source=drugbank+drugsfda: any=union of passes, intersect=both required.",
    )
    ap.add_argument(
        "--drugbank-xml",
        default="",
        help="Path to DrugBank full database XML (e.g. drugbank_all_full.xml). Required for drugbank* sources.",
    )
    ap.add_argument(
        "--drugbank-approved-ids-json",
        default="",
        help="Optional: precomputed JSON list of DB##### strings; skips XML parse if provided.",
    )
    ap.add_argument(
        "--drugsfda-zip",
        default="",
        help="Local path to drug-drugsfda-*.json.zip; if missing, download from --drugsfda-zip-url.",
    )
    ap.add_argument(
        "--drugsfda-zip-url",
        default=DRUGSFDA_ZIP_DEFAULT,
        help="URL for Drugs@FDA bulk JSON zip (used when --drugsfda-zip not set).",
    )
    ap.add_argument(
        "--drugsfda-match",
        choices=("exact", "prefix", "relaxed"),
        default="relaxed",
        help="Synonym vs Drugs@FDA active token only (not full NDC). relaxed=substring len≥8 (stricter than legacy NDC).",
    )
    ap.add_argument(
        "--ndc-ingredients-json",
        default="",
        help="(ndc only) Optional JSON list of ingredient strings.",
    )
    ap.add_argument("--pubchem-sleep", type=float, default=0.22, help="Delay between PubChem calls.")
    ap.add_argument("--extra-allow-cids", default="", help="Comma-separated CIDs always allowed.")
    ap.add_argument("--extra-allow-cids-file", default="", help="One CID per line.")
    ap.add_argument(
        "--no-default-exclude-cids",
        action="store_true",
        help="Do not subtract DEFAULT_THERAPEUTIC_EXCLUDE_CIDS (debug only).",
    )
    ap.add_argument("--exclude-cids", default="", help="Comma-separated CIDs to remove after allowlist.")
    ap.add_argument("--exclude-cids-file", default="", help="One CID per line to remove.")
    args = ap.parse_args()

    ranking_path = Path(args.ranking_csv)
    if not ranking_path.is_file():
        raise SystemExit(f"Missing ranking CSV: {ranking_path}")

    out_dir = Path(args.output_dir) if args.output_dir else ranking_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "fda_filter_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    reg_cache = cache_dir / "pubchem_registry_drugbank_ids_by_cid.json"
    syn_cache = cache_dir / "pubchem_synonyms_by_cid.json"

    df = pd.read_csv(ranking_path)
    need = ["sample_id", "canonical_drug_id", "rank", "ensemble_score"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"Ranking CSV missing column {c!r}")

    cids = df["canonical_drug_id"].astype(int).tolist()
    uniq = sorted(set(cids))
    extra = parse_extra_cids(
        Path(args.extra_allow_cids_file) if args.extra_allow_cids_file else None,
        args.extra_allow_cids or None,
    )

    src = args.source
    definition_bits: list[str] = []
    allowed: set[int] = set()

    if src == "ndc":
        ndc_path = Path(args.ndc_ingredients_json) if args.ndc_ingredients_json else cache_dir / "fda_ndc_ingredients.json"
        if ndc_path.is_file():
            ndc_names = {str(x).strip().lower() for x in json.loads(ndc_path.read_text()) if str(x).strip()}
        else:
            ndc_names = fetch_ndc_active_ingredient_names()
            ndc_path.write_text(json.dumps(sorted(ndc_names), indent=0) + "\n", encoding="utf-8")
        allowed = cids_allowed_ndc(uniq, ndc_names, syn_cache, args.pubchem_sleep)
        definition_bits.append("legacy OpenFDA NDC + loose PubChem synonym match")

    elif src == "drugsfda":
        zpath = Path(args.drugsfda_zip) if args.drugsfda_zip else cache_dir / "drug-drugsfda-0001-of-0001.json.zip"
        if not zpath.is_file():
            zpath.write_bytes(_http_bytes(args.drugsfda_zip_url))
        tokens = load_drugsfda_active_tokens(zpath, args.drugsfda_zip_url)
        allowed = cids_allowed_drugsfda(uniq, tokens, syn_cache, args.pubchem_sleep, args.drugsfda_match)
        definition_bits.append(
            f"Drugs@FDA active ingredients (PubChem synonym {args.drugsfda_match}), n_tokens={len(tokens)}"
        )

    elif src == "drugbank":
        if args.drugbank_approved_ids_json:
            data = json.loads(Path(args.drugbank_approved_ids_json).read_text())
            approved_db = {str(x).strip().upper() for x in data if DB_ID_RE.match(str(x).strip().upper())}
        else:
            if not args.drugbank_xml:
                raise SystemExit("--drugbank-xml or --drugbank-approved-ids-json required for --source drugbank")
            xml_path = Path(args.drugbank_xml)
            if not xml_path.is_file():
                raise SystemExit(f"Missing DrugBank XML: {xml_path}")
            approved_db = parse_drugbank_approved_db_ids(xml_path)
            (cache_dir / "drugbank_approved_db_ids.json").write_text(
                json.dumps(sorted(approved_db), indent=0) + "\n", encoding="utf-8"
            )
        allowed = cids_allowed_drugbank(uniq, approved_db, reg_cache, args.pubchem_sleep)
        definition_bits.append(
            f"DrugBank XML <group>approved</group> (n={len(approved_db)} DB ids) + PubChem RegistryID DB##### xref"
        )

    else:
        # drugbank+drugsfda
        if args.drugbank_approved_ids_json:
            data = json.loads(Path(args.drugbank_approved_ids_json).read_text())
            approved_db = {str(x).strip().upper() for x in data if DB_ID_RE.match(str(x).strip().upper())}
        else:
            if not args.drugbank_xml:
                raise SystemExit("--drugbank-xml or --drugbank-approved-ids-json required for drugbank+drugsfda")
            xml_path = Path(args.drugbank_xml)
            if not xml_path.is_file():
                raise SystemExit(f"Missing DrugBank XML: {xml_path}")
            approved_db = parse_drugbank_approved_db_ids(xml_path)
            (cache_dir / "drugbank_approved_db_ids.json").write_text(
                json.dumps(sorted(approved_db), indent=0) + "\n", encoding="utf-8"
            )
        a_db = cids_allowed_drugbank(uniq, approved_db, reg_cache, args.pubchem_sleep)
        zpath = Path(args.drugsfda_zip) if args.drugsfda_zip else cache_dir / "drug-drugsfda-0001-of-0001.json.zip"
        if not zpath.is_file():
            zpath.write_bytes(_http_bytes(args.drugsfda_zip_url))
        tokens = load_drugsfda_active_tokens(zpath, args.drugsfda_zip_url)
        (cache_dir / "drugsfda_active_tokens.count.txt").write_text(str(len(tokens)), encoding="utf-8")
        a_fda = cids_allowed_drugsfda(uniq, tokens, syn_cache, args.pubchem_sleep, args.drugsfda_match)
        if args.combine == "any":
            allowed = a_db | a_fda
        else:
            allowed = a_db & a_fda
        definition_bits.append(
            f"DrugBank approved (n_db={len(approved_db)}) {args.combine} Drugs@FDA ({args.drugsfda_match}, n_tok={len(tokens)})"
        )

    allowed |= extra

    exclude: set[int] = set()
    if not args.no_default_exclude_cids:
        exclude |= set(DEFAULT_THERAPEUTIC_EXCLUDE_CIDS)
    exclude |= parse_extra_cids(
        Path(args.exclude_cids_file) if args.exclude_cids_file else None,
        args.exclude_cids or None,
    )
    allowed -= exclude

    filt = df[df["canonical_drug_id"].astype(int).isin(allowed)].copy()
    filt = filt.sort_values("rank", ascending=True, kind="mergesort")

    filtered_path = out_dir / "fda_filtered_ranking.csv"
    filt.to_csv(filtered_path, index=False)

    per_drug = (
        filt.sort_values("rank", ascending=True, kind="mergesort")
        .groupby("canonical_drug_id", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    per_drug = per_drug.sort_values("rank", ascending=True, kind="mergesort").head(int(args.top_k))
    per_drug = per_drug.reset_index(drop=True)
    per_drug.insert(0, "fda_drug_rank", range(1, len(per_drug) + 1))
    per_drug = per_drug.rename(columns={"rank": "best_pair_rank"})
    shortlist_path = out_dir / "fda_top30_shortlist.csv"
    per_drug.to_csv(shortlist_path, index=False)

    meta = {
        "ranking_csv": str(ranking_path),
        "source": src,
        "combine": args.combine if src == "drugbank+drugsfda" else None,
        "n_input_rows": int(len(df)),
        "n_filtered_rows": int(len(filt)),
        "n_unique_drugs_input": int(df["canonical_drug_id"].nunique()),
        "n_unique_drugs_filtered": int(filt["canonical_drug_id"].nunique()),
        "shortlist_k": int(args.top_k),
        "shortlist_n": int(len(per_drug)),
        "filter_definition": "; ".join(definition_bits),
        "default_exclude_cids_applied": not bool(args.no_default_exclude_cids),
        "n_exclude_cids": int(len(exclude)),
        "outputs": {"filtered": str(filtered_path), "shortlist": str(shortlist_path)},
    }
    (out_dir / "fda_filter_run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
