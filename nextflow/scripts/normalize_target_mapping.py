from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize target mapping to canonical_drug_id + gene symbol.")
    p.add_argument("--drug-features-uri", required=True, help="bridge drug_features parquet (canonical_drug_id, drug_name_norm)")
    p.add_argument("--drug-target-uri", required=True, help="drugbank target parquet (Drug_Name, UniProt_ID)")
    p.add_argument("--out-parquet", required=True, help="output target map parquet")
    p.add_argument("--out-report", required=True, help="output mapping report json")
    p.add_argument("--batch-size", type=int, default=120, help="UniProt accessions per REST query")
    return p.parse_args()


def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _batch_uniprot_to_symbol(uniprot_ids: list[str], batch_size: int = 120) -> dict[str, str]:
    out: dict[str, str] = {u: "" for u in uniprot_ids}
    for i in range(0, len(uniprot_ids), batch_size):
        chunk = uniprot_ids[i : i + batch_size]
        query = " OR ".join([f"accession:{u}" for u in chunk])
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {"query": query, "fields": "accession,gene_primary", "format": "tsv", "size": str(len(chunk))}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            continue
        lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
        # header: Entry\tGene Names (primary)
        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            acc = parts[0].strip()
            gene = parts[1].strip().upper()
            if acc in out and gene:
                out[acc] = gene
    return out


def main() -> None:
    args = parse_args()
    out_parquet = Path(args.out_parquet)
    out_report = Path(args.out_report)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df_bridge = pd.read_parquet(args.drug_features_uri, columns=["canonical_drug_id", "drug_name_norm"])
    df_db = pd.read_parquet(args.drug_target_uri, columns=["Drug_Name", "UniProt_ID"])
    df_db = df_db.dropna(subset=["Drug_Name", "UniProt_ID"]).copy()

    b = df_bridge.copy()
    b["norm_name"] = b["drug_name_norm"].map(_norm_name)
    d = df_db.copy()
    d["norm_name"] = d["Drug_Name"].map(_norm_name)

    joined = b[["canonical_drug_id", "norm_name"]].drop_duplicates().merge(
        d[["norm_name", "UniProt_ID"]].drop_duplicates(),
        on="norm_name",
        how="left",
    )
    joined = joined.dropna(subset=["UniProt_ID"]).copy()

    uniprots = sorted(set(joined["UniProt_ID"].astype(str)))
    u2g = _batch_uniprot_to_symbol(uniprots, batch_size=args.batch_size)

    joined["target_gene_symbol"] = joined["UniProt_ID"].map(lambda x: u2g.get(str(x), ""))
    out = (
        joined.dropna(subset=["target_gene_symbol"])
        .query("target_gene_symbol != ''")
        [["canonical_drug_id", "target_gene_symbol"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    out.to_parquet(out_parquet, index=False)

    report = {
        "bridge_drug_count": int(df_bridge["canonical_drug_id"].nunique()),
        "candidate_uniprot_count": int(len(uniprots)),
        "mapped_uniprot_to_symbol_count": int(sum(1 for v in u2g.values() if v)),
        "final_target_rows": int(out.shape[0]),
        "final_mapped_drug_count": int(out["canonical_drug_id"].nunique()) if not out.empty else 0,
        "notes": "Drug mapping by normalized drug_name; UniProt mapped using UniProt REST gene_primary.",
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
