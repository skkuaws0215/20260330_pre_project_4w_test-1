#!/usr/bin/env python3
"""
Prepare METABRIC-native FE inputs:

1) metabric_matrix_gene_order.csv — one row per matrix row, same order as parquet rows.
   Default: remap Entrez_Gene_Id -> HGNC via mygene.info (batch POST) for pathway/LINCS alignment.
   Matrix must include columns Hugo_Symbol, Entrez_Gene_Id (as in team4 54_filtered.parquet).

2) metabric_gene_order_summary.json — row-count checks, duplicate stats, remap coverage.

3) metabric_pairs.parquet — cross join MB-* sample_id x canonical_drug_id from labels.

Requires network for --remap-mygene (default).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _ensure_local(uri: str, tdp: Path, fname: str) -> Path:
    uri = uri.strip()
    if uri.startswith("s3://"):
        dst = tdp / fname
        _run(["aws", "s3", "cp", uri, str(dst)])
        return dst
    return Path(uri)


def mygene_entrez_to_symbol_batch(entrez_ids: list[int], chunk: int = 400) -> dict[int, str]:
    """Batch query mygene.info; returns entrez -> symbol (uppercase)."""
    url = "https://mygene.info/v3/query"
    out: dict[int, str] = {}
    for i in range(0, len(entrez_ids), chunk):
        batch = entrez_ids[i : i + chunk]
        q = ",".join(str(int(x)) for x in batch)
        body = f"q={q}&scopes=entrezgene&fields=symbol&species=human".encode()
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                rows = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise SystemExit(f"mygene HTTP error: {e}") from e
        for row in rows:
            q = row.get("query")
            sym = row.get("symbol")
            if q is None or not sym:
                continue
            try:
                eid = int(str(q).strip())
            except ValueError:
                continue
            out[eid] = str(sym).strip().upper()
        time.sleep(0.15)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metabric-matrix-uri",
        default="s3://drug-discovery-joe-raw-data-team4/8/metabric/54_filtered.parquet",
    )
    ap.add_argument(
        "--labels-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet",
        help="Used to list canonical_drug_id for pair cross-join.",
    )
    ap.add_argument(
        "--output-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/native_inputs",
    )
    ap.add_argument(
        "--remap-mygene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map Entrez_Gene_Id to HGNC symbol via mygene.info (recommended).",
    )
    ap.add_argument(
        "--mygene-chunk-size",
        type=int,
        default=400,
        help="Entrez IDs per mygene POST batch.",
    )
    ap.add_argument(
        "--max-metabric-samples",
        type=int,
        default=0,
        help="If >0, keep only this many MB samples (sorted by id) for pairs.",
    )
    ap.add_argument(
        "--max-drugs",
        type=int,
        default=0,
        help="If >0, keep only this many drugs (sorted) for pairs.",
    )
    args = ap.parse_args()

    out_dir = _resolve(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as td:
        tdp = Path(td)
        mpath = _ensure_local(args.metabric_matrix_uri, tdp, "metabric.parquet")
        lpath = _ensure_local(args.labels_uri, tdp, "labels.parquet")

        df = pd.read_parquet(mpath)
        n_rows = int(len(df))

        for col in ("Hugo_Symbol", "Entrez_Gene_Id"):
            if col not in df.columns:
                raise SystemExit(
                    f"METABRIC matrix missing {col!r}; cannot build gene order from this parquet."
                )

        mb_cols = [c for c in df.columns if str(c).startswith("MB-")]
        hugo_raw = df["Hugo_Symbol"].astype(str).str.strip()
        entrez = pd.to_numeric(df["Entrez_Gene_Id"], errors="coerce")

        summary: dict = {
            "metabric_matrix_uri": args.metabric_matrix_uri,
            "n_matrix_rows": n_rows,
            "n_mb_sample_columns": len(mb_cols),
            "hugo_symbol_na": int(hugo_raw.isna().sum() + (hugo_raw == "").sum()),
            "hugo_symbol_duplicated": int(hugo_raw.duplicated().sum()),
            "entrez_na": int(entrez.isna().sum()),
            "remap_mygene": bool(args.remap_mygene),
        }

        row_index = np.arange(n_rows, dtype=np.int64)
        gene_symbol_out: list[str] = []

        if args.remap_mygene:
            eids = []
            for v in entrez:
                if pd.isna(v):
                    eids.append(None)
                else:
                    eids.append(int(v))
            unique_e = sorted({e for e in eids if e is not None})
            emap = mygene_entrez_to_symbol_batch(unique_e, chunk=args.mygene_chunk_size)
            summary["mygene_unique_entrez_queried"] = len(unique_e)
            summary["mygene_symbols_resolved"] = len(emap)
            for i, e in enumerate(eids):
                if e is not None and e in emap:
                    gene_symbol_out.append(emap[e])
                else:
                    gene_symbol_out.append(str(hugo_raw.iloc[i]).strip().upper())
        else:
            gene_symbol_out = hugo_raw.str.upper().tolist()

        order_df = pd.DataFrame(
            {
                "row_index": row_index,
                "gene_symbol": gene_symbol_out,
                "hugo_symbol_as_in_matrix": hugo_raw.str.upper().tolist(),
                "entrez_gene_id": entrez,
            }
        )
        gcsv = out_dir / "metabric_matrix_gene_order.csv"
        order_df.to_csv(gcsv, index=False)

        summary["gene_csv_path"] = str(gcsv)
        summary["gene_symbol_na_after_build"] = int(order_df["gene_symbol"].isna().sum())
        summary["gene_symbol_duplicated_after_build"] = int(order_df["gene_symbol"].duplicated().sum())
        summary["row_count_matches_matrix"] = int(len(order_df)) == n_rows
        summary["first_5_rows"] = order_df.head(5).to_dict(orient="records")
        summary["last_5_rows"] = order_df.tail(5).to_dict(orient="records")

        labels = pd.read_parquet(lpath, columns=["canonical_drug_id"])
        drugs = sorted(labels["canonical_drug_id"].astype(str).str.strip().unique())
        if args.max_drugs > 0:
            drugs = drugs[: args.max_drugs]
        mb_use = sorted(mb_cols)
        if args.max_metabric_samples > 0:
            mb_use = mb_use[: args.max_metabric_samples]

        pairs = pd.MultiIndex.from_product([mb_use, drugs], names=["sample_id", "canonical_drug_id"]).to_frame(
            index=False
        )
        ppath = out_dir / "metabric_pairs.parquet"
        pairs.to_parquet(ppath, index=False)

        summary["pairs_parquet_path"] = str(ppath)
        summary["pairs_n_rows"] = int(len(pairs))
        summary["pairs_n_mb_samples"] = len(mb_use)
        summary["pairs_n_drugs"] = len(drugs)
        summary["max_metabric_samples_cap"] = int(args.max_metabric_samples)
        summary["max_drugs_cap"] = int(args.max_drugs)

    (out_dir / "metabric_gene_order_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in summary if k not in ("first_5_rows", "last_5_rows")}, indent=2))


if __name__ == "__main__":
    main()
