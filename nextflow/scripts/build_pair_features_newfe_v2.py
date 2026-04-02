from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LOGGER = logging.getLogger("build_pair_features_newfe_v2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build pair-level new FE baseline: pathway(sample) + chemistry(drug) + LINCS(pair) + target(pair)."
        )
    )
    p.add_argument("--run-id", required=True, help="Run ID, ex) 20260331")
    p.add_argument("--pairs-uri", required=True, help="Parquet with at least sample_id, canonical_drug_id")
    p.add_argument("--sample-expression-uri", required=True, help="Wide expression/signature parquet keyed by sample_id")
    p.add_argument("--drug-uri", required=True, help="Drug parquet with canonical_drug_id and canonical_smiles")
    p.add_argument("--lincs-drug-signature-uri", required=True, help="Wide LINCS drug signature parquet keyed by canonical_drug_id")
    p.add_argument("--drug-target-uri", required=True, help="Parquet with canonical_drug_id,target_gene_symbol (1:N)")
    p.add_argument("--pathway-gmt", default="", help="Optional GMT file path for pathway scoring")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory. ex) results/features_nextflow_team4/fe_re_batch_runs/20260331",
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--smiles-col", default="canonical_smiles")
    p.add_argument("--target-gene-col", default="target_gene_symbol")
    p.add_argument("--pathway-name-col", default="pathway_name")
    p.add_argument("--high-z-threshold", type=float, default=1.0)
    p.add_argument("--low-z-threshold", type=float, default=-1.0)
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-nbits", type=int, default=2048)
    p.add_argument("--reverse-topk-small", type=int, default=50)
    p.add_argument("--reverse-topk-large", type=int, default=100)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _read_parquet(uri: str) -> pd.DataFrame:
    return pd.read_parquet(uri)


def _parse_gmt(path: str) -> dict[str, set[str]]:
    gmt: dict[str, set[str]] = {}
    if not path:
        return gmt
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pathway = parts[0]
            genes = {str(g).strip().upper() for g in parts[2:] if str(g).strip()}
            if genes:
                gmt[pathway] = genes
    return gmt


def build_sample_pathway_features(
    sample_expr_df: pd.DataFrame,
    sample_id_col: str,
    gmt_map: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr_cols = [c for c in sample_expr_df.columns if c != sample_id_col]
    expr_cols = [c for c in expr_cols if pd.api.types.is_numeric_dtype(sample_expr_df[c])]

    if not gmt_map:
        return sample_expr_df[[sample_id_col]].copy(), pd.DataFrame(columns=["pathway_name", "gene_symbol"])

    # Map HGNC-like symbol (from GMT) to expression column (e.g. crispr__TP53 -> TP53).
    symbol_to_col: dict[str, str] = {}
    for c in expr_cols:
        sym = str(c).split("__")[-1].strip().upper()
        if sym and sym not in symbol_to_col:
            symbol_to_col[sym] = c

    pathways: dict[str, np.ndarray] = {}
    members: list[dict[str, str]] = []
    for pathway, genes in gmt_map.items():
        kept_cols: list[str] = []
        for g in genes:
            col = symbol_to_col.get(str(g).strip().upper())
            if col is not None:
                kept_cols.append(col)
        kept_cols = sorted(set(kept_cols))
        if not kept_cols:
            continue
        for col in kept_cols:
            sym = str(col).split("__")[-1].strip().upper()
            members.append({"pathway_name": pathway, "gene_symbol": sym})
        pathways[pathway] = sample_expr_df[kept_cols].to_numpy(dtype=np.float32)

    out = sample_expr_df[[sample_id_col]].copy()
    for pathway, mat in pathways.items():
        out[f"pathway__{pathway}"] = np.nanmean(mat, axis=1)
    member_df = pd.DataFrame(members)
    return out, member_df


def build_drug_chem_features(
    drug_df: pd.DataFrame,
    drug_id_col: str,
    smiles_col: str,
    radius: int,
    nbits: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors
    except Exception as e:
        raise RuntimeError("RDKit is required for chemistry features. Please install rdkit.") from e

    required = [drug_id_col, smiles_col]
    missing = [c for c in required if c not in drug_df.columns]
    if missing:
        raise ValueError(f"drug input missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    invalid_smiles = 0
    descriptor_names = {
        "drug_desc_mol_wt": Descriptors.MolWt,
        "drug_desc_logp": Descriptors.MolLogP,
        "drug_desc_tpsa": Descriptors.TPSA,
        "drug_desc_hbd": Descriptors.NumHDonors,
        "drug_desc_hba": Descriptors.NumHAcceptors,
        "drug_desc_rot_bonds": Descriptors.NumRotatableBonds,
        "drug_desc_ring_count": Descriptors.RingCount,
        "drug_desc_heavy_atoms": Descriptors.HeavyAtomCount,
        "drug_desc_frac_csp3": Descriptors.FractionCSP3,
    }

    seen = set()
    for _, row in drug_df[[drug_id_col, smiles_col]].drop_duplicates(subset=[drug_id_col]).iterrows():
        did = str(row[drug_id_col]).strip()
        if did in seen:
            continue
        seen.add(did)
        smi = "" if pd.isna(row[smiles_col]) else str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        rec: dict[str, Any] = {drug_id_col: did, "drug_has_valid_smiles": 1 if mol else 0}

        if mol is None:
            invalid_smiles += 1
            for i in range(nbits):
                rec[f"drug_morgan_{i:04d}"] = 0
            for dn in descriptor_names:
                rec[dn] = np.nan
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
            arr = np.zeros((nbits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            for i in range(nbits):
                rec[f"drug_morgan_{i:04d}"] = int(arr[i])
            for dn, fn in descriptor_names.items():
                rec[dn] = float(fn(mol))
        rows.append(rec)

    out = pd.DataFrame(rows)
    qc = {
        "drug_rows": int(out.shape[0]),
        "invalid_smiles_count": int(invalid_smiles),
        "invalid_smiles_ratio": float(invalid_smiles / max(len(rows), 1)),
    }
    return out, qc


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    v = spearmanr(a, b).statistic
    if v is None or np.isnan(v):
        return 0.0
    return float(v)


def _reverse_score_topk(sample_vec: np.ndarray, drug_vec: np.ndarray, k: int) -> float:
    k = min(k, sample_vec.shape[0])
    if k <= 0:
        return 0.0
    idx = np.argpartition(np.abs(sample_vec), -k)[-k:]
    return float(-np.mean(sample_vec[idx] * drug_vec[idx]))


def build_pair_lincs_features(
    pairs_df: pd.DataFrame,
    sample_expr_df: pd.DataFrame,
    lincs_drug_df: pd.DataFrame,
    sample_id_col: str,
    drug_id_col: str,
    topk_small: int,
    topk_large: int,
) -> pd.DataFrame:
    sample_numeric = [c for c in sample_expr_df.columns if c != sample_id_col and pd.api.types.is_numeric_dtype(sample_expr_df[c])]
    drug_numeric = [c for c in lincs_drug_df.columns if c != drug_id_col and pd.api.types.is_numeric_dtype(lincs_drug_df[c])]
    common = sorted(set(sample_numeric).intersection(drug_numeric))
    if not common:
        raise ValueError("No common numeric columns between sample expression and LINCS drug signature.")

    sample_index = sample_expr_df[[sample_id_col] + common].drop_duplicates(subset=[sample_id_col]).set_index(sample_id_col)
    drug_index = lincs_drug_df[[drug_id_col] + common].drop_duplicates(subset=[drug_id_col]).set_index(drug_id_col)

    out_rows: list[dict[str, Any]] = []
    for _, pair in pairs_df[[sample_id_col, drug_id_col]].iterrows():
        sid = str(pair[sample_id_col]).strip()
        did = str(pair[drug_id_col]).strip()
        rec: dict[str, Any] = {sample_id_col: sid, drug_id_col: did}
        if sid not in sample_index.index or did not in drug_index.index:
            rec["lincs_cosine"] = 0.0
            rec["lincs_pearson"] = 0.0
            rec["lincs_spearman"] = 0.0
            rec[f"lincs_reverse_score_top{topk_small}"] = 0.0
            rec[f"lincs_reverse_score_top{topk_large}"] = 0.0
            out_rows.append(rec)
            continue

        sv = sample_index.loc[sid, common].to_numpy(dtype=np.float32)
        dv = drug_index.loc[did, common].to_numpy(dtype=np.float32)

        rec["lincs_cosine"] = _cosine(sv, dv)
        rec["lincs_pearson"] = _pearson(sv, dv)
        rec["lincs_spearman"] = _spearman(sv, dv)
        rec[f"lincs_reverse_score_top{topk_small}"] = _reverse_score_topk(sv, dv, topk_small)
        rec[f"lincs_reverse_score_top{topk_large}"] = _reverse_score_topk(sv, dv, topk_large)
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def build_target_features(
    pairs_df: pd.DataFrame,
    sample_expr_df: pd.DataFrame,
    drug_target_df: pd.DataFrame,
    sample_pathway_df: pd.DataFrame,
    pathway_member_df: pd.DataFrame,
    sample_id_col: str,
    drug_id_col: str,
    target_gene_col: str,
    high_z: float,
    low_z: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [drug_id_col, target_gene_col]
    missing = [c for c in required if c not in drug_target_df.columns]
    if missing:
        raise ValueError(f"drug target input missing required columns: {missing}")

    expr_cols = [c for c in sample_expr_df.columns if c != sample_id_col and pd.api.types.is_numeric_dtype(sample_expr_df[c])]
    # Handle prefixed expression columns (e.g., crispr__EGFR) by keeping both raw and tokenized symbol map.
    expr_symbol_to_col: dict[str, str] = {}
    for c in expr_cols:
        sym = c.split("__")[-1].strip().upper()
        if sym and sym not in expr_symbol_to_col:
            expr_symbol_to_col[sym] = c
    expr_symbol_set = set(expr_symbol_to_col.keys())

    sample_expr_idx = sample_expr_df[[sample_id_col] + expr_cols].drop_duplicates(subset=[sample_id_col]).set_index(sample_id_col)
    sample_high: dict[str, set[str]] = {}
    sample_low: dict[str, set[str]] = {}
    for sid, row in sample_expr_idx.iterrows():
        vals = row.to_numpy(dtype=np.float32)
        high_mask = vals >= high_z
        low_mask = vals <= low_z
        sample_high[str(sid)] = {g.split("__")[-1].strip().upper() for g, m in zip(expr_cols, high_mask) if m}
        sample_low[str(sid)] = {g.split("__")[-1].strip().upper() for g, m in zip(expr_cols, low_mask) if m}

    drug_targets: dict[str, set[str]] = {}
    for did, grp in drug_target_df[[drug_id_col, target_gene_col]].dropna().groupby(drug_id_col):
        tset = {str(g).strip().upper() for g in grp[target_gene_col].tolist() if str(g).strip()}
        drug_targets[str(did).strip()] = tset

    pathway_score_cols = [c for c in sample_pathway_df.columns if c.startswith("pathway__")]
    sample_pathway_idx = (
        sample_pathway_df[[sample_id_col] + pathway_score_cols]
        .drop_duplicates(subset=[sample_id_col])
        .set_index(sample_id_col)
        if pathway_score_cols
        else pd.DataFrame()
    )
    pathway_map: dict[str, set[str]] = {}
    if not pathway_member_df.empty:
        for pname, grp in pathway_member_df.groupby("pathway_name"):
            pathway_map[str(pname)] = {str(g).strip().upper() for g in grp["gene_symbol"].tolist()}

    drug_target_pathways: dict[str, list[str]] = {}
    if pathway_map:
        for did, tset in drug_targets.items():
            hits = []
            for pname, genes in pathway_map.items():
                if tset.intersection(genes):
                    col = f"pathway__{pname}"
                    if col in pathway_score_cols:
                        hits.append(col)
            drug_target_pathways[did] = hits

    rows: list[dict[str, Any]] = []
    target_missing_drug_count = 0
    for _, pair in pairs_df[[sample_id_col, drug_id_col]].iterrows():
        sid = str(pair[sample_id_col]).strip()
        did = str(pair[drug_id_col]).strip()
        targets = drug_targets.get(did, set())
        tcount = len(targets)
        if tcount == 0:
            target_missing_drug_count += 1

        sample_high_genes = sample_high.get(sid, set())
        sample_low_genes = sample_low.get(sid, set())
        overlap_up = len(targets.intersection(sample_high_genes))
        overlap_down = len(targets.intersection(sample_low_genes))

        in_expr = targets.intersection(expr_symbol_set)
        coverage = float(len(in_expr) / tcount) if tcount > 0 else 0.0

        expr_mean = 0.0
        expr_std = 0.0
        if sid in sample_expr_idx.index and in_expr:
            expr_cols_hit = [expr_symbol_to_col[g] for g in sorted(in_expr) if g in expr_symbol_to_col]
            vals = sample_expr_idx.loc[sid, expr_cols_hit].to_numpy(dtype=np.float32)
            expr_mean = float(np.mean(vals))
            expr_std = float(np.std(vals))

        pmean = 0.0
        phit = 0
        if did in drug_target_pathways and sid in sample_pathway_idx.index:
            cols = drug_target_pathways[did]
            if cols:
                vals = sample_pathway_idx.loc[sid, cols].to_numpy(dtype=np.float32)
                vals = vals[~np.isnan(vals)]
                phit = int(vals.size)
                if vals.size > 0:
                    pmean = float(np.mean(vals))

        rows.append(
            {
                sample_id_col: sid,
                drug_id_col: did,
                "target_overlap_count": int(overlap_up),
                "target_overlap_ratio": float(overlap_up / tcount) if tcount > 0 else 0.0,
                "target_overlap_down_count": int(overlap_down),
                "target_overlap_down_ratio": float(overlap_down / tcount) if tcount > 0 else 0.0,
                "target_expr_mean": expr_mean,
                "target_expr_std": expr_std,
                "target_pathway_score_mean": pmean,
                "target_pathway_hit_count": int(phit),
                "target_gene_coverage_ratio": coverage,
                "target_gene_count": int(tcount),
            }
        )

    out = pd.DataFrame(rows)
    qc = {
        "target_missing_drug_count": int(target_missing_drug_count),
        "target_missing_drug_ratio": float(target_missing_drug_count / max(len(rows), 1)),
        "target_overlap_count_summary": out["target_overlap_count"].describe().to_dict(),
        "target_expr_mean_summary": out["target_expr_mean"].describe().to_dict(),
        "target_feature_null_ratio": {c: float(out[c].isna().mean()) for c in out.columns if c.startswith("target_")},
    }
    return out, qc


def _summary(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out[c] = {k: float(v) for k, v in df[c].describe().to_dict().items()}
    return out


def build_pair_features_newfe_v2_from_frames(
    pairs_df: pd.DataFrame,
    sample_expr_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    lincs_drug_df: pd.DataFrame,
    drug_target_df: pd.DataFrame,
    *,
    pathway_gmt: str,
    sample_id_col: str = "sample_id",
    drug_id_col: str = "canonical_drug_id",
    smiles_col: str = "canonical_smiles",
    target_gene_col: str = "target_gene_symbol",
    high_z_threshold: float = 1.0,
    low_z_threshold: float = -1.0,
    morgan_radius: int = 2,
    morgan_nbits: int = 2048,
    reverse_topk_small: int = 50,
    reverse_topk_large: int = 100,
    include_pair_lincs: bool = True,
) -> dict[str, Any]:
    """
    Same merge logic as main(): pathway + drug chem + [LINCS] + target -> pair_features_newfe_v2.
    Inputs must already satisfy column requirements (validated by caller or main).
    If include_pair_lincs is False, LINCS block is skipped (matches training without LINCS features).
    """
    pairs_df = pairs_df[[sample_id_col, drug_id_col]].drop_duplicates().copy()
    pairs_df[sample_id_col] = pairs_df[sample_id_col].astype(str).str.strip()
    pairs_df[drug_id_col] = pairs_df[drug_id_col].astype(str).str.strip()
    sample_expr_df = sample_expr_df.copy()
    sample_expr_df[sample_id_col] = sample_expr_df[sample_id_col].astype(str).str.strip()
    drug_df = drug_df.copy()
    drug_df[drug_id_col] = drug_df[drug_id_col].astype(str).str.strip()
    if include_pair_lincs:
        lincs_drug_df = lincs_drug_df.copy()
        lincs_drug_df[drug_id_col] = lincs_drug_df[drug_id_col].astype(str).str.strip()
    drug_target_df = drug_target_df.copy()
    drug_target_df[drug_id_col] = drug_target_df[drug_id_col].astype(str).str.strip()

    gmt_map = _parse_gmt(pathway_gmt)
    sample_pathway_df, pathway_member_df = build_sample_pathway_features(
        sample_expr_df=sample_expr_df,
        sample_id_col=sample_id_col,
        gmt_map=gmt_map,
    )

    drug_chem_df, chem_qc = build_drug_chem_features(
        drug_df=drug_df,
        drug_id_col=drug_id_col,
        smiles_col=smiles_col,
        radius=morgan_radius,
        nbits=morgan_nbits,
    )

    if include_pair_lincs:
        pair_lincs_df = build_pair_lincs_features(
            pairs_df=pairs_df,
            sample_expr_df=sample_expr_df,
            lincs_drug_df=lincs_drug_df,
            sample_id_col=sample_id_col,
            drug_id_col=drug_id_col,
            topk_small=reverse_topk_small,
            topk_large=reverse_topk_large,
        )
    else:
        pair_lincs_df = pd.DataFrame(columns=[sample_id_col, drug_id_col])

    pair_target_df, target_qc = build_target_features(
        pairs_df=pairs_df,
        sample_expr_df=sample_expr_df,
        drug_target_df=drug_target_df,
        sample_pathway_df=sample_pathway_df,
        pathway_member_df=pathway_member_df,
        sample_id_col=sample_id_col,
        drug_id_col=drug_id_col,
        target_gene_col=target_gene_col,
        high_z=high_z_threshold,
        low_z=low_z_threshold,
    )

    pair_features_newfe = (
        pairs_df.merge(sample_pathway_df, on=sample_id_col, how="left")
        .merge(drug_chem_df, on=drug_id_col, how="left")
    )
    if include_pair_lincs:
        pair_features_newfe = pair_features_newfe.merge(
            pair_lincs_df, on=[sample_id_col, drug_id_col], how="left"
        )
    pair_features_newfe_v2 = pair_features_newfe.merge(
        pair_target_df,
        on=[sample_id_col, drug_id_col],
        how="left",
    )

    for df in (pair_features_newfe, pair_features_newfe_v2, pair_target_df):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df[num_cols] = df[num_cols].fillna(0.0)

    return {
        "pairs_df": pairs_df,
        "sample_pathway_df": sample_pathway_df,
        "pathway_member_df": pathway_member_df,
        "drug_chem_df": drug_chem_df,
        "pair_lincs_df": pair_lincs_df,
        "pair_target_df": pair_target_df,
        "pair_features_newfe": pair_features_newfe,
        "pair_features_newfe_v2": pair_features_newfe_v2,
        "chem_qc": chem_qc,
        "target_qc": target_qc,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading inputs")
    pairs_df = _read_parquet(args.pairs_uri)
    sample_expr_df = _read_parquet(args.sample_expression_uri)
    drug_df = _read_parquet(args.drug_uri)
    lincs_drug_df = _read_parquet(args.lincs_drug_signature_uri)
    drug_target_df = _read_parquet(args.drug_target_uri)

    for name, df, cols in [
        ("pairs", pairs_df, [args.sample_id_col, args.drug_id_col]),
        ("sample_expression", sample_expr_df, [args.sample_id_col]),
        ("drug", drug_df, [args.drug_id_col, args.smiles_col]),
        ("lincs_drug_signature", lincs_drug_df, [args.drug_id_col]),
    ]:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ValueError(f"{name} missing required columns: {miss}")

    LOGGER.info("Building merged pair features (pathway + chem + LINCS + target)")
    built = build_pair_features_newfe_v2_from_frames(
        pairs_df,
        sample_expr_df,
        drug_df,
        lincs_drug_df,
        drug_target_df,
        pathway_gmt=args.pathway_gmt,
        sample_id_col=args.sample_id_col,
        drug_id_col=args.drug_id_col,
        smiles_col=args.smiles_col,
        target_gene_col=args.target_gene_col,
        high_z_threshold=args.high_z_threshold,
        low_z_threshold=args.low_z_threshold,
        morgan_radius=args.morgan_radius,
        morgan_nbits=args.morgan_nbits,
        reverse_topk_small=args.reverse_topk_small,
        reverse_topk_large=args.reverse_topk_large,
    )
    pairs_df = built["pairs_df"]
    sample_pathway_df = built["sample_pathway_df"]
    pathway_member_df = built["pathway_member_df"]
    drug_chem_df = built["drug_chem_df"]
    pair_lincs_df = built["pair_lincs_df"]
    pair_target_df = built["pair_target_df"]
    pair_features_newfe = built["pair_features_newfe"]
    pair_features_newfe_v2 = built["pair_features_newfe_v2"]
    chem_qc = built["chem_qc"]
    target_qc = built["target_qc"]

    out_sample_pathway = out_dir / "sample_pathway_features.parquet"
    out_drug_chem = out_dir / "drug_chem_features.parquet"
    out_pair_lincs = out_dir / "pair_lincs_features.parquet"
    out_pair_target = out_dir / "pair_target_features.parquet"
    out_pair_newfe = out_dir / "pair_features_newfe.parquet"
    out_pair_newfe_v2 = out_dir / "pair_features_newfe_v2.parquet"
    out_manifest = out_dir / "feature_manifest.json"

    sample_pathway_df.to_parquet(out_sample_pathway, index=False)
    drug_chem_df.to_parquet(out_drug_chem, index=False)
    pair_lincs_df.to_parquet(out_pair_lincs, index=False)
    pair_target_df.to_parquet(out_pair_target, index=False)
    pair_features_newfe.to_parquet(out_pair_newfe, index=False)
    pair_features_newfe_v2.to_parquet(out_pair_newfe_v2, index=False)

    manifest = {
        "run_id": args.run_id,
        "purpose": "new FE baseline (pathway + Morgan/descriptor + LINCS) + target interaction extension",
        "keys": {"sample_id_col": args.sample_id_col, "drug_id_col": args.drug_id_col},
        "feature_groups": {
            "sample_pathway": {
                "enabled": True,
                "pathway_count": int(max(sample_pathway_df.shape[1] - 1, 0)),
                "gmt_provided": bool(args.pathway_gmt),
            },
            "drug_chem": {
                "enabled": True,
                "morgan_radius": args.morgan_radius,
                "morgan_nbits": args.morgan_nbits,
                "invalid_smiles_policy": "all-zero fingerprint + NaN descriptors then numeric fillna(0.0) at final merge",
                "invalid_smiles_qc": chem_qc,
            },
            "pair_lincs": {
                "enabled": True,
                "metrics": [
                    "lincs_cosine",
                    "lincs_pearson",
                    "lincs_spearman",
                    f"lincs_reverse_score_top{args.reverse_topk_small}",
                    f"lincs_reverse_score_top{args.reverse_topk_large}",
                ],
            },
            "pair_target": {
                "enabled": True,
                "columns": [
                    "target_overlap_count",
                    "target_overlap_ratio",
                    "target_overlap_down_count",
                    "target_overlap_down_ratio",
                    "target_expr_mean",
                    "target_expr_std",
                    "target_pathway_score_mean",
                    "target_pathway_hit_count",
                    "target_gene_coverage_ratio",
                    "target_gene_count",
                ],
                "definition": {
                    "high_gene_rule": f"zscore >= {args.high_z_threshold}",
                    "low_gene_rule": f"zscore <= {args.low_z_threshold}",
                    "missing_target_policy": "if no target mapping for drug -> all target_* features set to 0.0",
                    "overlap_ratio_rule": "overlap_count / target_gene_count (0 if target_gene_count=0)",
                },
                "qc": target_qc,
            },
        },
        "row_counts": {
            "pairs": int(pairs_df.shape[0]),
            "pair_features_newfe_rows": int(pair_features_newfe.shape[0]),
            "pair_features_newfe_v2_rows": int(pair_features_newfe_v2.shape[0]),
        },
        "outputs": {
            "sample_pathway_features": str(out_sample_pathway),
            "drug_chem_features": str(out_drug_chem),
            "pair_lincs_features": str(out_pair_lincs),
            "pair_target_features": str(out_pair_target),
            "pair_features_newfe": str(out_pair_newfe),
            "pair_features_newfe_v2": str(out_pair_newfe_v2),
        },
        "distribution_summary": {
            "pair_target": _summary(
                pair_target_df,
                [
                    "target_overlap_count",
                    "target_overlap_ratio",
                    "target_overlap_down_count",
                    "target_overlap_down_ratio",
                    "target_expr_mean",
                    "target_expr_std",
                    "target_pathway_score_mean",
                    "target_pathway_hit_count",
                    "target_gene_coverage_ratio",
                    "target_gene_count",
                ],
            )
        },
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Done. Outputs written to %s", out_dir)
    LOGGER.info("target 없는 drug 비율: %.4f", target_qc["target_missing_drug_ratio"])
    LOGGER.info("overlap count 분포: %s", json.dumps(target_qc["target_overlap_count_summary"], ensure_ascii=False))
    LOGGER.info("target_expr_mean 분포: %s", json.dumps(target_qc["target_expr_mean_summary"], ensure_ascii=False))
    LOGGER.info("target feature null 비율: %s", json.dumps(target_qc["target_feature_null_ratio"], ensure_ascii=False))


if __name__ == "__main__":
    main()
