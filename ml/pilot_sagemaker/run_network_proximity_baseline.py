"""
Network Proximity baseline (rule-based): drug–target ↔ fixed disease gene set
on a bipartite graph (drug, gene) + optional PPI gene–gene edges.

Reuses the same row order as ML/DL and the same CV fold indices from
model_selection_stage1/cv_fold_indices.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import numpy as np
import pandas as pd

from graph_baseline_data import load_cv_meta, load_drug_targets_dict, load_merged_pair_frame
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score

INF = 10**9


def _resolve_out_path(out_dir: Path, name: str | None, default_name: str) -> Path:
    if not name:
        return out_dir / default_name
    p = Path(name)
    return p if p.is_absolute() else out_dir / p


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Network proximity baseline with fixed CV folds.")
    p.add_argument(
        "--labels-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet"
        ),
        help="Local labels (n=14497 when merged with pathway_addon pair features); override with s3:// if needed.",
    )
    p.add_argument(
        "--features-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet"
        ),
    )
    p.add_argument(
        "--drug-target-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/drug_target_map_20260331.parquet"
        ),
        help="Parquet with canonical_drug_id, target_gene_symbol (repo-local default; use s3:// if needed).",
    )
    p.add_argument(
        "--disease-genes-path",
        default=str(repo / "data/graph_baseline/disease_genes_common_v1.txt"),
        help="Text file: one gene symbol per line (# comments allowed).",
    )
    p.add_argument(
        "--ppi-edges-uri",
        default="",
        help="Optional TSV/CSV with columns gene_a,gene_b (symbols). If missing or empty, PPI skipped.",
    )
    p.add_argument(
        "--cv-fold-json",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/model_selection_stage1/cv_fold_indices.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1"),
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--target-gene-col", default="target_gene_symbol")
    p.add_argument("--null-draws", type=int, default=80, help="Monte Carlo draws for z-score null.")
    p.add_argument("--null-seed", type=int, default=42)
    p.add_argument(
        "--comparison-csv",
        default=None,
        help="Output CSV for Network_Proximity rows only (default: graph_family_comparison.csv in out-dir).",
    )
    p.add_argument("--schema-json", default=None, help="Schema JSON path (default: graph_schema.json).")
    p.add_argument("--summary-json", default=None, help="Summary JSON path (default: graph_family_summary.json).")
    p.add_argument(
        "--omit-summary-json",
        action="store_true",
        help="Do not write graph_family_summary.json (e.g. group CV; merge script writes final summary).",
    )
    return p.parse_args()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(s) if s is not None and pd.notna(s) else float("nan")


def rank_metrics(df_valid: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tmp = df_valid[[sample_id_col]].copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    ndcgs, hits = [], []
    for _, g in tmp.groupby(sample_id_col):
        if len(g) < 2:
            continue
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        m = float(np.min(yt))
        if m < 0:
            yt = yt - m
        k = min(20, len(g))
        ndcgs.append(float(ndcg_score(yt.reshape(1, -1), yp.reshape(1, -1), k=k)))
        top_t = set(np.argsort(-yt)[:k].tolist())
        top_p = set(np.argsort(-yp)[:k].tolist())
        hits.append(1.0 if top_t.intersection(top_p) else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else float("nan"), float(np.mean(hits)) if hits else float("nan")


def fold_metrics(df_va: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ndcg, hit = rank_metrics(df_va, sample_id_col, y_true, pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "Spearman": safe_spearman(y_true, pred),
        "NDCG@20": ndcg,
        "Hit@20": hit,
    }


def load_disease_genes(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.upper())
    return sorted(set(out))


def build_adjacency(
    drug_targets: dict[str, set[str]],
    disease_genes: set[str],
    ppi_pairs: list[tuple[str, str]] | None,
) -> tuple[dict[str, list[str]], set[str], set[str]]:
    """Undirected adjacency: nodes 'D:{drug}' and 'G:{gene}' (uppercase gene)."""
    adj: dict[str, list[str]] = {}
    genes_in_graph: set[str] = set()
    drugs_in_graph: set[str] = set()

    def add_edge(a: str, b: str) -> None:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    for did, tset in drug_targets.items():
        dnode = f"D:{did}"
        drugs_in_graph.add(did)
        for g in tset:
            g = g.upper().strip()
            if not g:
                continue
            genes_in_graph.add(g)
            gnode = f"G:{g}"
            add_edge(dnode, gnode)

    for g in disease_genes:
        genes_in_graph.add(g.upper().strip())

    if ppi_pairs:
        for a, b in ppi_pairs:
            a, b = a.upper().strip(), b.upper().strip()
            if not a or not b:
                continue
            genes_in_graph.add(a)
            genes_in_graph.add(b)
            add_edge(f"G:{a}", f"G:{b}")

    for g in disease_genes:
        gn = f"G:{g.upper().strip()}"
        if gn not in adj:
            adj[gn] = []

    return adj, genes_in_graph, drugs_in_graph


def multi_source_shortest_hops(adj: dict[str, list[str]], sources: list[str]) -> dict[str, int]:
    dist: dict[str, int] = {}
    dq = deque()
    for s in sources:
        if s not in adj:
            continue
        if s not in dist:
            dist[s] = 0
            dq.append(s)
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def min_target_to_disease_hops(
    dist_from_disease: dict[str, int],
    target_genes: set[str],
) -> int:
    best = INF
    for g in target_genes:
        gn = f"G:{g.upper()}"
        d = dist_from_disease.get(gn)
        if d is not None and d < best:
            best = d
    return best


def drug_proximity_z(
    adj: dict[str, list[str]],
    gene_nodes: list[str],
    disease_gene_nodes: list[str],
    drug_targets: dict[str, set[str]],
    n_draws: int,
    rng: random.Random,
) -> dict[str, float]:
    """
    For each drug: obs = min hops from any target gene to nearest disease gene.
    z = (mean_null - obs) / std_null with null = random gene sets of same size as disease set
    (larger z => closer than random). Only n_draws BFS total for nulls.
    """
    k = len(disease_gene_nodes)
    if k == 0:
        return {d: 0.0 for d in drug_targets}

    true_sources = [f"G:{g}" for g in disease_gene_nodes]
    true_sources = [s for s in true_sources if s in adj]
    if not true_sources:
        return {d: 0.0 for d in drug_targets}

    d_true = multi_source_shortest_hops(adj, true_sources)

    obs: dict[str, float] = {}
    for did, tg in drug_targets.items():
        h = min_target_to_disease_hops(d_true, tg)
        obs[did] = float(h if h < INF else INF)

    z_out: dict[str, float] = {}
    pool = [n for n in gene_nodes if n.startswith("G:")]
    if len(pool) < k:
        for did in drug_targets:
            z_out[did] = 0.0
        return z_out

    null_maps: list[dict[str, int]] = []
    for _ in range(n_draws):
        sample = rng.sample(pool, k)
        null_maps.append(multi_source_shortest_hops(adj, sample))

    for did, tg in drug_targets.items():
        if not tg:
            z_out[did] = 0.0
            continue
        nulls = [float(min_target_to_disease_hops(dm, tg)) for dm in null_maps]
        mu = float(np.mean(nulls))
        sigma = float(np.std(nulls, ddof=0))
        o = obs[did]
        if sigma < 1e-12:
            z_out[did] = 0.0
        else:
            z_out[did] = (mu - o) / sigma
    return z_out


def try_load_ppi(uri: str) -> list[tuple[str, str]] | None:
    if not uri or not uri.strip():
        return None
    p = Path(uri)
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    for a, b in [("gene_a", "gene_b"), ("g1", "g2"), ("a", "b")]:
        if a in cols and b in cols:
            ca, cb = cols[a], cols[b]
            out = []
            for x, y in zip(df[ca].astype(str), df[cb].astype(str)):
                if pd.isna(x) or pd.isna(y):
                    continue
                out.append((x.strip(), y.strip()))
            return out
    return None


def main() -> None:
    args = parse_args()
    rng = random.Random(args.null_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disease_raw = load_disease_genes(Path(args.disease_genes_path))
    disease_set = set(disease_raw)

    df_merged, _feat_cols = load_merged_pair_frame(
        args.labels_uri,
        args.features_uri,
        args.sample_id_col,
        args.drug_id_col,
        args.target_col,
    )
    key = [args.sample_id_col, args.drug_id_col]
    df = df_merged[key + [args.target_col]].copy()

    cv_meta = load_cv_meta(Path(args.cv_fold_json), len(df))

    drug_targets = load_drug_targets_dict(args.drug_target_uri, args.drug_id_col, args.target_gene_col)

    ppi_pairs = try_load_ppi(args.ppi_edges_uri.strip())
    adj, genes_in_graph, _ = build_adjacency(drug_targets, disease_set, ppi_pairs)

    gene_nodes = sorted({n for n in adj if n.startswith("G:")})
    disease_in_graph = sorted({g for g in disease_set if f"G:{g}" in adj})

    z_by_drug = drug_proximity_z(
        adj,
        gene_nodes,
        disease_in_graph,
        drug_targets,
        n_draws=args.null_draws,
        rng=rng,
    )

    pred_series = df[args.drug_id_col].astype(str).map(lambda d: z_by_drug.get(d, 0.0))
    pred = pred_series.fillna(0.0).to_numpy(dtype=np.float64)
    y = df[args.target_col].to_numpy(dtype=np.float64)

    fold_rows: list[dict[str, Any]] = []
    spearmans: list[float] = []
    for fold_info in cv_meta["folds"]:
        va_idx = np.array(fold_info["valid_indices"], dtype=int)
        tr_idx = np.array(fold_info["train_indices"], dtype=int)
        df_va = df.iloc[va_idx].reset_index(drop=True)
        y_tr, p_tr = y[tr_idx], pred[tr_idx]
        y_va, p_va = y[va_idx], pred[va_idx]
        # Affine map raw drug z-scores to label scale using train fold only (Spearman invariant if slope > 0).
        A = np.vstack([p_tr, np.ones(len(p_tr))]).T
        slope, intercept = np.linalg.lstsq(A, y_tr, rcond=None)[0]
        if not np.isfinite(slope) or abs(slope) > 80.0 or float(np.std(p_tr)) < 1e-12:
            slope, intercept = 0.0, float(np.mean(y_tr))
        p_va_cal = slope * p_va + intercept
        if not np.all(np.isfinite(p_va_cal)):
            p_va_cal = np.full_like(p_va, float(np.mean(y_tr)), dtype=np.float64)
        rmse = float(np.sqrt(mean_squared_error(y_va, p_va_cal)))
        if rmse > 25.0 * max(float(np.std(y_va)), 1e-6):
            p_va_cal = np.full_like(p_va, float(np.mean(y_tr)), dtype=np.float64)
            rmse = float(np.sqrt(mean_squared_error(y_va, p_va_cal)))
            mae_v = float(mean_absolute_error(y_va, p_va_cal))
        else:
            mae_v = float(mean_absolute_error(y_va, p_va_cal))
        ndcg, hit = rank_metrics(df_va, args.sample_id_col, y_va, p_va)
        sp = safe_spearman(y_va, p_va)
        m = {"RMSE": rmse, "MAE": mae_v, "Spearman": sp, "NDCG@20": ndcg, "Hit@20": hit}
        spearmans.append(m["Spearman"])
        fold_rows.append({"model": "Network_Proximity", "fold": fold_info["fold"], **m})

    mean_row = {
        "model": "Network_Proximity",
        "fold": "mean",
        "RMSE": float(np.mean([r["RMSE"] for r in fold_rows])),
        "MAE": float(np.mean([r["MAE"] for r in fold_rows])),
        "Spearman": float(np.nanmean([r["Spearman"] for r in fold_rows])),
        "NDCG@20": float(np.nanmean([r["NDCG@20"] for r in fold_rows])),
        "Hit@20": float(np.nanmean([r["Hit@20"] for r in fold_rows])),
    }
    std_row = {
        "model": "Network_Proximity",
        "fold": "spearman_std",
        "RMSE": float("nan"),
        "MAE": float("nan"),
        "Spearman": float(np.nanstd(spearmans, ddof=0)),
        "NDCG@20": float("nan"),
        "Hit@20": float("nan"),
    }

    comp = pd.DataFrame(fold_rows + [mean_row, std_row])
    comp_path = _resolve_out_path(out_dir, args.comparison_csv, "graph_family_comparison.csv")
    comp.to_csv(comp_path, index=False)

    schema = {
        "version": "graph_baseline_round1",
        "nodes": ["drug", "gene"],
        "node_id_convention": {"drug": "D:{canonical_drug_id}", "gene": "G:{GENE_SYMBOL_UPPER}"},
        "edges": ["drug–target (undirected)"] + (["PPI gene–gene (undirected)"] if ppi_pairs else []),
        "ppi_loaded": bool(ppi_pairs),
        "disease_genes_file": str(Path(args.disease_genes_path).resolve()),
        "disease_genes_list_size": len(disease_raw),
        "disease_genes_in_graph": len(disease_in_graph),
        "n_drugs_with_targets": len(drug_targets),
        "n_gene_nodes_in_adj": len(gene_nodes),
        "n_edges_undirected_pairs": sum(len(v) for v in adj.values()) // 2,
        "cv_fold_json": str(Path(args.cv_fold_json).resolve()),
        "n_rows_merged": len(df),
        "null_draws": args.null_draws,
        "null_seed": args.null_seed,
        "proximity_calibration": "Per fold: OLS on train for RMSE/MAE on valid (a*raw_z+b). Spearman/NDCG/Hit use raw drug z-scores (label-scale affine can flip slope and distort ranks).",
    }
    schema_path = _resolve_out_path(out_dir, args.schema_json, "graph_schema.json")
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    if not args.omit_summary_json:
        summary_path = _resolve_out_path(out_dir, args.summary_json, "graph_family_summary.json")
        summary = {
            "baseline": "Network_Proximity",
            "metrics_mean_5fold": {k: mean_row[k] for k in ["RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]},
            "spearman_std_across_folds": std_row["Spearman"],
            "graph_schema_path": str(schema_path),
            "comparison_csv": str(comp_path),
            "notes": "Rule-based, non-sample-specific graph baseline: prediction is drug-level proximity z-score (same value for all pair rows sharing a drug). After GraphSAGE/GCN, run merge_graph_family_outputs.py for a single comparison table and representative selection.",
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"wrote": str(out_dir), "comparison": str(comp_path)}, indent=2))


if __name__ == "__main__":
    main()
