"""
Run small GCN hyperparameter grid on drug-group CV; write comparison CSV + summary JSON.

Primary metric: Spearman mean (fold mean row). Meaningful improvement vs baseline A: delta >= 0.01.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parents[1]


def parse_args() -> argparse.Namespace:
    out = (
        _REPO
        / "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1"
    )
    cv = out / "cv_fold_indices_drug_group.json"
    p = argparse.ArgumentParser(description="GCN drug-group CV mini tuning (A–D).")
    p.add_argument("--out-dir", default=str(out))
    p.add_argument("--cv-fold-json", default=str(cv))
    p.add_argument("--comparison-csv", default="gcn_tuning_comparison.csv")
    p.add_argument("--summary-json", default="gcn_tuning_summary.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only aggregate existing gcn_tune_*_groupcv_partial.csv files.",
    )
    return p.parse_args()


CONFIGS: list[dict[str, Any]] = [
    {"id": "A", "hidden_dim": 64, "lr": 1e-3, "weight_decay": 1e-5},
    {"id": "B", "hidden_dim": 64, "lr": 3e-4, "weight_decay": 1e-5},
    {"id": "C", "hidden_dim": 128, "lr": 1e-3, "weight_decay": 1e-5},
    {"id": "D", "hidden_dim": 64, "lr": 1e-3, "weight_decay": 1e-4},
]


def run_one(repo: Path, cv_json: Path, out_dir: Path, seed: int, cfg: dict[str, Any]) -> None:
    script = _PKG / "run_graph_gnn_cv.py"
    partial = f"gcn_tune_{cfg['id']}_groupcv_partial.csv"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        "gcn",
        "--cv-fold-json",
        str(cv_json),
        "--out-dir",
        str(out_dir),
        "--partial-csv",
        partial,
        "--hidden-dim",
        str(cfg["hidden_dim"]),
        "--lr",
        str(cfg["lr"]),
        "--weight-decay",
        str(cfg["weight_decay"]),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, cwd=str(repo), check=True)


def _rel_repo(repo: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def read_partial_metrics(out_dir: Path, cfg_id: str) -> dict[str, float]:
    p = out_dir / f"gcn_tune_{cfg_id}_groupcv_partial.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    mean_row = df[df["fold"] == "mean"].iloc[0]
    std_row = df[df["fold"] == "spearman_std"].iloc[0]
    return {
        "RMSE_mean": float(mean_row["RMSE"]),
        "MAE_mean": float(mean_row["MAE"]),
        "Spearman_mean": float(mean_row["Spearman"]),
        "NDCG@20_mean": float(mean_row["NDCG@20"]),
        "Hit@20_mean": float(mean_row["Hit@20"]),
        "Spearman_std_across_folds": float(std_row["Spearman"]),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_json = Path(args.cv_fold_json)
    repo = _REPO

    if not args.skip_run:
        for cfg in CONFIGS:
            print(f"=== GCN tune config {cfg['id']} ===", flush=True)
            run_one(repo, cv_json, out_dir, args.seed, cfg)

    rows: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        m = read_partial_metrics(out_dir, cfg["id"])
        rows.append(
            {
                "config": cfg["id"],
                "hidden_dim": cfg["hidden_dim"],
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "Spearman_mean": m["Spearman_mean"],
                "Spearman_std_folds": m["Spearman_std_across_folds"],
                "RMSE_mean": m["RMSE_mean"],
                "MAE_mean": m["MAE_mean"],
                "NDCG@20_mean": m["NDCG@20_mean"],
                "Hit@20_mean": m["Hit@20_mean"],
                "partial_csv": _rel_repo(repo, out_dir / f"gcn_tune_{cfg['id']}_groupcv_partial.csv"),
            }
        )

    cmp_df = pd.DataFrame(rows)
    cmp_path = Path(args.comparison_csv)
    if not cmp_path.is_absolute():
        cmp_path = out_dir / cmp_path.name
    cmp_df.to_csv(cmp_path, index=False)

    baseline_sp = float(cmp_df.loc[cmp_df["config"] == "A", "Spearman_mean"].iloc[0])
    best_idx = cmp_df["Spearman_mean"].idxmax()
    best_row = cmp_df.loc[best_idx]
    deltas = {str(r["config"]): float(r["Spearman_mean"]) - baseline_sp for _, r in cmp_df.iterrows()}
    meaningful = [k for k, d in deltas.items() if k != "A" and d >= 0.01]

    summary: dict[str, Any] = {
        "preset": "gcn_groupcv_mini_tuning",
        "cv_type": "GroupKFold by canonical_drug_id (same as graph_family_groupcv)",
        "cv_fold_json": _rel_repo(repo, cv_json),
        "primary_metric": "Spearman_mean",
        "meaningful_spearman_delta_vs_baseline_A": 0.01,
        "baseline_config": "A",
        "baseline_spearman_mean": baseline_sp,
        "best_config_by_spearman_mean": str(best_row["config"]),
        "best_spearman_mean": float(best_row["Spearman_mean"]),
        "spearman_mean_delta_vs_A_by_config": deltas,
        "meaningful_improvement_over_A_configs": meaningful,
        "configs": rows,
        "comparison_csv": _rel_repo(repo, cmp_path),
        "notes": (
            "Early stopping uses validation MSE per fold, not Spearman. "
            "Interpret Spearman_mean with Spearman_std_folds; delta >= 0.01 vs A is team rule for meaningful gain."
        ),
    }
    sum_path = Path(args.summary_json)
    if not sum_path.is_absolute():
        sum_path = out_dir / sum_path.name
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"wrote": str(cmp_path), "summary": str(sum_path), "best": summary["best_config_by_spearman_mean"]}, indent=2))


if __name__ == "__main__":
    main()
