"""Build drug-group CV folds, run Proximity + GraphSAGE + GCN, merge to graph_family_groupcv_*."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1"
    p = argparse.ArgumentParser(description="Full graph baselines under GroupKFold(canonical_drug_id).")
    p.add_argument("--skip-build-cv", action="store_true", help="Reuse existing cv_fold_indices_drug_group.json")
    p.add_argument(
        "--cv-fold-json",
        default=str(out_dir / "cv_fold_indices_drug_group.json"),
    )
    p.add_argument("--out-dir", default=str(out_dir))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    sm = repo / "ml" / "pilot_sagemaker"
    py = sys.executable
    out_dir = Path(args.out_dir)
    cv_json = Path(args.cv_fold_json)

    if not args.skip_build_cv:
        subprocess.run(
            [py, str(sm / "build_cv_fold_indices_drug_group.py"), "--out-json", str(cv_json)],
            cwd=str(repo),
            check=True,
        )

    common = [
        py,
        str(sm / "run_network_proximity_baseline.py"),
        "--cv-fold-json",
        str(cv_json),
        "--out-dir",
        str(out_dir),
        "--comparison-csv",
        "graph_family_groupcv_comparison.csv",
        "--schema-json",
        "graph_schema_groupcv.json",
        "--omit-summary-json",
    ]
    subprocess.run(common, cwd=str(repo), check=True)

    for model, name in [("sage", "graph_gnn_sage_groupcv_partial.csv"), ("gcn", "graph_gnn_gcn_groupcv_partial.csv")]:
        subprocess.run(
            [
                py,
                str(sm / "run_graph_gnn_cv.py"),
                "--model",
                model,
                "--cv-fold-json",
                str(cv_json),
                "--out-dir",
                str(out_dir),
                "--partial-csv",
                name,
            ],
            cwd=str(repo),
            check=True,
        )

    subprocess.run(
        [py, str(sm / "merge_graph_family_outputs.py"), "--preset", "groupcv", "--out-dir", str(out_dir)],
        cwd=str(repo),
        check=True,
    )


if __name__ == "__main__":
    main()
