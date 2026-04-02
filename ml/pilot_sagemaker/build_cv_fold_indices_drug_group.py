"""Build 5-fold CV indices grouped by canonical_drug_id (no drug appears in both train and val)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import numpy as np
from sklearn.model_selection import GroupKFold

from graph_baseline_data import load_merged_pair_frame


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Write cv_fold_indices JSON for GroupKFold by drug id.")
    p.add_argument(
        "--labels-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet"
        ),
    )
    p.add_argument(
        "--features-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet"
        ),
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/graph_baseline_round1/cv_fold_indices_drug_group.json"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df, _ = load_merged_pair_frame(
        args.labels_uri,
        args.features_uri,
        args.sample_id_col,
        args.drug_id_col,
        args.target_col,
    )
    n = len(df)
    groups = df[args.drug_id_col].astype(str).to_numpy()
    X = np.zeros((n, 1))
    y = np.zeros(n)
    gkf = GroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    folds: list[dict] = []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        tr_d = set(groups[tr].tolist())
        va_d = set(groups[va].tolist())
        if tr_d.intersection(va_d):
            raise SystemExit(f"Fold {k}: train/val drug overlap {tr_d.intersection(va_d)}")
        folds.append(
            {
                "fold": k,
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
                "train_indices": tr.astype(int).tolist(),
                "valid_indices": va.astype(int).tolist(),
            }
        )

    out = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "n_rows": n,
        "cv_type": "group_drug",
        "group_col": args.drug_id_col,
        "n_unique_drugs": int(len(np.unique(groups))),
        "folds": folds,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "n_rows": n, "n_drugs": out["n_unique_drugs"]}, indent=2))


if __name__ == "__main__":
    main()
