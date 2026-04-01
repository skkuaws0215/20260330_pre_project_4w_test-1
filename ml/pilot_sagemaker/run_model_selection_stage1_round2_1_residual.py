from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

import run_model_selection_stage1 as s1
from run_model_selection_stage1_round2 import prepare_scaled_round2, train_residual_with_clip


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Stage-1 Round2.1: residual-only micro tuning (3-4 configs).")
    p.add_argument(
        "--labels-uri",
        default="s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/abc_inputs/20260330_abc_v1/B/labels.parquet",
    )
    p.add_argument(
        "--features-uri",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet"
        ),
    )
    p.add_argument(
        "--stage1-dir",
        default=str(repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/model_selection_stage1"),
    )
    p.add_argument(
        "--round2-dir",
        default=str(repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/model_selection_stage1_round2"),
    )
    p.add_argument(
        "--out-dir",
        default=str(repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/model_selection_stage1_round2_1"),
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-epochs", type=int, default=45)
    p.add_argument("--torch-batch", type=int, default=256)
    return p.parse_args()


def residual_round2_1_configs() -> list[dict[str, Any]]:
    return [
        # Round2 best neighborhood (hidden=128, pathway_zscore, wd=1e-3, lr=1e-4)
        {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "pathway_zscore", "patience": 5, "grad_clip": None},
        {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "pathway_zscore", "patience": 5, "grad_clip": 1.0},
        {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "pathway_zscore", "patience": 5, "grad_clip": 1.0},
        {"hidden_dim": 256, "num_layers": 3, "dropout": 0.2, "lr": 3e-4, "weight_decay": 5e-4, "pathway_norm": "raw", "patience": 5, "grad_clip": 1.0},
    ]


def load_stage_best(csv_path: Path) -> dict[str, float]:
    d = pd.read_csv(csv_path).sort_values(by=["spearman_mean", "rmse_mean"], ascending=[False, True]).iloc[0]
    return {
        "spearman_mean": float(d["spearman_mean"]),
        "spearman_std": float(d["spearman_std"]),
        "rmse_mean": float(d["rmse_mean"]),
        "val_minus_train_mse_mean": float(d["val_minus_train_mse_mean"]),
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    df = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + [args.target_col])]
    df = df[key + [args.target_col] + feat_cols].copy().sort_values(key).reset_index(drop=True)
    X_raw = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)
    path_idx, _, _, _ = s1.assign_feature_blocks(feat_cols)
    in_dim = X_raw.shape[1]
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    rows: list[dict[str, Any]] = []
    cfgs = residual_round2_1_configs()
    for i, cfg in enumerate(cfgs, start=1):
        fold_s, fold_r, fold_a, gaps = [], [], [], []
        for fold, (tr, va) in enumerate(kf.split(X_raw), start=1):
            X_tr, X_va = prepare_scaled_round2(X_raw, tr, va, path_idx, cfg["pathway_norm"])
            y_tr, y_va = y[tr], y[va]
            ytr_t = torch.from_numpy(y_tr.astype(np.float32))
            yv_t = torch.from_numpy(y_va.astype(np.float32))
            xtr = torch.from_numpy(X_tr.astype(np.float32))
            xv = torch.from_numpy(X_va.astype(np.float32))
            model = s1.ResidualMLP(in_dim, int(cfg["hidden_dim"]), int(cfg["num_layers"]), float(cfg["dropout"]))
            ds = torch.utils.data.TensorDataset(xtr, ytr_t)
            ld = torch.utils.data.DataLoader(ds, batch_size=args.torch_batch, shuffle=True)
            pred, tlog = train_residual_with_clip(
                model,
                ld,
                xv,
                yv_t,
                epochs=args.torch_epochs,
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
                patience=int(cfg["patience"]),
                seed=args.seed + fold + i * 100,
                grad_clip=cfg["grad_clip"],
            )
            fold_s.append(s1.safe_spearman(y_va, pred))
            fold_r.append(float(np.sqrt(mean_squared_error(y_va, pred))))
            fold_a.append(float(mean_absolute_error(y_va, pred)))
            gaps.append(float(tlog["val_minus_train_mse"]))
        rows.append(
            {
                "config_id": i,
                "spearman_mean": float(np.nanmean(fold_s)),
                "spearman_std": float(np.nanstd(fold_s, ddof=0)),
                "rmse_mean": float(np.mean(fold_r)),
                "rmse_std": float(np.std(fold_r, ddof=0)),
                "mae_mean": float(np.mean(fold_a)),
                "val_minus_train_mse_mean": float(np.mean(gaps)),
                "params_json": json.dumps(cfg, sort_keys=True, default=str),
            }
        )

    out_dir = out_root / "residual_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(rows).sort_values(
        by=["spearman_mean", "spearman_std", "val_minus_train_mse_mean", "rmse_mean"],
        ascending=[False, True, True, True],
    )
    res_df.to_csv(out_dir / "cv_results.csv", index=False)
    best = res_df.iloc[0].to_dict()
    (out_dir / "best_config.json").write_text(json.dumps(json.loads(best["params_json"]), indent=2), encoding="utf-8")
    (out_dir / "training_log.json").write_text(json.dumps({"family": "residual_mlp_round2_1", "rows": rows}, indent=2), encoding="utf-8")

    s1_best = load_stage_best(Path(args.stage1_dir) / "residual_mlp" / "cv_results.csv")
    r2_best = load_stage_best(Path(args.round2_dir) / "residual_mlp" / "cv_results.csv")
    xgb_r2 = load_stage_best(Path(args.round2_dir) / "xgboost" / "cv_results.csv")

    summary = [
        "# Model selection stage 1 round2.1 (Residual only)",
        "",
        "ResidualMLP micro-search (4 configs) around Round2 best for score recovery while keeping overfitting control.",
        "",
        f"- Best config: `{best['params_json']}`",
        f"- Round2.1 best: Spearman={best['spearman_mean']:.6f}, std={best['spearman_std']:.6f}, RMSE={best['rmse_mean']:.6f}, MAE={best['mae_mean']:.6f}, gap={best['val_minus_train_mse_mean']:.6f}",
        f"- vs Stage1 best: dSpearman={best['spearman_mean']-s1_best['spearman_mean']:+.6f}, dRMSE={best['rmse_mean']-s1_best['rmse_mean']:+.6f}, dStd={best['spearman_std']-s1_best['spearman_std']:+.6f}, dGap={best['val_minus_train_mse_mean']-s1_best['val_minus_train_mse_mean']:+.6f}",
        f"- vs Round2 best: dSpearman={best['spearman_mean']-r2_best['spearman_mean']:+.6f}, dRMSE={best['rmse_mean']-r2_best['rmse_mean']:+.6f}, dStd={best['spearman_std']-r2_best['spearman_std']:+.6f}, dGap={best['val_minus_train_mse_mean']-r2_best['val_minus_train_mse_mean']:+.6f}",
        "",
        "## DL final criteria check",
        f"- Spearman close to/exceeds XGBoost Round2 ({xgb_r2['spearman_mean']:.6f}): {'YES' if best['spearman_mean'] >= xgb_r2['spearman_mean'] - 0.002 else 'NO'}",
        f"- fold std stable: {'YES' if best['spearman_std'] <= r2_best['spearman_std'] + 1e-12 else 'NO'}",
        f"- train-val gap not excessive: {'YES' if best['val_minus_train_mse_mean'] <= 0.05 else 'NO'}",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_root), "best_spearman": float(best["spearman_mean"])}, indent=2))


if __name__ == "__main__":
    main()
