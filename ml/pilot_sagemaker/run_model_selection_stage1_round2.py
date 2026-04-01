from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import run_model_selection_stage1 as s1

try:
    import xgboost as xgb
except ImportError as e:  # pragma: no cover
    raise SystemExit("xgboost is required. pip install xgboost") from e


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="Stage-1 Round2 narrow tuning (overfitting mitigation + generalization)."
    )
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
        "--out-dir",
        default=str(repo / "results/features_nextflow_team4/fe_re_batch_runs/20260331/model_selection_stage1_round2"),
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-epochs", type=int, default=45)
    p.add_argument("--torch-batch", type=int, default=256)
    p.add_argument("--torch-patience", type=int, default=5)
    return p.parse_args()


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Spearman": s1.safe_spearman(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def prepare_scaled_round2(
    X_raw: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    path_idx: list[int],
    norm_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    X_tr = X_raw[tr_idx].copy()
    X_va = X_raw[va_idx].copy()
    if norm_mode == "raw":
        return X_tr, X_va
    cont_idx, _ = s1.cont_binary_indices(X_tr)
    s1.scale_pathway_only(X_tr, X_va, path_idx, cont_idx)
    return X_tr, X_va


def scale_full_round2(X: np.ndarray, path_idx: list[int], norm_mode: str) -> np.ndarray:
    out = np.asarray(X, dtype=np.float32).copy()
    if norm_mode == "raw":
        return out
    cont_idx, _ = s1.cont_binary_indices(out)
    to_scale = [j for j in path_idx if j in cont_idx]
    if to_scale:
        sc = StandardScaler()
        out[:, to_scale] = sc.fit_transform(out[:, to_scale])
    return out


def train_residual_with_clip(
    model: s1.ResidualMLP,
    train_loader: torch.utils.data.DataLoader,
    xv: torch.Tensor,
    yv: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    grad_clip: float | None,
) -> tuple[np.ndarray, dict[str, float]]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    train_mse_at_best = float("nan")
    no_imp = 0
    yv = yv.to(device)

    for _ in range(epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_sum += float(loss.item()) * len(yb)
            tr_n += len(yb)
        tr_mse = tr_sum / max(tr_n, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv.to(device)), yv).item())
        if val_loss < best_val:
            best_val = val_loss
            train_mse_at_best = tr_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xv.to(device)).cpu().numpy()
    return pred.astype(np.float32), {
        "best_val_mse": float(best_val),
        "train_mse_at_best_val": float(train_mse_at_best),
        "val_minus_train_mse": float(best_val - train_mse_at_best),
    }


def blockwise_round2_configs() -> list[dict[str, Any]]:
    return [
        {"block_hidden_dim": 64, "fusion_hidden_dim": 128, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4, "pathway_norm": "raw", "patience": 5},
        {"block_hidden_dim": 64, "fusion_hidden_dim": 128, "dropout": 0.3, "lr": 3e-4, "weight_decay": 5e-4, "pathway_norm": "raw", "patience": 5},
        {"block_hidden_dim": 64, "fusion_hidden_dim": 128, "dropout": 0.3, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "raw", "patience": 3},
        {"block_hidden_dim": 64, "fusion_hidden_dim": 64, "dropout": 0.3, "lr": 3e-4, "weight_decay": 5e-4, "pathway_norm": "pathway_zscore", "patience": 5},
        {"block_hidden_dim": 64, "fusion_hidden_dim": 256, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-4, "pathway_norm": "pathway_zscore", "patience": 5},
        {"block_hidden_dim": 64, "fusion_hidden_dim": 128, "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "pathway_zscore", "patience": 3},
    ]


def residual_round2_configs() -> list[dict[str, Any]]:
    return [
        {"hidden_dim": 256, "num_layers": 3, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4, "pathway_norm": "raw", "patience": 5, "grad_clip": None},
        {"hidden_dim": 256, "num_layers": 3, "dropout": 0.3, "lr": 3e-4, "weight_decay": 5e-4, "pathway_norm": "raw", "patience": 5, "grad_clip": 1.0},
        {"hidden_dim": 256, "num_layers": 2, "dropout": 0.3, "lr": 3e-4, "weight_decay": 1e-3, "pathway_norm": "raw", "patience": 3, "grad_clip": 1.0},
        {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3, "lr": 3e-4, "weight_decay": 5e-4, "pathway_norm": "raw", "patience": 5, "grad_clip": 1.0},
        {"hidden_dim": 128, "num_layers": 2, "dropout": 0.4, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "pathway_zscore", "patience": 3, "grad_clip": 1.0},
        {"hidden_dim": 256, "num_layers": 2, "dropout": 0.4, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "pathway_zscore", "patience": 3, "grad_clip": 0.5},
        {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-3, "pathway_norm": "pathway_zscore", "patience": 5, "grad_clip": None},
    ]


def xgb_round2_configs() -> list[dict[str, Any]]:
    return [
        {"max_depth": 4, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7, "learning_rate": 0.05, "reg_alpha": 1.0, "reg_lambda": 10, "n_estimators": 800},
        {"max_depth": 4, "min_child_weight": 7, "subsample": 0.8, "colsample_bytree": 0.7, "learning_rate": 0.05, "reg_alpha": 3.0, "reg_lambda": 10, "n_estimators": 800},
        {"max_depth": 4, "min_child_weight": 7, "subsample": 0.7, "colsample_bytree": 0.5, "learning_rate": 0.05, "reg_alpha": 3.0, "reg_lambda": 20, "n_estimators": 800},
        {"max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7, "learning_rate": 0.05, "reg_alpha": 1.0, "reg_lambda": 10, "n_estimators": 800},
        {"max_depth": 4, "min_child_weight": 3, "subsample": 0.7, "colsample_bytree": 0.7, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 5, "n_estimators": 800},
        {"max_depth": 6, "min_child_weight": 7, "subsample": 0.7, "colsample_bytree": 0.5, "learning_rate": 0.05, "reg_alpha": 3.0, "reg_lambda": 20, "n_estimators": 800},
    ]


def rank_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["spearman_mean", "spearman_std", "val_minus_train_mse_mean", "rmse_mean", "mae_mean"],
        ascending=[False, True, True, True, True],
    )


def load_stage1_best(stage1_dir: Path, model_name: str) -> dict[str, float]:
    p = stage1_dir / model_name / "cv_results.csv"
    d = pd.read_csv(p).sort_values(by=["spearman_mean", "rmse_mean"], ascending=[False, True]).iloc[0]
    return {
        "spearman_mean": float(d["spearman_mean"]),
        "spearman_std": float(d["spearman_std"]),
        "rmse_mean": float(d["rmse_mean"]),
        "val_minus_train_mse_mean": float(d.get("val_minus_train_mse_mean", np.nan)),
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stage1_dir = Path(args.stage1_dir)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    df = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + [args.target_col])]
    df = df[key + [args.target_col] + feat_cols].copy().sort_values(key).reset_index(drop=True)
    X_raw = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)
    path_idx, chem_idx, lincs_idx, target_idx = s1.assign_feature_blocks(feat_cols)
    in_dim = X_raw.shape[1]
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_meta = []
    for i, (tr, va) in enumerate(kf.split(X_raw)):
        fold_meta.append({"fold": i + 1, "train_indices": tr.astype(int).tolist(), "valid_indices": va.astype(int).tolist()})
    (out_root / "cv_fold_indices.json").write_text(json.dumps({"n_splits": args.n_splits, "seed": args.seed, "folds": fold_meta}, indent=2), encoding="utf-8")

    # XGBoost
    xgb_dir = out_root / "xgboost"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    xgb_rows: list[dict[str, Any]] = []
    for i, params in enumerate(xgb_round2_configs(), start=1):
        fold_s, fold_r, fold_a, fold_g = [], [], [], []
        for tr, va in kf.split(X_raw):
            m = xgb.XGBRegressor(objective="reg:squarederror", random_state=args.seed, n_jobs=-1, tree_method="hist", **params)
            m.fit(X_raw[tr], y[tr])
            pred_tr = m.predict(X_raw[tr]).astype(np.float32)
            pred_va = m.predict(X_raw[va]).astype(np.float32)
            met = fold_metrics(y[va], pred_va)
            fold_s.append(met["Spearman"])
            fold_r.append(met["RMSE"])
            fold_a.append(met["MAE"])
            fold_g.append(float(mean_squared_error(y[va], pred_va) - mean_squared_error(y[tr], pred_tr)))
        xgb_rows.append(
            {
                "config_id": i,
                "spearman_mean": float(np.nanmean(fold_s)),
                "spearman_std": float(np.nanstd(fold_s, ddof=0)),
                "rmse_mean": float(np.mean(fold_r)),
                "rmse_std": float(np.std(fold_r, ddof=0)),
                "mae_mean": float(np.mean(fold_a)),
                "val_minus_train_mse_mean": float(np.mean(fold_g)),
                "params_json": json.dumps(s1.json_safe_params(params), sort_keys=True),
            }
        )
    xgb_df = rank_df(pd.DataFrame(xgb_rows))
    xgb_df.to_csv(xgb_dir / "cv_results.csv", index=False)
    xgb_best = xgb_df.iloc[0].to_dict()
    (xgb_dir / "best_config.json").write_text(json.dumps(json.loads(xgb_best["params_json"]), indent=2), encoding="utf-8")
    xgb_full = xgb.XGBRegressor(objective="reg:squarederror", random_state=args.seed, n_jobs=-1, tree_method="hist", **json.loads(xgb_best["params_json"]))
    xgb_full.fit(X_raw, y)
    joblib.dump(xgb_full, xgb_dir / "model_xgb_best.joblib")
    (xgb_dir / "training_log.json").write_text(json.dumps({"family": "xgboost", "rows": xgb_rows}, indent=2), encoding="utf-8")

    # BlockWiseMLP
    bw_dir = out_root / "blockwise_mlp"
    bw_dir.mkdir(parents=True, exist_ok=True)
    bw_rows: list[dict[str, Any]] = []
    bw_cfgs = blockwise_round2_configs()
    for i, cfg in enumerate(bw_cfgs, start=1):
        fold_s, fold_r, fold_a, gaps = [], [], [], []
        for fold, (tr, va) in enumerate(kf.split(X_raw), start=1):
            X_tr, X_va = prepare_scaled_round2(X_raw, tr, va, path_idx, cfg["pathway_norm"])
            y_tr, y_va = y[tr], y[va]
            ytr_t, yv_t = torch.from_numpy(y_tr.astype(np.float32)), torch.from_numpy(y_va.astype(np.float32))
            model = s1.BlockWiseMLP4(len(path_idx), len(chem_idx), len(lincs_idx), len(target_idx), int(cfg["block_hidden_dim"]), int(cfg["fusion_hidden_dim"]), float(cfg["dropout"]))
            ds = torch.utils.data.TensorDataset(s1.zblock(X_tr, path_idx), s1.zblock(X_tr, chem_idx), s1.zblock(X_tr, lincs_idx), s1.zblock(X_tr, target_idx), ytr_t)
            ld = torch.utils.data.DataLoader(ds, batch_size=args.torch_batch, shuffle=True)
            pred, tlog = s1.train_torch_block4(
                model,
                ld,
                s1.zblock(X_va, path_idx),
                s1.zblock(X_va, chem_idx),
                s1.zblock(X_va, lincs_idx),
                s1.zblock(X_va, target_idx),
                yv_t,
                epochs=args.torch_epochs,
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
                patience=int(cfg["patience"]),
                seed=args.seed + fold + i * 100,
            )
            met = fold_metrics(y_va, pred)
            fold_s.append(met["Spearman"])
            fold_r.append(met["RMSE"])
            fold_a.append(met["MAE"])
            gaps.append(tlog["val_minus_train_mse"])
        bw_rows.append(
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
    bw_df = rank_df(pd.DataFrame(bw_rows))
    bw_df.to_csv(bw_dir / "cv_results.csv", index=False)
    bw_best = bw_df.iloc[0].to_dict()
    (bw_dir / "best_config.json").write_text(json.dumps(json.loads(bw_best["params_json"]), indent=2), encoding="utf-8")
    bw_cfg = bw_cfgs[int(bw_best["config_id"]) - 1]
    X_full_bw = scale_full_round2(X_raw, path_idx, bw_cfg["pathway_norm"])
    bw_model = s1.BlockWiseMLP4(len(path_idx), len(chem_idx), len(lincs_idx), len(target_idx), int(bw_cfg["block_hidden_dim"]), int(bw_cfg["fusion_hidden_dim"]), float(bw_cfg["dropout"]))
    ds_bw = torch.utils.data.TensorDataset(s1.zblock(X_full_bw, path_idx), s1.zblock(X_full_bw, chem_idx), s1.zblock(X_full_bw, lincs_idx), s1.zblock(X_full_bw, target_idx), torch.from_numpy(y.astype(np.float32)))
    ld_bw = torch.utils.data.DataLoader(ds_bw, batch_size=args.torch_batch, shuffle=True)
    s1.train_torch_block4_full_epochs(bw_model, ld_bw, epochs=args.torch_epochs, lr=float(bw_cfg["lr"]), weight_decay=float(bw_cfg["weight_decay"]), seed=args.seed + 999)
    torch.save(bw_model.state_dict(), bw_dir / "model_blockwise_best.pt")
    (bw_dir / "training_log.json").write_text(json.dumps({"family": "blockwise_mlp", "rows": bw_rows}, indent=2, default=str), encoding="utf-8")

    # ResidualMLP
    res_dir = out_root / "residual_mlp"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_rows: list[dict[str, Any]] = []
    res_cfgs = residual_round2_configs()
    for i, cfg in enumerate(res_cfgs, start=1):
        fold_s, fold_r, fold_a, gaps = [], [], [], []
        for fold, (tr, va) in enumerate(kf.split(X_raw), start=1):
            X_tr, X_va = prepare_scaled_round2(X_raw, tr, va, path_idx, cfg["pathway_norm"])
            y_tr, y_va = y[tr], y[va]
            ytr_t, yv_t = torch.from_numpy(y_tr.astype(np.float32)), torch.from_numpy(y_va.astype(np.float32))
            xtr, xv = torch.from_numpy(X_tr.astype(np.float32)), torch.from_numpy(X_va.astype(np.float32))
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
            met = fold_metrics(y_va, pred)
            fold_s.append(met["Spearman"])
            fold_r.append(met["RMSE"])
            fold_a.append(met["MAE"])
            gaps.append(tlog["val_minus_train_mse"])
        res_rows.append(
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
    res_df = rank_df(pd.DataFrame(res_rows))
    res_df.to_csv(res_dir / "cv_results.csv", index=False)
    res_best = res_df.iloc[0].to_dict()
    (res_dir / "best_config.json").write_text(json.dumps(json.loads(res_best["params_json"]), indent=2), encoding="utf-8")
    res_cfg = res_cfgs[int(res_best["config_id"]) - 1]
    X_full_res = scale_full_round2(X_raw, path_idx, res_cfg["pathway_norm"])
    res_model = s1.ResidualMLP(in_dim, int(res_cfg["hidden_dim"]), int(res_cfg["num_layers"]), float(res_cfg["dropout"]))
    ds_res = torch.utils.data.TensorDataset(torch.from_numpy(X_full_res.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    ld_res = torch.utils.data.DataLoader(ds_res, batch_size=args.torch_batch, shuffle=True)
    s1.train_torch_residual_full_epochs(res_model, ld_res, epochs=args.torch_epochs, lr=float(res_cfg["lr"]), weight_decay=float(res_cfg["weight_decay"]), seed=args.seed + 1999)
    torch.save(res_model.state_dict(), res_dir / "model_residual_best.pt")
    (res_dir / "training_log.json").write_text(json.dumps({"family": "residual_mlp", "rows": res_rows}, indent=2, default=str), encoding="utf-8")

    # Summary / comparison
    s1_xgb = load_stage1_best(stage1_dir, "xgboost")
    s1_bw = load_stage1_best(stage1_dir, "blockwise_mlp")
    s1_res = load_stage1_best(stage1_dir, "residual_mlp")

    lines = [
        "# Model selection stage 1 round2",
        "",
        "Narrow round2 tuning before SageMaker baseline freeze. Goal: reduce overfitting and improve generalization with fixed feature set and same 5-fold split.",
        "",
        "## Best configs",
        f"- XGBoost: `{xgb_best['params_json']}`",
        f"- BlockWiseMLP: `{bw_best['params_json']}`",
        f"- ResidualMLP: `{res_best['params_json']}`",
        "",
        "## Metrics (round2 best)",
        f"- XGBoost: Spearman={xgb_best['spearman_mean']:.6f}, RMSE={xgb_best['rmse_mean']:.6f}, std={xgb_best['spearman_std']:.6f}, gap={xgb_best['val_minus_train_mse_mean']:.6f}, MAE={xgb_best['mae_mean']:.6f}",
        f"- BlockWiseMLP: Spearman={bw_best['spearman_mean']:.6f}, RMSE={bw_best['rmse_mean']:.6f}, std={bw_best['spearman_std']:.6f}, gap={bw_best['val_minus_train_mse_mean']:.6f}, MAE={bw_best['mae_mean']:.6f}",
        f"- ResidualMLP: Spearman={res_best['spearman_mean']:.6f}, RMSE={res_best['rmse_mean']:.6f}, std={res_best['spearman_std']:.6f}, gap={res_best['val_minus_train_mse_mean']:.6f}, MAE={res_best['mae_mean']:.6f}",
        "",
        "## Stage1 -> Round2 delta",
        f"- XGBoost: dSpearman={xgb_best['spearman_mean']-s1_xgb['spearman_mean']:+.6f}, dRMSE={xgb_best['rmse_mean']-s1_xgb['rmse_mean']:+.6f}, dStd={xgb_best['spearman_std']-s1_xgb['spearman_std']:+.6f}",
        f"- BlockWiseMLP: dSpearman={bw_best['spearman_mean']-s1_bw['spearman_mean']:+.6f}, dRMSE={bw_best['rmse_mean']-s1_bw['rmse_mean']:+.6f}, dStd={bw_best['spearman_std']-s1_bw['spearman_std']:+.6f}, dGap={bw_best['val_minus_train_mse_mean']-s1_bw['val_minus_train_mse_mean']:+.6f}",
        f"- ResidualMLP: dSpearman={res_best['spearman_mean']-s1_res['spearman_mean']:+.6f}, dRMSE={res_best['rmse_mean']-s1_res['rmse_mean']:+.6f}, dStd={res_best['spearman_std']-s1_res['spearman_std']:+.6f}, dGap={res_best['val_minus_train_mse_mean']-s1_res['val_minus_train_mse_mean']:+.6f}",
        "",
        "## Checks requested",
        f"- BlockWise stability kept/improved: {'YES' if bw_best['spearman_std'] <= s1_bw['spearman_std'] + 1e-12 else 'NO'}",
        f"- Residual overfitting reduced: {'YES' if res_best['val_minus_train_mse_mean'] < s1_res['val_minus_train_mse_mean'] - 1e-12 else 'NO'}",
        f"- XGBoost generalization improved (Spearman up or RMSE down): {'YES' if (xgb_best['spearman_mean'] > s1_xgb['spearman_mean'] or xgb_best['rmse_mean'] < s1_xgb['rmse_mean']) else 'NO'}",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    dl_pick = "BlockWiseMLP" if (bw_best["spearman_std"] <= res_best["spearman_std"] and bw_best["val_minus_train_mse_mean"] <= res_best["val_minus_train_mse_mean"]) else "ResidualMLP"
    cmp_lines = [
        "# Round2 comparison",
        "",
        "## DL representative re-check",
        f"- BlockWiseMLP: Spearman={bw_best['spearman_mean']:.6f}, std={bw_best['spearman_std']:.6f}, gap={bw_best['val_minus_train_mse_mean']:.6f}, RMSE={bw_best['rmse_mean']:.6f}, MAE={bw_best['mae_mean']:.6f}",
        f"- ResidualMLP: Spearman={res_best['spearman_mean']:.6f}, std={res_best['spearman_std']:.6f}, gap={res_best['val_minus_train_mse_mean']:.6f}, RMSE={res_best['rmse_mean']:.6f}, MAE={res_best['mae_mean']:.6f}",
        f"- DL recommendation for SageMaker baseline: **{dl_pick}**",
        "",
        "## ML baseline re-check",
        f"- XGBoost(best): Spearman={xgb_best['spearman_mean']:.6f}, std={xgb_best['spearman_std']:.6f}, gap={xgb_best['val_minus_train_mse_mean']:.6f}, RMSE={xgb_best['rmse_mean']:.6f}, MAE={xgb_best['mae_mean']:.6f}",
        "",
        "## SageMaker baseline candidates",
        f"- ML: XGBoost (round2 best config_id={int(xgb_best['config_id'])})",
        f"- DL: {dl_pick} (round2 best config_id={int(bw_best['config_id'] if dl_pick == 'BlockWiseMLP' else res_best['config_id'])})",
        "",
    ]
    (out_root / "comparison_round2.md").write_text("\n".join(cmp_lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_root), "dl_pick": dl_pick}, indent=2))


if __name__ == "__main__":
    main()
