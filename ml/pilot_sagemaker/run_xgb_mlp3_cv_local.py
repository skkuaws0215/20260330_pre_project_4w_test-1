from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="5-fold CV: XGBoost (tuned) vs FlatMLP / BlockWiseMLP / ResidualMLP on identical folds."
    )
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--xgb-best-params-json",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/xgb_cv_tuning/xgb_best_params.json",
    )
    p.add_argument("--flat-max-iter", type=int, default=200)
    p.add_argument("--torch-epochs", type=int, default=45)
    p.add_argument("--torch-batch", type=int, default=256)
    p.add_argument("--torch-lr", type=float, default=1e-3)
    p.add_argument("--torch-patience", type=int, default=8)
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
    return float(np.mean(ndcgs)), float(np.mean(hits))


def fold_metrics(df_va: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "Spearman": safe_spearman(y_true, pred),
        "NDCG@20": rank_metrics(df_va, sample_id_col, y_true, pred)[0],
        "Hit@20": rank_metrics(df_va, sample_id_col, y_true, pred)[1],
    }


def cont_binary_indices(X_tr: np.ndarray) -> tuple[list[int], list[int]]:
    cont_idx, bin_idx = [], []
    for j in range(X_tr.shape[1]):
        vals = np.unique(X_tr[:, j])
        if set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
            bin_idx.append(j)
        else:
            cont_idx.append(j)
    return cont_idx, bin_idx


def scale_continuous(X_tr: np.ndarray, X_va: np.ndarray, cont_idx: list[int]) -> None:
    if not cont_idx:
        return
    sc = StandardScaler()
    X_tr[:, cont_idx] = sc.fit_transform(X_tr[:, cont_idx])
    X_va[:, cont_idx] = sc.transform(X_va[:, cont_idx])


class BlockWiseMLP(nn.Module):
    def __init__(self, in_path: int, in_chem: int, in_target: int):
        super().__init__()
        ep, ec, et = 32, 128, 32
        self.ip, self.ic, self.it = in_path, in_chem, in_target
        self.path_enc = nn.Sequential(nn.Linear(max(1, in_path), 64), nn.ReLU(), nn.Linear(64, ep), nn.ReLU())
        self.chem_enc = nn.Sequential(nn.Linear(max(1, in_chem), 256), nn.ReLU(), nn.Linear(256, ec), nn.ReLU())
        self.target_enc = nn.Sequential(nn.Linear(max(1, in_target), 64), nn.ReLU(), nn.Linear(64, et), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(ep + ec + et, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x_path, x_chem, x_target):
        dev = x_path.device
        p = self.path_enc(x_path if self.ip > 0 else torch.zeros((x_path.shape[0], 1), device=dev))
        c = self.chem_enc(x_chem if self.ic > 0 else torch.zeros((x_chem.shape[0], 1), device=dev))
        t = self.target_enc(x_target if self.it > 0 else torch.zeros((x_target.shape[0], 1), device=dev))
        return self.head(torch.cat([p, c, t], dim=1)).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return self.act(x + self.fc2(h))


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU())
        self.b1 = ResidualBlock(256)
        self.b2 = ResidualBlock(256)
        self.b3 = ResidualBlock(256)
        self.out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        return self.out(h).squeeze(-1)


def train_torch_block_or_residual(
    model: nn.Module,
    mode: str,
    train_loader: torch.utils.data.DataLoader,
    valid_tensors: tuple,
    *,
    epochs: int,
    lr: float,
    patience: int,
) -> np.ndarray:
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    no_imp = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            if mode == "block":
                xb_p, xb_c, xb_t, yb = [t.to(device) for t in batch]
                pred = model(xb_p, xb_c, xb_t)
            else:
                xb, yb = [t.to(device) for t in batch]
                pred = model(xb)
            loss_fn(pred, yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            if mode == "block":
                xv_p, xv_c, xv_t, yv = [t.to(device) for t in valid_tensors]
                val_loss = float(loss_fn(model(xv_p, xv_c, xv_t), yv).item())
            else:
                xv, yv = [t.to(device) for t in valid_tensors]
                val_loss = float(loss_fn(model(xv), yv).item())
        if val_loss < best_val:
            best_val = val_loss
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
        if mode == "block":
            xv_p, xv_c, xv_t, _ = [t.to(device) for t in valid_tensors]
            out = model(xv_p, xv_c, xv_t).cpu().numpy()
        else:
            xv, _ = [t.to(device) for t in valid_tensors]
            out = model(xv).cpu().numpy()
    return out.astype(np.float32)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in ["RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]:
        out[f"{m}_mean"] = float(df[m].mean())
        out[f"{m}_std"] = float(df[m].std(ddof=0))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    df = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + [args.target_col])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    df = df[key + [args.target_col] + feat_cols].copy().sort_values(key).reset_index(drop=True)

    X_raw = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)

    pathway_cols = [c for c in feat_cols if ("pathway" in c.lower() and not c.lower().startswith("target_"))]
    target_cols = [c for c in feat_cols if c.lower().startswith("target_")]
    chem_cols = [c for c in feat_cols if c not in set(pathway_cols + target_cols)]
    path_idx = [feat_cols.index(c) for c in pathway_cols]
    chem_idx = [feat_cols.index(c) for c in chem_cols]
    target_idx = [feat_cols.index(c) for c in target_cols]

    xgb_params = json.loads(Path(args.xgb_best_params_json).read_text(encoding="utf-8"))["best_params"]
    import xgboost as xgb

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_rows: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_raw), start=1):
        # ----- XGBoost: raw features (tree; matches prior tuned CV script) -----
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
            tree_method="hist",
            **xgb_params,
        )
        xgb_model.fit(X_raw[tr_idx], y[tr_idx])
        pred_xgb = xgb_model.predict(X_raw[va_idx]).astype(np.float32)
        m = fold_metrics(df.iloc[va_idx], args.sample_id_col, y[va_idx], pred_xgb)
        fold_rows.append({"fold": fold, "model": "XGBoost_tuned", **m})

        # ----- MLPs: per-fold scaling on continuous columns -----
        X_tr = X_raw[tr_idx].copy()
        X_va = X_raw[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        df_va = df.iloc[va_idx]

        cont_idx, _bin_idx = cont_binary_indices(X_tr)
        scale_continuous(X_tr, X_va, cont_idx)

        # FlatMLP
        flat = MLPRegressor(
            hidden_layer_sizes=(512, 256, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=args.flat_max_iter,
            random_state=args.seed + fold,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
        )
        flat.fit(X_tr, y_tr)
        pred_flat = flat.predict(X_va).astype(np.float32)
        m = fold_metrics(df_va, args.sample_id_col, y_va, pred_flat)
        fold_rows.append({"fold": fold, "model": "FlatMLP", **m})

        # BlockWiseMLP
        def zblock(arr: np.ndarray, cols: list[int]) -> torch.Tensor:
            if not cols:
                return torch.zeros((len(arr), 0), dtype=torch.float32)
            return torch.from_numpy(arr[:, cols].astype(np.float32))

        ytr_t = torch.from_numpy(y_tr.astype(np.float32))
        yv_t = torch.from_numpy(y_va.astype(np.float32))
        xtr_p, xtr_c, xtr_t = zblock(X_tr, path_idx), zblock(X_tr, chem_idx), zblock(X_tr, target_idx)
        xv_p, xv_c, xv_t = zblock(X_va, path_idx), zblock(X_va, chem_idx), zblock(X_va, target_idx)

        torch.manual_seed(args.seed + fold)
        bw = BlockWiseMLP(len(path_idx), len(chem_idx), len(target_idx))
        ds_b = torch.utils.data.TensorDataset(xtr_p, xtr_c, xtr_t, ytr_t)
        ld_b = torch.utils.data.DataLoader(ds_b, batch_size=args.torch_batch, shuffle=True)
        pred_bw = train_torch_block_or_residual(
            bw, "block", ld_b, (xv_p, xv_c, xv_t, yv_t),
            epochs=args.torch_epochs, lr=args.torch_lr, patience=args.torch_patience,
        )
        m = fold_metrics(df_va, args.sample_id_col, y_va, pred_bw)
        fold_rows.append({"fold": fold, "model": "BlockWiseMLP", **m})

        # ResidualMLP
        torch.manual_seed(args.seed + fold)
        xtr = torch.from_numpy(X_tr.astype(np.float32))
        xv = torch.from_numpy(X_va.astype(np.float32))
        res = ResidualMLP(X_tr.shape[1])
        ds_r = torch.utils.data.TensorDataset(xtr, ytr_t)
        ld_r = torch.utils.data.DataLoader(ds_r, batch_size=args.torch_batch, shuffle=True)
        pred_res = train_torch_block_or_residual(
            res, "residual", ld_r, (xv, yv_t),
            epochs=args.torch_epochs, lr=args.torch_lr, patience=args.torch_patience,
        )
        m = fold_metrics(df_va, args.sample_id_col, y_va, pred_res)
        fold_rows.append({"fold": fold, "model": "ResidualMLP", **m})

    fold_df = pd.DataFrame(fold_rows).round(6)
    fold_df.to_csv(out_dir / "xgb_mlp3_cv_fold_metrics.csv", index=False)

    summary_models: dict[str, dict[str, float]] = {}
    comparison_rows = []
    for name in ["XGBoost_tuned", "FlatMLP", "BlockWiseMLP", "ResidualMLP"]:
        sub = fold_df[fold_df["model"] == name].copy()
        sm = summarize(sub)
        summary_models[name] = {k: round(float(v), 6) for k, v in sm.items()}
        comparison_rows.append({"model": name, **sm})

    comp = pd.DataFrame(comparison_rows).round(6)
    comp.to_csv(out_dir / "xgb_mlp3_cv_comparison.csv", index=False)

    meta = {
        "description": "5-fold CV (shuffle=True, random_state=42): XGBoost tuned vs FlatMLP / BlockWiseMLP / ResidualMLP; identical folds.",
        "preprocessing": {
            "XGBoost": "raw numeric features (no standardization; matches prior xgb CV comparability).",
            "FlatMLP_BlockWise_Residual": "per-fold continuous columns StandardScaler fit on train fold; binary passthrough.",
        },
        "xgb_tuned_params": xgb_params,
        "block_grouping": {
            "pathway_features": len(path_idx),
            "chem_features": len(chem_idx),
            "target_features": len(target_idx),
        },
        "flat_mlp": {"hidden_layer_sizes": [512, 256, 64], "max_iter": args.flat_max_iter},
        "torch": {
            "epochs_max": args.torch_epochs,
            "batch_size": args.torch_batch,
            "lr": args.torch_lr,
            "early_stop_patience": args.torch_patience,
        },
        "dataset": "pair_features_newfe_v2 (LINCS excluded)",
        "rows": int(len(df)),
        "feature_cols": int(len(feat_cols)),
        "n_splits": args.n_splits,
        "per_model_cv_summary": summary_models,
        "outputs": {
            "fold_metrics": str(out_dir / "xgb_mlp3_cv_fold_metrics.csv"),
            "comparison": str(out_dir / "xgb_mlp3_cv_comparison.csv"),
        },
    }
    (out_dir / "xgb_mlp3_cv_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["per_model_cv_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
