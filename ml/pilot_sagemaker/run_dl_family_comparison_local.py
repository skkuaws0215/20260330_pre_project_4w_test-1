from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse VAE training from existing module (same loss / architecture).
from run_vae_baseline_local import train_eval_vae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-split DL family comparison (FlatMLP, VAE, TabNet, BlockWise, Residual).")
    p.add_argument("--labels-uri", required=True)
    p.add_argument("--features-uri", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flat-hidden", default="512,256,64")
    p.add_argument("--flat-max-iter", type=int, default=200)
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--vae-latent-dims", default="32,64")
    p.add_argument("--vae-lr", type=float, default=1e-3)
    p.add_argument("--vae-beta", type=float, default=1e-3)
    p.add_argument("--torch-epochs", type=int, default=50)
    p.add_argument("--torch-batch", type=int, default=256)
    p.add_argument("--torch-lr", type=float, default=1e-3)
    p.add_argument("--torch-patience", type=int, default=10)
    p.add_argument("--tabnet-max-epochs", type=int, default=120)
    p.add_argument("--tabnet-patience", type=int, default=20)
    p.add_argument("--tabnet-batch", type=int, default=512)
    return p.parse_args()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(s) if s is not None and pd.notna(s) else float("nan")


def rank_metrics_by_sample(df_valid: pd.DataFrame, sample_id_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tmp = df_valid[[sample_id_col]].copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    ndcgs: list[float] = []
    hits: list[float] = []
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
    return (float(np.mean(ndcgs)) if ndcgs else float("nan"), float(np.mean(hits)) if hits else float("nan"))


def metric_row(model: str, y_valid: np.ndarray, pred: np.ndarray, df_va: pd.DataFrame, sample_id_col: str, notes: str = "") -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    mae = float(mean_absolute_error(y_valid, pred))
    sp = safe_spearman(y_valid, pred)
    ndcg20, hit20 = rank_metrics_by_sample(df_va, sample_id_col, y_valid, pred)
    return {
        "model": model,
        "RMSE": rmse,
        "MAE": mae,
        "Spearman": sp,
        "NDCG@20": ndcg20,
        "Hit@20": hit20,
        "notes": notes,
    }


class BlockWiseMLP(nn.Module):
    def __init__(self, in_path: int, in_chem: int, in_target: int):
        super().__init__()
        ep, ec, et = 32, 128, 32
        self.in_path, self.in_chem, self.in_target = in_path, in_chem, in_target
        self.path_enc = nn.Sequential(nn.Linear(max(1, in_path), 64), nn.ReLU(), nn.Linear(64, ep), nn.ReLU())
        self.chem_enc = nn.Sequential(nn.Linear(max(1, in_chem), 256), nn.ReLU(), nn.Linear(256, ec), nn.ReLU())
        self.target_enc = nn.Sequential(nn.Linear(max(1, in_target), 64), nn.ReLU(), nn.Linear(64, et), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(ep + ec + et, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x_path, x_chem, x_target):
        p = self.path_enc(x_path if self.in_path > 0 else torch.zeros((x_path.shape[0], 1), device=x_path.device))
        c = self.chem_enc(x_chem if self.in_chem > 0 else torch.zeros((x_chem.shape[0], 1), device=x_chem.device))
        t = self.target_enc(x_target if self.in_target > 0 else torch.zeros((x_target.shape[0], 1), device=x_target.device))
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


def train_torch_mlp(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_tensors: tuple,
    *,
    epochs: int,
    lr: float,
    patience: int,
    mode: str,
) -> np.ndarray:
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    no_imp = 0

    for _ in range(1, epochs + 1):
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


def run_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> np.ndarray:
    from pytorch_tabnet.tab_model import TabNetRegressor

    model = TabNetRegressor(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        seed=seed,
        verbose=0,
    )
    yt = y_train.reshape(-1, 1)
    yv = y_valid.reshape(-1, 1)
    model.fit(
        X_train=X_train,
        y_train=yt,
        eval_set=[(X_valid, yv)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=min(128, batch_size),
        drop_last=False,
    )
    return model.predict(X_valid).reshape(-1).astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    merged = labels[key + [args.target_col]].merge(feats, on=key, how="inner").sort_values(key).reset_index(drop=True)

    feat_cols = [c for c in merged.columns if c not in set(key + [args.target_col])]
    feat_cols = [c for c in feat_cols if "lincs" not in c.lower()]
    X_df = merged[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = merged[args.target_col].to_numpy(dtype=np.float32)

    continuous_cols: list[str] = []
    for c in X_df.columns:
        vals = pd.unique(X_df[c].astype(np.float32))
        vals = vals[~pd.isna(vals)]
        uniq = set(np.round(vals, 6).tolist())
        if not uniq.issubset({0.0, 1.0}):
            continuous_cols.append(c)

    idx = np.arange(len(merged))
    tr_idx, va_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    X_all = X_df.to_numpy(dtype=np.float32)
    X_train = X_all[tr_idx].copy()
    X_valid = X_all[va_idx].copy()
    y_train = y[tr_idx]
    y_valid = y[va_idx]
    df_va = merged.iloc[va_idx]

    if continuous_cols:
        cont_idx = [feat_cols.index(c) for c in continuous_cols]
        scaler = StandardScaler()
        X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
        X_valid[:, cont_idx] = scaler.transform(X_valid[:, cont_idx])

    rows: list[dict] = []

    # Flat MLP
    hidden = tuple(int(x.strip()) for x in args.flat_hidden.split(",") if x.strip())
    flat = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=256,
        max_iter=args.flat_max_iter,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )
    flat.fit(X_train, y_train)
    pred_flat = flat.predict(X_valid).astype(np.float32)
    rows.append(metric_row("FlatMLP", y_valid, pred_flat, df_va, args.sample_id_col, "sklearn MLPRegressor"))

    # VAE: best of latent dims by validation Spearman (family representative)
    latent_dims = [int(x.strip()) for x in args.vae_latent_dims.split(",") if x.strip()]
    vae_candidates: list[dict] = []
    for ld in latent_dims:
        pred_v, _ = train_eval_vae(
            X_train, y_train, X_valid, y_valid,
            latent_dim=ld,
            epochs=args.vae_epochs,
            batch_size=256,
            lr=args.vae_lr,
            beta=args.vae_beta,
            pred_weight=1.0,
            seed=args.seed,
        )
        sp = safe_spearman(y_valid, pred_v)
        vae_candidates.append({"latent_dim": ld, "pred": pred_v, "Spearman": sp})
    best_vae = max(vae_candidates, key=lambda x: x["Spearman"])
    rows.append(
        metric_row(
            "VAE",
            y_valid,
            best_vae["pred"],
            df_va,
            args.sample_id_col,
            f"best latent_dim={best_vae['latent_dim']} by Spearman among {latent_dims}",
        )
    )

    # TabNet
    pred_tab = run_tabnet(
        X_train, y_train, X_valid, y_valid,
        seed=args.seed,
        max_epochs=args.tabnet_max_epochs,
        patience=args.tabnet_patience,
        batch_size=args.tabnet_batch,
    )
    rows.append(metric_row("TabNet", y_valid, pred_tab, df_va, args.sample_id_col, "pytorch-tabnet baseline"))

    pathway_cols = [c for c in feat_cols if ("pathway" in c.lower() and not c.lower().startswith("target_"))]
    target_cols = [c for c in feat_cols if c.lower().startswith("target_")]
    chem_cols = [c for c in feat_cols if c not in set(pathway_cols + target_cols)]
    path_idx = [feat_cols.index(c) for c in pathway_cols]
    chem_idx = [feat_cols.index(c) for c in chem_cols]
    target_idx = [feat_cols.index(c) for c in target_cols]

    ytr_t = torch.from_numpy(y_train.astype(np.float32))
    yv_t = torch.from_numpy(y_valid.astype(np.float32))

    def zblock(arr: np.ndarray, cols: list[int]) -> torch.Tensor:
        if not cols:
            return torch.zeros((len(arr), 0), dtype=torch.float32)
        return torch.from_numpy(arr[:, cols].astype(np.float32))

    xtr_p, xtr_c, xtr_t = zblock(X_train, path_idx), zblock(X_train, chem_idx), zblock(X_train, target_idx)
    xv_p, xv_c, xv_t = zblock(X_valid, path_idx), zblock(X_valid, chem_idx), zblock(X_valid, target_idx)

    ds_b = torch.utils.data.TensorDataset(xtr_p, xtr_c, xtr_t, ytr_t)
    ld_b = torch.utils.data.DataLoader(ds_b, batch_size=args.torch_batch, shuffle=True)
    bw = BlockWiseMLP(len(path_idx), len(chem_idx), len(target_idx))
    pred_bw = train_torch_mlp(
        bw, ld_b, (xv_p, xv_c, xv_t, yv_t),
        epochs=args.torch_epochs, lr=args.torch_lr, patience=args.torch_patience, mode="block",
    )
    n_path, n_chem, n_tgt = len(path_idx), len(chem_idx), len(target_idx)
    rows.append(
        metric_row(
            "BlockWiseMLP",
            y_valid,
            pred_bw,
            df_va,
            args.sample_id_col,
            f"pathway={n_path}, chem={n_chem}, target={n_tgt} (pathway=0 → chem+target only)",
        )
    )

    xtr = torch.from_numpy(X_train.astype(np.float32))
    xv = torch.from_numpy(X_valid.astype(np.float32))
    ds_r = torch.utils.data.TensorDataset(xtr, ytr_t)
    ld_r = torch.utils.data.DataLoader(ds_r, batch_size=args.torch_batch, shuffle=True)
    res = ResidualMLP(X_train.shape[1])
    pred_res = train_torch_mlp(
        res, ld_r, (xv, yv_t),
        epochs=args.torch_epochs, lr=args.torch_lr, patience=args.torch_patience, mode="residual",
    )
    rows.append(metric_row("ResidualMLP", y_valid, pred_res, df_va, args.sample_id_col, "3 residual blocks"))

    comp = pd.DataFrame(rows).sort_values("Spearman", ascending=False).reset_index(drop=True)
    comp_round = comp.copy()
    for c in ["RMSE", "MAE", "Spearman", "NDCG@20", "Hit@20"]:
        comp_round[c] = comp_round[c].round(6)
    comp_round.to_csv(out_dir / "dl_family_comparison.csv", index=False)

    ranked = comp.sort_values(["Spearman", "NDCG@20", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    top1 = ranked.iloc[0]["model"]
    top2 = ranked.iloc[1]["model"] if len(ranked) > 1 else None

    summary = {
        "purpose": "Select 1–2 representative DL models per family stage; final champion vs ML/Graph later.",
        "dataset": "pair_features_newfe_v2.parquet (LINCS excluded)",
        "rows_total": int(len(merged)),
        "rows_train": int(len(tr_idx)),
        "rows_valid": int(len(va_idx)),
        "feature_cols": int(len(feat_cols)),
        "split": {"test_size": args.test_size, "random_state": args.seed},
        "block_grouping": {"pathway_features": n_path, "chem_features": n_chem, "target_features": n_tgt},
        "vae_representative": {"latent_dim_chosen": int(best_vae["latent_dim"]), "candidates_evaluated": latent_dims},
        "dl_family_top1_by_spearman": top1,
        "dl_family_top2_by_spearman": top2,
        "recommendation": f"DL family representatives (this run): primary={top1}, secondary={top2}. If pathway_features=0, BlockWiseMLP is chem+target split only; re-evaluate when pathway features exist.",
        "outputs": {
            "comparison_csv": str(out_dir / "dl_family_comparison.csv"),
            "summary_json": str(out_dir / "dl_family_summary.json"),
        },
    }
    (out_dir / "dl_family_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
