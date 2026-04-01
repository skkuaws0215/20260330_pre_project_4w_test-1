from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError as e:  # pragma: no cover
    raise SystemExit("xgboost is required. pip install xgboost") from e


Variant = Literal["A", "B"]


def json_safe_params(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    default_features = (
        repo
        / "results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet"
    )
    p = argparse.ArgumentParser(
        description="Stage-1 model selection: XGBoost + BlockWiseMLP + ResidualMLP, same 5-fold CV, equal config budget."
    )
    p.add_argument("--labels-uri", required=True, help="Path to labels.parquet (merged on sample_id + canonical_drug_id).")
    p.add_argument("--features-uri", type=str, default=str(default_features))
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260331/model_selection_stage1"
        ),
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    p.add_argument("--target-col", default="label_regression")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-configs", type=int, default=12, help="Equal search budget per model family.")
    p.add_argument("--torch-epochs", type=int, default=45)
    p.add_argument("--torch-batch", type=int, default=256)
    p.add_argument("--torch-patience", type=int, default=8)
    return p.parse_args()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(s) if s is not None and pd.notna(s) else float("nan")


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"Spearman": safe_spearman(y_true, y_pred), "RMSE": rmse}


def cont_binary_indices(X_tr: np.ndarray) -> tuple[list[int], list[int]]:
    cont_idx, bin_idx = [], []
    for j in range(X_tr.shape[1]):
        vals = np.unique(X_tr[:, j])
        if set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
            bin_idx.append(j)
        else:
            cont_idx.append(j)
    return cont_idx, bin_idx


def scale_all_continuous(X_tr: np.ndarray, X_va: np.ndarray, cont_idx: list[int]) -> None:
    if not cont_idx:
        return
    sc = StandardScaler()
    X_tr[:, cont_idx] = sc.fit_transform(X_tr[:, cont_idx])
    X_va[:, cont_idx] = sc.transform(X_va[:, cont_idx])


def scale_pathway_only(
    X_tr: np.ndarray, X_va: np.ndarray, path_idx: list[int], cont_idx: list[int]
) -> None:
    to_scale = [j for j in path_idx if j in cont_idx]
    if not to_scale:
        return
    sc = StandardScaler()
    X_tr[:, to_scale] = sc.fit_transform(X_tr[:, to_scale])
    X_va[:, to_scale] = sc.transform(X_va[:, to_scale])


def scale_full_matrix(X: np.ndarray, path_idx: list[int], variant: Variant) -> np.ndarray:
    """Fit scaler on the full matrix (final refit); single fit_transform (not tr+va on same buffer)."""
    out = np.asarray(X, dtype=np.float32).copy()
    cont_idx, _ = cont_binary_indices(out)
    if variant == "A":
        if cont_idx:
            sc = StandardScaler()
            out[:, cont_idx] = sc.fit_transform(out[:, cont_idx])
    else:
        to_scale = [j for j in path_idx if j in cont_idx]
        if to_scale:
            sc = StandardScaler()
            out[:, to_scale] = sc.fit_transform(out[:, to_scale])
    return out


def assign_feature_blocks(feat_cols: list[str]) -> tuple[list[int], list[int], list[int], list[int]]:
    pathway_cols, chem_cols, lincs_cols, target_cols = [], [], [], []
    for c in feat_cols:
        cl = c.lower()
        if cl.startswith("target_"):
            target_cols.append(c)
        elif "lincs" in cl:
            lincs_cols.append(c)
        elif "pathway" in cl:
            pathway_cols.append(c)
        else:
            chem_cols.append(c)
    path_idx = [feat_cols.index(c) for c in pathway_cols]
    chem_idx = [feat_cols.index(c) for c in chem_cols]
    lincs_idx = [feat_cols.index(c) for c in lincs_cols]
    target_idx = [feat_cols.index(c) for c in target_cols]
    return path_idx, chem_idx, lincs_idx, target_idx


class BlockWiseMLP4(nn.Module):
    def __init__(
        self,
        in_path: int,
        in_chem: int,
        in_lincs: int,
        in_target: int,
        block_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ip, self.ic, self.il, self.it = in_path, in_chem, in_lincs, in_target
        bh, fh, dr = block_hidden_dim, fusion_hidden_dim, dropout

        def enc(n_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(max(1, n_in), bh * 2),
                nn.ReLU(),
                nn.Dropout(dr),
                nn.Linear(bh * 2, bh),
                nn.ReLU(),
            )

        self.path_enc = enc(in_path)
        self.chem_enc = enc(in_chem)
        self.lincs_enc = enc(in_lincs)
        self.target_enc = enc(in_target)
        self.head = nn.Sequential(
            nn.Linear(bh * 4, fh),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(fh, 1),
        )

    def forward(
        self,
        x_path: torch.Tensor,
        x_chem: torch.Tensor,
        x_lincs: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        dev = x_path.device
        zp = x_path if self.ip > 0 else torch.zeros((x_path.shape[0], 1), device=dev, dtype=x_path.dtype)
        zc = x_chem if self.ic > 0 else torch.zeros((x_chem.shape[0], 1), device=dev, dtype=x_chem.dtype)
        zl = x_lincs if self.il > 0 else torch.zeros((x_lincs.shape[0], 1), device=dev, dtype=x_lincs.dtype)
        zt = x_target if self.it > 0 else torch.zeros((x_target.shape[0], 1), device=dev, dtype=x_target.dtype)
        h = torch.cat(
            [self.path_enc(zp), self.chem_enc(zc), self.lincs_enc(zl), self.target_enc(zt)],
            dim=1,
        )
        return self.head(h).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return self.act(x + self.fc2(h))


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(x)).squeeze(-1)


def train_torch_block4(
    model: BlockWiseMLP4,
    train_loader: torch.utils.data.DataLoader,
    xv_p: torch.Tensor,
    xv_c: torch.Tensor,
    xv_l: torch.Tensor,
    xv_t: torch.Tensor,
    yv: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
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
        for xb_p, xb_c, xb_l, xb_t, yb in train_loader:
            xb_p, xb_c, xb_l, xb_t, yb = (
                xb_p.to(device),
                xb_c.to(device),
                xb_l.to(device),
                xb_t.to(device),
                yb.to(device),
            )
            opt.zero_grad()
            pred = model(xb_p, xb_c, xb_l, xb_t)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_sum += float(loss.item()) * len(yb)
            tr_n += len(yb)
        tr_mse = tr_sum / max(tr_n, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(
                loss_fn(
                    model(
                        xv_p.to(device),
                        xv_c.to(device),
                        xv_l.to(device),
                        xv_t.to(device),
                    ),
                    yv,
                ).item()
            )
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
        pred = (
            model(
                xv_p.to(device),
                xv_c.to(device),
                xv_l.to(device),
                xv_t.to(device),
            )
            .cpu()
            .numpy()
        )
    gap = float(best_val - train_mse_at_best)
    log = {
        "best_val_mse": float(best_val),
        "train_mse_at_best_val": float(train_mse_at_best),
        "val_minus_train_mse": gap,
    }
    return pred.astype(np.float32), log


def train_torch_block4_full_epochs(
    model: BlockWiseMLP4,
    train_loader: torch.utils.data.DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb_p, xb_c, xb_l, xb_t, yb in train_loader:
            xb_p, xb_c, xb_l, xb_t, yb = (
                xb_p.to(device),
                xb_c.to(device),
                xb_l.to(device),
                xb_t.to(device),
                yb.to(device),
            )
            opt.zero_grad()
            loss_fn(model(xb_p, xb_c, xb_l, xb_t), yb).backward()
            opt.step()


def train_torch_residual(
    model: ResidualMLP,
    train_loader: torch.utils.data.DataLoader,
    xv: torch.Tensor,
    yv: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
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
    gap = float(best_val - train_mse_at_best)
    log = {
        "best_val_mse": float(best_val),
        "train_mse_at_best_val": float(train_mse_at_best),
        "val_minus_train_mse": gap,
    }
    return pred.astype(np.float32), log


def train_torch_residual_full_epochs(
    model: ResidualMLP,
    train_loader: torch.utils.data.DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()


def xgb_space_sample(n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    space = {
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "learning_rate": [0.03, 0.05, 0.1],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 5, 10],
        "n_estimators": [300, 500, 800],
    }
    keys = list(space.keys())
    out: list[dict[str, Any]] = []
    for _ in range(n):
        row = {k: rng.choice(space[k]) for k in keys}
        out.append(row)
    return out


def blockwise_configs(n: int) -> list[dict[str, Any]]:
    base = {
        "block_hidden_dim": 64,
        "fusion_hidden_dim": 128,
        "dropout": 0.2,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "variant": "A",
    }
    one_offs: list[dict[str, Any]] = [
        {**base, "variant": "B"},
        {**base, "block_hidden_dim": 32},
        {**base, "block_hidden_dim": 128},
        {**base, "fusion_hidden_dim": 64},
        {**base, "fusion_hidden_dim": 256},
        {**base, "dropout": 0.1},
        {**base, "dropout": 0.3},
        {**base, "lr": 1e-3},
        {**base, "lr": 1e-4},
        {**base, "weight_decay": 1e-5},
        {**base, "weight_decay": 1e-3},
    ]
    seq = [base] + one_offs
    return seq[:n]


def residual_configs(n: int) -> list[dict[str, Any]]:
    base = {
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.2,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "variant": "A",
    }
    one_offs: list[dict[str, Any]] = [
        {**base, "variant": "B"},
        {**base, "hidden_dim": 128},
        {**base, "hidden_dim": 512},
        {**base, "num_layers": 2},
        {**base, "num_layers": 4},
        {**base, "dropout": 0.1},
        {**base, "dropout": 0.3},
        {**base, "lr": 1e-3},
        {**base, "lr": 1e-4},
        {**base, "weight_decay": 1e-5},
        {**base, "weight_decay": 1e-3},
    ]
    return ([base] + one_offs)[:n]


def prepare_scaled(
    X_raw: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    path_idx: list[int],
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray]:
    X_tr = X_raw[tr_idx].copy()
    X_va = X_raw[va_idx].copy()
    cont_idx, _ = cont_binary_indices(X_tr)
    if variant == "A":
        scale_all_continuous(X_tr, X_va, cont_idx)
    else:
        scale_pathway_only(X_tr, X_va, path_idx, cont_idx)
    return X_tr, X_va


def zblock(arr: np.ndarray, cols: list[int]) -> torch.Tensor:
    if not cols:
        return torch.zeros((len(arr), 0), dtype=torch.float32)
    return torch.from_numpy(arr[:, cols].astype(np.float32))


def run_xgb_cv(
    params: dict[str, Any],
    X_raw: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    config_id: int,
) -> dict[str, Any]:
    fold_s, fold_r = [], []
    for fold, (tr, va) in enumerate(kf.split(X_raw), start=1):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(X_raw[tr], y[tr])
        pred = model.predict(X_raw[va]).astype(np.float32)
        m = fold_metrics(y[va], pred)
        fold_s.append(m["Spearman"])
        fold_r.append(m["RMSE"])
    sm, ss = float(np.nanmean(fold_s)), float(np.nanstd(fold_s, ddof=0))
    rm, rs = float(np.mean(fold_r)), float(np.std(fold_r, ddof=0))
    row: dict[str, Any] = {
        "config_id": config_id,
        "spearman_mean": sm,
        "spearman_std": ss,
        "rmse_mean": rm,
        "rmse_std": rs,
        "val_minus_train_mse_mean": float("nan"),
    }
    for fi, (s, r) in enumerate(zip(fold_s, fold_r), start=1):
        row[f"fold_{fi}_spearman"] = s
        row[f"fold_{fi}_rmse"] = r
    row["params_json"] = json.dumps(json_safe_params(params), sort_keys=True)
    return row


def run_blockwise_cv(
    cfg: dict[str, Any],
    X_raw: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    path_idx: list[int],
    chem_idx: list[int],
    lincs_idx: list[int],
    target_idx: list[int],
    args: argparse.Namespace,
    config_id: int,
) -> dict[str, Any]:
    variant: Variant = cfg["variant"]
    fold_s, fold_r, gaps = [], [], []
    for fold, (tr, va) in enumerate(kf.split(X_raw), start=1):
        X_tr, X_va = prepare_scaled(X_raw, tr, va, path_idx, variant)
        y_tr, y_va = y[tr], y[va]
        ytr_t = torch.from_numpy(y_tr.astype(np.float32))
        yv_t = torch.from_numpy(y_va.astype(np.float32))
        xtr_p, xtr_c, xtr_l, xtr_t = (
            zblock(X_tr, path_idx),
            zblock(X_tr, chem_idx),
            zblock(X_tr, lincs_idx),
            zblock(X_tr, target_idx),
        )
        xv_p, xv_c, xv_l, xv_t = (
            zblock(X_va, path_idx),
            zblock(X_va, chem_idx),
            zblock(X_va, lincs_idx),
            zblock(X_va, target_idx),
        )
        model = BlockWiseMLP4(
            len(path_idx),
            len(chem_idx),
            len(lincs_idx),
            len(target_idx),
            int(cfg["block_hidden_dim"]),
            int(cfg["fusion_hidden_dim"]),
            float(cfg["dropout"]),
        )
        ds = torch.utils.data.TensorDataset(xtr_p, xtr_c, xtr_l, xtr_t, ytr_t)
        ld = torch.utils.data.DataLoader(ds, batch_size=args.torch_batch, shuffle=True)
        pred, tlog = train_torch_block4(
            model,
            ld,
            xv_p,
            xv_c,
            xv_l,
            xv_t,
            yv_t,
            epochs=args.torch_epochs,
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            patience=args.torch_patience,
            seed=args.seed + fold + config_id * 100,
        )
        m = fold_metrics(y_va, pred)
        fold_s.append(m["Spearman"])
        fold_r.append(m["RMSE"])
        gaps.append(tlog["val_minus_train_mse"])
    sm, ss = float(np.nanmean(fold_s)), float(np.nanstd(fold_s, ddof=0))
    rm, rs = float(np.mean(fold_r)), float(np.std(fold_r, ddof=0))
    row: dict[str, Any] = {
        "config_id": config_id,
        "variant": variant,
        "block_hidden_dim": cfg["block_hidden_dim"],
        "fusion_hidden_dim": cfg["fusion_hidden_dim"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "spearman_mean": sm,
        "spearman_std": ss,
        "rmse_mean": rm,
        "rmse_std": rs,
        "val_minus_train_mse_mean": float(np.mean(gaps)),
    }
    for fi, (s, r) in enumerate(zip(fold_s, fold_r), start=1):
        row[f"fold_{fi}_spearman"] = s
        row[f"fold_{fi}_rmse"] = r
    row["params_json"] = json.dumps(
        {k: v for k, v in cfg.items()},
        sort_keys=True,
        default=str,
    )
    return row


def run_residual_cv(
    cfg: dict[str, Any],
    X_raw: np.ndarray,
    y: np.ndarray,
    kf: KFold,
    path_idx: list[int],
    args: argparse.Namespace,
    config_id: int,
    in_dim: int,
) -> dict[str, Any]:
    variant: Variant = cfg["variant"]
    fold_s, fold_r, gaps = [], [], []
    for fold, (tr, va) in enumerate(kf.split(X_raw), start=1):
        X_tr, X_va = prepare_scaled(X_raw, tr, va, path_idx, variant)
        y_tr, y_va = y[tr], y[va]
        ytr_t = torch.from_numpy(y_tr.astype(np.float32))
        yv_t = torch.from_numpy(y_va.astype(np.float32))
        xtr = torch.from_numpy(X_tr.astype(np.float32))
        xv = torch.from_numpy(X_va.astype(np.float32))
        model = ResidualMLP(
            in_dim,
            int(cfg["hidden_dim"]),
            int(cfg["num_layers"]),
            float(cfg["dropout"]),
        )
        ds = torch.utils.data.TensorDataset(xtr, ytr_t)
        ld = torch.utils.data.DataLoader(ds, batch_size=args.torch_batch, shuffle=True)
        pred, tlog = train_torch_residual(
            model,
            ld,
            xv,
            yv_t,
            epochs=args.torch_epochs,
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            patience=args.torch_patience,
            seed=args.seed + fold + config_id * 100,
        )
        m = fold_metrics(y_va, pred)
        fold_s.append(m["Spearman"])
        fold_r.append(m["RMSE"])
        gaps.append(tlog["val_minus_train_mse"])
    sm, ss = float(np.nanmean(fold_s)), float(np.nanstd(fold_s, ddof=0))
    rm, rs = float(np.mean(fold_r)), float(np.std(fold_r, ddof=0))
    row: dict[str, Any] = {
        "config_id": config_id,
        "variant": variant,
        "hidden_dim": cfg["hidden_dim"],
        "num_layers": cfg["num_layers"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "spearman_mean": sm,
        "spearman_std": ss,
        "rmse_mean": rm,
        "rmse_std": rs,
        "val_minus_train_mse_mean": float(np.mean(gaps)),
    }
    for fi, (s, r) in enumerate(zip(fold_s, fold_r), start=1):
        row[f"fold_{fi}_spearman"] = s
        row[f"fold_{fi}_rmse"] = r
    row["params_json"] = json.dumps(
        {k: v for k, v in cfg.items()},
        sort_keys=True,
        default=str,
    )
    return row


def write_summary_md(
    path: Path,
    xgb_best: dict[str, Any],
    bw_best: dict[str, Any],
    res_best: dict[str, Any],
    dl_winner: str,
    dl_rationale: str,
) -> None:
    lines = [
        "# Model selection stage 1",
        "",
        "Same 5-fold CV (`KFold`, `shuffle=True`, `random_state=42`) for all models; equal config count per family; shared DL training budget (epochs, batch, early stopping).",
        "",
        "## XGBoost (best configuration)",
        "",
        f"- Spearman mean ± std: **{xgb_best['spearman_mean']:.6f}** ± {xgb_best['spearman_std']:.6f}",
        f"- RMSE mean ± std: {xgb_best['rmse_mean']:.6f} ± {xgb_best['rmse_std']:.6f}",
        f"- Params: `{xgb_best.get('params_json', '')}`",
        "",
        "## BlockWiseMLP (best configuration)",
        "",
        f"- Spearman mean ± std: **{bw_best['spearman_mean']:.6f}** ± {bw_best['spearman_std']:.6f}",
        f"- RMSE mean ± std: {bw_best['rmse_mean']:.6f} ± {bw_best['rmse_std']:.6f}",
        f"- Mean val−train MSE gap (stability / overfitting hint): {bw_best.get('val_minus_train_mse_mean', float('nan')):.6f}",
        f"- Config: `{bw_best.get('params_json', '')}`",
        "",
        "## ResidualMLP (best configuration)",
        "",
        f"- Spearman mean ± std: **{res_best['spearman_mean']:.6f}** ± {res_best['spearman_std']:.6f}",
        f"- RMSE mean ± std: {res_best['rmse_mean']:.6f} ± {res_best['rmse_std']:.6f}",
        f"- Mean val−train MSE gap: {res_best.get('val_minus_train_mse_mean', float('nan')):.6f}",
        f"- Config: `{res_best.get('params_json', '')}`",
        "",
        "## Selected DL model (DL-only comparison)",
        "",
        f"**{dl_winner}**",
        "",
        dl_rationale,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    labels = pd.read_parquet(args.labels_uri)
    feats = pd.read_parquet(args.features_uri)
    key = [args.sample_id_col, args.drug_id_col]
    df = labels[key + [args.target_col]].merge(feats, on=key, how="inner")
    feat_cols = [c for c in df.columns if c not in set(key + [args.target_col])]
    df = df[key + [args.target_col] + feat_cols].copy().sort_values(key).reset_index(drop=True)
    X_raw = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[args.target_col].to_numpy(dtype=np.float32)

    path_idx, chem_idx, lincs_idx, target_idx = assign_feature_blocks(feat_cols)
    in_dim = X_raw.shape[1]

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_meta = []
    for i, (tr, va) in enumerate(kf.split(X_raw)):
        fold_meta.append(
            {
                "fold": i + 1,
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
                "train_indices": tr.astype(int).tolist(),
                "valid_indices": va.astype(int).tolist(),
            }
        )
    (out_root / "cv_fold_indices.json").write_text(
        json.dumps(
            {
                "n_splits": args.n_splits,
                "seed": args.seed,
                "n_rows": int(len(df)),
                "folds": fold_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ncfg = args.n_configs
    training_log: dict[str, Any] = {
        "torch_epochs_max": args.torch_epochs,
        "torch_batch": args.torch_batch,
        "early_stop_patience": args.torch_patience,
        "n_configs_per_model": ncfg,
        "preprocessing": {
            "XGBoost": "raw features (no standardization).",
            "DL_variant_A": "StandardScaler on all continuous columns (fit on train fold).",
            "DL_variant_B": "StandardScaler on pathway block continuous columns only; other features unscaled.",
        },
        "block_indices": {
            "pathway": len(path_idx),
            "drug_chem": len(chem_idx),
            "pair_lincs": len(lincs_idx),
            "pair_target": len(target_idx),
        },
        "dl_saved_checkpoint": (
            "CV: early stopping on validation fold. Final .pt: trained on 100% of rows for "
            "torch_epochs (same max epochs as CV cap), no early stopping, after scaling variant of best config."
        ),
    }

    # ----- XGBoost -----
    xgb_dir = out_root / "xgboost"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    xgb_configs = xgb_space_sample(ncfg, rng)
    xgb_rows: list[dict[str, Any]] = []
    for i, params in enumerate(xgb_configs, start=1):
        xgb_rows.append(run_xgb_cv(params, X_raw, y, kf, i))
    xgb_df = pd.DataFrame(xgb_rows).sort_values(by=["spearman_mean", "rmse_mean"], ascending=[False, True])
    xgb_df.to_csv(xgb_dir / "cv_results.csv", index=False)
    top3 = xgb_df.head(3).to_dict(orient="records")
    best = top3[0]
    (xgb_dir / "best_config.json").write_text(
        json.dumps(json.loads(best["params_json"]), indent=2),
        encoding="utf-8",
    )
    (xgb_dir / "best_configs_top3.json").write_text(
        json.dumps(
            [
                {"rank": j + 1, "spearman_mean": r["spearman_mean"], "params": json.loads(r["params_json"])}
                for j, r in enumerate(top3)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    best_params = json.loads(best["params_json"])
    xgb_full = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )
    xgb_full.fit(X_raw, y)
    joblib.dump(xgb_full, xgb_dir / "model_xgb_best.joblib")
    (xgb_dir / "training_log.json").write_text(
        json.dumps({**training_log, "family": "xgboost", "rows": xgb_rows}, indent=2, default=str),
        encoding="utf-8",
    )

    # ----- BlockWiseMLP -----
    bw_dir = out_root / "blockwise_mlp"
    bw_dir.mkdir(parents=True, exist_ok=True)
    bw_cfgs = blockwise_configs(ncfg)
    bw_rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(bw_cfgs, start=1):
        bw_rows.append(
            run_blockwise_cv(cfg, X_raw, y, kf, path_idx, chem_idx, lincs_idx, target_idx, args, i)
        )
    bw_df = pd.DataFrame(bw_rows).sort_values(
        by=["spearman_mean", "rmse_mean", "spearman_std"], ascending=[False, True, True]
    )
    bw_df.to_csv(bw_dir / "cv_results.csv", index=False)
    bw_best_row = bw_df.iloc[0].to_dict()
    (bw_dir / "best_config.json").write_text(
        json.dumps(json.loads(bw_best_row["params_json"]), indent=2),
        encoding="utf-8",
    )
    bcfg = bw_cfgs[int(bw_best_row["config_id"]) - 1]
    bw_model = BlockWiseMLP4(
        len(path_idx),
        len(chem_idx),
        len(lincs_idx),
        len(target_idx),
        int(bcfg["block_hidden_dim"]),
        int(bcfg["fusion_hidden_dim"]),
        float(bcfg["dropout"]),
    )
    X_full = scale_full_matrix(X_raw, path_idx, bcfg["variant"])
    xf_p = zblock(X_full, path_idx)
    xf_c = zblock(X_full, chem_idx)
    xf_l = zblock(X_full, lincs_idx)
    xf_t = zblock(X_full, target_idx)
    yt = torch.from_numpy(y.astype(np.float32))
    ds_f = torch.utils.data.TensorDataset(xf_p, xf_c, xf_l, xf_t, yt)
    ld_f = torch.utils.data.DataLoader(ds_f, batch_size=args.torch_batch, shuffle=True)
    train_torch_block4_full_epochs(
        bw_model,
        ld_f,
        epochs=args.torch_epochs,
        lr=float(bcfg["lr"]),
        weight_decay=float(bcfg["weight_decay"]),
        seed=args.seed + 999,
    )
    torch.save(bw_model.state_dict(), bw_dir / "model_blockwise_best.pt")
    (bw_dir / "training_log.json").write_text(
        json.dumps({**training_log, "family": "blockwise_mlp", "rows": bw_rows}, indent=2, default=str),
        encoding="utf-8",
    )

    # ----- ResidualMLP -----
    res_dir = out_root / "residual_mlp"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_cfgs = residual_configs(ncfg)
    res_rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(res_cfgs, start=1):
        res_rows.append(run_residual_cv(cfg, X_raw, y, kf, path_idx, args, i, in_dim))
    res_df = pd.DataFrame(res_rows).sort_values(
        by=["spearman_mean", "rmse_mean", "spearman_std"], ascending=[False, True, True]
    )
    res_df.to_csv(res_dir / "cv_results.csv", index=False)
    top2 = res_df.head(2).to_dict(orient="records")
    res_best_row = top2[0]
    (res_dir / "best_config.json").write_text(
        json.dumps(json.loads(res_best_row["params_json"]), indent=2),
        encoding="utf-8",
    )
    (res_dir / "best_configs_top2.json").write_text(
        json.dumps(
            [
                {"rank": j + 1, "spearman_mean": r["spearman_mean"], "params": json.loads(r["params_json"])}
                for j, r in enumerate(top2)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    rcfg = res_cfgs[int(res_best_row["config_id"]) - 1]
    X_full_r = scale_full_matrix(X_raw, path_idx, rcfg["variant"])
    xfull = torch.from_numpy(X_full_r.astype(np.float32))
    res_model = ResidualMLP(
        in_dim,
        int(rcfg["hidden_dim"]),
        int(rcfg["num_layers"]),
        float(rcfg["dropout"]),
    )
    ds_rf = torch.utils.data.TensorDataset(xfull, yt)
    ld_rf = torch.utils.data.DataLoader(ds_rf, batch_size=args.torch_batch, shuffle=True)
    train_torch_residual_full_epochs(
        res_model,
        ld_rf,
        epochs=args.torch_epochs,
        lr=float(rcfg["lr"]),
        weight_decay=float(rcfg["weight_decay"]),
        seed=args.seed + 1999,
    )
    torch.save(res_model.state_dict(), res_dir / "model_residual_best.pt")
    (res_dir / "training_log.json").write_text(
        json.dumps({**training_log, "family": "residual_mlp", "rows": res_rows}, indent=2, default=str),
        encoding="utf-8",
    )

    # DL-only winner: Spearman_mean (primary) → spearman_std (stability) → val−train gap (generalization)
    b_spear = float(bw_best_row["spearman_mean"])
    r_spear = float(res_best_row["spearman_mean"])
    b_std = float(bw_best_row["spearman_std"])
    r_std = float(res_best_row["spearman_std"])
    b_gap = float(bw_best_row.get("val_minus_train_mse_mean", np.nan))
    r_gap = float(res_best_row.get("val_minus_train_mse_mean", np.nan))

    if r_spear > b_spear + 1e-9:
        dl_winner, dl_rationale = (
            "ResidualMLP",
            f"Primary: higher Spearman mean ({r_spear:.6f} vs {b_spear:.6f}). Secondary RMSE means: Residual={res_best_row['rmse_mean']:.6f}, BlockWise={bw_best_row['rmse_mean']:.6f}. "
            f"Stability (std): BlockWise={b_std:.6f}, Residual={r_std:.6f}. Generalization (mean val−train MSE): BlockWise={b_gap:.6f}, Residual={r_gap:.6f}.",
        )
    elif b_spear > r_spear + 1e-9:
        dl_winner, dl_rationale = (
            "BlockWiseMLP",
            f"Primary: higher Spearman mean ({b_spear:.6f} vs {r_spear:.6f}). Secondary RMSE means: BlockWise={bw_best_row['rmse_mean']:.6f}, Residual={res_best_row['rmse_mean']:.6f}. "
            f"Stability: BlockWise={b_std:.6f}, Residual={r_std:.6f}. Generalization gaps: BlockWise={b_gap:.6f}, Residual={r_gap:.6f}.",
        )
    elif b_std < r_std - 1e-9:
        dl_winner, dl_rationale = (
            "BlockWiseMLP",
            f"Tied Spearman (~{b_spear:.6f}); lower fold Spearman std wins stability ({b_std:.6f} vs {r_std:.6f}). Gaps: BlockWise={b_gap:.6f}, Residual={r_gap:.6f}.",
        )
    elif r_std < b_std - 1e-9:
        dl_winner, dl_rationale = (
            "ResidualMLP",
            f"Tied Spearman (~{r_spear:.6f}); lower fold Spearman std ({r_std:.6f} vs {b_std:.6f}). Gaps: BlockWise={b_gap:.6f}, Residual={r_gap:.6f}.",
        )
    elif not np.isnan(b_gap) and not np.isnan(r_gap) and abs(b_gap - r_gap) > 1e-12:
        pick = "BlockWiseMLP" if b_gap < r_gap else "ResidualMLP"
        dl_winner = pick
        dl_rationale = (
            f"Tied Spearman and std; smaller mean val−train MSE gap (generalization proxy) → **{pick}** "
            f"(BlockWise={b_gap:.6f}, Residual={r_gap:.6f})."
        )
    else:
        dl_winner, dl_rationale = (
            "BlockWiseMLP",
            f"Near-tie on Spearman, std, and gap; default **BlockWiseMLP** (arbitrary tie-break).",
        )

    write_summary_md(
        out_root / "summary.md",
        best,
        bw_best_row,
        res_best_row,
        dl_winner,
        dl_rationale,
    )
    print(json.dumps({"summary_written": str(out_root / "summary.md"), "dl_winner": dl_winner}, indent=2))


if __name__ == "__main__":
    main()
