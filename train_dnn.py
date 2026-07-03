#!/usr/bin/env python3

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import uproot

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class FastDNN(nn.Module):
    def __init__(self, n_features, dropout=0.05):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

'''
def read_root_tree(filename, tree_name):
    with uproot.open(filename) as f:
        tree = f[tree_name]
        arrays = tree.arrays(library="np")

    out = {}

    for name, arr in arrays.items():
        arr = np.asarray(arr)

        if arr.ndim == 1:
            out[name] = arr

        elif arr.ndim == 2:
            for i in range(arr.shape[1]):
                out[f"{name}_{i}"] = arr[:, i]

        elif arr.ndim == 3:
            for i in range(arr.shape[1]):
                for j in range(arr.shape[2]):
                    out[f"{name}_{i}_{j}"] = arr[:, i, j]

        else:
            print("Skipping unsupported branch:", name, arr.shape)

    return pd.DataFrame(out)
'''
def read_root_tree(filename, tree_name):
    tree = uproot.open(f"{filename}:{tree_name}")
    arrays = tree.arrays(library="np")

    out = {}

    for name, arr in arrays.items():
        arr = np.asarray(arr)

        if arr.ndim == 1:
            out[name] = arr

        elif arr.ndim == 2:
            for i in range(arr.shape[1]):
                out[f"{name}_{i}"] = arr[:, i]

        elif arr.ndim == 3:
            for i in range(arr.shape[1]):
                for j in range(arr.shape[2]):
                    out[f"{name}_{i}_{j}"] = arr[:, i, j]

        else:
            print("Skipping unsupported branch:", name, arr.shape)

    return pd.DataFrame(out)

def get_target_branch(target):
    if target == "S0":
        return "S0_true"
    if target == "impact":
        return "impact_true"
    if target == "meanpt":
        return "mean_pT_true"
    raise ValueError("Unknown target")


def add_if_exists(features, df, names):
    for name in names:
        if name in df.columns:
            features.append(name)


def build_features(df, target, stage, use_npart, allow_pt_shape):
    features = []

    global_mult = [
        "Nch",
        "dNch_deta",
        "Nch_trans",
        "dNch_trans_deta",
        "Nch_toward",
        "Nch_away",
        "Nch_soft_015_03",
        "Nch_soft_03_05",
        "Nch_mid_05_10",
        "Nch_hard_gt_1",
        "Nch_hard_gt_2",
        "Nch_eta_neg",
        "Nch_eta_pos",
        "eta_asym",
        "mean_eta",
        "mean_abs_eta",
        "max_abs_eta",
        "std_eta",
        "forward_frac",
        "Nch_raw_charged",
        "Nch_eta1",
        "Nch_eta2",
        "Nch_eta3",
        "dNch_deta_eta1",
        "dNch_deta_eta2",
        "dNch_deta_eta3",
        "nch_trans_frac",
        "nch_away_frac",
    ]

    pt_shape = [
        "mean_pT",
        "std_pT",
        "var_pT",
        "sum_pT",
        "meanpt_trans",
        "sumpt_trans",
        "meanpt_toward",
        "meanpt_away",
        "sumpt_toward",
        "sumpt_away",
        "meanpt_eta_neg",
        "meanpt_eta_pos",
        "sumpt_eta_neg",
        "sumpt_eta_pos",
        "leading_pT",
        "subleading_pT",
        "leading_frac",
        "pT_skew",
        "pT_kurt",
        "frac_pt_gt_1",
        "frac_pt_gt_2",
    ]

    phi_shape = [
        "cos1_phi",
        "sin1_phi",
        "cos2_phi",
        "sin2_phi",
        "cos3_phi",
        "sin3_phi",
        "ptw_cos1_phi",
        "ptw_sin1_phi",
        "ptw_cos2_phi",
        "ptw_sin2_phi",
        "ptw_cos3_phi",
        "ptw_sin3_phi",
        "R1_phi",
        "R2_phi",
        "R3_phi",
        "mean_phi_abs",
        "trans_frac",
        "away_frac",
        "sumpt_ratio",
    ]

    add_if_exists(features, df, global_mult)

    if stage in ["shape", "full"]:
        add_if_exists(features, df, phi_shape)

    if stage in ["pt", "full"]:
        add_if_exists(features, df, pt_shape)

    if stage in ["bins", "full"]:
        features += [c for c in df.columns if c.startswith("pt_bin_")]
        features += [c for c in df.columns if c.startswith("eta_bin_")]

    if stage in ["map", "full"]:
        features += [c for c in df.columns if c.startswith("eta_phi_count_")]

    if stage == "full":
        features += [c for c in df.columns if c.startswith("eta_phi_sumpt_")]

    if use_npart and "Npart" in df.columns:
        features.append("Npart")

    forbidden = {
        "system_id",
        "event_id",
        "impact_true",
        "S0_true",
        "mean_pT_true",
    }

    if target == "meanpt" and not allow_pt_shape:
        forbidden.update({
            "mean_pT",
            "std_pT",
            "var_pT",
            "sum_pT",
            "meanpt_trans",
            "sumpt_trans",
            "meanpt_toward",
            "meanpt_away",
            "sumpt_toward",
            "sumpt_away",
            "meanpt_eta_neg",
            "meanpt_eta_pos",
            "sumpt_eta_neg",
            "sumpt_eta_pos",
            "leading_pT",
            "subleading_pT",
            "leading_frac",
            "pT_skew",
            "pT_kurt",
        })

        features = [f for f in features if not f.startswith("pt_bin_")]
        features = [f for f in features if not f.startswith("eta_phi_sumpt_")]

    features = [f for f in features if f not in forbidden]
    features = [f for f in features if f in df.columns]
    features = sorted(list(set(features)))

    return features


def make_weights(y, target, peripheral_weight):
    weights = np.ones_like(y, dtype=float)

    if target == "impact" and peripheral_weight:
        percentiles = np.linspace(0, 100, 21)
        edges = np.percentile(y, percentiles)

        inds = np.digitize(y, edges) - 1
        inds = np.clip(inds, 0, len(edges) - 2)

        counts = np.bincount(inds, minlength=len(edges) - 1)
        counts = np.maximum(counts, 1)

        weights = 1.0 / counts[inds]
        weights /= np.mean(weights)
        weights = np.clip(weights, 0.3, 4.0)

    return weights.astype(np.float32)


def save_root_output(root_path, df_test, y_test, y_pred, metrics):
    residual = y_pred - y_test

    tree_data = {
        "y_true": y_test.astype(np.float64),
        "y_pred": y_pred.astype(np.float64),
        "residual": residual.astype(np.float64),
    }

    keep = [
        "Nch",
        "Nch_trans",
        "Nch_toward",
        "Nch_away",
        "mean_pT",
        "impact_true",
        "S0_true",
        "mean_pT_true",
        "R2_phi",
        "ptw_cos2_phi",
        "trans_frac",
        "away_frac",
    ]

    for k in keep:
        if k in df_test.columns:
            tree_data[k] = df_test[k].values.astype(np.float64)

    metrics_data = {
        "test_rmse": np.array([metrics["test_rmse"]], dtype=np.float64),
        "test_mae": np.array([metrics["test_mae"]], dtype=np.float64),
        "test_r2": np.array([metrics["test_r2"]], dtype=np.float64),
        "train_rmse": np.array([metrics["train_rmse"]], dtype=np.float64),
        "train_mae": np.array([metrics["train_mae"]], dtype=np.float64),
        "train_r2": np.array([metrics["train_r2"]], dtype=np.float64),
    }
    '''
    with uproot.recreate(root_path) as f:
        f["predTree"] = tree_data
        f["metricsTree"] = metrics_data
    '''

    with uproot.recreate(root_path) as f:
        f.mktree("predTree", {k: v.dtype for k, v in tree_data.items()})
        f["predTree"].extend(tree_data)

        f.mktree("metricsTree", {k: v.dtype for k, v in metrics_data.items()})
        f["metricsTree"].extend(metrics_data)


def predict_model(model, X, batch_size, device):
    model.eval()

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--tree", default="mlTree")
    parser.add_argument("--target", choices=["S0", "impact", "meanpt"], required=True)
    parser.add_argument(
        "--stage",
        choices=["base", "shape", "pt", "bins", "map", "full"],
        default="full"
    )

    parser.add_argument("--outdir", default="step2_dnn_1M")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use-npart", action="store_true")
    parser.add_argument("--allow-pt-shape", action="store_true")
    parser.add_argument("--peripheral-weight", action="store_true")
    parser.add_argument("--max-impact", type=float, default=-1.0)
    parser.add_argument("--impact-transform", choices=["none", "b2", "log"], default="none")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.05)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    df = read_root_tree(args.input, args.tree)

    target_branch = get_target_branch(args.target)
    if target_branch not in df.columns:
        raise RuntimeError(f"Missing target branch: {target_branch}")

    features = build_features(
        df=df,
        target=args.target,
        stage=args.stage,
        use_npart=args.use_npart,
        allow_pt_shape=args.allow_pt_shape,
    )

    y = df[target_branch].values.astype(np.float32)

    good = np.isfinite(y)

    if args.target == "S0":
        good = good & (y >= 0.0) & (y <= 1.0)

    if args.target == "impact" and args.max_impact > 0:
        good = good & (y < args.max_impact)

    df = df.loc[good].reset_index(drop=True)
    y = y[good]

    if args.target == "impact":
        bmax_train = float(np.max(y))

        if args.impact_transform == "b2":
            y_model = (y / bmax_train) ** 2

        elif args.impact_transform == "log":
            y_model = np.log1p(y)

        else:
            y_model = y.copy()
    else:
        bmax_train = -1.0
        y_model = y.copy()

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)

    idx = np.arange(len(y))

    X_trainval, X_test, y_trainval_model, y_test_model, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X,
        y_model,
        y,
        idx,
        test_size=args.test_size,
        random_state=args.seed,
    )

    X_train, X_val, y_train_model, y_val_model, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval_model,
        y_trainval,
        test_size=args.val_size,
        random_state=args.seed,
    )

    xscaler = StandardScaler()
    yscaler = StandardScaler()

    X_train_s = xscaler.fit_transform(X_train).astype(np.float32)
    X_val_s = xscaler.transform(X_val).astype(np.float32)
    X_test_s = xscaler.transform(X_test).astype(np.float32)

    y_train_s = yscaler.fit_transform(y_train_model.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = yscaler.transform(y_val_model.reshape(-1, 1)).ravel().astype(np.float32)

    if args.target == "impact" and args.impact_transform != "none":
        w_train = make_weights(y_train_model, args.target, args.peripheral_weight)
    else:
        w_train = make_weights(y_train, args.target, args.peripheral_weight)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )

    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val_s, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = FastDNN(
        n_features=len(features),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss(reduction="none")

    best_val_loss = 1e30
    best_state = None
    bad_epochs = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss_each = loss_fn(pred, yb)
            loss = torch.mean(loss_each * wb)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = torch.mean(loss_fn(pred, yb))
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"epoch {epoch:03d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping")
            break

    model.load_state_dict(best_state)

    pred_train_model_s = predict_model(model, X_train_s, args.batch_size, device)
    pred_test_model_s = predict_model(model, X_test_s, args.batch_size, device)

    pred_train_model = yscaler.inverse_transform(pred_train_model_s.reshape(-1, 1)).ravel()
    pred_test_model = yscaler.inverse_transform(pred_test_model_s.reshape(-1, 1)).ravel()

    if args.target == "impact" and args.impact_transform == "b2":
        pred_train = bmax_train * np.sqrt(np.clip(pred_train_model, 0.0, None))
        pred_test = bmax_train * np.sqrt(np.clip(pred_test_model, 0.0, None))

    elif args.target == "impact" and args.impact_transform == "log":
        pred_train = np.expm1(pred_train_model)
        pred_test = np.expm1(pred_test_model)

    else:
        pred_train = pred_train_model
        pred_test = pred_test_model

    if args.target == "impact":
        pred_train = np.clip(pred_train, 0.0, bmax_train)
        pred_test = np.clip(pred_test, 0.0, bmax_train)

    metrics = {
        "model": "DNN",
        "target": args.target,
        "target_branch": target_branch,
        "stage": args.stage,
        "use_npart": bool(args.use_npart),
        "allow_pt_shape": bool(args.allow_pt_shape),
        "peripheral_weight": bool(args.peripheral_weight),
        "max_impact": float(args.max_impact),
        "impact_transform": args.impact_transform,
        "bmax_train": float(bmax_train),
        "n_events": int(len(y)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_features": int(len(features)),
        "best_val_loss_scaled": float(best_val_loss),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_r2": float(r2_score(y_test, pred_test)),
    }

    tag = args.stage
    if args.use_npart:
        tag += "_withNpart"
    if args.allow_pt_shape:
        tag += "_ptshape"
    if args.peripheral_weight:
        tag += "_quantileW"
    if args.impact_transform != "none":
        tag += f"_{args.impact_transform}"
    if args.max_impact > 0:
        tag += f"_bLT{args.max_impact:g}"

    outdir = os.path.join(args.outdir, args.target, tag)
    os.makedirs(outdir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    joblib.dump(xscaler, os.path.join(outdir, "xscaler.pkl"))
    joblib.dump(yscaler, os.path.join(outdir, "yscaler.pkl"))

    with open(os.path.join(outdir, "features.txt"), "w") as f:
        for feat in features:
            f.write(feat + "\n")

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(history).to_csv(os.path.join(outdir, "loss_history.csv"), index=False)

    df_test = df.iloc[idx_test].reset_index(drop=True)

    out_csv = pd.DataFrame({
        "y_true": y_test,
        "y_pred": pred_test,
        "residual": pred_test - y_test,
    })

    for col in ["Nch", "Nch_trans", "mean_pT", "impact_true", "S0_true", "mean_pT_true"]:
        if col in df_test.columns:
            out_csv[col] = df_test[col].values

    out_csv.to_csv(os.path.join(outdir, "predictions.csv"), index=False)

    save_root_output(
        os.path.join(outdir, "predictions.root"),
        df_test,
        y_test,
        pred_test,
        metrics,
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    print("\nSaved:")
    print(outdir)


if __name__ == "__main__":
    main()
