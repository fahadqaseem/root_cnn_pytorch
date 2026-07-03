#!/usr/bin/env python3

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import uproot

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

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

'''
def make_weights(y, target, peripheral_weight):
    weights = np.ones_like(y, dtype=float)

    if target == "impact" and peripheral_weight:
        weights += 2.0 * (y > 12.0)
        weights += 2.0 * (y > 15.0)

    return weights
'''
'''
def make_weights(y, target, peripheral_weight):
    weights = np.ones_like(y, dtype=float)

    if target == "impact" and peripheral_weight:
        bins = np.linspace(np.min(y), np.max(y), 31)
        counts, edges = np.histogram(y, bins=bins)

        inds = np.digitize(y, edges) - 1
        inds = np.clip(inds, 0, len(counts) - 1)

        counts_safe = np.maximum(counts, 1)
        weights = 1.0 / counts_safe[inds]
        weights = weights / np.mean(weights)

        weights = np.clip(weights, 0.2, 5.0)

    return weights
    '''
def make_weights(y, target, peripheral_weight):
    weights = np.ones_like(y, dtype=float)

    if target == "impact" and peripheral_weight:

        # define bins in percentile space
        percentiles = np.linspace(0, 100, 21)
        edges = np.percentile(y, percentiles)

        inds = np.digitize(y, edges) - 1
        inds = np.clip(inds, 0, len(edges) - 2)

        counts = np.bincount(inds, minlength=len(edges)-1)
        counts = np.maximum(counts, 1)

        weights = 1.0 / counts[inds]

        # normalize
        weights /= np.mean(weights)

        # avoid extreme weights
        weights = np.clip(weights, 0.3, 4.0)

    return weights

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


def save_feature_importance(model, features, out_csv):
    imp = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    })

    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(out_csv, index=False)


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

    parser.add_argument("--outdir", default="step2_xgb_1M")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use-npart", action="store_true")
    parser.add_argument("--allow-pt-shape", action="store_true")
    parser.add_argument("--peripheral-weight", action="store_true")
    parser.add_argument("--max-impact", type=float, default=-1.0)
    parser.add_argument("--impact-transform", choices=["none", "b2", "log"], default="none")

    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample", type=float, default=0.85)

    args = parser.parse_args()

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

    y = df[target_branch].values.astype(float)

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

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    idx = np.arange(len(y))

    X_train, X_test, y_train_model, y_test_model, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y_model,
        y,
        idx,
        test_size=args.test_size,
        random_state=args.seed,
    )

    if args.target == "impact" and args.impact_transform != "none":
        w_train = make_weights(y_train_model, args.target, args.peripheral_weight)
    else:
        w_train = make_weights(y_train, args.target, args.peripheral_weight)
    '''
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    '''
    constraints = []

    for f in features:
        if "Nch" in f or "sumpt" in f:
            constraints.append(-1)
        else:
            constraints.append(0)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
        monotone_constraints=tuple(constraints),
    )
    model.fit(X_train, y_train_model, sample_weight=w_train)

    pred_train_model = model.predict(X_train)
    pred_test_model = model.predict(X_test)

    if args.target == "impact" and args.impact_transform == "b2":
        pred_train = bmax_train * np.sqrt(np.clip(pred_train_model, 0.0, None))
        pred_test = bmax_train * np.sqrt(np.clip(pred_test_model, 0.0, None))

    elif args.target == "impact" and args.impact_transform == "log":
        pred_train = np.expm1(pred_train_model)
        pred_test = np.expm1(pred_test_model)

    else:
        pred_train = pred_train_model
        pred_test = pred_test_model

    # correct indentation here
    if args.target == "impact":
        pred_test = np.clip(pred_test, 0.0, bmax_train)
        pred_train = np.clip(pred_train, 0.0, bmax_train)

    metrics = {
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
        "n_test": int(len(y_test)),
        "n_features": int(len(features)),
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

    model_path = os.path.join(outdir, "model.pkl")
    features_path = os.path.join(outdir, "features.txt")
    metrics_path = os.path.join(outdir, "metrics.json")
    csv_path = os.path.join(outdir, "predictions.csv")
    root_path = os.path.join(outdir, "predictions.root")
    importance_path = os.path.join(outdir, "feature_importance.csv")

    joblib.dump(model, model_path)

    with open(features_path, "w") as f:
        for feat in features:
            f.write(feat + "\n")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    df_test = df.iloc[idx_test].reset_index(drop=True)

    out_csv = pd.DataFrame({
        "y_true": y_test,
        "y_pred": pred_test,
        "residual": pred_test - y_test,
    })

    for col in ["Nch", "Nch_trans", "mean_pT", "impact_true", "S0_true", "mean_pT_true"]:
        if col in df_test.columns:
            out_csv[col] = df_test[col].values

    out_csv.to_csv(csv_path, index=False)

    save_root_output(root_path, df_test, y_test, pred_test, metrics)
    save_feature_importance(model, features, importance_path)

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    print("\nTop 15 features:")
    imp = pd.read_csv(importance_path)
    print(imp.head(15).to_string(index=False))

    print("\nSaved:")
    print(outdir)


if __name__ == "__main__":
    main()
