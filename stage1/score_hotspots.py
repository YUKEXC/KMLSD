# score_hotspots.py
# Join: MSA site features + alanine labels (+ optional ddG/PLM priors) -> Stage I scoring & Top-k.
#
# Usage:
#   python stage1/score_hotspots.py --in_dir stage1_data/p450 --out_dir stage1_data/p450 --topk 6 --srs_only
#       [--w_model 1.0 --w_alpha 1.0 --w_delta 0.2
#        --w_lambda 0.5
#        --kernel_lambda 1e-2 --kernel_sigma -1.0 --min_train_points 5
#        --auto_weight --auto_weight_lambda 1e-2]
#
import argparse
import os
import warnings
import numpy as np
import pandas as pd

SRS_RANGES = [(68, 96), (173, 181), (186, 195), (233, 256), (287, 301), (390, 401)]


def zscore(series):
    series = series.astype(float)
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd <= 1e-12:
        sd = 1.0
    return (series - mu) / sd


def infer_srs_region(pos):
    for idx, (lo, hi) in enumerate(SRS_RANGES, start=1):
        if lo <= pos <= hi:
            return idx
    return 0


def pairwise_sq_dists(A, B):
    a_norm = np.sum(A ** 2, axis=1).reshape(-1, 1)
    b_norm = np.sum(B ** 2, axis=1).reshape(1, -1)
    dist = a_norm + b_norm - 2.0 * (A @ B.T)
    return np.maximum(dist, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--srs_only", action="store_true", help="Restrict scoring to canonical SRS windows")
    ap.add_argument("--ddg_csv", default=None, help="Optional ddG site summary CSV with columns [ref_pos, ddg_min]")
    ap.add_argument("--w_ddg", type=float, default=0.2, help="Weight for ddG contribution (smaller is better)")
    ap.add_argument("--plm_csv", default=None, help="Optional PLM site summary CSV with columns [ref_pos, plm_mean]")
    ap.add_argument("--w_plm", type=float, default=0.3, help="Weight for PLM positive-mean contribution (larger is better)")

    # Supervised model options for Stage I (lightweight)
    ap.add_argument("--supervised_model", default="krr",
                    choices=["none","krr","ridge","lasso","elastic","rf","gbr","mlp"],
                    help="Supervised learner for model_pred. 'krr' (default) uses internal RBF kernel ridge. Others try scikit-learn; fallback to krr if unavailable.")
    ap.add_argument("--rf_n_estimators", type=int, default=200, help="RandomForest trees (if supervised_model=rf)")
    ap.add_argument("--gbr_n_estimators", type=int, default=300, help="GradientBoosting trees (if supervised_model=gbr)")
    ap.add_argument("--gbr_lr", type=float, default=0.05, help="GradientBoosting learning rate (if supervised_model=gbr)")
    ap.add_argument("--mlp_hidden", type=str, default="64,32", help="MLP hidden sizes, comma-separated (if supervised_model=mlp)")
    ap.add_argument("--mlp_max_iter", type=int, default=800, help="MLP max iterations")
    ap.add_argument("--mlp_lr", type=float, default=0.01, help="MLP learning rate init")

    ap.add_argument("--w_model", type=float, default=0.8, help="Weight for kernel-ridge prediction component")
    ap.add_argument("--w_alpha", type=float, default=0.8, help="Weight for UDCA yield z-score")
    ap.add_argument("--w_alpha_udca_sel", type=float, default=0.5, help="Weight for UDCA selectivity z-score")
    ap.add_argument("--w_alpha_mdca", type=float, default=0, help="Weight for MDCA yield z-score")
    ap.add_argument("--w_alpha_mdca_sel", type=float, default=0, help="Weight for MDCA selectivity z-score")
    ap.add_argument("--w_delta", type=float, default=0.8, help="Weight for entropy z-score")
    ap.add_argument("--w_lambda", type=float, default=0.5, help="Penalty applied to (risk + conflict_pen)")

    ap.add_argument("--kernel_lambda", type=float, default=1e-2, help="Kernel ridge regularization strength")
    ap.add_argument("--kernel_sigma", type=float, default=-1.0, help="RBF kernel sigma; <=0 uses median heuristic")
    ap.add_argument("--min_train_points", type=int, default=5, help="Minimum alanine points required for training")

    ap.add_argument("--auto_weight", action="store_true",
                    help="Fit linear weights for [model_pred_z, y_z, y_udca_sel_z, y_mdca_yield_z, y_mdca_sel_z, entropy_z, ddg_inv_z, plm_pos_z]")
    ap.add_argument("--auto_weight_lambda", type=float, default=1e-2,
                    help="Regularization strength when fitting automatic weights")

    args = ap.parse_args()

    msa = pd.read_csv(f"{args.in_dir}/msa_site_features.csv")
    lab = pd.read_csv(f"{args.in_dir}/alanine_labels.csv")

    df = msa[["ref_pos", "entropy"]].copy()
    df = pd.merge(df, lab, on="ref_pos", how="left")
    df = df.sort_values("ref_pos").reset_index(drop=True)

    # Optional ddG merge: expect columns ref_pos, ddg_min (smaller is better)
    if args.ddg_csv and os.path.exists(args.ddg_csv):
        try:
            ddg = pd.read_csv(args.ddg_csv)
            if 'ref_pos' in ddg.columns and 'ddg_min' in ddg.columns:
                df = pd.merge(df, ddg[['ref_pos','ddg_min']], on='ref_pos', how='left')
            else:
                print(f"[WARN] {args.ddg_csv} missing required columns; skip ddG.")
        except Exception as e:
            print(f"[WARN] failed to read ddg_csv: {e}")

    # Optional PLM merge: expect columns ref_pos, plm_mean (we only keep positive part)
    if args.plm_csv and os.path.exists(args.plm_csv):
        try:
            plm = pd.read_csv(args.plm_csv)
            # allow alternative column names
            pm_col = 'plm_mean' if 'plm_mean' in plm.columns else None
            if pm_col is None:
                print(f"[WARN] {args.plm_csv} missing plm_mean; skip PLM.")
            else:
                plm = plm[['ref_pos', pm_col]].rename(columns={pm_col: 'plm_mean'})
                # Keep only positives; negatives/zeros will be treated as no contribution
                df = pd.merge(df, plm, on='ref_pos', how='left')
        except Exception as e:
            print(f"[WARN] failed to read plm_csv: {e}")

    df["srs_region"] = df["ref_pos"].apply(infer_srs_region)
    if args.srs_only:
        df = df[df["srs_region"] > 0].reset_index(drop=True)
        if df.empty:
            raise SystemExit("No positions fall within specified SRS regions.")

    fill_cols = [
        "entropy",
        "y",
        "risk",
        "y_udca_selectivity",
        "y_mdca_yield",
        "y_mdca_selectivity",
        "risk_udca_yield",
        "risk_udca_selectivity",
        "risk_mdca_yield",
        "risk_mdca_selectivity",
        "ddg_min",
        "plm_mean",
    ]
    for col in fill_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    y_raw = df["y"].copy()
    mask_y = y_raw.notna()
    df["y"] = y_raw.fillna(0.0)
    df["y_z"] = 0.0
    if mask_y.any():
        df.loc[mask_y, "y_z"] = zscore(y_raw[mask_y])

    df["y_udca_selectivity"] = df["y_udca_selectivity"].fillna(0.0)
    df["y_mdca_yield"] = df["y_mdca_yield"].fillna(0.0)
    df["y_mdca_selectivity"] = df["y_mdca_selectivity"].fillna(0.0)
    df["risk_udca_yield"] = df["risk_udca_yield"].fillna(0.0)
    df["risk_udca_selectivity"] = df["risk_udca_selectivity"].fillna(0.0)
    df["risk_mdca_yield"] = df["risk_mdca_yield"].fillna(0.0)
    df["risk_mdca_selectivity"] = df["risk_mdca_selectivity"].fillna(0.0)

    df["y_udca_sel_z"] = zscore(df["y_udca_selectivity"])
    df["y_mdca_yield_z"] = zscore(df["y_mdca_yield"])
    df["y_mdca_sel_z"] = zscore(df["y_mdca_selectivity"])

    df["entropy_z"] = zscore(df["entropy"])
    # ddG smaller is better -> invert for z-score
    if 'ddg_min' in df.columns:
        df['ddg_min'] = df['ddg_min'].fillna(0.0)
        df['ddg_inv_z'] = zscore(-df['ddg_min'])
    else:
        df['ddg_inv_z'] = 0.0

    # PLM zero-shot: map raw plm_mean into [-1, 1] (0 neutral, >0 reward, <0 penalty)
    if 'plm_mean' in df.columns:
        ser = pd.to_numeric(df['plm_mean'], errors='coerce').fillna(0.0)
        min_val = float(ser.min())
        max_val = float(ser.max())
        if max_val > min_val:
            df['plm_pos_z'] = 2.0 * (ser - min_val) / (max_val - min_val) - 1.0
        else:
            df['plm_pos_z'] = 0.0
    else:
        df['plm_pos_z'] = 0.0

    feature_cols = [
        "entropy",
        "risk",
        "y_udca_selectivity",
        "y_mdca_yield",
        "y_mdca_selectivity",
        "risk_udca_yield",
        "risk_mdca_yield",
        "ddg_min",
        "plm_mean",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        feature_cols = ["entropy"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    model_pred = np.zeros(len(df))
    n_train = int(mask_y.sum())
    if n_train >= max(2, args.min_train_points):
        X_all = df[feature_cols].astype(float).values
        X_train = X_all[mask_y.values]
        y_train = df.loc[mask_y, "y"].astype(float).values

        # Standardize for distance-based/linear models
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0, ddof=0)
        feat_std[feat_std < 1e-8] = 1.0
        X_train_std = (X_train - feat_mean) / feat_std
        X_all_std = (X_all - feat_mean) / feat_std

        def predict_krr():
            nonlocal X_train_std, X_all_std, y_train, n_train
            dist_train = pairwise_sq_dists(X_train_std, X_train_std)
            sigma = args.kernel_sigma
            if sigma is None or sigma <= 0:
                upper = dist_train[np.triu_indices(n_train, k=1)]
                upper = upper[upper > 1e-12]
                sigma_val = float(np.sqrt(np.median(upper))) if upper.size > 0 else 1.0
            else:
                sigma_val = float(sigma)
            reg = float(max(args.kernel_lambda, 1e-8))
            K_train = np.exp(-dist_train / (2.0 * sigma_val ** 2))
            K_train += reg * np.eye(n_train)
            try:
                alpha = np.linalg.solve(K_train, y_train)
                dist_all = pairwise_sq_dists(X_all_std, X_train_std)
                K_all = np.exp(-dist_all / (2.0 * sigma_val ** 2))
                return K_all @ alpha
            except np.linalg.LinAlgError:
                print("[WARN] Kernel ridge solve failed; using zeros.")
                return np.zeros(X_all_std.shape[0])

        mdl = args.supervised_model.lower().strip()
        if mdl == "none":
            model_pred = np.zeros(len(df))
        elif mdl == "krr":
            model_pred = predict_krr()
        else:
            # Try scikit-learn models; fallback to KRR if not available
            try:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                if mdl == "ridge":
                    from sklearn.linear_model import Ridge
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("reg", Ridge(alpha=1.0, random_state=42))
                    ])
                    pipe.fit(X_train, y_train)
                    model_pred = pipe.predict(X_all)
                elif mdl == "lasso":
                    from sklearn.linear_model import Lasso
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("reg", Lasso(alpha=0.01, random_state=42, max_iter=5000))
                    ])
                    pipe.fit(X_train, y_train)
                    model_pred = pipe.predict(X_all)
                elif mdl == "elastic":
                    from sklearn.linear_model import ElasticNet
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("reg", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000))
                    ])
                    pipe.fit(X_train, y_train)
                    model_pred = pipe.predict(X_all)
                elif mdl == "rf":
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor(n_estimators=int(args.rf_n_estimators), random_state=42)
                    rf.fit(X_train, y_train)
                    model_pred = rf.predict(X_all)
                elif mdl == "gbr":
                    from sklearn.ensemble import GradientBoostingRegressor
                    gbr = GradientBoostingRegressor(n_estimators=int(args.gbr_n_estimators),
                                                    learning_rate=float(args.gbr_lr), random_state=42)
                    gbr.fit(X_train, y_train)
                    model_pred = gbr.predict(X_all)
                elif mdl == "mlp":
                    from sklearn.neural_network import MLPRegressor
                    hidden = tuple(int(x) for x in str(args.mlp_hidden).split(',') if x.strip())
                    mlp = MLPRegressor(hidden_layer_sizes=hidden, max_iter=int(args.mlp_max_iter),
                                       learning_rate_init=float(args.mlp_lr), random_state=42)
                    # Scale features for MLP
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("reg", mlp)
                    ])
                    pipe.fit(X_train, y_train)
                    model_pred = pipe.predict(X_all)
                else:
                    model_pred = predict_krr()
            except Exception as e:
                warnings.warn(f"[WARN] sklearn model '{mdl}' failed or not available: {e}. Falling back to KRR.")
                model_pred = predict_krr()
    else:
        print(f"[WARN] Only {n_train} alanine points; skip supervised training.")

    df["model_pred"] = model_pred
    df["model_pred_z"] = zscore(pd.Series(model_pred))

    risk_combo = (
        df["risk_udca_yield"].fillna(0.0) +
        0.5 * df["risk_udca_selectivity"].fillna(0.0) +
        0.5 * df["risk_mdca_yield"].fillna(0.0) +
        0.5 * df["risk_mdca_selectivity"].fillna(0.0)
    )
    df["risk_combo"] = risk_combo

    df["Score_manual"] = (
        args.w_model * df["model_pred_z"] +
        args.w_alpha * df["y_z"] +
        args.w_alpha_udca_sel * df["y_udca_sel_z"] +
        args.w_alpha_mdca * df["y_mdca_yield_z"] +
        args.w_alpha_mdca_sel * df["y_mdca_sel_z"] +
        args.w_delta * df["entropy_z"] -
        args.w_lambda * risk_combo +
        args.w_ddg * df["ddg_inv_z"] +
        args.w_plm * df["plm_pos_z"]
    )

    df["Score"] = df["Score_manual"]
    if args.auto_weight:
        comp_cols = [
            "model_pred_z",
            "y_z",
            "y_udca_sel_z",
            "y_mdca_yield_z",
            "y_mdca_sel_z",
            "entropy_z",
            "ddg_inv_z",
            "plm_pos_z",
        ]
        X_train = df.loc[mask_y, comp_cols].values
        y_train = df.loc[mask_y, "y"].astype(float).values
        if len(y_train) >= len(comp_cols) + 1:
            X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            X_all_aug = np.hstack([np.ones((df.shape[0], 1)), df[comp_cols].values])
            reg = float(max(args.auto_weight_lambda, 1e-8))
            XtX = X_train_aug.T @ X_train_aug + reg * np.eye(X_train_aug.shape[1])
            XtX[0, 0] -= reg  # do not regularize intercept
            XtY = X_train_aug.T @ y_train
            try:
                coeff = np.linalg.solve(XtX, XtY)
                raw_score = X_all_aug @ coeff
                df["Score_auto"] = raw_score - args.w_lambda * risk_combo
                df["Score"] = df["Score_auto"]
                names = ["intercept"] + comp_cols
                print("[INFO] Learned auto weights:")
                for name, val in zip(names, coeff):
                    print(f"    {name:<20}: {val:.4f}")
            except np.linalg.LinAlgError:
                print("[WARN] Auto weight solve failed; using manual score.")
        else:
            print("[WARN] Too few points for auto weight fit; using manual score.")

    df.to_csv(f"{args.out_dir}/site_features_stage1.csv", index=False)

    topk = df.sort_values("Score", ascending=False).head(args.topk).copy()
    topk_cols = [
        "ref_pos", "srs_region", "Score",
        "model_pred_z", "y_z", "y_udca_sel_z", "y_mdca_yield_z", "y_mdca_sel_z",
        "entropy_z", "ddg_inv_z", "plm_pos_z",
        "risk", "risk_udca_yield", "risk_mdca_yield"
    ]
    available_cols = [c for c in topk_cols if c in topk.columns]
    topk = topk[available_cols]
    topk.to_csv(f"{args.out_dir}/top{args.topk}.csv", index=False)

    df.sort_values("Score", ascending=False).to_csv(f"{args.out_dir}/stage1_scores.csv", index=False)
    print(f"[OK] Wrote: {args.out_dir}/site_features_stage1.csv, stage1_scores.csv, top{args.topk}.csv")


if __name__ == "__main__":
    main()
