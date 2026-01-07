import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from pathlib import Path

from src.models import fit_random_forest, predict_random_forest
"""
This evaluation module was developed with substantial assistance from AI tools.

The initial implementation of the walk-forward evaluation was written by the author,
but severe computational and numerical issues were encountered (very long runtimes,
unstable loss values, and repeated model refitting).

AI assistance was therefore used to:
- simplify the evaluation design,
- reduce unnecessary computations,
- stabilize the QLIKE loss numerically,
- and improve code efficiency while preserving econometric validity.

All modeling choices, assumptions, and final results were reviewed, understood,
and validated by the author.
"""
def dm_test(actual, pred1, pred2, nw_lags=5):
    
    # Diebold-Mariano test with Newey-West long-run variance
    # H0: equal predictive accuracy
    

    # Convert to numpy for consistent computations
    actual = np.asarray(actual, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)

    # QLIKE is standard for volatility forecasts; it requires strictly positive forecasts
    # We clip forecasts and actual volatility to avoid log(0) and numerical explosions
    pred1 = np.clip(pred1, 1e-3, None)
    pred2 = np.clip(pred2, 1e-3, None)
    actual = np.clip(actual, 1e-3, None)

    loss1 = np.log(pred1) + actual / pred1
    loss2 = np.log(pred2) + actual / pred2
    d = loss1 - loss2

    d = np.asarray(d, dtype=float)
    T = len(d)

    # Too few points -> DM test is not reliable
    if T < 30:
        return np.nan, np.nan

    d_bar = d.mean()
    dc = d - d_bar

    # Newey-West long-run variance for serial correlation in loss differences
    gamma0 = np.mean(dc * dc)
    var_hat = gamma0

    for L in range(1, int(nw_lags) + 1):
        w = 1.0 - L / (nw_lags + 1.0)
        gammaL = np.mean(dc[L:] * dc[:-L])
        var_hat += 2.0 * w * gammaL

    if var_hat < 1e-18:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(var_hat / T)

    # Two-sided p-value (we test "different", not one-direction)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=T - 1))
    return float(dm_stat), float(p_value)


def run_walk_forward(
    file_path: Path,
    commodity_name: str,
    window_size: int = 750,
    step_size: int = 5,
    nw_lags_dm: int = 5,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    rf_refit_every: int = 25,):
    """
    Walk-forward OOS evaluation for:
      - HAR (baseline)
      - HAR-X (HAR + 1 conflict proxy, lag1 only)
      - Random Forest (same inputs as HAR-X, non-linear benchmark)

    Output: DataFrame with Date, Actual, Pred_HAR, Pred_HARX, Pred_RF
    """

    print(f"OUT-OF-SAMPLE FORECAST (WALK-FORWARD): {commodity_name}")

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Attention: File not found: {file_path}")
        return None

    df = pd.read_csv(file_path, parse_dates=["Date"])

    # WHY: restrict to a recent regime to reduce runtime and avoid mixing very different market periods
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

    # HAR features and target
    features_base = ["RV_Daily", "RV_Weekly", "RV_Monthly"]
    target = "Target_RV"

    missing = [c for c in ["Date"] + features_base + [target] if c not in df.columns]
    if missing:
        print(f"[Critical Error] Missing columns: {missing}")
        return None

    # Candidate conflict columns (EWMA 0.94)
    candidates = [
        c for c in df.columns
        if ("log_deaths" in c.lower())
        and ("ewma_94" in c.lower())
        and c.endswith("_lag1")] # Using only lag-1 conflict variables to avoid any look-ahead bias

    # Select 1 conflict proxy consistent with H2
    name_lower = commodity_name.lower()

    if "wti" in name_lower or "oil" in name_lower:
        conflict_cols = [c for c in candidates if "middle_east" in c.lower()]

    elif "gas" in name_lower:
        conflict_cols = [c for c in candidates if "europe" in c.lower()]

    elif "gold" in name_lower:
        conflict_cols = [c for c in candidates if "global" in c.lower()] \
            or [c for c in candidates if "middle_east" in c.lower()]

    else:
        conflict_cols = candidates

    # OOS must use lag1 only
    final_conflicts = sorted([c for c in conflict_cols if c.endswith("_lag1")])[:1]
    have_conflict = len(final_conflicts) == 1

    if have_conflict:
        print(f"   Selected Feature for OOS: {final_conflicts[0]}")
    else:
        print("[Info] No suitable lag1 conflict column found. We run HAR only (no HAR-X, no RF).")

    cols_required = ["Date"] + features_base + [target] + (final_conflicts if have_conflict else [])
    data = df.dropna(subset=cols_required).copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    n_obs = len(data)
    if n_obs < window_size + 50:
        print(f"[Error] Not enough data ({n_obs} rows). Need at least {window_size + 50}.")
        return None

    n_eval_points = n_obs - window_size
    n_expected = int(np.ceil(n_eval_points / max(int(step_size), 1)))

    print(f"   Config: Window={window_size}, Step={step_size}, DM_NW_lags={nw_lags_dm}")
    print(f"   Expected forecasts: ~{n_expected}")
    if have_conflict:
        print(f"   RF refit every     : {rf_refit_every} steps (benchmark speed-up)")

    # Inputs for HAR-X / RF (same information set)
    X_aug = features_base + (final_conflicts if have_conflict else [])
    rf_features = X_aug[:]

    rows = []
    n_fail = {"base": 0, "aug": 0, "rf": 0}

    # RF cache (refit less often)
    rf_model = None
    rf_last_fit_t = None

    for t in range(window_size, n_obs, step_size):
        train = data.iloc[t - window_size:t]
        x_now = data.iloc[[t]]

        # Floor: keep QLIKE stable (no near-zero values)
        floor = 1e-3

        y_now = float(x_now.iloc[0][target])
        y_now = max(y_now, floor)

        # --- HAR (baseline) ---
        try:
            m_base = sm.OLS(
                train[target],
                sm.add_constant(train[features_base], has_constant="add"),
            ).fit()
            pb = float(
                m_base.predict(
                    sm.add_constant(x_now[features_base], has_constant="add")
                ).iloc[0]
            )
            pb = max(pb, floor)
        
        except Exception:
            n_fail["base"] += 1
            continue

        pa = np.nan
        pr = np.nan

        if have_conflict:
            # --- HAR-X ---
            try:
                m_aug = sm.OLS(
                    train[target],
                    sm.add_constant(train[X_aug], has_constant="add"),
                ).fit()
                pa = float(
                    m_aug.predict(
                        sm.add_constant(x_now[X_aug], has_constant="add")
                    ).iloc[0]
                )
                pa = max(pa, floor)
            except Exception:
                n_fail["aug"] += 1
                pa = np.nan

            # --- RF benchmark (refit less often) ---
            try:
                need_refit = (
                    rf_model is None
                    or rf_last_fit_t is None
                    or ((t - rf_last_fit_t) >= rf_refit_every)
                )
                if need_refit:
                    rf_model = fit_random_forest(
                        train[rf_features], train[target], random_state=42
                    )
                    rf_last_fit_t = t

                pr = float(predict_random_forest(rf_model, x_now[rf_features])[0])
                pr = max(pr, floor)
            except Exception:
                n_fail["rf"] += 1
                pr = np.nan

        rows.append(
            {
                "Date": x_now.iloc[0]["Date"],
                "Actual": y_now,
                "Pred_HAR": pb,
                "Pred_HARX": pa,
                "Pred_RF": pr})

        # Minimal progress feedback
        if len(rows) % 25 == 0:
            print(f"   Progress: {len(rows)} / ~{n_expected}")

    if len(rows) < 50:
        print("[Error] Too few forecasts generated.")
        print(f"   Failures: {n_fail}")
        return None

    res = pd.DataFrame(rows)

    # Compare all models on the SAME dates (common sample)
    if have_conflict:
        common = (
            np.isfinite(res["Pred_HAR"].values)
            & np.isfinite(res["Pred_HARX"].values)
            & np.isfinite(res["Pred_RF"].values))
        
    else:
        common = np.isfinite(res["Pred_HAR"].values)

    res_c = res.loc[common].copy()

    print("\n   --- QUALITY CHECK ---")
    print(f"   Forecasts stored : {len(res)}")
    print(f"   Forecasts used   : {len(res_c)} (common sample)")
    print(f"   Date range (OOS) : {res_c['Date'].min()} -> {res_c['Date'].max()}")
    print(f"   Failures         : base={n_fail['base']}, aug={n_fail['aug']}, rf={n_fail['rf']}")

    if len(res_c) < 50:
        print("[Error] Too few common-sample forecasts to compare models.")
        return res

    a = res_c["Actual"].to_numpy(float)
    ph = res_c["Pred_HAR"].to_numpy(float)

    qlike_har = float(np.mean(np.log(ph) + a / ph))

    print(f"\n   --- RESULTS: {commodity_name.upper()} (COMMON SAMPLE) ---")
    print(f"   QLIKE HAR   : {qlike_har:.6f}")

    if not have_conflict:
        print("   Note: HAR-X and RF skipped (no lag1 conflict feature available).")
        return res

    px = res_c["Pred_HARX"].to_numpy(float)
    prf = res_c["Pred_RF"].to_numpy(float)

    qlike_harx = float(np.mean(np.log(px) + a / px))
    qlike_rf = float(np.mean(np.log(prf) + a / prf))

    print(f"   QLIKE HAR-X : {qlike_harx:.6f}")
    print(f"   QLIKE RF    : {qlike_rf:.6f}")

    ranking = sorted(
        [("HAR", qlike_har), ("HAR-X", qlike_harx), ("RF", qlike_rf)],
        key=lambda x: x[1],
    )
    print(f"\n   Best model (by QLIKE): {ranking[0][0]}")

    # Two DM tests only (focused on the research question)
    dm_x, p_x = dm_test(a, ph, px, nw_lags=nw_lags_dm)
    dm_r, p_r = dm_test(a, ph, prf, nw_lags=nw_lags_dm)

    print(f"\n   --- Diebold-Mariano (QLIKE, NW={nw_lags_dm}) ---")
    print(f"   HAR vs HAR-X : stat={dm_x:.3f}, p={p_x:.4f}")
    print(f"   HAR vs RF    : stat={dm_r:.3f}, p={p_r:.4f}")

    return res
