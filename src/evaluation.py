import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from pathlib import Path


def dm_test(actual, pred1, pred2, h=1, loss_type="QLIKE", nw_lags=0):
    """
    Diebold-Mariano test with Newey-West long-run variance (HAC).
    H0: equal predictive accuracy.
    """
    actual = np.asarray(actual, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)

    e1 = actual - pred1
    e2 = actual - pred2

    if loss_type == "MSE":
        d = e1**2 - e2**2
    elif loss_type == "QLIKE":
        pred1 = np.clip(pred1, 1e-12, None)
        pred2 = np.clip(pred2, 1e-12, None)
        actual = np.clip(actual, 1e-12, None)
        loss1 = np.log(pred1) + actual / pred1
        loss2 = np.log(pred2) + actual / pred2
        d = loss1 - loss2
    elif loss_type == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Unknown loss type")

    d = np.asarray(d, dtype=float)
    T = len(d)
    if T < 30:
        return np.nan, np.nan

    d_bar = d.mean()
    dc = d - d_bar

    # Newey-West long-run variance
    gamma0 = np.mean(dc * dc)
    var_hat = gamma0
    for L in range(1, int(nw_lags) + 1):
        w = 1.0 - L / (nw_lags + 1.0)
        gammaL = np.mean(dc[L:] * dc[:-L])
        var_hat += 2.0 * w * gammaL

    if var_hat < 1e-18:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(var_hat / T)

    # Harvey–Leybourne–Newbold small-sample correction
    correction = np.sqrt((T + 1 - 2*h + (h*(h-1)/T)) / T)
    dm_stat *= correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=T-1))
    return float(dm_stat), float(p_value)


def run_walk_forward(file_path: Path, commodity_name: str, window_size=1000, step_size=1, nw_lags_dm=0):
    print(f"\n{'='*65}")
    print(f"   OUT-OF-SAMPLE FORECAST (WALK-FORWARD): {commodity_name.upper()}")
    print(f"{'='*65}")

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return

    # 1) Load data
    df = pd.read_csv(file_path, parse_dates=["Date"])

    # 2) Variables aligned with build_model_dataset.py
    features_base = ["RV_Daily", "RV_Weekly", "RV_Monthly"]
    target = "Target_RV"

    missing = [c for c in ["Date"] + features_base + [target] if c not in df.columns]
    if missing:
        print(f"[Critical Error] Missing columns: {missing}")
        return

    # 3) Identify EWMA(0.94) conflict columns
    candidates = [
        c for c in df.columns
        if ("log_fatal" in c.lower())
        and ("ewma_94" in c.lower())
    ]

    # 4) H2/H1 selection logic (families)
    name_lower = commodity_name.lower()
    conflict_cols = []

    if "wti" in name_lower or "oil" in name_lower:
        conflict_cols = [c for c in candidates if "oil_focus" in c.lower()]
        if not conflict_cols:
            conflict_cols = [c for c in candidates if "middle_east" in c.lower()]

    elif "gas" in name_lower:
        conflict_cols = [c for c in candidates if "gas_focus" in c.lower()]
        if not conflict_cols:
            conflict_cols = [c for c in candidates if "europe" in c.lower()]

    elif "gold" in name_lower:
        conflict_cols = [c for c in candidates if "global" in c.lower()]
        if not conflict_cols:
            conflict_cols = [c for c in candidates if "middle_east" in c.lower()]

    else:
        conflict_cols = candidates

    # 5) Strict lag1 only (NO fallback to lag0)
    final_conflicts = [c for c in conflict_cols if c.endswith("_lag1")]
    if not final_conflicts:
        print("[Info] No suitable lag1 conflict columns found. Baseline only.")
        return
    else:
        final_conflicts = sorted(final_conflicts)[:1]  # enforce 1 proxy
        print(f"   Selected Feature for OOS: {final_conflicts[0]}")

    # 6) Common sample
    cols_required = ["Date"] + features_base + [target] + final_conflicts
    data = df.dropna(subset=cols_required).copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    n_obs = len(data)
    if n_obs < window_size + 50:
        print(f"[Error] Not enough data ({n_obs} rows). Need at least {window_size+50}.")
        return

    h = 1  # Target_RV is RV_Daily_{t+1}
    print(f"   Config: Window={window_size}, Step={step_size}, Horizon={h}, DM_NW_lags={nw_lags_dm}")
    print(f"   Evaluation Set: {n_obs - window_size} days")

    forecasts = []

    # 7) Rolling walk-forward
    for t in range(window_size, n_obs, step_size):
        train = data.iloc[t-window_size:t]
        X_now = data.iloc[[t]]
        actual = float(X_now.iloc[0][target])

        # Baseline HAR
        try:
            m_base = sm.OLS(train[target], sm.add_constant(train[features_base], has_constant="add")).fit()
            pb = float(m_base.predict(sm.add_constant(X_now[features_base], has_constant="add")).iloc[0])
            pb = max(pb, 1e-12)
        except:
            continue

        # HAR-X
        try:
            X_aug = features_base + final_conflicts
            m_aug = sm.OLS(train[target], sm.add_constant(train[X_aug], has_constant="add")).fit()
            pa = float(m_aug.predict(sm.add_constant(X_now[X_aug], has_constant="add")).iloc[0])
            pa = max(pa, 1e-12)
        except:
            continue

        forecasts.append({
            "Date": X_now.iloc[0]["Date"],
            "Actual": actual,
            "Pred_Base": pb,
            "Pred_Aug": pa
        })

    if len(forecasts) < 50:
        print("[Error] Too few forecasts generated (data issues / singular fit).")
        return

    res = pd.DataFrame(forecasts)

    # 8) QLIKE metrics
    actual_safe = np.clip(res["Actual"].values, 1e-12, None)
    pb = np.clip(res["Pred_Base"].values, 1e-12, None)
    pa = np.clip(res["Pred_Aug"].values, 1e-12, None)

    qlike_base = float(np.mean(np.log(pb) + actual_safe / pb))
    qlike_aug = float(np.mean(np.log(pa) + actual_safe / pa))
    delta_qlike = qlike_aug - qlike_base

    # 9) DM test (QLIKE)
    dm_stat, dm_p = dm_test(actual_safe, pb, pa, h=h, loss_type="QLIKE", nw_lags=nw_lags_dm)

    # 10) Display
    print(f"\n   --- RESULTS: {commodity_name.upper()} ---")
    print(f"   QLIKE Base  : {qlike_base:.6f}")
    print(f"   QLIKE Aug   : {qlike_aug:.6f}")
    sign = "(improvement)" if delta_qlike < 0 else "(degradation)"
    print(f"   Delta QLIKE : {delta_qlike:.6f} {sign}")

    print(f"\n   --- Diebold-Mariano (QLIKE) ---")
    print(f"   DM stat     : {dm_stat:.3f}")
    print(f"   p-value     : {dm_p:.4f}")

    if (delta_qlike < 0) and (dm_p < 0.10):
        print("   Conclusion: conflict improves OOS forecasts (QLIKE), statistically (p < 0.10).")
    elif (delta_qlike < 0):
        print("   Conclusion: conflict improves OOS forecasts (QLIKE), not statistically.")
    else:
        print("   Conclusion: no OOS improvement.")

    return res