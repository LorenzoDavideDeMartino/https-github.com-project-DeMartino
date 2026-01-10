import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pathlib import Path
from src.models import fit_random_forest, predict_random_forest

def run_walk_forward(
    file_path: Path,
    commodity_name: str,
    window_size: int = 750,
    step_size: int = 5,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    rf_refit_every: int = 25):

    print(f"OUT-OF-SAMPLE FORECAST: {commodity_name}")

    # Load the final modeling dataset and restrict to a recent regime
    # This avoids mixing very different volatility regimes and reduces runtime
    data = pd.read_csv(file_path, parse_dates=["Date"])
    data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)].copy()

    # Standard HAR features 
    features_har = ["RV_Daily", "RV_Weekly", "RV_Monthly"]
    target_var = "Target_RV"

    # Conflict variables must be lagged (lag 1 only) to avoid look-ahead bias
    conflict_candidates = [
        c for c in data.columns
        if "log_deaths" in c.lower()
        and "ewma_94" in c.lower()
        and c.endswith("_lag1")]

    # Commodity-specific regional exposure (Hypothesis H2)
    name = commodity_name.lower()
    if "wti" in name or "oil" in name:
        conflict_vars = [c for c in conflict_candidates if "middle_east" in c.lower()]
    elif "gas" in name:
        conflict_vars = [c for c in conflict_candidates if "europe" in c.lower()]
    elif "gold" in name:
        conflict_vars = [c for c in conflict_candidates if "global" in c.lower()]
    else:
        conflict_vars = []

    # We include at most one conflict proxy to keep interpretation clean
    conflict_var = conflict_vars[0] if len(conflict_vars) > 0 else None
    use_conflict = conflict_var is not None

    if use_conflict:
        print(f"Selected conflict feature: {conflict_var}")
        features_harx = features_har + [conflict_var]
    else:
        print("No conflict variable found : HAR only")
        features_harx = features_har

    # Keep only complete observations used in estimation and forecasting
    required_cols = ["Date"] + features_harx + [target_var]
    data = data[required_cols].dropna().reset_index(drop=True)

    # Walk-forward evaluation mimics real-time forecasting
    results = []

    rf_model = None
    rf_last_fit_index = None

    for t in range(window_size, len(data), step_size):

        train = data.iloc[t - window_size : t]
        test = data.iloc[[t]]

        actual_vol = float(test[target_var].iloc[0])

        # HAR baseline (linear benchmark)
        har_model = sm.OLS(
            train[target_var],
            sm.add_constant(train[features_har], has_constant="add")).fit()

        har_pred = float(
            har_model.predict(
                sm.add_constant(test[features_har], has_constant="add")).iloc[0])

        harx_pred = np.nan
        rf_pred = np.nan

        if use_conflict:
            # HAR-X tests whether conflict intensity adds predictive information
            harx_model = sm.OLS(
                train[target_var],
                sm.add_constant(train[features_harx], has_constant="add")).fit()

            harx_pred = float(
                harx_model.predict(
                    sm.add_constant(test[features_harx], has_constant="add")).iloc[0])

            # Random Forest used as a non-linear benchmark
            # Refit only occasionally to reduce computation time
            if rf_model is None or (t - rf_last_fit_index) >= rf_refit_every:
                rf_model = fit_random_forest(
                    train[features_harx],
                    train[target_var],
                    random_state=42)
                
                rf_last_fit_index = t

            rf_pred = float(predict_random_forest(rf_model, test[features_harx])[0])

        results.append({
            "Date": test["Date"].iloc[0],
            "Actual": actual_vol,
            "HAR": har_pred,
            "HAR_X": harx_pred,
            "RF": rf_pred})

        if len(results) % 25 == 0:
            print(f"Progress: {len(results)}")

    # Drop rows with missing forecasts to ensure fair comparison
    results_df = pd.DataFrame(results).dropna()

    # Evaluation metrics
    y = results_df["Actual"].to_numpy()
    y_har = results_df["HAR"].to_numpy()
    y_harx = results_df["HAR_X"].to_numpy()
    y_rf = results_df["RF"].to_numpy()

    # MAE measures average absolute forecast error (robust to outliers)
    def mae(a, b):
        return np.mean(np.abs(a - b))

    # RMSE penalizes large forecast errors more strongly
    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    # RMSE on logs evaluates relative (proportional) forecast accuracy
    # Small floor is applied only for numerical stability
    eps = max(1e-12, np.quantile(y, 0.01) * 0.1)

    def rmse_log(a, b):
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return np.sqrt(np.mean((np.log(a) - np.log(b)) ** 2))

    # Evaluation matrix summarizing forecast performance by model
    metrics = pd.DataFrame({
        "Model": ["HAR", "HAR-X", "RF"],
        "MAE": [
            mae(y, y_har),
            mae(y, y_harx),
            mae(y, y_rf),
        ],
        "RMSE": [
            rmse(y, y_har),
            rmse(y, y_harx),
            rmse(y, y_rf),
        ],
        "RMSE_log": [
            rmse_log(y, y_har),
            rmse_log(y, y_harx),
            rmse_log(y, y_rf),
        ]})

    print("\nEvaluation metrics:")
    print(metrics)
    print(f"\n")

    # Export results

    # One folder per commodity for GitHub clarity
    out_dir = Path("results") / "out_of_sample" / commodity_name.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Forecast paths (time series)
    results_df.to_csv(out_dir / "forecasts.csv", sep=";", float_format="%.6f")

    # Evaluation matrix (model comparison table)
    metrics.to_csv(out_dir / "metrics.csv", sep=";", float_format="%.6f")

    # Visual comparison of out-of-sample forecasts
    plt.figure(figsize=(10, 4))
    
    # 1. Tracé des courbes
    plt.plot(results_df["Date"], results_df["Actual"], label="Actual", linewidth=2)
    plt.plot(results_df["Date"], results_df["HAR"], label="HAR")
    plt.plot(results_df["Date"], results_df["HAR_X"], label="HAR-X")
    plt.plot(results_df["Date"], results_df["RF"], label="RF")

    # 2. Zoom automatique (Calcul du max raisonnable sans le pic Covid)
    robust_max = results_df["Actual"].quantile(0.995)
    plt.ylim(0, robust_max * 1.1)

    # 3. Finitions et sauvegarde
    plt.legend()
    plt.title(f"{commodity_name.upper()} — Out-of-sample forecasts")
    plt.tight_layout()
    plt.savefig(out_dir / "forecast_plot.png", dpi=150)
    plt.close()

    return results_df