import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

# I keep the EWMA identifier as a constant so the filtering logic is explicit and easy to change
EWMA_KEEP = "ewma_94"

def _fit_ols_hac(y: pd.Series, X: pd.DataFrame, maxlags: int = 21):
    # HAR models are linear regressions, so OLS is an appropriate estimator.
    # HAC standard errors are used to handle volatility persistence and changing variance in financial time series
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}) # <-IA help to implement

def run_har_comparison(file_path: Path, commodity_name: str):
    # This function performs an in-sample diagnostic comparison between a baseline HAR model and several HAR-X variants
    # The goal is not forecasting here, but understanding whether conflict variables add explanatory power in-sample

    if not file_path.exists():
        return # If the dataset is missing, the comparison cannot be run

    # I need to load the final regression dataset prepared by the pipeline
    df = pd.read_csv(file_path, parse_dates=["Date"])

    out_dir = Path("results") / "in_sample" / commodity_name.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline HAR regressors, as defined in the original HAR framework
    features_base = ["RV_Daily", "RV_Weekly", "RV_Monthly"]
    target = "Target_RV"

    # (IA suggestion) We fail early if core variables are missing to avoid misleading results
    missing = [c for c in features_base + [target] if c not in df.columns]
    if missing:
        print(f"[Error] Missing columns: {missing}")
        return

    # Candidate conflict variables are identified by name patterns
    # (IA suggestion) This avoids hard-coding column names and keeps the logic flexible
    candidates = [
        c for c in df.columns
        if ("log_deaths" in c.lower())
        and (EWMA_KEEP in c.lower())
        and (c.endswith("_lag0") or c.endswith("_lag1"))]

    name = commodity_name.lower()

    # Conflict variables are grouped into economically meaningful blocks depending on the commodity under study
    if "wti" in name:
        families = ["middle_east"]
    elif "gas" in name:
        families = ["europe"]
    elif "gold" in name:
        families = ["global", "middle_east"]
    else:
        families = []

    # If no predefined mapping applies, we infer families directly from column names
    # (IA input) This makes the code robust to different datasets or naming conventions, 
    # and avoids that the code only works when the data has exactly with the names, structure, and format expected in advance.
    if not families:
        inferred = sorted({c.split("__")[0] for c in candidates if "__" in c})
        families = inferred if inferred else ["conflict"]

    variants = {}
    for fam in families:
        # Each family is tested separately to keep interpretations clean
        fam_cols = [c for c in candidates if fam in c.lower()]
        if not fam_cols:
            continue

        # (IA suggestion) Lag-specific variants allow us to test whether conflict information
        # affects volatility immediately (lag 0) or with a delay (lag 1), rather than pooling both effects into a single regression.
        lag0_cols = [c for c in fam_cols if c.endswith("_lag0")]
        lag1_cols = [c for c in fam_cols if c.endswith("_lag1")]

        if lag0_cols:
            variants[f"HAR-X ({fam}, lag 0)"] = lag0_cols
        if lag1_cols:
            variants[f"HAR-X ({fam}, lag 1)"] = lag1_cols

    if not variants:
        # If no valid variants can be formed, we stop the comparison
        return

    # Each HAR-X variant is compared against the same HAR baseline
    results = []
    for label, conf_cols in variants.items():
        cols_required = features_base + conf_cols + [target]
        data_common = df.dropna(subset=cols_required).copy()
        # Using a common sample ensures a fair comparison between HAR and HAR-X models

        if len(data_common) < 200: # Very small samples would make inference meaningless
            continue

        base = _fit_ols_hac(
            data_common[target],
            data_common[features_base])
        
        aug = _fit_ols_hac(
            data_common[target],
            data_common[features_base + conf_cols])

        # Adjusted R² is used to account for different numbers of regressors
        delta_r2a = aug.rsquared_adj - base.rsquared_adj

        # Joint F-test checks whether the conflict block adds explanatory power
        f_test = aug.f_test([f"{c} = 0" for c in conf_cols])

        results.append({
            "Variant": label,
            "N": len(data_common),
            "R2_Base": base.rsquared_adj,
            "R2_Aug": aug.rsquared_adj,
            "Delta_R2": delta_r2a,
            "p-value": float(f_test.pvalue)})

    if not results:
        print(f"No sufficient data for {commodity_name}")
        return

    # Results are ranked by improvement over the baseline HAR
    res = pd.DataFrame(results).sort_values(["Delta_R2"], ascending=False)

    # Save HAR vs HAR-X comparison table
    res_rounded = res.copy()
    res_rounded = res_rounded.apply(
        lambda s: s.round(6) if np.issubdtype(s.dtype, np.number) else s)

    res_rounded.to_csv(
        out_dir / "comparison.csv",
        sep=";",
        index=False,
        float_format="%.6f")

    print(f"\n---------- {commodity_name} ----------")
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")

    print(
        res[["Variant", "R2_Base", "R2_Aug", "Delta_R2", "p-value"]]
        .to_string(index=False))

    # For illustration, we display detailed output for the best-performing variant
    best_label = res.iloc[0]["Variant"] # Selects the HAR-X variant that achieved the largest improvement over the baseline HAR model
    best_cols = variants[best_label] # These are the conflict variables included in the best model

    best = res.iloc[0]
    if best["Delta_R2"] > 0 and best["p-value"] < 0.05:
        conclusion = "HAR-X outperforms HAR in-sample and the improvement is statistically significant."
    elif best["Delta_R2"] > 0 and best["p-value"] >= 0.05:
        conclusion = "HAR-X slightly outperforms HAR in-sample, but the improvement is not statistically significant."
    else:
        conclusion = "HAR-X does not improve upon the baseline HAR model in-sample."

    print(f"\n>>> Best Variant Details: {best_label}")
    print(f"\nConclusion : ({commodity_name.upper()}): {conclusion}\n")

    # I keep only rows where all variables needed for the best model are available
    data_best = df.dropna(subset=features_base + best_cols + [target]).copy()

    best_model = _fit_ols_hac(
        data_best[target],
        data_best[features_base + best_cols] ) # We re-estimate the best HAR-X model to inspect its coefficients.
    
    # WHY: Save coefficients of the best in-sample HAR-X model
    coef_table = best_model.summary2().tables[1]
    coef_table.to_csv(out_dir / "best_model_coefficients.csv", sep=";", float_format="%.6f", index=True)

def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series ,
    random_state: int = 42):
    
    # Train a Random Forest model for volatility forecasting
    # It can capture interactions and non-linear effects that linear HAR-type models cannot
 
    # A moderate number of trees is sufficient for a robust benchmark and keeps computation time reasonable
    model = RandomForestRegressor(
        n_estimators= 150,
        max_depth= 10,
        min_samples_leaf= 5,
        random_state=random_state, # Guarantees reproducibility across runs and machines
        n_jobs= 1) # WHY: repeated refits in walk-forward; -1 creates too much overhead

    # The model learns the relationship between past information (X) and next-day volatility (y).
    model.fit(X_train, y_train)

    return model

def predict_random_forest(model: RandomForestRegressor, X_test: pd.DataFrame):
    # Generate out-of-sample volatility predictions.

    # The model outputs a point forecast for next-day realized volatility.
    return model.predict(X_test)