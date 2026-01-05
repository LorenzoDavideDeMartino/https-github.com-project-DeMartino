import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

EWMA_KEEP = "ewma_94"

def _fit_ols_hac(y: pd.Series, X: pd.DataFrame, maxlags: int = 21):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

def run_har_comparison(file_path: Path, commodity_name: str):
    """
    Baseline HAR vs several HAR-X variants (one conflict block at a time).
    Reduced verbosity version.
    """
    if not file_path.exists():
        return

    df = pd.read_csv(file_path, parse_dates=["Date"])

    features_base = ["RV_Daily", "RV_Weekly", "RV_Monthly"]
    target = "Target_RV"

    missing = [c for c in features_base + [target] if c not in df.columns]
    if missing:
        print(f"[Error] Missing columns: {missing}")
        return

    # --- Candidate conflict columns ---
    candidates = [
        c for c in df.columns
        if ("log_fatal" in c.lower())
        and (EWMA_KEEP in c.lower())
        and (c.endswith("_lag0") or c.endswith("_lag1"))
    ]

    if not candidates:
        # Fallback Baseline Only
        data_common = df.dropna(subset=features_base + [target]).copy()
        model_base = _fit_ols_hac(data_common[target], data_common[features_base])
        print(f"\n=== {commodity_name.upper()} : BASELINE ONLY (No Conflicts) ===")
        print(model_base.summary().tables[1])
        return

    name = commodity_name.lower()

    # --- Define blocks ---
    if "wti" in name or "oil" in name:
        families = ["oil_focus", "middle_east"]
    elif "gas" in name:
        families = ["gas_focus", "europe"]
    elif "gold" in name:
        families = ["global", "middle_east"]
    else:
        families = []

    if not families:
        inferred = sorted({c.split("__")[0] for c in candidates if "__" in c})
        families = inferred if inferred else ["conflict"]

    variants = {}
    for fam in families:
        fam_cols = [c for c in candidates if fam in c.lower()]
        if not fam_cols: continue

        lag0_cols = [c for c in fam_cols if c.endswith("_lag0")]
        lag1_cols = [c for c in fam_cols if c.endswith("_lag1")]

        if lag0_cols: variants[f"{fam}_lag0"] = lag0_cols
        if lag1_cols: variants[f"{fam}_lag1"] = lag1_cols

    if not variants:
        return

    # --- Run Models ---
    results = []
    for label, conf_cols in variants.items():
        cols_required = features_base + conf_cols + [target]
        data_common = df.dropna(subset=cols_required).copy()

        if len(data_common) < 200: continue

        base = _fit_ols_hac(data_common[target], data_common[features_base])
        aug = _fit_ols_hac(data_common[target], data_common[features_base + conf_cols])

        delta_r2a = aug.rsquared_adj - base.rsquared_adj
        f_test = aug.f_test([f"{c} = 0" for c in conf_cols])

        results.append({
            "Variant": label,
            "N": len(data_common),
            "R2_Base": base.rsquared_adj,
            "R2_Aug": aug.rsquared_adj,
            "Delta_R2": delta_r2a,
            "p-value": float(f_test.pvalue),
        })

    if not results:
        print(f"[Info] No sufficient data for {commodity_name}.")
        return

    res = pd.DataFrame(results).sort_values(["Delta_R2"], ascending=False)

    # --- FINAL CLEAN OUTPUT ---
    print(f"\n{'-'*20} {commodity_name.upper()} : IN-SAMPLE RESULTS {'-'*20}")
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    
    print(res[["Variant", "R2_Base", "R2_Aug", "Delta_R2", "p-value"]].to_string(index=False))

    # Best Model Stats
    best_label = res.iloc[0]["Variant"]
    best_cols = variants[best_label]
    data_best = df.dropna(subset=features_base + best_cols + [target]).copy()
    best_model = _fit_ols_hac(data_best[target], data_best[features_base + best_cols])

    print(f"\n>>> Best Variant Details: {best_label}")
    print(best_model.summary().tables[1])
    print("="*60)