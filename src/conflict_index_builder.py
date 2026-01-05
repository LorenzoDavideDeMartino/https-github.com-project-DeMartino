import pandas as pd
import numpy as np
from pathlib import Path

# --- PARAMETERS ---
LAMBDA_LIST = [0.94]  # single EWMA decay (fixed ex ante)
SHOCK_PERCENTILE = 0.95
EXPANDING_WINDOW = 365  # require 1 year of history for the expanding quantile threshold

# Focus country sets (Main Produceur; source, EIA https://www.eia.gov/tools/faqs/faq.php?id=709&t=6.) 
OIL_FOCUS_COUNTRIES = {
    "United States of America", "Saudi Arabia", "Russia", "Canada", "China",
    "Iraq","Brazil", "United Arab Emirates", "Iran", "Kuwait", "Qatar", "Norway",
    "Oman", "Libya", "Algeria", "Nigeria", "Angola", "Venezuela", "Azerbaijan"
}
# Focus country sets (Main Produceur) 
GAS_FOCUS_COUNTRIES = {
    "United States of America", "Russia", "Iran", "China", "Canada", "Qatar", 
    "Algeria", "Saudi Arabia", "Iraq", "Syria", "Norway"
}

def ewma(series: pd.Series, lam: float) -> pd.Series:
    # Exponentially Weighted Moving Average (EWMA)
    return series.ewm(alpha=(1 - lam), adjust=False).mean()

def build_daily_panels(
    input_file: Path,
    out_dir: Path,
    start_date: str = "1990-01-01",
    end_date: str = "2024-12-31",
):
    print(f"Building Indices from {input_file}")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    # Date parsing and basic cleaning
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce").fillna(0.0).clip(lower=0)

    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- INTERNAL HELPER: build a daily panel for any subset ---
    def process_panel(df_subset: pd.DataFrame, filename_suffix: str) -> pd.DataFrame:
        # Handle empty subsets: write a zero panel to keep the pipeline robust
        if df_subset.empty:
            print(f"[WARN] Empty subset for {filename_suffix}. Writing zeros.")
            daily = pd.DataFrame(index=all_days)
            daily.index.name = "Date"
            daily["fatal"] = 0.0
            daily["log_fatal"] = 0.0
            daily["fatal_shock"] = 0
            daily["Major"] = 0  # H3 regime dummy

            lam = LAMBDA_LIST[0]
            daily[f"log_fatal_ewma_{int(lam*100)}"] = 0.0

            out_path = out_dir / f"conflict_daily_{filename_suffix}.csv"
            daily.to_csv(out_path)
            return daily

        # Daily aggregation (sum of deaths), then reindex to full calendar
        daily = (
            df_subset.groupby("Date")
            .agg(fatal=("Deaths", "sum"))
            .reindex(all_days, fill_value=0.0)
        )
        daily.index.name = "Date"

        # Intensity proxy (log(1 + deaths))
        daily["log_fatal"] = np.log1p(daily["fatal"])

        # Expanding quantile shock indicator (anti-leakage via shift(1))
        threshold_series = (
            daily["fatal"]
            .expanding(min_periods=EXPANDING_WINDOW)
            .quantile(SHOCK_PERCENTILE)
            .shift(1)
        )

        daily["fatal_shock"] = 0
        mask = threshold_series.notna()
        daily.loc[mask, "fatal_shock"] = (
            daily.loc[mask, "fatal"] >= threshold_series.loc[mask]
        ).astype(int)

        # H3 regime dummy (keep as a clear 0/1 variable)
        daily["Major"] = daily["fatal_shock"]

        # Single EWMA intensity (fixed lambda)
        lam = LAMBDA_LIST[0]
        daily[f"log_fatal_ewma_{int(lam*100)}"] = ewma(daily["log_fatal"], lam)

        # Save panel
        out_path = out_dir / f"conflict_daily_{filename_suffix}.csv"
        daily.to_csv(out_path)
        
        return daily

    # A) GLOBAL PANEL
    g = process_panel(df, "global")

    # B) KEY REGIONS PANEL
    if "Region" in df.columns:
        df_reg = df.dropna(subset=["Region"]).copy()

        # Aggregate deaths by day and region
        r = (
            df_reg.groupby(["Date", "Region"])
            .agg(fatal=("Deaths", "sum"))
            .reset_index())

        r_fatal = (
            r.pivot(index="Date", columns="Region", values="fatal")
            .reindex(all_days)
            .fillna(0.0))

        out_reg = pd.DataFrame(index=all_days)
        out_reg.index.name = "Date"

        lam = LAMBDA_LIST[0]

        # For each region: log intensity, shock dummy, EWMA intensity
        for col in r_fatal.columns:
            safe_col = str(col).replace(" ", "_")
            series = r_fatal[col]

            log_series = np.log1p(series)
            out_reg[f"log_fatal_{safe_col}"] = log_series

            # Expanding-quantile shock (anti-leakage via shift(1))
            thresh = (
                series.expanding(min_periods=EXPANDING_WINDOW)
                .quantile(SHOCK_PERCENTILE)
                .shift(1)
            )

            shock_col = pd.Series(0, index=all_days)
            mask = thresh.notna()
            shock_col[mask] = (series[mask] >= thresh[mask]).astype(int)
            out_reg[f"shock_{safe_col}"] = shock_col

            # Single EWMA intensity
            out_reg[f"log_fatal_{safe_col}_ewma_{int(lam*100)}"] = ewma(log_series, lam)

        out_reg.to_csv(out_dir / "conflict_daily_by_region.csv")

    # C) FOCUS INDICES
    g_oil = pd.DataFrame()
    g_gas = pd.DataFrame()

    if "Country" in df.columns:
        df_oil = df[df["Country"].isin(OIL_FOCUS_COUNTRIES)]
        g_oil = process_panel(df_oil, "oil_focus")

        df_gas = df[df["Country"].isin(GAS_FOCUS_COUNTRIES)]
        g_gas = process_panel(df_gas, "gas_focus")

    # D) QUICK DESCRIPTIVE STATS
    def print_stats(name: str, series: pd.Series):
        if series is None or series.empty:
            print(f"{name}: [EMPTY/ERROR]")
            return
        desc = series.describe()
        ratio = desc["std"] / desc["mean"] if desc["mean"] != 0 else 0
        print(f"\n--- {name} ---")
        print(f"Mean : {desc['mean']:.4f}")
        print(f"Std  : {desc['std']:.4f}")
        print(f"Min  : {desc['min']:.4f}")
        print(f"Max  : {desc['max']:.4f}")
        print(f"CV (Std/Mean): {ratio:.2f}")

    # Focus stats (if present)
    if "log_fatal_ewma_94" in g_oil.columns:
        print_stats("OIL FOCUS", g_oil["log_fatal_ewma_94"])
    if "log_fatal_ewma_94" in g_gas.columns:
        print_stats("GAS FOCUS", g_gas["log_fatal_ewma_94"])

    # Middle East regional EWMA column check (if regions were built)
    if "out_reg" in locals():
        target_me = "log_fatal_Middle_East_ewma_94"
        if target_me in out_reg.columns:
            print_stats("MIDDLE EAST", out_reg[target_me])
        else:
            print(f"\n[INFO] Column '{target_me}' not found (check 'Regions found' output).")

# HELPER FOR FUTURE MERGE
def prepare_region_view(region_file: Path, region_name: str, out_path: Path) -> Path | None:
    # Extract a single region EWMA series and rename to a generic column name for merging
    if not region_file.exists():
        return None

    df = pd.read_csv(region_file)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    safe_reg = region_name.replace(" ", "_")

    # Single-lambda mapping (0.94 only)
    cols_map = {
        f"log_fatal_{safe_reg}_ewma_94": "log_fatal_ewma_94"}

    available = [c for c in cols_map.keys() if c in df.columns]
    if not available:
        return None

    view = df[["Date"] + available].rename(columns=cols_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    view.to_csv(out_path, index=False)
    return out_path