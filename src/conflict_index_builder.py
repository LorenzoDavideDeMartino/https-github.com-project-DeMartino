import pandas as pd
import numpy as np
from pathlib import Path

LAMBDA = 0.94  # Single EWMA decay (simple and reproducible)

# Focus country sets (Main producers, top 10; source: EIA)
OIL_FOCUS_COUNTRIES = {
    "United States of America", "Saudi Arabia", "Russia", "Canada", "China",
    "Iraq","Brazil", "United Arab Emirates", "Iran", "Kuwait"}

GAS_FOCUS_COUNTRIES = {
    "United States of America", "Russia", "Iran", "China", "Canada", "Qatar",
    "Australia", "Saudi Arabia", "Norway", "Algeria"}


def ewma(series: pd.Series, lam: float):
    # EWMA smooths conflict intensity while emphasizing recent events.
    return series.ewm(alpha=(1 - lam), adjust=False).mean() # Exponentially weighted moving average of conflict intensity.


def build_daily_panels(
    input_file: Path,
    out_dir: Path,
    start_date: str = "1990-01-01", 
    end_date: str = "2024-12-31"):

    print(f"Building Indices from {input_file}")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    # Basic typing and cleaning (robust to malformed entries)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce").fillna(0.0).clip(lower=0) # <- IA suggestion, ensures deaths are numeric, non-missing, and non-negative.

    # A full daily calendar is used so conflict indices align exactly with commodity price dates during merges.
    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daily conflict intensity panel.
    def process_panel(df_subset: pd.DataFrame, filename_suffix: str):
        # Aggregate event-level data to daily totals, then align to full calendar
        daily = (
            df_subset.groupby("Date")
            .agg(fatal=("Deaths", "sum"))
            .reindex(all_days, fill_value=0.0))
        
        daily.index.name = "Date"

        # Intensity proxy (log(1 + deaths)) + smoothing (EWMA)
        daily["log_fatal"] = np.log1p(daily["fatal"])
        daily[f"log_fatal_ewma_{int(LAMBDA*100)}"] = ewma(daily["log_fatal"], LAMBDA) 
        # Log transformation stabilizes extreme counts; EWMA smooths intensity and emphasizes recent conflicts

        out_path = out_dir / f"conflict_daily_{filename_suffix}.csv"
        daily.to_csv(out_path)
        return daily

    # A) GLOBAL PANEL (H1)
    g = process_panel(df, "global")

    # B) KEY REGIONS PANEL (H2)
    # Keep only continuous intensity measures (log and EWMA) for each region
    if "Region" in df.columns:
        df_reg = df.copy()

        # For H2, we create one daily series per region to test location-specific effects (e.g., Middle East vs Europe)
        r = (
            df_reg.groupby(["Date", "Region"])
            .agg(fatal=("Deaths", "sum"))
            .reset_index())

        # Region names become column suffixes so each region can be selected later without manual rewriting
        r_fatal = (
            r.pivot(index="Date", columns="Region", values="fatal")
            .reindex(all_days)
            .fillna(0.0))

        out_reg = pd.DataFrame(index=all_days)
        out_reg.index.name = "Date"

        for col in r_fatal.columns:
            safe_col = str(col).replace(" ", "_") # Spaces are replaced to produce valid and consistent column names
            series = r_fatal[col]
            log_series = np.log1p(series)

            out_reg[f"log_fatal_{safe_col}"] = log_series
            out_reg[f"log_fatal_{safe_col}_ewma_{int(LAMBDA*100)}"] = ewma(log_series, LAMBDA)

        out_reg.to_csv(out_dir / "conflict_daily_by_region.csv")

    # C) FOCUS INDICES (H2)
    # Focus panels restrict conflicts to key producer countries to proxy commodity-specific exposure (H2).
    g_oil = pd.DataFrame()
    g_gas = pd.DataFrame()

    if "Country" in df.columns:
        df_oil = df[df["Country"].isin(OIL_FOCUS_COUNTRIES)]
        g_oil = process_panel(df_oil, "oil_focus")

        df_gas = df[df["Country"].isin(GAS_FOCUS_COUNTRIES)]
        g_gas = process_panel(df_gas, "gas_focus")

def prepare_region_view(region_file: Path, region_name: str, out_path: Path):
      # Extract a single region EWMA series and rename to a generic column name for merging
      if not region_file.exists():
          return None
 
      df = pd.read_csv(region_file)
      df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
 
      safe_reg = region_name.replace(" ", "_")
      cols_map = {f"log_fatal_{safe_reg}_ewma_94": "log_fatal_ewma_94"}
      available = [c for c in cols_map.keys() if c in df.columns]
      if not available:
          return None

      view = df[["Date"] + available].rename(columns=cols_map)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      view.to_csv(out_path, index=False)
      return out_path
