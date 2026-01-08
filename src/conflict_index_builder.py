import pandas as pd
import numpy as np
from pathlib import Path

LAMBDA = 0.94  # Single EWMA decay (simple and reproducible)
KEEP_REGIONS = {"europe", "middle_east"}  # H2 only

def ewma(series: pd.Series, lam: float):
    # EWMA smooths conflict intensity while emphasizing recent events.
    return series.ewm(alpha=(1 - lam), adjust=False).mean() # Exponentially weighted moving average of conflict intensity.

def build_daily_panels(
    input_file: Path,
    out_dir: Path,
    start_date: str = "1990-01-01",
    end_date: str = "2024-12-31"):

    input_file = Path(input_file)
    out_dir = Path(out_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    # Basic typing and cleaning (robust to malformed entries)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Deaths"] = (
        pd.to_numeric(df["Deaths"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)) # <- IA suggestion, ensures deaths are numeric, non-missing, and non-negative.

    # Full daily calendar so conflict indices align with commodity price dates
    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    out_dir.mkdir(parents=True, exist_ok=True)

    def process_panel(df_subset: pd.DataFrame, filename_suffix: str):
        # Aggregate event-level data to daily totals, then align to full calendar
        daily = (
            df_subset.groupby("Date")
            .agg(deaths=("Deaths", "sum"))
            .reindex(all_days, fill_value=0.0))
        
        daily.index.name = "Date"

        # Intensity proxy (log(1 + deaths)) + smoothing (EWMA)
        daily["log_deaths"] = np.log1p(daily["deaths"])
        daily[f"{filename_suffix}__log_deaths_ewma_{int(LAMBDA*100)}"] = ewma(daily["log_deaths"], LAMBDA)
        # Log transformation stabilizes extreme counts; EWMA smooths intensity and emphasizes recent conflicts

        out_path = out_dir / f"conflict_daily_{filename_suffix}.csv"
        daily.to_csv(out_path)
        return daily

    # A) Global panel (H1 + Gold in H2)
    process_panel(df, "global")

    # B) Region panels (H2): keep ONLY Europe and Middle East
    if "Region" in df.columns:
        # For H2, we create one daily series per region to test location-specific effects (Middle East + Europe)
        r = (
            df.groupby(["Date", "Region"])
            .agg(deaths=("Deaths", "sum"))
            .reset_index())

        # Region names become column suffixes so each region can be selected later without manual rewriting
        r_deaths = (
            r.pivot(index="Date", columns="Region", values="deaths")
            .reindex(all_days)
            .fillna(0.0))

        out_reg = pd.DataFrame(index=all_days)
        out_reg.index.name = "Date"

        for col in r_deaths.columns:
            safe_col = str(col).replace(" ", "_").lower()

            # Keep only the two regions needed for H2
            if safe_col not in KEEP_REGIONS:
                continue

            series = r_deaths[col]
            log_series = np.log1p(series)

            # (IA suggestion: Stable, prefixed name (recommended for merges + lag creation)
            out_reg[f"{safe_col}__log_deaths_ewma_{int(LAMBDA*100)}"] = ewma(
                log_series, LAMBDA)

        out_reg.to_csv(out_dir / "conflict_daily_by_region.csv")