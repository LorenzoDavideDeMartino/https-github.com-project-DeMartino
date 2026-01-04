import pandas as pd
import numpy as np
from pathlib import Path

def load_commodity_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Commodity file not found: {path}")
    df = pd.read_csv(path)
    # Robust date parsing (Dayfirst=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    # [ADJUSTMENT 2] Normalize date to remove time component (00:00:00) for clean merges
    df["Date"] = df["Date"].dt.normalize()
    
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Price"])[["Date", "Price"]].reset_index(drop=True)

def add_returns_and_rv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Log returns
    df["r"] = np.log(df["Price"] / df["Price"].shift(1))
    
    # [ADJUSTMENT 1] Handle infinite values if Price was 0 (rare but possible)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df["r2"] = df["r"] ** 2
    
    # HAR inputs (Past volatility)
    df["RV_1"]  = df["r2"]
    df["RV_5"]  = df["r2"].rolling(5).sum()
    df["RV_21"] = df["r2"].rolling(21).sum()
    
    # Target: Future volatility (21 days)
    # First Rolling Forward, then Shift
    # This captures the sum of r^2 from t+1 to t+21
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    df["RV_fut_21"] = df["r2"].rolling(window=indexer).sum().shift(-1)
    
    return df

def load_conflict_index(path: Path, cols_keep: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Conflict file not found: {path}")
    c = pd.read_csv(path)
    # Robust date parsing
    c["Date"] = pd.to_datetime(c["Date"], errors="coerce", dayfirst=True)
    # [ADJUSTMENT 2] Normalize date to ensure match with commodity dates
    c["Date"] = c["Date"].dt.normalize()
    
    c = c.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
    # Keep only requested columns + Date
    return c[["Date"] + [col for col in cols_keep if col in c.columns]].reset_index(drop=True)

def build_dataset_for_commodity(
    commodity_name: str,
    commodity_csv: Path,
    conflict_files: dict[str, Path],
    conflict_cols: list[str],
    conflict_lags=(1, 5),
    out_path: Path | None = None,
) -> pd.DataFrame:
    
    print(f"   [Merge] Processing {commodity_name}...")
    df = load_commodity_clean(commodity_csv)
    df = add_returns_and_rv(df)

    # 1. Merge conflict files (Left Join)
    for tag, fpath in conflict_files.items():
        if fpath is None or not fpath.exists(): continue
        c = load_conflict_index(fpath, conflict_cols)
        # Rename to avoid collisions (e.g., oil_focus__log_fatal...)
        c = c.rename(columns={col: f"{tag}__{col}" for col in conflict_cols})
        df = df.merge(c, on="Date", how="left")

    # 2. Fill conflict NaNs with 0 BEFORE creating lags
    # Identify conflict columns using "__" separator
    # If left join gives NaN, it means no conflict that day -> 0
    merged_conf_cols = [c for c in df.columns if "__" in c]
    if merged_conf_cols:
        df[merged_conf_cols] = df[merged_conf_cols].fillna(0.0)

    # 3. Create Lags (Anti-Leakage)
    for col in merged_conf_cols:
        for L in conflict_lags:
            df[f"{col}_lag{L}"] = df[col].shift(L)

    # 4. Final column selection
    # r2 included for GARCH/Diagnostics
    keep = ["Date", "Price", "r", "r2", "RV_1", "RV_5", "RV_21", "RV_fut_21"]
    keep += [f"{col}_lag{L}" for col in merged_conf_cols for L in conflict_lags]
    
    # 5. Final cleanup (remove gaps created by lags/rolling/inf)
    # Now, dropna only removes series start/end (rolling),
    # not the peaceful days.
    out = df[[k for k in keep if k in df.columns]].dropna().reset_index(drop=True)
    
    # [MANDATORY CHECK] Data validation
    if not out.empty:
        print(f"   Date range: {out['Date'].min().date()} -> {out['Date'].max().date()}")
        print(f"   Columns: {len(out.columns)}")
    else:
        print("   [WARN] Dataset is empty after dropna!")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"   [Saved] {out_path.name} ({len(out)} rows)")

    return out