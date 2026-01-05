import pandas as pd
import numpy as np
from pathlib import Path

# Load already-prepared commodity features (HAR + Garman-Klass)
def load_features_ready(path: Path) -> pd.DataFrame:
    # Load the commodity FEATURES file.
    # Expected columns:
    # - Date, Price, RV_Daily, RV_Weekly, RV_Monthly
    
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)

    # Date parsing (ISO format assumed)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Normalize to midnight for safe merges
    df["Date"] = df["Date"].dt.normalize()

    # Mandatory column check
    required_cols = {"Date", "Price", "RV_Daily", "RV_Weekly", "RV_Monthly"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    return df.reset_index(drop=True)

# Load conflict indices
def load_conflict_index(path: Path, cols_keep: list[str]) -> pd.DataFrame:
    # Load a conflict index file and keep only selected columns.
    
    if not path.exists():
        raise FileNotFoundError(f"Conflict file not found: {path}")

    c = pd.read_csv(path)

    c["Date"] = pd.to_datetime(c["Date"], errors="coerce")
    c["Date"] = c["Date"].dt.normalize()

    c = (
        c.dropna(subset=["Date"])
         .sort_values("Date")
         .drop_duplicates("Date")
    )

    actual_cols = [col for col in cols_keep if col in c.columns]
    if len(actual_cols) < len(cols_keep):
        missing = set(cols_keep) - set(actual_cols)
        print(f"Attention: Missing conflict columns: {missing}")

    return c[["Date"] + actual_cols].reset_index(drop=True)

# Build final HAR + Conflict modeling dataset
def build_dataset_for_commodity(
    commodity_name: str,
    commodity_features_csv: Path,
    conflict_files: dict[str, Path],
    conflict_cols: list[str],
    conflict_lags: list[int] = [0, 1],  # lag0 = today, lag1 = yesterday
    out_path: Path | None = None,
) -> pd.DataFrame:

    print(f"Building dataset for {commodity_name}...")

    # 1) Load financial features (already computed!)
    df = load_features_ready(commodity_features_csv)

    # 2) Define TARGET: next-day volatility (HAR standard)
    # Target_t = RV_Daily_{t+1}
    df["Target_RV"] = df["RV_Daily"].shift(-1)

    # 3) Merge conflict indices
    for tag, fpath in conflict_files.items():
        if fpath is None or not fpath.exists():
            continue

        c = load_conflict_index(fpath, conflict_cols)

        # Rename to avoid collisions (e.g. oil_focus__log_fatal_ewma_94)
        rename_map = {col: f"{tag}__{col}" for col in conflict_cols}
        c = c.rename(columns=rename_map)

        # Left join: keep all trading days
        df = df.merge(c, on="Date", how="left")

        # No conflict that day = 0
        new_cols = list(rename_map.values())
        df[new_cols] = df[new_cols].fillna(0.0)

        # Create lags (anti-leakage)
        for col in new_cols:
            for L in conflict_lags:
                df[f"{col}_lag{L}"] = df[col].shift(L)

        # Drop raw (non-lagged) conflict columns
        df = df.drop(columns=new_cols)

    # 4) Final column selection
    base_cols = [
        "Date",
        "Price",
        "Target_RV",
        "RV_Daily",
        "RV_Weekly",
        "RV_Monthly",
    ]

    conflict_features = [c for c in df.columns if "__" in c and "lag" in c]
    final_cols = base_cols + conflict_features

    # 5) Final cleanup
    out = (
        df[final_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    # Validation
    if out.empty:
        print("ERROR: Dataset empty after cleaning!")
    else:
        print(f"Date range: {out['Date'].min().date()} → {out['Date'].max().date()}")
        print(f"Rows: {len(out)} | Columns: {len(out.columns)}")
        print(f"Conflict features: {conflict_features}")

    # Save
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"Saved {out_path.name}")

    return out
