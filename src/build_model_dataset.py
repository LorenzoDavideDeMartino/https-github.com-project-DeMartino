from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np

# This module builds the final dataset used for HAR-X regressions.
# It combines financial volatility features with lagged conflict intensity indices.

def load_features_ready(path: Path):
    # We keep feature construction separate from merging so each step stays simple and reproducible.

    df = pd.read_csv(path)

    # Dates are enforced and sorted so the dataset is a proper time series before any shift/merge.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Normalizing to midnight avoids merge mismatches caused by hidden time components.
    df["Date"] = df["Date"].dt.normalize()

    # HAR regressions require these variables; failing early avoids silent mistakes later.
    required_cols = {"Date", "Price", "RV_Daily", "RV_Weekly", "RV_Monthly"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    return df.reset_index(drop=True)


def load_conflict_index(path: Path, cols_keep: list[str]):
    # Conflict indices are merged as external regressors, so they must have clean daily dates.
    conflict_df = pd.read_csv(path)

    # We standardize dates to the same daily format as the commodity dataset to ensure a clean join key.
    conflict_df["Date"] = pd.to_datetime(conflict_df["Date"], errors="coerce")
    conflict_df["Date"] = conflict_df["Date"].dt.normalize()

    conflict_df= (
        conflict_df.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates("Date")) # We enforce one row per day to avoid duplicate merges that would distort the regressions.

    # We keep only the columns needed for the model so the merge stays lightweight and transparent.
    actual_cols = [col for col in cols_keep if col in conflict_df.columns]
    if len(actual_cols) < len(cols_keep):
        # We warn instead of failing so the pipeline remains usable across slightly different inputs.
        missing = set(cols_keep) - set(actual_cols)
        print(f"Attention: Missing conflict columns: {missing}")

    return conflict_df[["Date"] + actual_cols].reset_index(drop=True )


def build_dataset_for_commodity(
    commodity_name: str,
    commodity_features_csv: Path,
    conflict_files: dict[str, Path],
    conflict_cols: list[str],
    conflict_lags: list[int] = [0, 1],  # lag0 = today, lag1 = yesterday
    out_path: Path | None = None ):
    # This function assembles the final regression table used for HAR vs HAR-X comparisons.

    print(f"Building dataset for {commodity_name}...")

    # Load the HAR components (RV daily/weekly/monthly) that form the baseline model
    df = load_features_ready(commodity_features_csv)

    # The target is next-day volatility, so the model forecasts t+1 using information available at t
    df["Target_RV"] = df["RV_Daily"].shift(-1)

    # Conflict indices are added as X variables; each source is merged and lagged to avoid look-ahead
    conflict_features: list[str] = []

    for tag, fpath in conflict_files.items():
        # Build expected column names to load from the conflict file
        cols_to_load = []
        for col in conflict_cols:
            if "__" in col:
                cols_to_load.append(col)
            else:
                cols_to_load.append(f"{tag}__{col}")

        conflict_df = load_conflict_index(fpath, cols_to_load)

        # A left join keeps the full commodity time series and simply adds conflict information when available
        df = df.merge(conflict_df, on="Date", how="left")

        # Missing conflict entries are treated as zero intensity rather than dropping trading days
        new_cols = [conflict_df for conflict_df in cols_to_load if conflict_df in df.columns]
        df[new_cols] = df[new_cols].fillna(0.0)

        # Lags ensure regressors only use information that would have been known at prediction time
        for col in new_cols:
            for L in conflict_lags:
                lag_name = f"{col}_lag{L}"
                df[lag_name] = df[col].shift(L)
                conflict_features.append(lag_name)

        # We drop contemporaneous columns to keep the dataset strictly “lagged-only” for forecasting
        df = df.drop(columns=new_cols)

    # The final dataset is restricted to the baseline HAR variables plus lagged conflict regressors
    base_cols = ["Date", "Price", "Target_RV", "RV_Daily", "RV_Weekly", "RV_Monthly"]
    
    conflict_features = list(dict.fromkeys(conflict_features)) # <- IA Input, # Remove potential duplicate conflict variables while preserving their original order
    final_cols = base_cols + conflict_features

    # Rows with undefined targets or regressors cannot be used for estimation and evaluation
    final_df = (
        df[final_cols]
        # Replacing inf values avoids numerical issues during OLS estimation
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True))

    # A short summary helps confirm the dataset looks correct before running the model
    if final_df.empty:
        print("ERROR: Dataset empty after cleaning!")
    else:
        print(f"Date range: {final_df['Date'].min().date()} - {final_df['Date'].max().date()}")

    # Saving to disk ensures results can be reproduced without rebuilding everything from scratch
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_path, index=False)
        print(f"Saved {out_path.name}")

    return final_df