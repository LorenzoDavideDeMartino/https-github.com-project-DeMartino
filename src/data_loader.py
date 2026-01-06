"""
Data cleaning pipeline for Investing.com raw CSV files.

Raw CSV files downloaded from Investing.com are not standard CSV files.
They often contain a single column with comma-separated values, mixed numeric
formats, and inconsistent headers. Centralizing all cleaning logic here
improves reproducibility and avoids fragile fixes later in the project.

Assumptions:
- Files come from the [IMPORTANT] US VERSION of Investing.com.
- Dates follow the US format: MM/DD/YYYY.
- Any deviation from this format is converted to missing values on purpose
  to avoid silent parsing errors.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# 1. Numeric & Date Cleaning
def _convert_numeric(value: object):
    # Investing.com mixes commas, percentages, and K/M suffixes in numeric fields.
    # Returning NaN on failure makes data issues explicit and traceable. (<-IA Input)
    if not isinstance(value, str) or not value:
        if isinstance(value, (int, float)):
            return float(value)
        return np.nan

    try:
        v = value.strip()

        # Percentage values are stored as strings and must be cleaned explicitly.
        if "%" in v:
            v = v.replace("%", "")
            return float(v.replace(",", ""))

        u = v.upper()

        # Volumes with K suffix represent thousands and should be scaled explicitly.
        if "K" in u:
            u = u.replace("K", "")
            return round(float(u.replace(",", "")) * 1_000.0)

        # Volumes with M suffix represent millions and should be scaled explicitly.
        if "M" in u:
            u = u.replace("M", "")
            return round(float(u.replace(",", "")) * 1_000_000.0)

        return float(v.replace(",", ""))
    except Exception:
        # Failed parsing is made explicit instead of silently injecting noise. (<-IA Input)
        return np.nan


def _standardize_date(date_str: object):
    # Dates are normally in US format, but explicit parsing is used as a safeguard.
    # Invalid formats are converted to NaT instead of being guessed.
    if not isinstance(date_str, str) or not date_str:
        return pd.NaT

    s = date_str.strip('"\' ')
    return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")

# 2. Raw CSV Parsing (<- IA Input; safeguard in case a file contains extra rows)
def _find_header_row_idx(raw_df: pd.DataFrame) -> Optional[int]:
    # Some files include metadata rows before the actual header.
    # Searching for "Date" ensures the correct header is used.
    for idx, row in raw_df.iterrows():
        v = row.iloc[0]
        if isinstance(v, str) and "Date" in v:
            return idx
    return None


def _parse_header(header_row: str):
    # Column names are stored as a single comma-separated string. This step extracts clean column names from that string.
    return [col.strip('"\' ') for col in header_row.split(",")]


def _split_row(row_str: str, n_cols: int):
    # Quoted values may contain commas, which would break simple splitting.
    # A simple state-based parser avoids column shifts and malformed rows.
    if not isinstance(row_str, str):
        return [None] * n_cols

    values: List[str] = []
    current = ""
    in_quotes = False

    for char in row_str:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            values.append(current.strip('"\' '))
            current = ""
            continue
        current += char

    if current:
        values.append(current.strip('"\' '))

    # This prevents malformed rows from breaking the downstream pipeline. (<- IA suggestion)
    if len(values) < n_cols:
        values.extend([None] * (n_cols - len(values)))
    return values[:n_cols]


def read_investing_raw_csv(path: Path):
    # Raw files are read entirely as strings to avoid implicit pandas casting.
    # All conversions are handled explicitly and consistently.
    try:
        raw_df = pd.read_csv(path, header=None, low_memory=False, dtype=str)
    except pd.errors.EmptyDataError:
        print(f"Attention: Empty file ignored: {path.name}")
        return pd.DataFrame()

    # Locate the header row dynamically to handle variable file structures. (Same IA suggestion as for the _find_header_row_idx)
    header_row_idx = _find_header_row_idx(raw_df)
    if header_row_idx is None:
        header_row_idx = 0
        header_row = raw_df.iloc[0, 0]
    else:
        header_row = raw_df.iloc[header_row_idx, 0]

    if not isinstance(header_row, str):
        header_row = str(header_row)

    column_names = _parse_header(header_row)
    n_cols = len(column_names)

    # Raw files store each row as a single string; rows are rebuilt manually
    # Manual parsing is used to keep full control over column alignment.
    data_rows: List[List[Optional[str]]] = []
    for idx, row in raw_df.iterrows():
        if idx == header_row_idx:
            continue
        v = row.iloc[0]
        if not isinstance(v, str):
            continue
        data_rows.append(_split_row(v, n_cols))

    df = pd.DataFrame(data_rows, columns=column_names)

    # Standardize column names to a canonical format. Investing.com uses "Last" for the closing price; renaming to "Price
    rename_map = {
        "Last": "Price",
        "Vol.": "Vol.",
        "Volume": "Vol.",
        "Var. %": "Change %"
    }
    df = df.rename(columns=rename_map)

    # Without a Date column, the data cannot be used for time-series analysis.
    if "Date" not in df.columns:
        print(
            f"Attention: 'Date' column not found in -> {path.name} "
        )
        return pd.DataFrame()

    # Convert dates using the strict US format.
    df["Date"] = df["Date"].apply(_standardize_date)

    # Convert numeric columns explicitly.
    cols_numeric = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    for col in cols_numeric:
        if col in df.columns:
            df[col] = df[col].apply(_convert_numeric)

    # Rows without valid dates are unusable and removed. We don't take the risk. 
    df = df.dropna(subset=["Date"]).copy()

    # Price is the core variable for volatility estimation and must be present. Normally is shouldn't happen but just to be safe.
    if "Price" in df.columns:
        df = df.dropna(subset=["Price"])

    # Sorting ensures chronological consistency.
    df = df.sort_values("Date")

    # A single observation per date avoids double counting. (There is possible that there are duplicates in the different raw csv)
    df = df.drop_duplicates(subset=["Date"], keep="first")

    # Keep a stable and predictable column order.
    ordered = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    keep = [c for c in ordered if c in df.columns]

    return df.loc[:, keep]

# 3. Main Pipeline
def build_clean_commodity_from_parts(parts_dir: Path, out_file: Path):
    # Large downloads are often split into multiple files.
    # Merging all parts ensures full historical coverage.
    parts_dir = Path(parts_dir)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True) # (<- IA Input: Prevents save errors if the output directory does not exist.)

    # The pipeline cannot run without input files, so we fail early and explicitly.
    files = sorted(parts_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV parts found in: {parts_dir}")

    frames: List[pd.DataFrame] = []
    for f in files:
        df_part = read_investing_raw_csv(f)
        if not df_part.empty:
            frames.append(df_part)

    # If no valid data is extracted, the issue should be visible immediately.
    if not frames:
        print(f"Attention: No valid data extracted from {parts_dir}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Losing the Date column would invalidate the entire pipeline.
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column lost after merge in: {parts_dir}")

    # Final sorting and deduplication act as a safety check after merging.
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="first")

    ordered = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    keep = [c for c in ordered if c in df.columns]
    df = df.loc[:, keep]
    # All checks are applied to ensure a clean and reliable merged dataset.

    # Saving a single clean CSV guarantees reproducibility downstream.
    df.to_csv(out_file, index=False)
    return df
