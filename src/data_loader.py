"""
Data cleaning pipeline for Investing.com raw CSV files.

Assumption:
- All input files originate from the US version of Investing.com.
- Dates strictly follow the US format: MM/DD/YYYY.

Any deviation from this format will result in missing values (NaT),
by design, to avoid silent misparsing.
"""

# Raw parts are expected to be the untouched Investing.com downloads:
# - usually a single column per row containing comma-separated fields
# - numeric fields with commas, %, K/M suffixes in Volume, etc.
# Output: one clean CSV per commodity with columns:
# Date, Price, Open, High, Low, Vol., Change %

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd

# 1. Numeric & Date Cleaning
def _convert_numeric(value: object) -> float:
    
    # Robust numeric conversion (US format with ',' for thousands).
    # Handles %, K, M suffixes and rounds volumes to avoid floating point errors.
    
    if not isinstance(value, str) or not value:
        if isinstance(value, (int, float)):
            return float(value)
        return np.nan

    try:
        v = value.strip()
        if "%" in v:
            v = v.replace("%", "")
            return float(v.replace(",", ""))

        u = v.upper()
        if "K" in u:
            u = u.replace("K", "")
            return round(float(u.replace(",", "")) * 1000.0)

        if "M" in u:
            u = u.replace("M", "")
            return round(float(u.replace(",", "")) * 1_000_000.0)

        return float(v.replace(",", ""))
    except Exception:
        return np.nan

def _standardize_date(date_str: object) -> pd.Timestamp:

    # Parses a date string assuming strictly US format (MM/DD/YYYY).
    if not isinstance(date_str, str) or not date_str:
        return pd.NaT

    s = date_str.strip('"\' ')
    return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")


# 2. Raw CSV Parsing
def _find_header_row_idx(raw_df: pd.DataFrame) -> Optional[int]:
    #Find row index containing the header (looking for 'Date')
    for idx, row in raw_df.iterrows():
        v = row.iloc[0]
        if isinstance(v, str) and "Date" in v:
            return idx
    return None

def _parse_header(header_row: str) -> List[str]:
    return [col.strip('"\' ') for col in header_row.split(",")]

def _split_row(row_str: str, n_cols: int) -> List[Optional[str]]:
    #Split a raw CSV line handling quotes properly.
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

    if len(values) < n_cols:
        values.extend([None] * (n_cols - len(values)))
    return values[:n_cols]

def read_investing_raw_csv(path: Path) -> pd.DataFrame:
    
    # Reads a raw Investing.com CSV file.
    # Prints a warning if the file is unusable.
    
    # Force read as string to handle bad formatting
    try:
        raw_df = pd.read_csv(path, header=None, low_memory=False, dtype=str)
    except pd.errors.EmptyDataError:
        print(f"WARNING: Empty file ignored -> {path.name}")
        return pd.DataFrame()

    # Find Header
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

    # Parse rows
    data_rows: List[List[Optional[str]]] = []
    for idx, row in raw_df.iterrows():
        if idx == header_row_idx:
            continue
        v = row.iloc[0]
        if not isinstance(v, str):
            continue
        data_rows.append(_split_row(v, n_cols))

    df = pd.DataFrame(data_rows, columns=column_names)

    # Standardize column names
    rename_map = {
        "Last": "Price", "Dernier": "Price",
        "Vol.": "Vol.", "Volume": "Vol.",
        "Var. %": "Change %"
    }
    df = df.rename(columns=rename_map)

    # Critical Check
    if "Date" not in df.columns:
        print(f"WARNING: 'Date' column not found in -> {path.name} (Columns found: {df.columns.tolist()})")
        return pd.DataFrame()

    # Conversions
    df["Date"] = df["Date"].apply(_standardize_date)

    cols_numeric = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    for col in cols_numeric:
        if col in df.columns:
            df[col] = df[col].apply(_convert_numeric)

    # Final Cleaning
    # Drop rows without Date
    df = df.dropna(subset=["Date"]).copy()
    
    # Drop rows without Price (Volume can be missing, but Price is mandatory)
    if "Price" in df.columns:
         df = df.dropna(subset=["Price"])

    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="first")
    
    # Keep canonical order
    ordered = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    keep = [c for c in ordered if c in df.columns]
    
    return df.loc[:, keep]


# 3. Main Pipeline
def build_clean_commodity_from_parts(parts_dir: Path, out_file: Path) -> pd.DataFrame:
    
    # Merges all CSV parts from a folder into a single clean file.
    parts_dir = Path(parts_dir)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(parts_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV parts found in: {parts_dir}")

    frames: List[pd.DataFrame] = []
    for f in files:
        df_part = read_investing_raw_csv(f)
        if not df_part.empty:
            frames.append(df_part)

    if not frames:
        print(f"ERROR: No valid data extracted from {parts_dir}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    if "Date" not in df.columns:
        raise ValueError(f"'Date' column lost after merge in: {parts_dir}")
    
    # Final Dedupe
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="first")

    ordered = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    keep = [c for c in ordered if c in df.columns]
    df = df.loc[:, keep]

    df.to_csv(out_file, index=False)
    return df