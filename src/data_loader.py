# src/data_loader.py
# Build clean commodity price series from Investing.com "raw" CSV parts.
# Raw parts are expected to be the untouched Investing.com downloads:
# - usually a single column per row containing comma-separated fields
# - mixed date formats (MM.DD.YYYY and/or MM/DD/YYYY)
# - numeric fields with commas, %, K/M suffixes in Volume, etc.
#
# Output: one clean CSV per commodity with columns:
# Date, Price, Open, High, Low, Vol., Change %
#
# Usage is typically from main.py via build_clean_commodity_from_parts().

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Core parsing (your logic)
# -----------------------------
def _find_header_row_idx(raw_df: pd.DataFrame) -> Optional[int]:
    """Find row index containing header (string with 'Date')."""
    for idx, row in raw_df.iterrows():
        v = row.iloc[0]
        if isinstance(v, str) and "Date" in v:
            return idx
    return None


def _parse_header(header_row: str) -> List[str]:
    return [col.strip('"\' ') for col in header_row.split(",")]


def _split_row(row_str: str, n_cols: int) -> List[Optional[str]]:
    """Split one CSV-like line while respecting quotes."""
    if not isinstance(row_str, str):
        return [None] * n_cols

    values: List[str] = []
    current = ""
    in_quotes = False

    for char in row_str:
        if char in ['"', "'"]:
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            values.append(current.strip('"\' '))
            current = ""
            continue
        current += char

    if current:
        values.append(current.strip('"\' '))

    # pad / trim
    if len(values) < n_cols:
        values.extend([None] * (n_cols - len(values)))
    return values[:n_cols]


def _standardize_date(date_str: object) -> pd.Timestamp:
    """Your date standardization: MM.DD.YYYY -> MM/DD/YYYY, keep slashes, fallback formats."""
    if not isinstance(date_str, str) or not date_str:
        return pd.NaT

    s = date_str.strip('"\' ')

    if "." in s:
        # MM.DD.YYYY
        parts = s.split(".")
        if len(parts) == 3:
            month, day, year = parts
            try:
                return pd.to_datetime(f"{month}/{day}/{year}", format="%m/%d/%Y")
            except Exception:
                return pd.NaT

    if "/" in s:
        # MM/DD/YYYY
        try:
            return pd.to_datetime(s, format="%m/%d/%Y")
        except Exception:
            return pd.NaT

    # fallback
    for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y"]:
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue

    return pd.NaT


def _convert_numeric(value: object) -> float:
    """Your numeric conversion: %, K/M suffix, commas."""
    if not isinstance(value, str) or not value:
        return np.nan

    try:
        v = value.strip()

        if "%" in v:
            v = v.replace("%", "")
            return float(v.replace(",", ""))

        u = v.upper()
        if "K" in u:
            u = u.replace("K", "")
            return float(u.replace(",", "")) * 1000.0

        if "M" in u:
            u = u.replace("M", "")
            return float(u.replace(",", "")) * 1_000_000.0

        return float(v.replace(",", ""))
    except Exception:
        return np.nan


def read_investing_raw_csv(path: Path) -> pd.DataFrame:
    """
    Read one Investing.com raw CSV file (often 1-column) and return a cleaned DataFrame.
    """
    raw_df = pd.read_csv(path, header=None, low_memory=False)

    header_row_idx = _find_header_row_idx(raw_df)
    if header_row_idx is None:
        header_row_idx = 0
        header_row = raw_df.iloc[0, 0]
    else:
        header_row = raw_df.iloc[header_row_idx, 0]

    if not isinstance(header_row, str):
        raise ValueError(f"Header row not found / invalid in file: {path}")

    column_names = _parse_header(header_row)
    n_cols = len(column_names)

    data_rows: List[List[Optional[str]]] = []
    for idx, row in raw_df.iterrows():
        if idx == header_row_idx:
            continue
        v = row.iloc[0]
        if not isinstance(v, str):
            continue
        data_rows.append(_split_row(v, n_cols))

    df = pd.DataFrame(data_rows, columns=column_names)

    # Clean types
    if "Date" in df.columns:
        df["Date"] = df["Date"].apply(_standardize_date)

    for col in ["Price", "Open", "High", "Low", "Vol.", "Change %"]:
        if col in df.columns:
            df[col] = df[col].apply(_convert_numeric)

    # Drop invalid dates
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found after parsing: {path}")

    df = df.dropna(subset=["Date"]).copy()

    # Sort / drop duplicate dates (keep first)
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="first")

    # Keep canonical column order if present
    ordered = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    keep = [c for c in ordered if c in df.columns]
    df = df.loc[:, keep]

    return df


# -----------------------------
# Merge parts (multiple downloads)
# -----------------------------
def build_clean_commodity_from_parts(parts_dir: Path, out_file: Path) -> pd.DataFrame:
    """
    Merge many Investing.com raw CSV parts into one clean CSV.
    - Reads each *.csv in parts_dir
    - Cleans each
    - Concatenates and drops duplicate dates (keeps first occurrence)
    - Sorts by Date
    - Writes out_file
    """
    parts_dir = Path(parts_dir)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(parts_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV parts found in: {parts_dir}")

    frames: List[pd.DataFrame] = []
    for f in files:
        df_part = read_investing_raw_csv(f)
        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True)

    # Final dedupe after concatenation (important)
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="first")

    df.to_csv(out_file, index=False)
    return df


def add_log_returns(df: pd.DataFrame, price_col: str = "Price") -> pd.DataFrame:
    df = df.copy()
    df["return"] = np.log(df[price_col]).diff()
    return df


def add_realized_volatility(
    df: pd.DataFrame,
    window: int = 21,
    return_col: str = "return",
    annualize: bool = False,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Realized volatility proxy: sqrt(sum_{i=1..window} r_{t-i}^2)
    (you used sum of squared returns; sqrt is standard if you want volatility in same units as returns)
    Set annualize=True if needed later.
    """
    df = df.copy()
    rv = df[return_col].pow(2).rolling(window=window).sum()
    rv = np.sqrt(rv)
    if annualize:
        rv = rv * np.sqrt(trading_days)
    df[f"rv_{window}"] = rv
    return df