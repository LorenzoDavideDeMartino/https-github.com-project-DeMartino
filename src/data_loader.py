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
    """Version corrigée : gère mieux les formats JJ/MM/AAAA et MM/JJ/AAAA."""
    if not isinstance(date_str, str) or not date_str:
        return pd.NaT

    s = date_str.strip('"\' ')

    # Liste de formats à tester par ordre de priorité
    formats_to_try = [
        "%m/%d/%Y",  # Format US (souvent par défaut sur Investing.com EN)
        "%d/%m/%Y",  # Format EU (souvent sur Investing.com FR)
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%d.%m.%Y",  # Format avec points
        "%m.%d.%Y"
    ]
    
    # Nettoyage préalable (remplace les points par des slashs pour simplifier)
    s_clean = s.replace(".", "/")

    for fmt in formats_to_try:
        try:
            # On tente de convertir
            dt = pd.to_datetime(s, format=fmt)
            return dt
        except (ValueError, TypeError):
            # Si ça rate, on essaie le format suivant (pas de return NaT ici !)
            try:
                # Tentative avec s_clean (si le format original avait des points)
                dt = pd.to_datetime(s_clean, format=fmt.replace(".", "/"))
                return dt
            except:
                continue

    return pd.NaT

def _convert_numeric(value: object) -> float:
    """Your numeric conversion: %, K/M suffix, commas. CORRIGÉ POUR ARRONDIS."""
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
            # --- MODIFICATION ICI : Ajout de round() ---
            return round(float(u.replace(",", "")) * 1000.0)

        if "M" in u:
            u = u.replace("M", "")
            # --- MODIFICATION ICI : Ajout de round() ---
            return round(float(u.replace(",", "")) * 1_000_000.0)

        return float(v.replace(",", ""))
    except Exception:
        return np.nan


def read_investing_raw_csv(path: Path) -> pd.DataFrame:
    """
    Read one Investing.com raw CSV file (often 1-column) and return a cleaned DataFrame.
    """
    # Force loading as string to prevent pandas from guessing types wrongly initially
    # and to ensure the splitting logic works on strings.
    raw_df = pd.read_csv(path, header=None, low_memory=False, dtype=str)

    header_row_idx = _find_header_row_idx(raw_df)
    if header_row_idx is None:
        header_row_idx = 0
        header_row = raw_df.iloc[0, 0]
    else:
        header_row = raw_df.iloc[header_row_idx, 0]

    if not isinstance(header_row, str):
        # Fallback if header row isn't a string (e.g. if file is empty or weird)
        # Try to cast to string
        header_row = str(header_row)

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
        # Instead of crashing, let's try to return empty if really broken
        # raise ValueError(f"'Date' column not found after parsing: {path}")
        return pd.DataFrame()

    df = df.dropna(subset=["Date"]).copy()

    # Sort / drop duplicate dates (keep first)
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="first")

    # Keep canonical column order if present
    ordered = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    keep = [c for c in ordered if c in df.columns]
    df = df.loc[:, keep]

    return df

# Merge parts (multiple downloads)
def build_clean_commodity_from_parts(parts_dir: Path, out_file: Path) -> pd.DataFrame:
    """
    Merge many Investing.com raw CSV parts into one clean CSV.
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

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure required columns exist
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column missing after merging parts in: {parts_dir}")
    
    # Critical: remove rows where Price is missing after merge
    # Check if 'Price' exists, otherwise look for 'Last'
    price_col = "Price" if "Price" in df.columns else "Last"
    
    if price_col not in df.columns:
        # Last resort: try to rename anything looking like a price
        pass 
    
    if price_col in df.columns:
        df = df.dropna(subset=["Date", price_col]).copy()
        # Rename to Price if it was Last
        if price_col != "Price":
             df.rename(columns={price_col: "Price"}, inplace=True)

    # Final dedupe + sort
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="first")

    ordered = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    keep = [c for c in ordered if c in df.columns]
    df = df.loc[:, keep]

    df.to_csv(out_file, index=False)
    return df