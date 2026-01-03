import pandas as pd
import numpy as np
from pathlib import Path

def build_features_df(
    df: pd.DataFrame, 
    price_col: str = "Price", 
    date_col: str = "Date", 
    window: int = 21,
    annualize: bool = False
) -> pd.DataFrame:
    """
    Calculates Realized Volatility and Returns.
    
    Key Logic:
    - Only drops rows if PRICE is missing.
    - Rows with missing Volume are KEPT (treated as valid price days).
    - Drops the first ~21 rows where volatility cannot be calculated.
    """
    out = df.copy()

    # Validation
    if date_col not in out.columns or price_col not in out.columns:
        raise KeyError(f"Missing columns. Looked for: {date_col}, {price_col}. Available: {list(out.columns)}")

    # Type Conversion
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    
    # 1. STRICT CLEANING
    # Only drop if Date OR Price is empty.
    # We do NOT drop based on Volume.
    out = out.dropna(subset=[date_col, price_col]).sort_values(date_col)

    # 2. Calculate Returns (Log Returns)
    # Formula: ln(P_t / P_{t-1})
    out["Log_Ret"] = np.log(out[price_col] / out[price_col].shift(1))

    # 3. Calculate Target (Realized Volatility)
    # Sum of squared returns over the rolling window
    out["Squared_Ret"] = out["Log_Ret"] ** 2
    out["Realized_Vol_21d"] = out["Squared_Ret"].rolling(window=window).sum()

    if annualize:
        # Annualized Standard Deviation (Vol)
        out["Vol_Ann_Std"] = np.sqrt(out["Realized_Vol_21d"]) * np.sqrt(252 / window)

    # 4. Final Cleaning (Drop NaN targets)
    # This drops the first 'window' rows (e.g., 21 days)
    out = out.dropna(subset=["Realized_Vol_21d"])

    # 5. Drop intermediate/unused columns
    cols_to_drop = ["High", "Low", "Squared_Ret"] 
    out = out.drop(columns=cols_to_drop, errors="ignore")
    
    return out

def process_features_file(
    input_path: Path, 
    output_path: Path
) -> None:
    """Utility function to load, calculate, and save features."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    # Load
    df = pd.read_csv(input_path)
    
    # Calculate
    try:
        df_features = build_features_df(df)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)
        print(f"Features generated: {output_path} | Rows: {len(df_features)}")
        
    except KeyError as e:
        print(f"Error processing {input_path.name}: {e}")