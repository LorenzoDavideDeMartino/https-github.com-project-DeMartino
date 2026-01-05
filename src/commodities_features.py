import pandas as pd
import numpy as np
from pathlib import Path

def build_features_df(
    df: pd.DataFrame, 
    price_col: str = "Price", 
    date_col: str = "Date", 
) -> pd.DataFrame:
    
    # Calculates Realized Volatility and Returns.
    # Key Logic:
    # - Only drops rows if PRICE is missing.
    # - Rows with missing Volume are KEPT (treated as valid price days).

    out = df.copy()

    # Validation
    if date_col not in out.columns or price_col not in out.columns:
        raise KeyError(f"Missing columns. Looked for: {date_col}, {price_col}. Available: {list(out.columns)}")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    
    out = out.dropna(subset=[date_col, price_col]).sort_values(date_col)
    # 2. Calculate Returns (Log Returns)
    out["Log_Ret"] = np.log(out[price_col] / out[price_col].shift(1))

    # Daily variance proxy (HAR building block)
    out["RV_Daily"] = out["Log_Ret"] ** 2

    out["RV_Weekly"]  = out["RV_Daily"].rolling(5).mean()
    out["RV_Monthly"] = out["RV_Daily"].rolling(22).mean()

    # 4. Final Cleaning (Drop NaN targets)
    # This drops the first 'window' rows (e.g., 21 days)
    out = out.dropna(subset=["RV_Monthly"])

    # 5. Drop intermediate/unused columns
    cols_to_drop = ["High", "Low", "Squared_Ret"] 
    out = out.drop(columns=cols_to_drop, errors="ignore")
    
    return out

def process_features_file(input_path: Path, output_path: Path) -> None:
    # Utility function to load, calculate, and save features.

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