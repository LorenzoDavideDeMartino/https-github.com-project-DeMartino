from pathlib import Path
import pandas as pd

def load_commodity_prices(file_path: str | Path) -> pd.DataFrame:
    """
    Loads a cleaned commodity CSV with at least: Date, Price (or Close).
    Returns a DataFrame sorted by date.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    # flexible date column naming
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    return df
