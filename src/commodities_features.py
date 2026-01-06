import pandas as pd
import numpy as np
from pathlib import Path

def build_features_df(df: pd.DataFrame, price_col: str = "Price", date_col: str = "Date"):
    # This function builds the volatility features used in the HAR and HAR-X models
    # The goal is to construct realized volatility measures in a transparent and reproducible way
    # Rows with missing Volume are KEPT (treated as valid price days)

    out = df.copy() # I work on a copy to avoid side effects on the original DataFrame

    # The analysis relies on prices indexed in time; without these columns,
    # the dataset cannot be used for volatility estimation.
    if date_col not in out.columns or price_col not in out.columns:
        raise KeyError(
            f"Missing columns. Looked for: {date_col}, {price_col}. "
            f"Available: {list(out.columns)}")

    # (IA Input) Dates and prices are explicitly coerced to avoid silent parsing errors
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    # Rows without a valid date or price cannot contribute to returns or volatility
    out = out.dropna(subset=[date_col, price_col]).sort_values(date_col)

    # Log returns are used because they are standard in finance and additive over time
    out["Log_Ret"] = np.log(out[price_col] / out[price_col].shift(1))

    # Squared log returns provide a daily proxy for realized variance, which is the basic building block of HAR models
    out["RV_Daily"] = out["Log_Ret"] ** 2

    # Weekly and monthly components capture volatility persistence at different horizons, as in the HAR framework
    out["RV_Weekly"] = out["RV_Daily"].rolling(5).mean()
    out["RV_Monthly"] = out["RV_Daily"].rolling(22).mean()

    # Rows with missing long-horizon volatility are removed to ensure that the target variable is well defined. 
    out = out.dropna(subset=["RV_Monthly"])

    # Intermediate or unused columns are removed to keep the dataset focused on variables that are actually used in the models
    cols_to_drop = ["High", "Low"]
    out = out.drop(columns=cols_to_drop, errors="ignore")

    return out

def process_features_file(input_path: Path, output_path: Path):
    # This function loads the data, computes features, and saves the result.

    # If the input file does not exist, the pipeline cannot proceed
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    # Raw data are loaded from disk before feature construction
    df = pd.read_csv(input_path)

    try:
        # Feature construction is isolated in a single function to keep the logic clear and testable.
        df_features = build_features_df(df)

        # The output directory is created upfront to avoid save errors (<- IA Sugesstion)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)

        print(f"Features generated {output_path} - Rows: {len(df_features)}")

    except KeyError as e:
        # Structural data issues are reported explicitly to make failures visible.
        print(f"Error processing {input_path.name}")

