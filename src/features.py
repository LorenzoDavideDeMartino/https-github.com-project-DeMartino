from __future__ import annotations

import numpy as np
import pandas as pd


def build_features_df(
    df: pd.DataFrame,
    price_col: str = "Price",
    date_col: str = "Date",
    window: int = 21,
    annualize: bool = False,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Build a features table for volatility modeling.

    Outputs:
      - Return: log return ln(Pt/Pt-1)
      - Return2: squared return
      - RV: realized variance proxy = rolling sum(Return2, window)
      - RV_sqrt: realized volatility = sqrt(RV)
      - (optional) annualized versions
    """
    out = df.copy()

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)

    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=[price_col])

    out["Return"] = np.log(out[price_col] / out[price_col].shift(1))
    out["Return2"] = out["Return"] ** 2

    out["RV"] = out["Return2"].rolling(window).sum()
    out["RV_sqrt"] = np.sqrt(out["RV"])

    if annualize:
        out["RV_ann"] = out["RV"] * (trading_days / window)
        out["RV_sqrt_ann"] = out["RV_sqrt"] * np.sqrt(trading_days / window)

    return out
