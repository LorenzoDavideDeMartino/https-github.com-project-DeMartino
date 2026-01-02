from pathlib import Path
import pandas as pd

from src.data_loader import build_clean_commodity_from_parts
from src.features import build_features_df


def main() -> None:
    repo = Path(__file__).resolve().parent

    raw = repo / "data" / "raw" / "commodities"
    out_clean = repo / "data" / "processed" / "commodities"
    out_feat = repo / "data" / "processed" / "features"

    out_clean.mkdir(parents=True, exist_ok=True)
    out_feat.mkdir(parents=True, exist_ok=True)

    targets = [
        ("gold", "gold_clean.csv", "gold_features.csv"),
        ("crude_oil_wti", "crude_oil_wti_clean.csv", "crude_oil_wti_features.csv"),
        ("natural_gas", "natural_gas_clean.csv", "natural_gas_features.csv"),
    ]

    for folder, clean_name, feat_name in targets:
        parts_dir = raw / folder / "parts"
        clean_file = out_clean / clean_name
        feat_file = out_feat / feat_name

        df_clean = build_clean_commodity_from_parts(parts_dir=parts_dir, out_file=clean_file)

        df_feat = build_features_df(df_clean, price_col="Price", date_col="Date", window=21)
        df_feat.to_csv(feat_file, index=False)

        print(f"OK - {folder}")
        print(f"Clean rows: {len(df_clean):,} | Features rows: {len(df_feat):,}")
        print(f"Saved clean: {clean_file}")
        print(f"Saved feat : {feat_file}\n")


if __name__ == "__main__":
    main()
