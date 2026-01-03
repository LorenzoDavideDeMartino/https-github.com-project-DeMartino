from pathlib import Path
import pandas as pd

from src.data_loader import build_clean_commodity_from_parts
from src.features import build_features_df
from src.conflict_loader import build_ucdp_reduced_sorted

def main() -> None:
    # -------------------------------------------------
    # Repository root
    # -------------------------------------------------
    repo = Path(__file__).resolve().parent

    # -------------------------------------------------
    # COMMODITIES PIPELINE
    # -------------------------------------------------
    raw_commodities = repo / "data" / "raw" / "commodities"
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
        parts_dir = raw_commodities / folder / "parts"
        clean_file = out_clean / clean_name
        feat_file = out_feat / feat_name

        df_clean = None
        if clean_file.exists():
            print(f"[INFO] Existing Clean file found ({clean_name}). Loading...")
            df_clean = pd.read_csv(clean_file)
        else:
            print(f"[Processing] Creating Clean file...")
            df_clean = build_clean_commodity_from_parts(
                parts_dir=parts_dir,
                out_file=clean_file,
            )

        # STEP B: Features (Clean -> Features)
        if feat_file.exists():
            print(f"[INFO] Existing Features file found ({feat_name}). Skipping.")
        else:
            if df_clean is not None and not df_clean.empty:
                print(f"[Processing] Calculating Features (Volatility)...")
                df_feat = build_features_df(
                    df_clean,
                    price_col="Price",
                    date_col="Date",
                    window=21,
                    annualize=False,
                )
                df_feat.to_csv(feat_file, index=False)
                print(f"   Saved: {feat_file}")
            else:
                print("[ERROR] Cannot calculate features (missing clean data).")

    # -------------------------------------------------
    # CONFLICTS PIPELINE (UCDP GED – fully reproducible)
    # -------------------------------------------------
    # Chemins
    ucdp_raw = repo / "data" / "raw" / "conflicts" / "GEDEvent_v25_1.csv"
    ucdp_reduced = repo / "data" / "processed" / "conflicts" / "ucdp_ged_reduced_sorted.csv"

    # Execution
    if ucdp_raw.exists():
        build_ucdp_reduced_sorted(ucdp_raw, ucdp_reduced)
    else:
        print("Fichier raw UCDP absent.")

if __name__ == "__main__":
    main()
