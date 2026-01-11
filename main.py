import pandas as pd
from pathlib import Path

# For "Step B: Commodities data loading"
from src.data_loader import build_clean_commodity_from_parts
# For "Step B.1: Commodities features"
from src.commodities_features import build_features_df
# For "Step C: Conflicts data loading"
from src.conflict_loader import build_ucdp_reduced_sorted
# For "Step C.1: Index Construction"
from src.conflict_index_builder import build_daily_panels
# For "Step 4: Final Merge"
from src.build_model_dataset import build_dataset_for_commodity
# For "Step 5: Analysis" + "Step 6: Evaluation"
from src.models import run_har_comparison
from src.evaluation import run_walk_forward

def main():
    # Step 1. Repository root
    repo = Path(__file__).resolve().parent
    # Definition of paths
    raw_commodities = repo / "data" / "raw" / "commodities"
    # Outputs processed commodities
    out_clean = repo / "data" / "processed" / "commodities"
    # Outputs features 
    out_feat = repo / "data" / "processed" / "features"
    # Outputs Conflicts
    out_indices = repo / "data" / "processed" / "indices"

    # Creation of files
    out_clean.mkdir(parents=True, exist_ok=True)
    out_feat.mkdir(parents=True, exist_ok=True)
    out_indices.mkdir(parents=True, exist_ok=True)


    # Step 2.COMMODITIES PIPELINE
    print("----------Step 2----------")
    targets = [
        ("gold", "gold_clean.csv", "gold_features.csv"),
        ("crude_oil_wti", "crude_oil_wti_clean.csv", "crude_oil_wti_features.csv"),
        ("natural_gas", "natural_gas_clean.csv", "natural_gas_features.csv")]
    
    # For each commodity 
    for folder, clean_name, feat_name in targets:
        parts_dir = raw_commodities / folder / "parts"
        clean_file = out_clean / clean_name
        feat_file = out_feat / feat_name
        # (IA Suggestion) The definition of file paths follows a standard Python project structure suggested by AI

        # Check or creation of clean commodity csv
        df_clean = None
        if clean_file.exists():
            print(f"Existing clean file already exists ({clean_name})")
            df_clean = pd.read_csv(clean_file)
        else:
            print(f"Creating clean file for ({clean_name})")
            df_clean = build_clean_commodity_from_parts(parts_dir=parts_dir, out_file=clean_file)

        # 2.1: Features (From the clean csv to the features)
        if feat_file.exists():
            print(f"Existing feature file already exists ({feat_name})")

        else:
            if df_clean is not None and len(df_clean) > 0:
                print(f"Creating commodities Features ({feat_name})")
                df_feat = build_features_df(df_clean, price_col="Price", date_col="Date")
                
                df_feat.to_csv(feat_file, index=False)
                print(f"Saved: {feat_file}")
            else:
                print("Attention: Cannot calculate features")


    # Step 3. CONFLICTS PIPELINE
    print("----------Step 3----------")
    ucdp_raw = repo / "data" / "raw" / "conflicts" / "GEDEvent_v25_1.csv"
    ucdp_reduced = repo / "data" / "processed" / "conflicts" / "ucdp_ged_reduced_sorted.csv"
    
    # Check if the file exist
    ucdp_reduced.parent.mkdir(parents=True, exist_ok=True)

    # Execution reduction function
    if ucdp_reduced.exists():
        print(f"Existing conflict file already exists : {ucdp_reduced.name}")
    elif ucdp_raw.exists():
        build_ucdp_reduced_sorted(ucdp_raw, ucdp_reduced)
    else:
        print("Fichier raw UCDP missing. You need to download it if you have delete the cleaned one, it was to heavy for github")

    # Step 3.1 : Index Construction
    required_files = [
        out_indices / "conflict_daily_global.csv",
        out_indices / "conflict_daily_by_region.csv"]

    if all(p.exists() for p in required_files):
        print(f"Existing indices in {out_indices.name}")

    else:
        if ucdp_reduced.exists():
            print("Creating daily conflict intensity indices")
            build_daily_panels(
                input_file=ucdp_reduced,
                out_dir=out_indices,
                start_date="1990-01-01",
                end_date="2024-12-31")
        else:
            print("Attention: Cannot build indices, reduced file is missing") # Cannot build indices without the reduced UCDP dataset.


    # STEP 4: FINAL MERGE (Datasets pour Models)
    print("----------Step 4----------")
    out_models = repo / "data" / "processed" / "model_datasets"
    out_models.mkdir(parents=True, exist_ok=True)

    # 1) Commodity feature files
    comm_map = {
        "WTI":  out_feat / "crude_oil_wti_features.csv",
        "GAS":  out_feat / "natural_gas_features.csv",
        "GOLD": out_feat / "gold_features.csv"}

    # 2) Conflict index files (match conflict_index_builder outputs)
    merge_config = {
        "WTI":  {"middle_east": out_indices / "conflict_daily_by_region.csv"},
        "GAS":  {"europe": out_indices / "conflict_daily_by_region.csv"},
        "GOLD": {
            "global": out_indices / "conflict_daily_global.csv",
            "middle_east": out_indices / "conflict_daily_by_region.csv"}}

    final_files = []

    # 3) Merge loop
    for commodity, conflict_sources in merge_config.items():

        available_conflicts = {
            region: path
            for region, path in conflict_sources.items()
            if path is not None and path.exists()}

        commodity_file = comm_map.get(commodity)
        out_p = out_models / f"{commodity.lower()}_dataset.csv"

        if commodity_file and commodity_file.exists() and available_conflicts:
            build_dataset_for_commodity(
                commodity_name=commodity,
                commodity_features_csv=commodity_file,
                conflict_files=available_conflicts,
                conflict_cols=["log_deaths_ewma_94"],
                conflict_lags=[0, 1],
                out_path=out_p)

            final_files.append((commodity, out_p))
 
        else:
            print(f"Skip {commodity} (files missing)")

    # STEP 5: ANALYSIS (DIAGNOSTIC)
    print("----------Step 5----------")
    print("Note: p-values test whether HAR-X provides a statistically significant improvement over HAR.")
    print("----In-Sample results-----")
    for name, fpath in final_files:
        if fpath.exists():
            run_har_comparison(fpath, name)
    
    # STEP 6
    print("----------Step 6----------")
    print("Note: The out-of-sample evaluation may take approximately 2–3 minutes to run.")
    print(
        "Please keep the terminal open.\n"
        "Due to the large number of print statements, the display may appear irregular,\n"
        "but the computation is running normally.")

    print(f"Analysis window restricted from 2015-01-01 to 2024-12-31"
    "(Reducing to have an acceptable computation time)")

    WINDOW_SIZE = 750  # ~3 years; enough history and faster
    
    # Update model every 5 trading days 
    STEP_SIZE_OPTIMIZED = 5 # This drastically reduces computation time without compromising statistical validity.
    
    # Iterate over the datasets created in Step 4
    for name, path in final_files:
        run_walk_forward(
            file_path=path, 
            commodity_name=name, 
            window_size=WINDOW_SIZE, 
            step_size=STEP_SIZE_OPTIMIZED)

if __name__ == "__main__":
    main()