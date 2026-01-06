import pandas as pd
from pathlib import Path

    # For "Step B: COMMODITIES PIPELINE"
from src.data_loader import build_clean_commodity_from_parts
    # For "Step B.1: COMMODITIES PIPELINE"
from src.commodities_features import build_features_df
   #  For "Step C. CONFLICTS PIPELINE"
from src.conflict_loader import build_ucdp_reduced_sorted
    # For "Step C.1 : Index Construction"
from src.conflict_index_builder import build_daily_panels, prepare_region_view
    # For "Step 4: Final Merge"
from src.build_model_dataset import build_dataset_for_commodity
    # For "Step 5: ANALYSIS" 
from src.models import run_har_comparison

from src.evaluation import run_walk_forward

def main() -> None:
    # Step 1. Repository root
    repo = Path(__file__).resolve().parent

    # Definition of paths
    raw_commodities = repo / "data" / "raw" / "commodities"
    
    # Outputs Commodities
    out_clean = repo / "data" / "processed" / "commodities"
    out_feat = repo / "data" / "processed" / "features"
    
    # Outputs Conflicts
    out_indices = repo / "data" / "processed" / "indices"

    # Creation of files
    out_clean.mkdir(parents=True, exist_ok=True)
    out_feat.mkdir(parents=True, exist_ok=True)
    out_indices.mkdir(parents=True, exist_ok=True)

    # Step 2.COMMODITIES PIPELINE
    print("----------STEP 2----------")
    targets = [
        ("gold", "gold_clean.csv", "gold_features.csv"),
        ("crude_oil_wti", "crude_oil_wti_clean.csv", "crude_oil_wti_features.csv"),
        ("natural_gas", "natural_gas_clean.csv", "natural_gas_features.csv"),
    ]
    for folder, clean_name, feat_name in targets:
        parts_dir = raw_commodities / folder / "parts"
        clean_file = out_clean / clean_name
        feat_file = out_feat / feat_name
        # (IA Suggestion) The definition of file paths follows a standard Python project structure suggested by AI,
        # as this was not known prior to this project.

        # Check or creation of clean commodity csv
        df_clean = None
        if clean_file.exists():
            print(f"Existing Clean file already exists ({clean_name}).")
            df_clean = pd.read_csv(clean_file)
        else:
            print(f"Creating Clean file...")
            df_clean = build_clean_commodity_from_parts(
                parts_dir=parts_dir,
                out_file=clean_file,
            )

        # 2.1: Features (From the clean csv to the features)
        if feat_file.exists():
            print(f"Existing Feature file already exists ({feat_name})")
        else:
            if df_clean is not None and len(df_clean) > 0:
                print(f"Creating Commodities Features...")
                df_feat = build_features_df(
                    df_clean,
                    price_col="Price",
                    date_col="Date"
                )
                df_feat.to_csv(feat_file, index=False)
                print(f"Saved: {feat_file}")
            else:
                print("Attention: Cannot calculate features.")

    # Step 3. CONFLICTS PIPELINE
    print("----------STEP 3----------")
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

    # Step 3.1 : Index Construction (Oil Focus, Gas Focus, etc.)
    required_files = [
        out_indices / "conflict_daily_global.csv",
        out_indices / "conflict_daily_by_region.csv",
        out_indices / "conflict_daily_oil_focus.csv",
        out_indices / "conflict_daily_gas_focus.csv",
    ]

    if all(p.exists() for p in required_files):
        print(f"Existing Indices in {out_indices.name}.")
    else:
        if ucdp_reduced.exists():
            print("Building conflict panels...")
            try:
                build_daily_panels(
                    input_file=ucdp_reduced,
                    out_dir=out_indices,
                    start_date="1990-01-01",
                    end_date="2024-12-31")
            except Exception as e:
                print(f"Attention: Index Builder Failed: {e}") # <- IA Suggestion : Catch unexpected failures during index construction.
        else:
            print("Attention: Cannot build indices, reduced file missing.") # Cannot build indices without the reduced UCDP dataset.

    # STEP 4: FINAL MERGE (Datasets pour Models)
    print("----------STEP 4----------")
    out_models = repo / "data" / "processed" / "model_datasets"
    out_models.mkdir(parents=True, exist_ok=True)

    # 2. Create regional views (Middle East, Europe)
    region_source = out_indices / "conflict_daily_by_region.csv"
    me_view = prepare_region_view(region_source, "Middle East", out_indices / "view_middle_east.csv")
    eu_view = prepare_region_view(region_source, "Europe", out_indices / "view_europe.csv")

    # 3. Commodity
    comm_map = {
        "WTI":  out_feat / "crude_oil_wti_features.csv",
        "GAS":  out_feat / "natural_gas_features.csv",
        "GOLD": out_feat / "gold_features.csv"
    }

    merge_config = {
        "WTI":  {"oil_focus": out_indices / "conflict_daily_oil_focus.csv", "middle_east": me_view},
        "GAS":  {"gas_focus": out_indices / "conflict_daily_gas_focus.csv", "europe": eu_view},
        "GOLD": {"middle_east": me_view, "global": out_indices / "conflict_daily_global.csv"}
    }
    final_files = []

    # 4. Execution of the function for merge 
    for name, conflict_map in merge_config.items():
        # Only existing conflict files are kept
        valid_map = {k: v for k, v in conflict_map.items() if v is not None and v.exists()}
        comm_file = comm_map.get(name)
        
        if comm_file and comm_file.exists() and valid_map:
            try:
                out_p = out_models / f"{name.lower()}_dataset.csv"
                
                build_dataset_for_commodity(
                    commodity_name=name,
                    commodity_features_csv=comm_file,
                    conflict_files=valid_map,
                    conflict_cols=["log_fatal_ewma_94"],
                    conflict_lags=(0, 1),
                    out_path=out_models / f"{name.lower()}_dataset.csv"
                )
                final_files.append((name, out_p))

            except Exception as e:
                print(f"Merge failed for {name}: {e}")
        else:
            print(f"Skip {name} (files missing)")

    # STEP 5: ANALYSIS (DIAGNOSTIC)
        print("----------STEP 5----------")
        for name, fpath in final_files:
                if fpath.exists():
                    run_har_comparison(fpath, name)


    # STEP 6: OUT-OF-SAMPLE FORECAST EVALUATION (WALK-FORWARD)
    print("----------STEP 6----------")
    # CONFIGURATION
    # Window size: 1000 days (approx 4 years of history for training)
    WINDOW_SIZE = 1000
    
    # OPTIMIZATION: Update model every 5 trading days 
    # This drastically reduces computation time without compromising statistical validity.
    STEP_SIZE_OPTIMIZED = 5 
    
    # Newey-West Lags for Diebold-Mariano test
    # Set to 5 for robustness against serial correlation in weekly steps.
    NW_LAGS = 5

    # Iterate over the datasets created in Step 4
    for name, path in final_files:
        if path.exists():
            try:
                run_walk_forward(
                    file_path=path, 
                    commodity_name=name, 
                    window_size=WINDOW_SIZE, 
                    step_size=STEP_SIZE_OPTIMIZED, 
                    nw_lags_dm=NW_LAGS
                )
            except Exception as e:
                print(f"[Error] Evaluation failed for {name}: {e}")

if __name__ == "__main__":
    main()