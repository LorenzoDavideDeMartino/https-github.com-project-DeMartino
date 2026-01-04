import pandas as pd

from pathlib import Path

from src.data_loader import build_clean_commodity_from_parts
from src.features import build_features_df
from src.conflict_loader import build_ucdp_reduced_sorted

from src.conflict_index_builder import build_daily_panels, prepare_region_view

from src.build_model_dataset import build_dataset_for_commodity

def main() -> None:
    # Step A. Repository root
    repo = Path(__file__).resolve().parent

    # DEFINITION DES CHEMINS (PATHS)
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

    # Step B.COMMODITIES PIPELINE
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
            print(f"Existing Clean file already exists ({clean_name}).")
            df_clean = pd.read_csv(clean_file)
        else:
            print(f"[Processing] Creating Clean file...")
            df_clean = build_clean_commodity_from_parts(
                parts_dir=parts_dir,
                out_file=clean_file,
            )

        # B.1: Features (Clean -> Features)
        if feat_file.exists():
            print(f"Existing Feature file already exists ({feat_name})")
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

    # Step C. CONFLICTS PIPELINE (UCDP GED – fully reproducible)
    # Chemins
    ucdp_raw = repo / "data" / "raw" / "conflicts" / "GEDEvent_v25_1.csv"
    ucdp_reduced = repo / "data" / "processed" / "conflicts" / "ucdp_ged_reduced_sorted.csv"
    
    # Check if the file exist
    ucdp_reduced.parent.mkdir(parents=True, exist_ok=True)

    # Execution Reduction
    if ucdp_reduced.exists():
        print(f"Fichier réduit déjà présent : {ucdp_reduced.name}")
    elif ucdp_raw.exists():
        build_ucdp_reduced_sorted(ucdp_raw, ucdp_reduced)
    else:
        print("Fichier raw UCDP absent.")

    # Etape C.1 : Index Construction (Oil Focus, Gas Focus, etc.)
    if any(out_indices.iterdir()):
        print(f"   [Skip] Indices seems to exist in {out_indices.name} (folder not empty).")
    else:
        if ucdp_reduced.exists():
            print("   [Build] Building Advanced Conflict Panels...")
            try:
                build_daily_panels(
                    input_file=ucdp_reduced,
                    out_dir=out_indices,
                    start_date="1990-01-01",
                    end_date="2024-12-31"
                )
            except Exception as e:
                print(f"   [Error] Index Builder Failed: {e}")
        else:
            print("   [Error] Cannot build indices, reduced file missing.")

    print("\n--- PIPELINE COMPLETE ---")

    # STEP 4: FINAL MERGE (Datasets pour Models)
    print("\n--- [STEP 4] MERGING DATASETS ---")
    
    # 1. Préparation des dossiers et fichiers
    out_models = repo / "data" / "processed" / "model_datasets"
    out_models.mkdir(parents=True, exist_ok=True)

    # 2. Création des vues régionales (Moyen-Orient, Europe)
    region_source = out_indices / "conflict_daily_by_region.csv"
    me_view = prepare_region_view(region_source, "Middle East", out_indices / "view_middle_east.csv")
    eu_view = prepare_region_view(region_source, "Europe", out_indices / "view_europe.csv")

    # 3. Configuration : Quelle matière première va avec quel conflit ?
    # On mappe les clés (WTI, GAS, GOLD) vers tes fichiers nettoyés existants
    comm_map = {
        "WTI":  out_clean / "crude_oil_wti_clean.csv",
        "GAS":  out_clean / "natural_gas_clean.csv",
        "GOLD": out_clean / "gold_clean.csv"
    }

    merge_config = {
        "WTI":  {"oil_focus": out_indices / "conflict_daily_oil_focus.csv", "middle_east": me_view},
        "GAS":  {"gas_focus": out_indices / "conflict_daily_gas_focus.csv", "europe": eu_view},
        "GOLD": {"middle_east": me_view, "global": out_indices / "conflict_daily_global.csv"}
    }

    # 4. Exécution de la fusion
    for name, conflict_map in merge_config.items():
        # On ne garde que les fichiers conflits qui existent
        valid_map = {k: v for k, v in conflict_map.items() if v is not None and v.exists()}
        comm_file = comm_map.get(name)
        
        if comm_file and comm_file.exists() and valid_map:
            try:
                build_dataset_for_commodity(
                    commodity_name=name,
                    commodity_csv=comm_file,
                    conflict_files=valid_map,
                    conflict_cols=["log_fatal_ewma_94", "log_fatal_ewma_97"],
                    conflict_lags=(1, 5),
                    out_path=out_models / f"{name.lower()}_dataset.csv"
                )
            except Exception as e:
                print(f"   [Error] Merge failed for {name}: {e}")
        else:
            print(f"   [Skip] {name} (files missing)")

if __name__ == "__main__":
    main()