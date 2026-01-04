# src/conflict_loader.py
import pandas as pd
from pathlib import Path

def build_ucdp_reduced_sorted(
    input_file: Path,
    out_file: Path,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Step 1: Reduction, Filtering (>= 1990), and Aggregation.
    
    Logic:
    1. Read UCDP raw file in chunks.
    2. Filter dates >= 01-01-1990.
    3. Aggregate deaths (sum) if: Same Date + Same Country + Same Conflict Type + Same Region.
    4. Sort chronologically.
    5. Rename columns for consistency.
    """
    input_file = Path(input_file)
    out_file = Path(out_file)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Columns to keep 
    # (REGION est bien présent ici, c'est ce qui compte)
    keep_cols = [
        "active_year", 
        "type_of_violence", 
        "conflict_name",
        "country", 
        "region", 
        "date_start", 
        "best"
    ]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing UCDP file (chunks of {chunk_size})...")
    
    # 1. Read, Filter (1990), and Write Chunks
    first_chunk = True
    
    # On lit le fichier brut
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, usecols=lambda c: c in keep_cols):
        
        # Date Conversion
        chunk["date_start"] = pd.to_datetime(chunk["date_start"], errors="coerce")
        chunk = chunk.dropna(subset=["date_start"])

        # --- FILTER 1990 (Strict) ---
        chunk = chunk[chunk["date_start"] >= "1990-01-01"]

        if chunk.empty:
            continue

        # Save chunk temporarily
        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(out_file, index=False, mode=mode, header=header)
        first_chunk = False

    print("Reduction complete. Starting Aggregation and Global Sort...")

    # 2. Global Aggregation & Sort
    if not out_file.exists():
        print("No data found after 1990.")
        return pd.DataFrame()

    df_reduced = pd.read_csv(out_file)
    df_reduced["date_start"] = pd.to_datetime(df_reduced["date_start"])
    
    # --- LOGICAL AGGREGATION ---
    # On garde 'region' et 'type_of_violence' dans le groupe pour ne pas perdre l'info
    group_cols = ["date_start", "country", "type_of_violence", "conflict_name", "region"]
    
    # We sum the 'best' (deaths) column
    df_aggregated = df_reduced.groupby(group_cols)["best"].sum().reset_index()
    
    # 3. Final Sort
    df_aggregated = df_aggregated.sort_values(by=["date_start", "country"], ascending=[True, True])
    
    # --- AJOUT : RENOMMAGE STANDARD ---
    # Pour que ce soit plus propre dans la suite du projet (GARCH/Features)
    rename_map = {
        "date_start": "Date",
        "best": "Deaths",
        "type_of_violence": "Type",
        "region": "Region",
        "country": "Country",
        "conflict_name": "Conflict_Name"
    }
    df_aggregated = df_aggregated.rename(columns=rename_map)

    # Save final version
    df_aggregated.to_csv(out_file, index=False)
    
    print(f"File generated and sorted: {out_file}")
    print(f"Total Rows: {len(df_aggregated):,}")
    if not df_aggregated.empty:
        print(f"Start Date: {df_aggregated['Date'].min()}")
        print(f"End Date  : {df_aggregated['Date'].max()}")
    
    return df_aggregated