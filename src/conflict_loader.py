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
    3. Aggregate deaths (sum) if: Same Date + Same Country + Same Conflict Type.
    4. Sort chronologically.
    """
    input_file = Path(input_file)
    out_file = Path(out_file)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Columns to keep (ID removed because we are aggregating)
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
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, usecols=lambda c: c in keep_cols):
        
        # Date Conversion
        chunk["date_start"] = pd.to_datetime(chunk["date_start"], errors="coerce")
        chunk = chunk.dropna(subset=["date_start"])

        # --- FILTER 1990 (Strict) ---
        chunk = chunk[chunk["date_start"] >= "1990-01-01"]

        if chunk.empty:
            continue

        # Save chunk
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
    # "Sum deaths if same date, same country, same type"
    # We include 'conflict_name' and 'region' in the group to preserve that info.
    group_cols = ["date_start", "country", "type_of_violence", "conflict_name", "region"]
    
    # We sum the 'best' (deaths) column
    df_aggregated = df_reduced.groupby(group_cols)["best"].sum().reset_index()
    
    # 3. Final Sort
    df_aggregated = df_aggregated.sort_values(by=["date_start", "country"], ascending=[True, True])
    
    # Save final version
    df_aggregated.to_csv(out_file, index=False)
    
    print(f"File generated and sorted: {out_file}")
    print(f"Total Rows: {len(df_aggregated):,}")
    if not df_aggregated.empty:
        print(f"Start Date: {df_aggregated['date_start'].min()}")
        print(f"End Date  : {df_aggregated['date_start'].max()}")
    
    return df_aggregated


def build_daily_conflict_series(
    reduced_file: Path, 
    out_file: Path,
    region_name: str = None
) -> pd.DataFrame:
    # ... (Cette fonction ne change pas, elle est déjà correcte dans ton code précédent)
    # Si tu as besoin que je te la redonne, dis-le moi.
    pass