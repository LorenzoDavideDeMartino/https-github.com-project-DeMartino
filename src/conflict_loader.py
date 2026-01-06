import pandas as pd
from pathlib import Path

# This function reduces and aggregates the raw UCDP event-level dataset.
# The goal is to keep only the information needed for the project while making the dataset manageable, consistent, and reproducible.

def build_ucdp_reduced_sorted(input_file: Path, out_file: Path, chunk_size: int = 100_000,):
    # The file is processed in chunks to limit memory usage when handling large datasets.

    input_file = Path(input_file)
    out_file = Path(out_file)

    # The pipeline cannot proceed if the raw UCDP file is missing.
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Only columns needed for the analysis are kept to reduce memory usage and focus on relevant conflict information.
    keep_cols = [
        "type_of_violence",
        "country",
        "region",
        "date_start",
        "best"]

    # (As before, IA suggestion) Ensure that the output directory exists before writing intermediate files
    out_file.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    for chunk in pd.read_csv(
        input_file,
        chunksize=chunk_size,
        low_memory=False, # <- IA Input, prevents pandas from guessing column types across chunks, which can cause inconsistencies
        usecols=lambda c: c in keep_cols): # Reads only the columns needed for the analysis.

        # Dates are converted explicitly to ensure consistent time filtering.
        chunk["date_start"] = pd.to_datetime(chunk["date_start"], errors="coerce") # <- IA Suggestion Invalid values are converted to missing instead of raising errors.
        chunk = chunk.dropna(subset=["date_start"])

        # Events before 1990 are removed to match the study period and reduce unnecessary historical noise.
        chunk = chunk[chunk["date_start"] >= "1990-01-01"]

        if chunk.empty:
            continue

        # Chunks are written incrementally to disk to keep memory usage low.
        mode = "w" if first_chunk else "a" # avoid overwriting data or duplicating column names
        header = first_chunk
        chunk.to_csv(out_file, index=False, mode=mode, header=header)
        first_chunk = False

    # The reduced file is reloaded for global aggregation and sorting.
    df_reduced = pd.read_csv(out_file)
    df_reduced["date_start"] = pd.to_datetime(df_reduced["date_start"])
    df_reduced["best"] = pd.to_numeric(df_reduced["best"], errors="coerce").fillna(0.0)

    # Events are aggregated by date, country, violence type, and region to obtain total daily conflict intensity measures.
    group_cols = ["date_start", "country", "type_of_violence", "region"]
    df_aggregated = (
        df_reduced.groupby(group_cols)["best"]
        .sum()
        .reset_index())

    # Sorting ensures a clean and consistent chronological structure.
    df_aggregated = df_aggregated.sort_values(by=["date_start", "country"], ascending=[True, True]) 

    # Column names are standardized to keep consistency with the rest of the project and improve readability.
    rename_map = {
        "date_start": "Date",
        "best": "Deaths",
        "type_of_violence": "Type",
        "region": "Region",
        "country": "Country"}
    
    df_aggregated = df_aggregated.rename(columns=rename_map)

    # The final aggregated dataset is saved for downstream analysis.
    df_aggregated.to_csv(out_file, index=False)

    print(f"UCDP sorted file generated: {out_file}")
    print(f"Total Rows: {len(df_aggregated):,}")

    return df_aggregated
