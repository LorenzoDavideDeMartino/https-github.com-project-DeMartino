from __future__ import annotations

import pandas as pd
from pathlib import Path


def build_ucdp_reduced_sorted(
    input_file: Path,
    out_file: Path,
    chunk_size: int = 200_000,
) -> Path:
    """
    Build a reduced, cleaned, sorted (by date_start, id) UCDP GED event-level file.

    Keeps columns:
      id, active_year, type_of_violence, conflict_name, country, region, date_start, best

    Output:
      CSV sorted by (date_start, id).
    """
    keep_cols = [
        "id",
        "active_year",
        "type_of_violence",
        "conflict_name",
        "country",
        "region",
        "date_start",
        "best",
    ]

    out_file.parent.mkdir(parents=True, exist_ok=True)

    written = False

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
        cols = [c for c in keep_cols if c in chunk.columns]
        df = chunk.loc[:, cols].copy()

        # active_year -> bool if possible
        if "active_year" in df.columns:
            df["active_year"] = (
                df["active_year"].astype(str).str.lower().map({"true": True, "false": False})
                .fillna(df["active_year"])
            )

        # date
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
        df = df.dropna(subset=["date_start"])

        # numerics
        df["type_of_violence"] = pd.to_numeric(df["type_of_violence"], errors="coerce").astype("Int64")
        df["best"] = pd.to_numeric(df["best"], errors="coerce").fillna(0.0)

        # sort within chunk (stable ordering)
        if "id" in df.columns:
            df = df.sort_values(["date_start", "id"])
        else:
            df = df.sort_values(["date_start"])

        df.to_csv(
            out_file,
            index=False,
            mode="w" if not written else "a",
            header=not written,
        )
        written = True

        print(f"[UCDP] Chunk {i+1}: wrote {len(df):,} rows")

    print(f"[UCDP] DONE: {out_file}")
    return out_file
