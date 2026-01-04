# src/conflict_index_builder.py
import pandas as pd
import numpy as np
from pathlib import Path

# --- PARAMÈTRES ---
LAMBDA_LIST = [0.94, 0.97]
SHOCK_PERCENTILE = 0.95 
EXPANDING_WINDOW = 365 # On attend 1 an d'historique pour définir un seuil statistique

# Filtrage strict pour réduire le bruit (Economically relevant regions)
# Adapte ces noms si le print("Regions found") t'en montre d'autres
KEEP_REGIONS = ["Middle East", "Europe", "Africa"]

# Pays Focus (inchangés)
OIL_FOCUS_COUNTRIES = {
    "Iraq", "Iran", "Saudi Arabia", "Kuwait", "United Arab Emirates", "Qatar",
    "Oman", "Yemen", "Syria", "Libya", "Algeria", "Nigeria", "Angola", "Russia", 
    "Venezuela", "Azerbaijan"
}

GAS_FOCUS_COUNTRIES = {
    "Russia", "Ukraine", "Qatar", "Iran", "Algeria", "Nigeria", "Iraq", "Syria",
    "Norway", "United States"
}

def ewma(series: pd.Series, lam: float) -> pd.Series:
    """Calcule la Moyenne Mobile Exponentielle"""
    return series.ewm(alpha=(1 - lam), adjust=False).mean()

def build_daily_panels(
    input_file: Path,
    out_dir: Path,
    start_date="1990-01-01",
    end_date="2024-12-31",
):
    print(f"--- Building Indices from {input_file} ---")
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    df = pd.read_csv(input_file)
    
    # 1. DEBUG REGIONS (Pour vérifier l'orthographe exacte)
    if "Region" in df.columns:
        print("Regions found in CSV:", sorted(df["Region"].dropna().unique()))
    
    # 2. CORRECTION DATE & CLEANING
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce").fillna(0.0).clip(lower=0)

    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FONCTION INTERNE ROBUSTE ---
    def process_panel(df_subset, filename_suffix):
        # GESTION SUBSET VIDE
        if df_subset.empty:
            print(f"[WARN] Subset vide pour {filename_suffix}. Génération de zéros.")
            daily = pd.DataFrame(index=all_days)
            daily.index.name = "Date"
            daily["fatal"] = 0.0
            daily["log_fatal"] = 0.0
            daily["fatal_shock"] = 0
            for lam in LAMBDA_LIST:
                daily[f"log_fatal_ewma_{int(lam*100)}"] = 0.0
                daily[f"fatal_shock_ewma_{int(lam*100)}"] = 0.0
            
            out_path = out_dir / f"conflict_daily_{filename_suffix}.csv"
            daily.to_csv(out_path)
            return daily

        # Agrégation normale
        daily = (df_subset.groupby("Date")
                 .agg(fatal=("Deaths", "sum"))
                 .reindex(all_days, fill_value=0))
        daily.index.name = "Date"
        
        # Log Intensité
        daily["log_fatal"] = np.log1p(daily["fatal"])
        
        # 3. SHOCK "EXPANDING" (ANTI-LEAKAGE + INITIALISATION PROPRE)
        # Calcul du seuil sur le passé (shift 1)
        threshold_series = daily["fatal"].expanding(min_periods=EXPANDING_WINDOW).quantile(SHOCK_PERCENTILE).shift(1)
        
        # Initialisation à 0 par défaut
        daily["fatal_shock"] = 0
        
        # On n'active le calcul que là où on a assez d'historique (threshold non-NaN)
        mask = threshold_series.notna()
        daily.loc[mask, "fatal_shock"] = (daily.loc[mask, "fatal"] >= threshold_series.loc[mask]).astype(int)

        # Calcul des EWMA
        for lam in LAMBDA_LIST:
            daily[f"log_fatal_ewma_{int(lam*100)}"] = ewma(daily["log_fatal"], lam)
            daily[f"fatal_shock_ewma_{int(lam*100)}"] = ewma(daily["fatal_shock"], lam)
            
        # Sauvegarde
        out_path = out_dir / f"conflict_daily_{filename_suffix}.csv"
        daily.to_csv(out_path)
        return daily

    # =========================
    # A) GLOBAL (Juste pour doc)
    # =========================
    print("Building Global Index...")
    g = process_panel(df, "global")

    # =========================
    # B) REGIONS CLÉS
    # =========================
    if "Region" in df.columns:
        print(f"Building Regional Indices for {KEEP_REGIONS}...")
        
        df_reg = df[df["Region"].isin(KEEP_REGIONS)].copy()
        
        # Pivot
        r = (df_reg.groupby(["Date", "Region"])
               .agg(fatal=("Deaths", "sum"))
               .reset_index())
        
        r_fatal = r.pivot(index="Date", columns="Region", values="fatal").reindex(all_days).fillna(0)
        
        out_reg = pd.DataFrame(index=all_days)
        out_reg.index.name = "Date"
        
        # Pour chaque colonne régionale
        for col in r_fatal.columns:
            safe_col = str(col).replace(" ", "_")
            series = r_fatal[col]
            
            # Log
            log_series = np.log1p(series)
            out_reg[f"log_fatal_{safe_col}"] = log_series
            
            # Shock Expanding (Même logique stricte)
            thresh = series.expanding(min_periods=EXPANDING_WINDOW).quantile(SHOCK_PERCENTILE).shift(1)
            
            # Init à 0
            shock_col = pd.Series(0, index=all_days)
            mask = thresh.notna()
            # On compare series vs thresh (là où mask est True)
            shock_col[mask] = (series[mask] >= thresh[mask]).astype(int)
            
            out_reg[f"shock_{safe_col}"] = shock_col
            
            # EWMA
            for lam in LAMBDA_LIST:
                out_reg[f"log_fatal_{safe_col}_ewma_{int(lam*100)}"] = ewma(log_series, lam)

        out_reg.to_csv(out_dir / "conflict_daily_by_region.csv")
        
        # Ref pour stats Middle East
        me_col = "Middle East"
        # Vérif si la colonne existe (dépend du print regions)
        if me_col in r_fatal.columns:
            me_stats = pd.DataFrame({"fatal": r_fatal[me_col]})
            me_stats["log_fatal"] = np.log1p(me_stats["fatal"])
            me_stats["log_fatal_ewma_94"] = ewma(me_stats["log_fatal"], 0.94)

    # =========================
    # C) FOCUS INDICES
    # =========================
    g_oil = pd.DataFrame()
    g_gas = pd.DataFrame()

    if "Country" in df.columns:
        print("Building Focus Indices...")
        
        # Oil Focus
        df_oil = df[df["Country"].isin(OIL_FOCUS_COUNTRIES)]
        g_oil = process_panel(df_oil, "oil_focus")
        
        # Gas Focus
        df_gas = df[df["Country"].isin(GAS_FOCUS_COUNTRIES)]
        g_gas = process_panel(df_gas, "gas_focus")

    # =========================
    # D) STATS DESCRIPTIVES
    # =========================
    print("\n" + "="*40)
    print("   CHECK STATS (log_fatal_ewma_94)")
    print("="*40)

    def print_stats(name, series):
        if series is None or series.empty:
            print(f"{name}: [EMPTY/ERROR]")
            return
        desc = series.describe()
        ratio = desc['std'] / desc['mean'] if desc['mean'] != 0 else 0
        print(f"\n--- {name} ---")
        print(f"Mean : {desc['mean']:.4f}")
        print(f"Std  : {desc['std']:.4f}")
        print(f"Min  : {desc['min']:.4f}")
        print(f"Max  : {desc['max']:.4f}")
        print(f"CV (Std/Mean): {ratio:.2f}")

    # 1. Oil Focus
    if "log_fatal_ewma_94" in g_oil.columns:
        print_stats("OIL FOCUS", g_oil["log_fatal_ewma_94"])

    # 2. Gas Focus
    if "log_fatal_ewma_94" in g_gas.columns:
        print_stats("GAS FOCUS", g_gas["log_fatal_ewma_94"])

    # 3. Middle East
    # On va chercher la colonne dans out_reg si elle existe
    if 'out_reg' in locals():
        # Construction du nom de colonne probable
        target_me = "log_fatal_Middle_East_ewma_94"
        if target_me in out_reg.columns:
             print_stats("MIDDLE EAST", out_reg[target_me])
        else:
            print(f"\n[INFO] Pas de colonne '{target_me}' trouvée (Vérifie le nom exact dans 'Regions found').")

# --- HELPER for the future merge ---
def prepare_region_view(region_file: Path, region_name: str, out_path: Path) -> Path | None:
    """Helper pour extraire une région (ex: Middle East) et renommer les colonnes."""
    if not region_file.exists(): return None
    
    df = pd.read_csv(region_file)
    safe_reg = region_name.replace(" ", "_")
    
    # Mapping des colonnes spécifiques vers des noms génériques
    cols_map = {
        f"log_fatal_{safe_reg}_ewma_94": "log_fatal_ewma_94",
        f"log_fatal_{safe_reg}_ewma_97": "log_fatal_ewma_97"
    }
    
    available = [c for c in cols_map.keys() if c in df.columns]
    if not available: return None

    view = df[["Date"] + available].rename(columns=cols_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    view.to_csv(out_path, index=False)
    return out_path