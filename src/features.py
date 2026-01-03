# src/features.py
import pandas as pd
import numpy as np
from pathlib import Path

def build_features_df(
    df: pd.DataFrame, 
    price_col: str = "Price", 
    date_col: str = "Date", 
    window: int = 21,
    annualize: bool = False
) -> pd.DataFrame:
    """
    Calcule la volatilité réalisée.
    Règle : On ne supprime une ligne que si le PRIX est manquant.
    Les lignes avec Volume manquant sont conservées.
    """
    out = df.copy()

    # Vérifications
    if date_col not in out.columns or price_col not in out.columns:
        raise KeyError(f"Colonnes manquantes. Cherché: {date_col}, {price_col}.")

    # Conversions
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    
    # 1. NETTOYAGE STRICT : PRIX UNIQUEMENT
    # On ne supprime la ligne que si Date OU Price est vide.
    # Si 'Vol.' est vide, la ligne est CONSERVÉE.
    out = out.dropna(subset=[date_col, price_col]).sort_values(date_col)

    # 2. Calcul des Rendements
    out["Log_Ret"] = np.log(out[price_col] / out[price_col].shift(1))

    # 3. Calcul de la Cible (Volatilité)
    out["Squared_Ret"] = out["Log_Ret"] ** 2
    out["Realized_Vol_21d"] = out["Squared_Ret"].rolling(window=window).sum()

    if annualize:
        out["Vol_Ann_Std"] = np.sqrt(out["Realized_Vol_21d"]) * np.sqrt(252 / window)

    # 4. Nettoyage final (Target)
    # On est obligé de supprimer les 21 premières lignes où la volatilité 
    # n'a pas pu être calculée (car pas assez d'historique).
    out = out.dropna(subset=["Realized_Vol_21d"])

    # 5. Suppression des colonnes inutiles
    cols_to_drop = ["High", "Low", "Squared_Ret"] 
    out = out.drop(columns=cols_to_drop, errors="ignore")
    
    return out

def process_features_file(
    input_path: Path, 
    output_path: Path
) -> None:
    """Fonction utilitaire pour charger, calculer et sauvegarder."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Fichier introuvable : {input_path}")
        return

    # Chargement
    df = pd.read_csv(input_path)
    
    # Calcul
    try:
        df_features = build_features_df(df)
        
        # Sauvegarde
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)
        print(f"Features générées : {output_path} | Shape: {df_features.shape}")
        
    except KeyError as e:
        print(f"Erreur sur {input_path.name}: {e}")