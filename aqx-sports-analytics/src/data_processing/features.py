import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from src.data_processing.cleaner import load_and_clean_data

def generate_features(df):
    """
    Transforme les colonnes textuelles en valeurs numériques (Encodage).
    """
    le = LabelEncoder()
    df_processed = df.copy()
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # Crée une nouvelle colonne encodée (ex: 'Team' devient 'Team_encoded' avec des chiffres)
            new_col_name = f"{col}_encoded"
            df_processed[new_col_name] = le.fit_transform(df_processed[col].astype(str))
            
    print(f"Feature engineering terminé. {len(df_processed.columns)} colonnes prêtes.")
    return df_processed

if __name__ == "__main__":
    # Définition des chemins (en supposant qu'on lance depuis la racine du projet)
    RAW_DATA_PATH = "data/raw/NBA Finals and MVP.csv"
    PROCESSED_DATA_PATH = "data/processed/nba_processed.csv"
    
    # Exécution du pipeline
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    df_final = generate_features(df_clean)
    
    # Sauvegarde
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ SUCCÈS : Données traitées sauvegardées dans {PROCESSED_DATA_PATH}")
    