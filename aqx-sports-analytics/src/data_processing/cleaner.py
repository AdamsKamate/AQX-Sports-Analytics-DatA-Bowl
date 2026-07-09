import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def load_and_clean_data(filepath):
    """
    Charge le CSV brut et nettoie les données manquantes.
    """
    print(f"Chargement des données depuis : {filepath}")
    # Ajout de l'encodage latin1 qui a réglé le problème précédent
    df = pd.read_csv(filepath, encoding='latin1')
    
    # Suppression des lignes et colonnes 100% vides
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Remplissage des valeurs manquantes
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            # C'est un nombre : on remplace les vides par la médiane
            df[col] = df[col].fillna(df[col].median())
        else:
            # C'est du texte : on remplace les vides par 'Unknown'
            df[col] = df[col].fillna('Unknown')
            
    print("Nettoyage terminé. Aucune valeur nulle restante.")
    return df