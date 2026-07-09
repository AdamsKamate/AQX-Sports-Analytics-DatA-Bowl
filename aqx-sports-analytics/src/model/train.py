import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_path, model_save_path):
    print(f"Chargement des données depuis : {data_path}")
    
    # Vérifier si le fichier existe
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier {data_path} n'existe pas. As-tu bien fini la Phase 2 ?")
        return

    df = pd.read_csv(data_path)
    
    # PRÉPARATION DES DONNÉES
    # On suppose que la dernière colonne encodée (par exemple "MVP_encoded" ou "Champion_encoded")
    # est notre cible (Target). On prend la dernière colonne générée par features.py.
    target_col = df.columns[-1] 
    
    # On sépare les features (X) et la cible (y)
    # On ne garde que les colonnes numériques pour l'entraînement
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    features = [col for col in numeric_cols if col != target_col]
    
    X = df[features]
    y = df[target_col]
    
    print(f"Cible à prédire (Target) : {target_col}")
    print(f"Nombre de variables (Features) : {len(features)}")
    
    # Séparation Entraînement / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #  ENTRAÎNEMENT DU MODÈLE
    print("\nEntraînement du modèle Random Forest en cours...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # ÉVALUATION
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n✅ Précision du modèle (Accuracy) : {accuracy * 100:.2f}%")
    # print(classification_report(y_test, predictions)) # Décommente pour plus de détails
    
    # SAUVEGARDE
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    
    # On sauvegarde aussi les noms des colonnes pour l'interface Streamlit plus tard
    features_path = os.path.join(os.path.dirname(model_save_path), 'model_features.pkl')
    joblib.dump(features, features_path)
    
    print(f"\n💾 Modèle sauvegardé avec succès dans : {model_save_path}")

if __name__ == "__main__":
    PROCESSED_DATA = "data/processed/nba_processed.csv"
    MODEL_WEIGHTS = "src/model/weights/random_forest_model.pkl"
    
    train_model(PROCESSED_DATA, MODEL_WEIGHTS)