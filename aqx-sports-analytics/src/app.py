import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="DeepPitch: NBA Analytics", page_icon="🏀", layout="centered")

st.title("🏀 DeepPitch : Prédiction des Finales NBA")
st.write("""
**AQX Sports Analytics Data Bowl 2.0** Ce dashboard utilise un modèle d'intelligence artificielle (Random Forest) pour analyser les statistiques historiques des finales NBA et prédire si la série s'est terminée par un **Sweep** (une victoire écrasante 4-0).
""")

#  CHARGEMENT DU MODÈLE ET DES DONNÉES
MODEL_PATH = "src/model/weights/random_forest_model.pkl"
FEATURES_PATH = "src/model/weights/model_features.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    return None, None

model, features = load_model()

if model is None:
    st.error("⚠️ Modèle introuvable. Assure-toi d'avoir bien exécuté l'entraînement.")
else:
    st.sidebar.header("⚙️ Paramètres du match")
    st.sidebar.write("Ajustez les statistiques clés de l'équipe :")
    
    # Création dynamique des inputs selon les 3 features utilisées par ton modèle
    input_data = {}
    for feature in features:
        # Nettoyage du nom pour l'affichage (enlève le "_encoded" si présent)
        display_name = feature.replace("_encoded", "").replace("_", " ")
        input_data[feature] = st.sidebar.number_input(f"{display_name}", value=0.0, step=1.0)
        
    #  PRÉDICTION
    if st.button("🔮 Prédire le Résultat"):
        # Transformation des inputs en tableau pour le modèle
        df_input = pd.DataFrame([input_data])
        
        # Prédiction (Sweep ou pas)
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0]
    
        st.divider()
        st.subheader("📊 Résultat de l'analyse")
        
        # Affichage stylisé selon le résultat
        # On suppose ici que 1 = Sweep et 0 = Pas de sweep
        if prediction == 1:
            st.success("🧹 **Prédiction : SWEEP (4-0) !** L'équipe a de fortes chances d'écraser son adversaire.")
        else:
            st.info("⚔️ **Prédiction : PAS DE SWEEP.** La série risque d'être disputée.")
            
        # Affichage de la confiance de l'IA
        confidence = max(probability) * 100
        st.write(f"*Confiance du modèle : {confidence:.1f}%*")
        
        st.divider()
        st.write("💡 **Application Pratique :** Cet outil permet aux analystes sportifs et aux coachs d'évaluer la probabilité d'une domination totale lors des Finales NBA en se basant sur les métriques clés des équipes.")