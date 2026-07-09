import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DeepPitch: NBA Analytics", page_icon="🏀", layout="centered")

st.title("🏀 DeepPitch: NBA Finals Prediction")
st.write("""
**AQX Sports Analytics Data Bowl 2.0** This dashboard uses an Artificial Intelligence model (Random Forest) to analyze historical NBA Finals statistics and predict whether the series will end in a **Sweep** (a crushing 4-0 victory).
""")

# --- MODEL & DATA LOADING ---
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
    st.error("⚠️ Model not found. Please ensure the training script has been executed successfully.")
else:
    st.sidebar.header("⚙️ Match Parameters")
    st.sidebar.write("Adjust the key statistics:")
    
    # Dynamic input creation based on the features used by the model
    input_data = {}
    for feature in features:
        # Clean up feature names for the UI (removes "_encoded")
        display_name = feature.replace("_encoded", "").replace("_", " ")
        input_data[feature] = st.sidebar.number_input(f"{display_name}", value=0.0, step=1.0)
        
    # --- PREDICTION ---
    if st.button("🔮 Predict Result"):
        # Convert inputs into a DataFrame for the model
        df_input = pd.DataFrame([input_data])
        
        # Predict (Sweep or not)
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0]
        
        st.divider()
        st.subheader("📊 Analysis Result")
        
        # Display stylized result
        # Assuming 1 = Sweep and 0 = No sweep
        if prediction == 1:
            st.success("🧹 **Prediction: SWEEP (4-0)!** The team has a high probability of crushing their opponent.")
        else:
            st.info("⚔️ **Prediction: NO SWEEP.** The series is expected to be competitive.")
            
        # Display model confidence
        confidence = max(probability) * 100
        st.write(f"*Model Confidence: {confidence:.1f}%*")
        
        st.divider()
        st.write("💡 **Practical Application:** This tool allows sports analysts and coaches to evaluate the probability of total domination during the NBA Finals based on key metrics.")