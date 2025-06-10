
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prédiction Blessure Rugby", layout="wide")
st.title("🏉 Prédiction du Type de Blessure chez un Joueur de Rugby")

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("rugby_injury_dataset.csv")
    X = df.drop(columns=["predicted_injury_type"])
    y = df["predicted_injury_type"]
    X = pd.get_dummies(X, columns=["poste", "previous_injury_type"], drop_first=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), df

model, feature_names, full_df = load_data_and_model()

st.subheader("🎯 Visualisation des blessures simulées")
injury_counts = full_df["predicted_injury_type"].value_counts()
fig, ax = plt.subplots()
injury_counts.plot(kind="bar", color="salmon", ax=ax)
ax.set_title("Distribution simulée des types de blessure")
ax.set_ylabel("Nombre de cas")
st.pyplot(fig)

st.subheader("🧪 Entrez les données de votre joueur")

inputs = {}
inputs["age"] = st.slider("Âge", 18, 40, 25)
inputs["poste"] = st.selectbox("Poste", ["avants", "arrières"])
inputs["fatigue"] = st.slider("Fatigue (1-10)", 1, 10, 5)
inputs["soreness"] = st.slider("Douleur musculaire (1-10)", 1, 10, 5)
inputs["sleep"] = st.slider("Sommeil (heures)", 3.0, 10.0, 7.0)
inputs["training_load"] = st.slider("Charge d'entraînement", 500, 2000, 1000)
inputs["rest_days"] = st.slider("Jours de repos", 0, 7, 2)
inputs["glute_strength"] = st.slider("Force fessier moyen (kg)", 15, 50, 30)
inputs["knee_to_wall"] = st.slider("Knee to Wall (cm)", 4.0, 20.0, 10.0)
inputs["y_balance"] = st.slider("Y Balance (% symétrie)", 50, 100, 90)
inputs["quad_force"] = st.slider("Force quadriceps (N)", 60, 180, 130)
inputs["ham_force"] = st.slider("Force ischios (N)", 40, 150, 85)
inputs["navicular_drop"] = st.slider("Navicular Drop (mm)", 4, 14, 8)
inputs["previous_injury_type"] = st.selectbox("Antécédent de blessure", ["aucune", "cheville", "genou", "muscle", "épaule"])

if st.button("🔍 Prédire le type de blessure probable"):
    user_df = pd.DataFrame([inputs])
    user_df = pd.get_dummies(user_df)
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[feature_names]
    prediction = model.predict(user_df)[0]
    st.success(f"🩺 Risque probable de blessure : **{prediction.upper()}**")

    if prediction == "genou":
        st.warning("⚠️ Surveillez le contrôle moteur, force du fessier et l’équilibre.")
    elif prediction == "cheville":
        st.warning("⚠️ Améliorer la mobilité cheville et la proprioception.")
    elif prediction == "musculaire":
        st.warning("⚠️ Travailler l'équilibre quadriceps/ischio et gestion de la fatigue.")
    elif prediction == "tendinopathie":
        st.warning("⚠️ Optimiser la charge et la récupération.")
    elif prediction == "épaule":
        st.warning("⚠️ Renforcer le tronc et stabilisateurs d'épaule.")
    else:
        st.success("✅ Aucun risque particulier détecté.")
