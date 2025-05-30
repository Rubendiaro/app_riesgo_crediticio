import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image

# Configuración de página y estilo
st.set_page_config(page_title="Clasificador CSV", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            font-family: "Segoe UI", sans-serif;
        }
        h1, h2, h3 {
            color: #1a237e;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 1.1em;
        }
        .stDataFrame {
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 10px;
        }
        .logo-container {
            position: absolute;
            top: 10px;
            right: 20px;
            z-index: 10;
        }
    </style>
""", unsafe_allow_html=True)

# Logo superior derecho (imagen local) con HTML fijo para no mover el layout
display_logo = """
<div class="logo-container">
    <img src="data:image/png;base64,{}" width="80">
</div>
"""
import base64
with open("ac91235d-147c-4185-aaaa-2702724c14ba.png", "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode()
st.markdown(display_logo.format(encoded_logo), unsafe_allow_html=True)

st.title("📊 Clasificador de Riesgo de Crédito desde CSV")
st.markdown("Sube un archivo con información crediticia y obtén la probabilidad de incumplimiento para cada cliente.")

# --------------------------
# Cargar modelo entrenado
# --------------------------
model = joblib.load("xgb_model.pkl")

# --------------------------
# Subida de CSV
# --------------------------
file = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # Validación básica de columnas mínimas necesarias
    required_cols = [f"PAY_AMT{i}" for i in range(1, 7)] + [f"BILL_AMT{i}" for i in range(1, 7)] + [
        f"PAY_{i}" for i in [0,2,3,4,5,6]] + ["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE"]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ Faltan columnas necesarias en el CSV.")
    else:
        # --------------------------
        # Transformaciones
        # --------------------------
        df["avg_bill_amt"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].apply(lambda row: row[row > 0].mean(), axis=1).fillna(0)
        q99_bill = 200000
        df["avg_bill_amt_winz"] = df["avg_bill_amt"].clip(upper=q99_bill)
        df["avg_bill_amt_log"] = np.log1p(df["avg_bill_amt_winz"])

        df["avg_pay_amt"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].mean(axis=1)
        q99_pay = 20000
        df["avg_pay_amt_winz"] = df["avg_pay_amt"].clip(upper=q99_pay)
        df["avg_pay_amt_log"] = np.log1p(df["avg_pay_amt_winz"])

        df["pay_ratio"] = df["avg_pay_amt_winz"] / (df["avg_bill_amt_winz"] + 1)
        df["pay_ratio"] = df["pay_ratio"].clip(upper=5)

        pay_cols = [f"PAY_{i}" for i in [0,2,3,4,5,6]]
        df["months_delayed"] = df[pay_cols].gt(0).sum(axis=1)
        df["max_delay"] = df[pay_cols].clip(lower=0).max(axis=1)
        df["mean_delay"] = df[pay_cols].apply(lambda row: row[row > 0].mean() if any(row > 0) else 0, axis=1)
        df["any_delay"] = (df["months_delayed"] > 0).astype(int)

        df["credit_utilization"] = df["avg_bill_amt_winz"] / (df["LIMIT_BAL"] + 1)
        df["credit_utilization"] = df["credit_utilization"].clip(upper=2)

        # --------------------------
        # Preparar datos como en entrenamiento
        # --------------------------
        columns_modelo = [
            'LIMIT_BAL', 'AGE', 'avg_bill_amt', 'avg_bill_amt_winz', 'avg_bill_amt_log',
            'avg_pay_amt', 'avg_pay_amt_winz', 'avg_pay_amt_log', 'pay_ratio',
            'months_delayed', 'max_delay', 'mean_delay', 'any_delay', 'credit_utilization',
            'SEX_2', 'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_2', 'MARRIAGE_3'
        ]

        df_encoded = pd.get_dummies(df)
        for col in columns_modelo:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[columns_modelo]

        # --------------------------
        # Predicción con umbral ajustado
        # --------------------------
        y_proba = model.predict_proba(df_encoded)[:, 1]
        threshold = 0.30
        y_pred = (y_proba > threshold).astype(int)

        df_result = df.copy()
        df_result["Probabilidad de impago"] = y_proba
        df_result["Predicción"] = y_pred
        df_result["Predicción"] = df_result["Predicción"].map({0: "✔️ Cumple", 1: "❗ Incumple"})

        st.success("✅ Predicción completada para todos los clientes.")

        for idx, row in df_result.iterrows():
            st.markdown("""
                <div style='background-color: white; border-left: 6px solid #1a73e8; padding: 1em; margin-bottom: 1em; border-radius: 6px;'>
                    <h4>Cliente #{}</h4>
                    <ul>
                        <li><strong>Edad:</strong> {} años</li>
                        <li><strong>Límite de crédito:</strong> {:,.0f} NT$</li>
                        <li><strong>Probabilidad de impago:</strong> {:.2%}</li>
                        <li><strong>Predicción:</strong> <strong style='color:{};'>{}</strong></li>
                    </ul>
                </div>
            """.format(
                idx+1,
                row["AGE"],
                row["LIMIT_BAL"],
                row["Probabilidad de impago"],
                "red" if row["Predicción"] == "❗ Incumple" else "green",
                row["Predicción"]
            ), unsafe_allow_html=True)

        # --------------------------
        # Explicación SHAP del primer cliente
        # --------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_encoded)

        feature_importances = pd.Series(shap_values[0], index=df_encoded.columns)
        top_features = feature_importances.abs().sort_values(ascending=False).head(2)

        st.markdown("### 🔍 Diagnóstico del primer cliente")
        for feature in top_features.index:
            valor = df_encoded.iloc[0][feature]
            peso = feature_importances[feature]
            razon = "incrementa el riesgo" if peso > 0 else "reduce el riesgo"
            st.markdown(f"- **{feature}** = `{valor}` → {razon} de impago")
