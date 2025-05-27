import streamlit as st
import numpy as np
import joblib
import pandas as pd

# --------------------------
# Cargar modelo entrenado
# --------------------------
model = joblib.load("xgb_model.pkl")  # Asegúrate de tener el modelo entrenado guardado así

st.set_page_config(page_title="Predicción de Incumplimiento", layout="centered")
st.title("Clasificador de Riesgo de Crédito")

st.markdown("Introduce los datos del cliente para predecir si incumplirá el próximo mes.")

# --------------------------
# Entradas del usuario
# --------------------------
col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input("Límite de crédito (NT$)", min_value=10000, max_value=1000000, step=10000)
    age = st.slider("Edad", min_value=18, max_value=80, value=35)
    sex = st.selectbox("Sexo", options=[1, 2], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
    education = st.selectbox("Nivel educativo", options=[1, 2, 3, 4], format_func=lambda x: {1: "Posgrado", 2: "Universidad", 3: "Secundaria", 4: "Otros"}[x])
    marriage = st.selectbox("Estado civil", options=[1, 2, 3], format_func=lambda x: {1: "Casado/a", 2: "Soltero/a", 3: "Otros"}[x])

with col2:
    pay_0 = st.slider("Retraso PAY_0", -1, 8, 0)
    pay_2 = st.slider("Retraso PAY_2", -1, 8, 0)
    pay_3 = st.slider("Retraso PAY_3", -1, 8, 0)
    pay_4 = st.slider("Retraso PAY_4", -1, 8, 0)
    pay_5 = st.slider("Retraso PAY_5", -1, 8, 0)
    pay_6 = st.slider("Retraso PAY_6", -1, 8, 0)

# --------------------------
# Cálculo de variables derivadas
# --------------------------
pay_cols = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]

# Ejemplo fijo para simulación (debería venir de inputs dinámicos si implementas facturación/pagos mensuales)
avg_bill_amt = 50000.0
avg_pay_amt = 2500.0

# Derivadas calculadas
pay_ratio = avg_pay_amt / (avg_bill_amt + 1)
pay_ratio = np.clip(pay_ratio, 0, 5)

months_delayed = sum(1 for p in pay_cols if p > 0)
max_delay = max(0, max(pay_cols))

credit_utilization = avg_bill_amt / (limit_bal + 1)
credit_utilization = np.clip(credit_utilization, 0, 2)

# --------------------------
# Construcción del vector de entrada
# --------------------------
input_vector = pd.DataFrame([{
    "LIMIT_BAL": limit_bal,
    "SEX": sex,
    "EDUCATION": education,
    "MARRIAGE": marriage,
    "AGE": age,
    "PAY_0": pay_0,
    "PAY_2": pay_2,
    "PAY_3": pay_3,
    "PAY_4": pay_4,
    "PAY_5": pay_5,
    "PAY_6": pay_6,
    "avg_bill_amt": avg_bill_amt,
    "avg_pay_amt": avg_pay_amt,
    "pay_ratio": pay_ratio,
    "months_delayed": months_delayed,
    "max_delay": max_delay,
    "credit_utilization": credit_utilization
}])

# --------------------------
# Predicción
# --------------------------
if st.button("Predecir incumplimiento"):
    proba = model.predict_proba(input_vector)[0, 1]
    pred = model.predict(input_vector)[0]

    st.markdown(f"**Probabilidad de impago**: {proba:.2%}")
    if pred == 1:
        st.error("Resultado: El cliente probablemente INCUMPLIRÁ")
    else:
        st.success("Resultado: El cliente probablemente CUMPLIRÁ")
