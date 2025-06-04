# 💳 FRAUDTECH - Clasificador de Riesgo Crediticio

Este repositorio contiene una aplicación web desarrollada con **Streamlit** que permite clasificar clientes según su **riesgo de impago crediticio** utilizando un modelo de **XGBoost** previamente entrenado. Incluye herramientas de interpretación de resultados con **SHAP**, y permite cargar archivos CSV con información financiera para su evaluación.

---

## 📁 Estructura del Proyecto

```
.
├── app3.py                  # Aplicación Streamlit
├── Notebook_TFM.ipynb       # Notebook de entrenamiento y análisis del modelo
├── xgb_model.pkl            # Modelo XGBoost entrenado y serializado
├── requirements.txt         # Requisitos del entorno
├── ac91235d-147c-4185...png # Logo usado en la app
├── 1_cliente.csv            # Ejemplo de CSV individual
├── 2_clientes.csv           # CSV de múltiples clientes
```

---

## 🚀 Cómo usar la aplicación

### 1. Instalar dependencias

Se recomienda crear un entorno virtual. Luego ejecuta:

```bash
pip install -r requirements.txt
```

### 2. Lanzar la aplicación

Ejecuta el archivo principal:

```bash
streamlit run app3.py
```

Esto abrirá la app en tu navegador en la URL: `http://localhost:8501/`

---

## 📊 ¿Qué hace la aplicación?

- Permite subir un archivo CSV con datos crediticios.
- Realiza un preprocesamiento automático para generar variables como:
  - `avg_bill_amt_log`, `avg_pay_amt_log`
  - `pay_ratio`, `credit_utilization`
  - Indicadores de morosidad: `months_delayed`, `max_delay`, etc.
- Clasifica cada cliente como:
  - ✅ **Cumple**
  - ❗ **Incumple**
- Muestra la probabilidad de impago individual.
- Visualiza explicaciones con SHAP para el primer cliente analizado.

---

## 📂 Formato del CSV

El CSV debe contener al menos las siguientes columnas:

```csv
LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE,
PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
BILL_AMT1-6, PAY_AMT1-6
```

📝 Puedes usar `1_cliente.csv` o `2_clientes.csv` como ejemplo de estructura correcta.

---

## 🧠 Sobre el modelo

- Modelo: **XGBoost Classifier**
- Entrenado sobre el dataset de default de clientes de tarjetas de crédito de Taiwán (UCI).
- Métrica optimizada: `ROC-AUC`
- Umbral ajustado a `0.30` para maximizar recall de los casos de incumplimiento.
- Incluye ingeniería de características y codificación dummies (`get_dummies`).

---

## 🔍 Interpretabilidad con SHAP

Se utiliza la librería **SHAP** para mostrar los dos factores más relevantes en la clasificación del primer cliente del CSV, con indicaciones sobre si **incrementan o reducen** el riesgo.

---

## 🛡️ Logo e Identidad

El logo de la app representa seguridad y tecnología antifraude, aportando identidad visual al clasificador.

---

## 📷 Vista previa

<img src="ac91235d-147c-4185-aaaa-2702724c14ba.png" width="150">

---

## 🤝 Autores y créditos

Este proyecto fue desarrollado como parte de un TFM de Machine Learning para detección de fraude crediticio, combinando visualización interactiva, interpretabilidad y despliegue ágil.

---

## 📌 Requisitos

- Python 3.8+
- Navegador moderno
- Entorno de desarrollo recomendado: VSCode + Streamlit
