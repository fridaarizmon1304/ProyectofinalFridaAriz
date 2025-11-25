import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# Cargar dataset
# ============================================================
@st.cache_resource
def load_data():
    df = pd.read_csv("vehicles_clean.csv")
    return df

df = load_data()

st.title("🚗 Análisis y Predicción de Precios de Vehículos Usados")
st.write("Proyecto Final · Ciencia de Datos · Frida Arizmendi")

# ============================================================
# Sección 1: Dataset Preview
# ============================================================
st.header("📊 Vista general del datos")
st.write(df.head())
st.write(f"Total de registros: **{len(df):,}**")

# ============================================================
# Sección 2: Gráficas EDA
# ============================================================
st.header("📈 Análisis de la informacion")
st.header("¿Cuantos vehiculos puedo encontrar referente a un precio?")

# ------- 1. Histograma de precios -------
fig, ax = plt.subplots()
ax.hist(df["price"], bins=40, color="skyblue", edgecolor="black")
ax.set_title("Distribución de Precios")
ax.set_xlabel("Precio")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

# ------- 2. Precio por año -------

st.header("¿Cual es el costo promedio de un vehiculo usado segun su Año?")
avg_year = df.groupby("year")["price"].mean()
fig, ax = plt.subplots()
avg_year.plot(ax=ax)
ax.set_title("Precio Promedio por Año Modelo")
ax.set_xlabel("Año")
ax.set_ylabel("Precio promedio")
st.pyplot(fig)

# ------- 3. Precio vs odometer -------
fig, ax = plt.subplots()
ax.scatter(df["odometer"], df["price"], alpha=0.3)
ax.set_title("Precio vs Kilometraje")
ax.set_xlabel("Kilometraje")
ax.set_ylabel("Precio")
st.pyplot(fig)

# ------- 4. Top fabricantes -------
st.header("¿Marcas con mas vehiculos usados en mercado?")
top_manu = df["manufacturer"].value_counts().head(10)
fig, ax = plt.subplots()
top_manu.plot(kind="bar", ax=ax, color="orange")
ax.set_title("Top 10 Fabricantes")
ax.set_ylabel("Número de vehículos")
st.pyplot(fig)

# ============================================================
# Sección 3: Entrenar modelo
# ============================================================
st.header("🤖 Modelo de Machine Learning")

@st.cache_resource
def train_model(df):

    # Usamos una muestra para que Streamlit sea rápido
    df_model = df.sample(n=20000, random_state=42) if len(df) > 20000 else df.copy()

    y = df_model["price"]
    X = df_model.drop(columns=["price"])

    numeric_features = ["year", "odometer"]
    categorical_features = [
        "manufacturer", "model", "condition", "cylinders",
        "fuel", "title_status", "transmission",
        "state", "type", "paint_color"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=40,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rf", rf)
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2


with st.spinner("Entrenando modelo..."):
    model, rmse, r2 = train_model(df)

st.success("Modelo entrenado correctamente 🎉")

st.subheader("📌 Métricas del modelo")
st.write(f"**RMSE:** {rmse:,.2f}")
st.write(f"**R²:** {r2:.4f}")

# ============================================================
# Sección 4: Predicción
# ============================================================
st.header("🔮 Predicción del precio del vehículo")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Año del vehículo", min_value=1980, max_value=2024, value=2015)
    odometer = st.number_input("Kilometraje", min_value=0, max_value=300000, value=60000)

with col2:
    manufacturer = st.selectbox("Fabricante", sorted(df["manufacturer"].unique()))
    model_car = st.selectbox("Modelo", sorted(df["model"].unique()))
    condition = st.selectbox("Condición", sorted(df["condition"].unique()))

fuel = st.selectbox("Combustible", sorted(df["fuel"].unique()))
transmission = st.selectbox("Transmisión", sorted(df["transmission"].unique()))
state = st.selectbox("Estado (USA)", sorted(df["state"].unique()))
paint = st.selectbox("Color", sorted(df["paint_color"].unique()))
type_car = st.selectbox("Tipo de vehículo", sorted(df["type"].unique()))
cylinders = st.selectbox("Cilindros", sorted(df["cylinders"].unique()))
title_status = st.selectbox("Título legal", sorted(df["title_status"].unique()))

input_data = pd.DataFrame([{
    "year": year,
    "odometer": odometer,
    "manufacturer": manufacturer,
    "model": model_car,
    "condition": condition,
    "fuel": fuel,
    "title_status": title_status,
    "transmission": transmission,
    "state": state,
    "type": type_car,
    "paint_color": paint,
    "cylinders": cylinders
}])

if st.button("Predecir precio"):
    pred = model.predict(input_data)[0]
    st.success(f"💰 El precio estimado del vehículo es: **${pred:,.2f} USD**")

