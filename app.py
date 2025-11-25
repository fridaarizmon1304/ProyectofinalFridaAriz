import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

@st.cache_resource
def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Sample for speed
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






# --------------------------------------------------
# Cargar datos y modelo
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("vehicles_clean.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("analisis_vehicles.pkl")
    return model

df = load_data()
model = load_model()

# --------------------------------------------------
# Configuración básica de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Análisis de Autos Usados",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Proyecto Final – Análisis de Autos Usados")
st.write("Aplicación desarrollada como parte del Proyecto Final de Ciencia de Datos.")

# --------------------------------------------------
# Sidebar – Navegación y filtros globales
# --------------------------------------------------
st.sidebar.title("🧭 Navegación")
section = st.sidebar.radio(
    "Selecciona sección:",
    ("🏠 Inicio", "📊 Exploración de Datos (EDA)", "📈 Dashboard", "🤖 Modelo de Predicción", "📝 Conclusiones")
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros globales (para Dashboard)")
fabricantes = ["Todos"] + sorted(df["manufacturer"].unique().tolist())
fab_sel = st.sidebar.selectbox("Fabricante", fabricantes)

min_year = int(df["year"].min())
max_year = int(df["year"].max())
year_range = st.sidebar.slider("Rango de año", min_year, max_year, (min_year, max_year))

combustibles = ["Todos"] + sorted(df["fuel"].unique().tolist())
fuel_sel = st.sidebar.selectbox("Combustible", combustibles)

# Aplicar filtros (solo para Dashboard, no para todo el df)
df_filt = df[
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

if fab_sel != "Todos":
    df_filt = df_filt[df_filt["manufacturer"] == fab_sel]

if fuel_sel != "Todos":
    df_filt = df_filt[df_filt["fuel"] == fuel_sel]

# --------------------------------------------------
# 🏠 SECCIÓN: INICIO
# --------------------------------------------------
if section == "🏠 Inicio":
    st.header("🏠 Descripción del Proyecto")
    st.write("""
    Este proyecto analiza un conjunto de datos de **autos usados** con el objetivo de:

    - Explorar la distribución de precios y características de los vehículos.
    - Identificar patrones entre variables como año, kilometraje, fabricante, etc.
    - Entrenar un modelo de **Machine Learning (Regresión)** para predecir el precio de un auto usado.
    - Proveer una interfaz interactiva en Streamlit para visualizar KPIs y realizar predicciones.

    **Dataset:**
    - Fuente: Kaggle – Craigslist Car/Truck Data (vehicles)
    - Observaciones: miles de registros de anuncios de autos usados
    - Variables clave: `price`, `year`, `manufacturer`, `model`, `odometer`, `fuel`, `transmission`, `state`, etc.
    """)

    st.subheader("✅ Objetivos del dashboard")
    st.markdown("""
    - Entender cómo se comportan los precios según año, fabricante y kilometraje.
    - Construir un modelo que estime el precio de un auto según sus características.
    - Facilitar la exploración de los datos mediante filtros interactivos.
    """)

    st.subheader("📌 Dimensiones del dataset")
    st.write(f"Filas: **{df.shape[0]}**, Columnas: **{df.shape[1]}**")
    st.dataframe(df.head())

# --------------------------------------------------
# 📊 SECCIÓN: EDA
# --------------------------------------------------
elif section == "📊 Exploración de Datos (EDA)":
    st.header("📊 Exploración de Datos (EDA)")

    st.subheader("1. Distribución de precios")
    fig, ax = plt.subplots(figsize=(8, 4))
    df["price"].hist(bins=50, ax=ax)
    ax.set_xlabel("Precio")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de precios de autos usados")
    st.pyplot(fig)

    st.subheader("2. Relación entre año y precio")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["year"], df["price"], alpha=0.2)
    ax.set_xlabel("Año del vehículo")
    ax.set_ylabel("Precio")
    ax.set_title("Relación entre año y precio")
    st.pyplot(fig)

    st.subheader("3. Relación entre kilometraje (odometer) y precio")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["odometer"], df["price"], alpha=0.2)
    ax.set_xlabel("Kilometraje (odometer)")
    ax.set_ylabel("Precio")
    ax.set_title("Precio vs Kilometraje")
    st.pyplot(fig)

    st.subheader("4. Precio promedio por año (variación)")
    avg_price_by_year = df.groupby("year")["price"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    avg_price_by_year.plot(kind="line", marker="o", ax=ax)
    ax.set_xlabel("Año del vehículo")
    ax.set_ylabel("Precio promedio")
    ax.set_title("Variación del precio promedio por año del vehículo")
    ax.grid(True)
    st.pyplot(fig)

# --------------------------------------------------
# 📈 SECCIÓN: Dashboard
# --------------------------------------------------
elif section == "📈 Dashboard":
    st.header("📈 Dashboard Interactivo")

    st.write("Los filtros en el sidebar afectan estos indicadores y gráficas.")

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precio promedio (USD)", f"{df_filt['price'].mean():,.0f}")
    with col2:
        st.metric("Año promedio", f"{df_filt['year'].mean():.1f}")
    with col3:
        st.metric("Kilometraje promedio", f"{df_filt['odometer'].mean():,.0f}")

    st.subheader("Precio promedio por año (según filtros)")
    avg_price_year_filt = df_filt.groupby("year")["price"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_price_year_filt.plot(kind="bar", ax=ax)
    ax.set_xlabel("Año")
    ax.set_ylabel("Precio promedio")
    ax.set_title("Precio promedio por año (filtrado)")
    st.pyplot(fig)

    st.subheader("Top 10 fabricantes por número de anuncios (según filtros)")
    top_manu = df_filt["manufacturer"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    top_manu.plot(kind="bar", ax=ax)
    ax.set_ylabel("Cantidad de anuncios")
    ax.set_title("Top 10 fabricantes más frecuentes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.subheader("Muestra de datos filtrados")
    st.dataframe(df_filt.head(50))

# --------------------------------------------------
# 🤖 SECCIÓN: Modelo de Predicción
# --------------------------------------------------
elif section == "🤖 Modelo de Predicción":
    st.header("🤖 Predicción de precio de auto usado")

    st.write("""
    Completa las características del vehículo y el modelo de Machine Learning 
    estimará un precio aproximado.
    """)

    # Formularios de entrada (coinciden con las columnas usadas en el modelo)
    col1, col2 = st.columns(2)

    with col1:
        manufacturer = st.selectbox("Fabricante (manufacturer)", sorted(df["manufacturer"].unique()))
        model_name = st.text_input("Modelo (model)", value="corolla")
        condition = st.selectbox("Condición (condition)", sorted(df["condition"].unique()))
        cylinders = st.selectbox("Cilindros (cylinders)", sorted(df["cylinders"].unique()))
        fuel = st.selectbox("Combustible (fuel)", sorted(df["fuel"].unique()))

    with col2:
        title_status = st.selectbox("Estatus del título (title_status)", sorted(df["title_status"].unique()))
        transmission = st.selectbox("Transmisión (transmission)", sorted(df["transmission"].unique()))
        state = st.selectbox("Estado (state)", sorted(df["state"].unique()))
        car_type = st.selectbox("Tipo de vehículo (type)", sorted(df["type"].unique()))
        paint_color = st.selectbox("Color (paint_color)", sorted(df["paint_color"].unique()))

    year_input = st.number_input("Año del vehículo (year)", min_value=int(df["year"].min()),
                                 max_value=int(df["year"].max()), value=2015, step=1)
    odometer_input = st.number_input("Kilometraje (odometer)", min_value=0, max_value=500000, value=80000, step=1000)

    if st.button("Calcular precio estimado"):
        # Construir un DataFrame con una sola fila con las mismas columnas que X
        input_dict = {
            "manufacturer": manufacturer,
            "model": model_name,
            "condition": condition,
            "cylinders": cylinders,
            "fuel": fuel,
            "title_status": title_status,
            "transmission": transmission,
            "state": state,
            "type": car_type,
            "paint_color": paint_color,
            "year": year_input,
            "odometer": odometer_input
        }

        input_df = pd.DataFrame([input_dict])

        # Predicción
        try:
            pred_price = model.predict(input_df)[0]
            st.success(f"💰 Precio estimado: **${pred_price:,.0f} USD**")
            st.caption("El valor es aproximado, basado en patrones aprendidos por el modelo de regresión.")
        except Exception as e:
            st.error("Ocurrió un error al predecir. Verifica las entradas o el modelo.")
            st.text(str(e))

# --------------------------------------------------
# 📝 SECCIÓN: Conclusiones
# --------------------------------------------------
elif section == "📝 Conclusiones":
    st.header("📝 Conclusiones del análisis")

    st.markdown("""
    **Hallazgos clave:**

    - Existe una relación clara entre el **año del vehículo** y el **precio**, donde los modelos más recientes tienden a tener precios significativamente más altos.
    - El **kilometraje (odometer)** influye negativamente en el precio: a mayor recorrido, menor valor esperado del vehículo.
    - Algunos **fabricantes** concentran la mayor parte de los anuncios y muestran precios promedio más altos, lo que indica posibles segmentos de mercado diferenciados.

    **Resultados del modelo de Machine Learning:**

    - Se entrenó un modelo de **Random Forest Regressor**.
    - Métrica de desempeño:
        - RMSE ≈ 7,200 USD (error promedio en la estimación de precio).
        - R² ≈ 0.74, lo que indica que el modelo explica alrededor del 74% de la variabilidad del precio.
    - El modelo es adecuado para realizar estimaciones aproximadas de precio en contexto de autos usados.

    **Recomendaciones:**

    - Utilizar el modelo como herramienta de apoyo para estimar rangos de precios razonables en función de año, kilometraje y características del vehículo.
    - Profundizar en análisis por fabricante y tipo de vehículo para identificar nichos específicos de alto valor.
    - Seguir refinando el modelo con más variables (por ejemplo, ubicación exacta, equipamiento, historial de accidentes) si se dispone de esos datos.

    """)

    st.markdown("Gracias por revisar el proyecto 🙌. Esta aplicación forma parte del Proyecto Final de Ciencia de Datos.")


