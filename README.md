# 🚗 Proyecto Final – Ciencia de Datos 
## Predicción de precios de vehículos usados

**Autora:** Frida Arizmendi  25 de noviembre 2025

Este proyecto analiza un conjunto de datos de vehículos usados y construye un modelo de Machine Learning para estimar el precio de un auto en función de sus características principales. Además, se despliega una aplicación interactiva en Streamlit.

---

## 📊 Dataset

- Fuente: dataset público de autos usados (Craigslist)
- Registros: más de 3,000 filas
- Variables principales:
  - `price`, `year`, `manufacturer`, `model`, `condition`, `cylinders`
  - `fuel`, `odometer`, `title_status`, `transmission`, `state`, `type`, `paint_color`

El archivo limpio se encuentra en: **`vehicles_clean.csv`**.

---

## 🤖 Modelo de Machine Learning

Se entrena un **Random Forest Regressor** con:

- OneHotEncoder para variables categóricas
- Split 80% entrenamiento / 20% prueba
- Métricas aproximadas:
  - **RMSE:** ~7,300 USD
  - **R²:** ~0.74

El modelo se entrena dentro de la propia app de Streamlit para evitar problemas de compatibilidad de versiones.

---

## 🖥️ Aplicación en Streamlit

La aplicación incluye:

- Sección de análisis exploratorio (EDA) con 4 gráficas:
  - Histograma de precios
  - Precio promedio por año
  - Precio vs. kilometraje
  - Top 10 fabricantes
- Entrenamiento del modelo
- Visualización de métricas (RMSE y R²)
- Formulario para predecir el precio de un vehículo individual

### 🔗 Enlace a la app desplegada

>https://proyectofinalfridaariz-eaostdg3ca4mkmuzvbjgwk.streamlit.app/


---

## ▶️ Cómo ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
