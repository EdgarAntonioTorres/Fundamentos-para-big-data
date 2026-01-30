import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

st.title("Dashboard de Análisis de Datos")

CSV_FILE = "./20250929-enroll.csv"

if os.path.exists("./20250929-enroll.csv"):
    CSV_FILE = "./20250929-enroll.csv"
else:
    CSV_FILE = "./sample.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE)
    return df

df = load_data()

# MOSTRAR PARTE DE LA TABLA
st.subheader("Vista previa del dataset")
st.dataframe(df.head(100))

# MUESTRA PARA GRÁFICAS
df_sample = df.sample(10000, random_state=1)

# Solo columnas numéricas
df_num = df_sample.select_dtypes(include="number")

# GRÁFICA DE BARRAS
st.subheader("Gráfica de Barras")

col_bar = st.selectbox(
    "Selecciona una columna numérica",
    df_num.columns
)

fig, ax = plt.subplots()
df_num[col_bar].value_counts().head(10).plot(kind="bar", ax=ax)
ax.set_xlabel(col_bar)
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

# GRÁFICA DE DISPERSIÓN
st.subheader("Gráfica de Dispersión")

x_axis = st.selectbox("Eje X", df_num.columns, key="x")
y_axis = st.selectbox("Eje Y", df_num.columns, key="y")

fig, ax = plt.subplots()
ax.scatter(df_num[x_axis], df_num[y_axis], alpha=0.5, color = "cyan", edgecolors = "black")
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
st.pyplot(fig)

st.subheader("Matriz de correlación")
corr = df_num.corr()

corr_sin_diag = corr.copy()
np.fill_diagonal(corr_sin_diag.values, 0)

cols_validas = corr_sin_diag.columns[(corr_sin_diag.abs() > 0).any()]

corr = corr.loc[cols_validas, cols_validas]

# HEATMAP
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if corr.iloc[i, j] != 0:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center")

plt.title("Matriz de correlación")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Tabla de correlación")
st.dataframe(corr)