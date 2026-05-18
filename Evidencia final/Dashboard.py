import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Clientes",
    page_icon="📊",
    layout="wide",
)

sns.set(style="whitegrid")

# Carga de datos
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    df["age_group"] = pd.cut(df["age"], bins=[18, 25, 35, 45, 55, 65, 100])
    return df


st.title("Dashboard de Clientes Call Center")
st.markdown("---")

uploaded = st.sidebar.file_uploader("Cargar archivo clientes.csv", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded, delimiter=";")
    df["age_group"] = pd.cut(df["age"], bins=[18, 25, 35, 45, 55, 65, 100])
else:
    try:
        df = load_data("./clientes.csv")
    except FileNotFoundError:
        st.warning("No se encontró `clientes.csv`. Sube el archivo desde la barra lateral.")
        st.stop()

# KPIs en la parte superior
total = len(df)
conversiones = (df["y"] == "yes").sum()
tasa = conversiones / total * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total de clientes", f"{total:,}")
col2.metric("Conversiones (Sí)", f"{conversiones:,}")
col3.metric("Tasa de conversión", f"{tasa:.1f}%")

st.markdown("---")

# Conversiones
st.subheader("Conversiones")
resultado_counts = df["y"].value_counts()

c1, c2 = st.columns(2)

with c1:
    st.markdown("##### Cantidad de conversiones (Sí vs No)")
    fig, ax = plt.subplots()
    sns.barplot(x=resultado_counts.index, y=resultado_counts.values, ax=ax)
    ax.set_xlabel("Resultado")
    ax.set_ylabel("Número de clientes")
    st.pyplot(fig)
    plt.close(fig)

with c2:
    st.markdown("##### Proporción de conversión")
    fig, ax = plt.subplots()
    ax.pie(resultado_counts.values, labels=resultado_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# Demografía
st.subheader("Demografía")

c3, c4 = st.columns(2)

with c3:
    st.markdown("##### Distribución por estado civil")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="marital", ax=ax)
    ax.set_xlabel("Estado civil")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)
    plt.close(fig)

with c4:
    st.markdown("##### Distribución por grupos de edad")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="age_group", ax=ax)
    ax.set_xlabel("Rango de edad")
    ax.set_ylabel("Cantidad")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# Por trabajo 
st.subheader("Análisis por trabajo")

c5, c6 = st.columns(2)

with c5:
    st.markdown("##### Top 10 trabajos más frecuentes")
    top_jobs = df["job"].value_counts().nlargest(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_jobs.index, y=top_jobs.values, ax=ax)
    ax.set_xlabel("Trabajo")
    ax.set_ylabel("Cantidad")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

with c6:
    st.markdown("##### Tasa de conversión por trabajo")
    conversion_job = (
        df.groupby("job")["y"]
        .apply(lambda x: (x == "yes").mean())
        .reset_index()
    )
    conversion_job.columns = ["job", "tasa_conversion"]
    fig, ax = plt.subplots()
    sns.barplot(data=conversion_job, x="job", y="tasa_conversion", ax=ax)
    ax.set_xlabel("Trabajo")
    ax.set_ylabel("Probabilidad de conversión")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# Duración 
st.subheader("Duración de llamadas")

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="y", y="duration", ax=ax)
ax.set_xlabel("Resultado")
ax.set_ylabel("Duración (segundos)")
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# Vista previa de datos
with st.expander("Vista previa de los datos"):
    st.dataframe(df.head(20), use_container_width=True)