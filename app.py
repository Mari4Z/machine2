
import joblib
import streamlit as st
import numpy as np
import pandas as pd

model = joblib.load("modelo_pred.pkl")

NAVY = "#002B5C"
GOLD = "#FFC72C"

st.markdown(f"""
<style>
  section[data-testid="stSidebar"] {{ background-color: {NAVY} !important; }}
  section[data-testid="stSidebar"] * {{ color: {GOLD} !important; }}
  section[data-testid="stSidebar"] label {{ color: white !important; }}
  .stButton > button {{
      background-color: {GOLD} !important;
      color: {NAVY} !important;
      font-weight: bold !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 10px 28px !important;
      font-size: 1rem !important;
  }}
  .stApp {{ background-color: #f4f6f9; }}
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://upload.wikimedia.org/wikipedia/en/0/01/Golden_State_Warriors_logo.svg"
st.markdown(f"""
<div style="
    background: linear-gradient(90deg, {NAVY} 70%, #1D428A 100%);
    padding: 20px 32px;
    border-radius: 12px;
    border-bottom: 4px solid {GOLD};
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
">
  <img src="{LOGO_URL}" style="height:70px; width:auto;" />
  <div>
    <h2 style="color:{GOLD}; margin:0; font-size:1.8rem; letter-spacing:.05em;">GOLDEN STATE WARRIORS</h2>
    <p style="color:white; margin:4px 0 0 0; font-size:0.9rem; opacity:.85;">
      Clasificador de Tipo de Cliente · Ticket Timing Model
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

CLASSES = ["In-Between", "Last-Minute", "Planner"]

# Retornos esperados por tipo (calculados de df_raw en el notebook)
RETORNO_POR_TIPO = {
    "Planner":     880.0,   # reemplaza con retorno_planner
    "In-Between":  530.0,   # reemplaza con retorno_in_between
    "Last-Minute": 265.0,   # reemplaza con retorno_last_minute
}

# ── Sidebar con dos pestañas ─────────────────────────────────
with st.sidebar:
    pestaña = st.radio("", ["Predecir cliente", " Retorno esperado"])

# ════════════════════════════════════════════════════════════
# PESTAÑA 1 — Predecir cliente
# ════════════════════════════════════════════════════════════
if pestaña == "Predecir cliente":

    st.markdown("Introduce los datos del fan para predecir su *Customer Type*.")

    # Sliders ahora en la página principal
    age     = st.slider("Edad", 18, 76, 43)
    mailing = st.selectbox("¿Suscrito a lista de correos?", ["Yes", "No"])
    days    = st.slider("Días antes del juego", 0, 36, 9)

    if st.button("Predecir Customer Type"):
        mailing_encoded = 1 if mailing == "Yes" else 0

        input_data = pd.DataFrame([{
            'Days_Before_Game': days,
            'Age':              age,
            'Fan_Mailing_List': mailing_encoded,
        }])

        pred_numeric    = model.predict(input_data)[0]
        predicted_class = CLASSES[pred_numeric]
        st.success(f"Clase predicha: **{predicted_class}**")

        st.subheader("Probabilidades de cada clase:")
        probabilities = model.predict_proba(input_data)[0]
        prob_df = pd.DataFrame({"Customer Type": CLASSES, "Probabilidad": probabilities})
        prob_df["Probabilidad"] = prob_df["Probabilidad"].map(lambda x: f"{x:.2%}")
        st.dataframe(prob_df.set_index("Customer Type"))

# ════════════════════════════════════════════════════════════
# PESTAÑA 2 — Retorno esperado
# ════════════════════════════════════════════════════════════
else:

    st.markdown("###  Calculadora de Retorno Esperado")
    st.markdown("Define cuántas ventas quieres lograr por tipo de cliente y calcula el ingreso estimado.")

    # La empresa ingresa sus metas de ventas por tipo
    meta_planner     = st.number_input("Meta de ventas — Planner",     min_value=0, value=100, step=10)
    meta_in_between  = st.number_input("Meta de ventas — In-Between",  min_value=0, value=100, step=10)
    meta_last_minute = st.number_input("Meta de ventas — Last-Minute", min_value=0, value=100, step=10)

    if st.button("Calcular retorno"):

        # Retorno = meta de ventas × retorno promedio por tipo
        ret_planner     = meta_planner     * RETORNO_POR_TIPO["Planner"]
        ret_in_between  = meta_in_between  * RETORNO_POR_TIPO["In-Between"]
        ret_last_minute = meta_last_minute * RETORNO_POR_TIPO["Last-Minute"]
        total           = ret_planner + ret_in_between + ret_last_minute

        # Tabla de resultados
        resultado_df = pd.DataFrame({
            "Tipo de cliente":       ["Planner",      "In-Between",    "Last-Minute"],
            "Meta (ventas)":         [meta_planner,   meta_in_between, meta_last_minute],
            "Retorno promedio ($)":  [RETORNO_POR_TIPO["Planner"], RETORNO_POR_TIPO["In-Between"], RETORNO_POR_TIPO["Last-Minute"]],
            "Retorno estimado ($)":  [ret_planner,    ret_in_between,  ret_last_minute],
        })
        st.dataframe(resultado_df.set_index("Tipo de cliente"), use_container_width=True)

        # Total destacado
        st.markdown(f"""
        <div style="background:{NAVY}; border-left:4px solid {GOLD};
                    padding:16px 24px; border-radius:8px; margin-top:12px;">
          <p style="color:{GOLD}; margin:0; font-size:.85rem; text-transform:uppercase; letter-spacing:.06em;">
            Retorno total estimado
          </p>
          <p style="color:white; margin:4px 0 0 0; font-size:2rem; font-weight:700;">
            ${total:,.2f}
          </p>
        </div>
        """, unsafe_allow_html=True)
