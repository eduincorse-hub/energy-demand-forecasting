import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Configuración de la página ──────────────────────────────
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Demand Forecasting Dashboard")
st.markdown("**Predicción de demanda eléctrica con SARIMA y LSTM**")
st.markdown("---")

# ── Cargar datos ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('C:\\Users\\HP\\Documents\\EDWIN\\AUTOCAD\\PHYTON\\Proyecto 1\\AEP_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').set_index('Datetime')
    df_daily = df.resample('D').mean()
    return df_daily

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
modelo_elegido = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ["SARIMA", "LSTM"]
)
dias_prediccion = st.sidebar.slider(
    "Días a predecir:",
    min_value=7,
    max_value=60,
    value=30
)

# ── Sección 1: Exploración ────────────────────────────────────
st.header("📊 Exploración de datos")

col1, col2, col3 = st.columns(3)
col1.metric("Total de días", f"{len(df):,}")
col2.metric("Demanda promedio", f"{df['AEP_MW'].mean():,.0f} MW")
col3.metric("Demanda máxima", f"{df['AEP_MW'].max():,.0f} MW")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df['AEP_MW'], color='steelblue', linewidth=0.8)
ax.set_title('Serie histórica de demanda eléctrica')
ax.set_xlabel('Fecha')
ax.set_ylabel('Demanda (MW)')
plt.tight_layout()
st.pyplot(fig)

# ── Sección 2: Predicción ─────────────────────────────────────
st.header(f"🔮 Predicción con {modelo_elegido}")

train = df[df.index < '2017-12-01']
test = df[df.index >= '2017-12-01']

if modelo_elegido == "SARIMA":
    with st.spinner("Entrenando modelo SARIMA..."):
        model = SARIMAX(train['AEP_MW'],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit(disp=False)
        pred = results.forecast(steps=dias_prediccion)
        pred.index = test.index[:dias_prediccion]
        real = test['AEP_MW'][:dias_prediccion]

else:  # LSTM
    with st.spinner("Cargando modelo LSTM..."):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['AEP_MW']])

        def create_sequences(data, window=60):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i-window:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        split = int(len(X) * 0.85)
        X_test = X[split:]
        y_test = y[split:]

        lstm_model = load_model('lstm_energy_model.keras')
        pred_scaled = lstm_model.predict(X_test[:dias_prediccion])
        pred = scaler.inverse_transform(pred_scaled).flatten()
        real = scaler.inverse_transform(
            y_test[:dias_prediccion].reshape(-1, 1)
        ).flatten()
        pred = pd.Series(pred, index=test.index[:dias_prediccion])
        real = pd.Series(real, index=test.index[:dias_prediccion])

# ── Métricas ──────────────────────────────────────────────────
mae = mean_absolute_error(real, pred)
rmse = np.sqrt(mean_squared_error(real, pred))
mape = np.mean(np.abs((real - pred) / real)) * 100

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f} MW")
col2.metric("RMSE", f"{rmse:.2f} MW")
col3.metric("MAPE", f"{mape:.2f}%")

# ── Gráfico predicción vs real ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(real.index, real.values,
         label='Real', color='steelblue', linewidth=2)
ax2.plot(pred.index, pred.values,
         label=f'Predicción {modelo_elegido}',
         color='coral', linewidth=2, linestyle='--')
ax2.set_title(f'{modelo_elegido}: Predicción vs Demanda Real')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Demanda (MW)')
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Desarrollado con Python · SARIMA · LSTM · Streamlit")