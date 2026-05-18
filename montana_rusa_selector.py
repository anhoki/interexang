# montana_rusa_selector.py
# Guarda este archivo y ejecuta: streamlit run montana_rusa_selector.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Selector de Montaña Rusa", layout="wide")

# ============================================================
# 1. FUNCIONES DE AJUSTE EXPONENCIAL
# ============================================================

def exponencial_crecimiento(t, A, k, B):
    """Modelo: y = A * exp(k * t) + B"""
    return A * np.exp(k * t) + B

def ema(series, alpha=0.3):
    """Suavizado exponencial EMA"""
    result = np.zeros_like(series, dtype=float)
    if len(series) == 0:
        return result
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result

def ajustar_exponencial(y_vals):
    """Ajusta exponencial y devuelve k, A, R², longitud_arco, pendiente_final, y_pred"""
    x_vals = np.arange(len(y_vals))
    
    y_pos = np.maximum(y_vals, 0.01)
    
    y_max = y_pos.max()
    y_min = y_pos.min()
    A0 = max((y_max - y_min) * 0.5, 0.1)
    k0 = 0.1
    B0 = max(y_min, 0)
    
    try:
        params, _ = curve_fit(exponencial_crecimiento, x_vals, y_pos, 
                              p0=[A0, k0, B0], maxfev=5000)
        A, k, B = params
        
        y_pred = exponencial_crecimiento(x_vals, A, k, B)
        ss_res = np.sum((y_pos - y_pred) ** 2)
        ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2 = max(0, min(1, r2))
        
        horizonte = len(y_vals) - 1
        pasos = 100
        dt = horizonte / pasos if horizonte > 0 else 1
        longitud = 0
        for i in range(pasos):
            t = i * dt
            t_next = (i + 1) * dt
            ft = exponencial_crecimiento(t, A, k, B)
            ft_next = exponencial_crecimiento(t_next, A, k, B)
            dy = ft_next - ft
            dx = dt
            longitud += np.sqrt(dx*dx + dy*dy)
        
        pendiente_final = A * k * np.exp(k * (len(y_vals)-1)) if k != 0 else 0
        
        return k, A, B, r2, longitud, pendiente_final, y_pred
    
    except Exception as e:
        return 0, 0, 0, 0, 0, 0, np.zeros_like(y_vals)

# ============================================================
# 2. CARGA DE DATOS
# ============================================================

st.title("🎢 Selector de Montaña Rusa")
st.markdown("¿Cuál de estos candidatos tiene la **mejor curva** para subirse ahora?")

opcion = st.radio("Origen de datos", ["Usar datos de ejemplo", "Subir CSV con mis datos"])

df = None

if opcion == "Subir CSV con mis datos":
    archivo = st.file_uploader("Sube un CSV con columnas: Ticker, Vela, Votos_Neto, Precio, Volumen", type="csv")
    if archivo:
        df = pd.read_csv(archivo)
        st.success(f"✅ Datos cargados: {df['Ticker'].nunique()} activos, {len(df)} filas")
    else:
        st.warning("Esperando archivo... usando datos de ejemplo")
        opcion = "Usar datos de ejemplo"

if opcion == "Usar datos de ejemplo":
    datos = {
        "Ticker": ["AAPL"]*20 + ["MSFT"]*20 + ["TSLA"]*20 + ["AMZN"]*20 + ["GOOGL"]*20,
        "Vela": list(range(20))*5,
        "Votos_Neto": [
            1,1,3,3,5,5,3,3,1,1,3,3,5,5,3,4,4,5,5,4,
            -1,-1,-3,-3,-5,-5,-3,-3,-1,-1,-3,-3,-5,-5,-3,-4,-4,-5,-5,-4,
            3,3,5,5,5,5,3,3,5,5,5,5,3,3,5,5,5,5,5,5,
            2,2,3,3,4,4,3,3,2,2,3,3,4,4,3,3,4,4,4,4,
            1,3,1,3,5,3,1,3,5,3,1,3,5,3,1,3,5,3,1,3,
        ],
        "Precio": [
            179.80,179.95,180.20,180.50,181.00,181.30,181.00,180.80,180.30,180.10,
            180.50,180.70,181.20,181.60,181.30,181.50,181.80,182.00,182.30,182.10,
            420.00,419.80,419.00,418.50,417.50,417.00,417.50,418.00,418.50,419.00,
            418.50,418.00,417.00,416.50,417.00,416.80,416.50,416.00,415.50,416.00,
            250.00,251.00,253.00,255.00,257.00,259.00,258.00,257.00,259.00,261.00,
            263.00,265.00,263.00,261.00,263.00,265.00,267.00,269.00,271.00,273.00,
            178.00,178.20,178.80,179.00,179.50,179.80,179.50,179.30,179.00,178.80,
            179.00,179.20,179.60,179.90,179.60,179.40,179.70,180.00,180.30,180.50,
            165.00,165.50,165.20,165.80,166.50,166.20,165.80,166.00,166.80,166.50,
            166.00,166.30,167.00,166.80,166.30,166.60,167.20,167.00,166.60,166.90,
        ],
        "Volumen": [
            2.5,2.8,3.1,3.4,3.8,3.9,3.6,3.4,3.0,2.8,3.0,3.1,3.5,3.7,3.5,3.6,3.8,4.0,4.1,3.9,
            2.2,2.3,2.5,2.6,2.9,3.0,2.8,2.7,2.5,2.4,2.6,2.7,3.0,3.1,2.9,2.8,2.9,3.1,3.2,3.0,
            5.0,5.2,5.8,6.1,6.5,6.8,6.2,5.9,6.3,6.6,6.9,7.1,6.5,6.1,6.5,6.8,7.0,7.3,7.5,7.8,
            3.0,3.1,3.3,3.4,3.6,3.7,3.5,3.4,3.2,3.1,3.3,3.4,3.6,3.7,3.5,3.4,3.6,3.7,3.8,3.9,
            2.0,2.2,2.1,2.3,2.6,2.4,2.2,2.3,2.6,2.4,2.2,2.3,2.6,2.4,2.2,2.3,2.6,2.4,2.2,2.3,
        ]
    }
    df = pd.DataFrame(datos)

if df is None:
    st.stop()

# ============================================================
# 3. VISUALIZACIÓN DE DATOS CRUDOS (NUEVO)
# ============================================================

st.subheader("📋 Datos cargados")

tab1, tab2, tab3 = st.tabs(["📊 Vista general", "📈 Por activo", "🔢 Estadísticas"])

with tab1:
    st.dataframe(df, use_container_width=True, height=400)
    st.caption(f"Total: {len(df)} filas | {df['Ticker'].nunique()} activos | {df['Vela'].max()+1} velas por activo")

with tab2:
    activo_seleccionado = st.selectbox("Selecciona un activo", df['Ticker'].unique())
    df_filtrado = df[df['Ticker'] == activo_seleccionado]
    st.dataframe(df_filtrado, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df_filtrado['Vela'], df_filtrado['Votos_Neto'], 'o-', color='steelblue', linewidth=2, markersize=6, label='Votos Netos')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Vela")
    ax.set_ylabel("Votos Netos")
    ax.set_title(f"{activo_seleccionado} - Serie original")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Activos", df['Ticker'].nunique())
    with col2:
        st.metric("Velas por activo", df.groupby('Ticker').size().iloc[0] if len(df) > 0 else 0)
    with col3:
        st.metric("Rango Votos Netos", f"{df['Votos_Neto'].min()} a {df['Votos_Neto'].max()}")
    with col4:
        st.metric("Total filas", len(df))
    
    stats_df = df.groupby('Ticker').agg({
        'Votos_Neto': ['mean', 'std', 'min', 'max'],
        'Precio': ['first', 'last', 'min', 'max'],
        'Volumen': ['mean', 'max']
    }).round(2)
    st.dataframe(stats_df, use_container_width=True)

# ============================================================
# 4. PROCESAMIENTO Y CÁLCULOS
# ============================================================

st.subheader("📊 Procesando datos...")

df['EMA5'] = df.groupby('Ticker')['Votos_Neto'].transform(lambda x: ema(x.values))

resultados = []
tickers = df['Ticker'].unique()

with st.spinner("Calculando curvas..."):
    for ticker in tickers:
        sub = df[df['Ticker'] == ticker].copy()
        y_vals = sub['EMA5'].values
        
        k, A, B, r2, longitud, pend_final, y_pred = ajustar_exponencial(y_vals)
        
        resultados.append({
            "Ticker": ticker,
            "k (pendiente)": k,
            "A (amplitud)": A,
            "R²": r2,
            "Longitud_arco": longitud,
            "Pendiente_final": pend_final,
            "y_pred": y_pred,
            "y_real": y_vals,
            "velas": sub['Vela'].values
        })

df_ranking = pd.DataFrame(resultados)

max_long = df_ranking['Longitud_arco'].max() if df_ranking['Longitud_arco'].max() > 0 else 1
max_pend = df_ranking['Pendiente_final'].max() if df_ranking['Pendiente_final'].max() > 0 else 1
max_k = df_ranking['k (pendiente)'].max() if df_ranking['k (pendiente)'].max() > 0 else 1

df_ranking['Puntaje'] = (
    (df_ranking['Longitud_arco'] / max_long) * 0.5 +
    (df_ranking['Pendiente_final'] / max_pend) * 0.3 +
    (df_ranking['k (pendiente)'] / max_k) * 0.2
)
df_ranking = df_ranking.sort_values('Puntaje', ascending=False)
df_ranking['Puntaje'] = df_ranking['Puntaje'] / df_ranking['Puntaje'].max()

# ============================================================
# 5. DASHBOARD DE RESULTADOS
# ============================================================

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🏆 Ranking")
    
    display_df = df_ranking[['Ticker', 'k (pendiente)', 'R²', 'Longitud_arco', 'Puntaje']].copy()
    display_df['k (pendiente)'] = display_df['k (pendiente)'].map(lambda x: f"{x:.3f}")
    display_df['R²'] = display_df['R²'].map(lambda x: f"{x:.2f}")
    display_df['Longitud_arco'] = display_df['Longitud_arco'].map(lambda x: f"{x:.2f}")
    display_df['Puntaje'] = display_df['Puntaje'].map(lambda x: f"{x:.0%}")
    
    st.dataframe(display_df, use_container_width=True)
    
    mejor = df_ranking.iloc[0]
    st.success(f"🎢 **MEJOR: {mejor['Ticker']}**")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Puntaje", f"{mejor['Puntaje']:.0%}")
    col_b.metric("Longitud", f"{mejor['Longitud_arco']:.2f}")
    col_c.metric("Pendiente (k)", f"{mejor['k (pendiente)']:.3f}")

with col2:
    st.subheader("📈 Curvas de cada candidato")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, row in df_ranking.iterrows():
        ticker = row['Ticker']
        sub = df[df['Ticker'] == ticker]
        color = colores[i % len(colores)]
        ax.plot(sub['Vela'], sub['EMA5'], 'o-', color=color, label=f"{ticker} (real)", alpha=0.7, markersize=4)
        ax.plot(sub['Vela'], row['y_pred'], '--', color=color, label=f"{ticker} (exp)", alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel("Vela (tiempo)")
    ax.set_ylabel("Votos Netos Suavizados (EMA5)")
    ax.set_title("Comparativa de curvas reales vs exponenciales")
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ============================================================
# 6. DETALLE POR ACTIVO
# ============================================================

st.subheader("🔍 Detalle individual")

for i, row in df_ranking.iterrows():
    ticker = row['Ticker']
    sub = df[df['Ticker'] == ticker]
    
    with st.expander(f"{ticker} | k={row['k (pendiente)']:.3f} | R²={row['R²']:.2f} | Longitud={row['Longitud_arco']:.2f}"):
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(sub['Vela'], sub['EMA5'], 'o-', color='blue', label='Real (EMA5)', linewidth=2, markersize=4)
        ax2.plot(sub['Vela'], row['y_pred'], '--', color='red', label='Exponencial ajustada', linewidth=2)
        ax2.set_xlabel("Vela")
        ax2.set_ylabel("Votos Netos Suavizados")
        ax2.set_title(f"{ticker}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Pendiente (k)", f"{row['k (pendiente)']:.3f}")
        col_b.metric("Amplitud (A)", f"{row['A (amplitud)']:.2f}")
        col_c.metric("R²", f"{row['R²']:.2f}")
        col_d.metric("Pendiente final", f"{row['Pendiente_final']:.2f}")

# ============================================================
# 7. DECISIÓN FINAL
# ============================================================

st.divider()
st.subheader("🎯 Decisión final")

if st.button("✅ Seleccionar este candidato"):
    st.success(f"""
    **Orden sugerida:**
    - Activo: {mejor['Ticker']}
    - Tamaño base: 1.0
    - Factor montaña rusa: {mejor['Puntaje']:.0%}
    - Tamaño final: {mejor['Puntaje']:.2f}x
    - Confianza ajuste: {mejor['R²']:.0%}
    """)
    
    st.code(f"""
    # Código para integrar en tu estrategia
    mejor_candidato = "{mejor['Ticker']}"
    factor_tamanio = {mejor['Puntaje']:.2f}
    confianza_ajuste = {mejor['R²']:.2f}
    """, language="python")

st.caption("""
**Interpretación:**
- Mayor longitud de arco = más "recorrido" potencial
- Mayor pendiente final = más fuerza al final del período
- Mayor R² = la curva exponencial se ajusta bien (señal más limpia)
""")
