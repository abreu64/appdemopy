import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Analytics de Vendas Atacadistas | Portfólio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO (DARK MODE PREMIUM) ---
st.markdown("""
<style>
    /* Importando Fonte Modernistas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fundo e Container Principal */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF; /* Aumentando contraste geral */
    }

    /* Escondendo Menu e Footer do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Cartões de KPI Estilizados */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
        border-color: #4A90E2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #CCCCCC; /* Cinza mais claro para contraste */
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 10px 0;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
    }
    .delta-up { color: #00E676; } /* Verde mais vibrante */
    .delta-down { color: #FF5252; }

    /* Estilo para Títulos de Seção */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4A90E2;
        margin-bottom: 25px;
        border-left: 4px solid #4A90E2;
        padding-left: 15px;
    }

    /* Bloco de Insights */
    .insight-box {
        background: rgba(74, 144, 226, 0.1);
        border: 1px solid rgba(74, 144, 226, 0.2);
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }

    /* ESTILIZAÇÃO DOS TABS (Contraste das Legendas) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        color: #FFFFFF !important; /* Texto Branco */
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(74, 144, 226, 0.2) !important;
        border-bottom: 3px solid #4A90E2 !important;
    }

    /* ESTILIZAÇÃO DE TABELAS (Contraste do Resumo Estatístico) */
    [data-testid="stTable"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #FFFFFF !important;
        font-weight: 500;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- MOTOR DE DADOS (DATA ENGINE) ---
@st.cache_data
def generate_data(seed_offset=0):
    np.random.seed(42 + seed_offset)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq='D')
    n = len(dates)
    
    # Componentes: Tendência + Sazonalidade + Ruído
    trend = np.linspace(50000, 85000, n)  # Crescimento orgânico
    
    # Sazonalidade Semanal (Maior volume às sextas e sábados no atacado)
    day_seasonality = [1.0, 1.0, 1.1, 1.05, 1.3, 1.4, 0.8] # Seg-Dom
    weekly = np.array([day_seasonality[d.weekday()] for d in dates])
    
    # Sazonalidade Mensal (Picos em Dezembro e Junho)
    month_seasonality = {1:0.9, 2:0.85, 3:1.0, 4:1.05, 5:1.1, 6:1.2, 7:1.0, 8:1.0, 9:1.1, 10:1.15, 11:1.3, 12:1.6}
    monthly = np.array([month_seasonality[d.month] for d in dates])
    
    # Ruído Aleatório
    noise = np.random.normal(0, 5000, n)
    
    # Cálculo Final das Vendas (R$)
    sales = (trend * weekly * monthly) + noise
    
    df = pd.DataFrame({'Data': dates, 'Vendas': sales})
    df.set_index('Data', inplace=True)
    return df

# --- CÁLCULOS ESTATÍSTICOS ---
def get_insights(df):
    current_val = df['Vendas'].iloc[-1]
    mean_val = df['Vendas'].mean()
    std_val = df['Vendas'].std()
    
    # Tendência (Crescimento Últimos 30 dias)
    last_30 = df['Vendas'].iloc[-30:].mean()
    prev_30 = df['Vendas'].iloc[-60:-30].mean()
    growth = ((last_30 - prev_30) / prev_30) * 100
    
    # Análise de Volatilidade
    volatility_score = (std_val / mean_val) * 100
    
    insights = []
    
    # Insight 1: Tendência
    if growth > 5:
        insights.append(f"🚀 **Tendência de Alta:** Detectamos um crescimento de {growth:.1f}% nos últimos 30 dias em comparação ao mês anterior.")
    elif growth < -5:
        insights.append(f"⚠️ **Alerta:** Queda de {abs(growth):.1f}% na média mensal recente.")
    else:
        insights.append("📊 **Estabilidade:** As vendas mantêm-se estáveis no curto prazo.")
        
    # Insight 2: Volatilidade
    if volatility_score > 20:
        insights.append(f"📉 **Alta Variância:** Identificada alta volatilidade ({volatility_score:.1f}%) nos pedidos, sugerindo demanda imprevisível.")
    else:
        insights.append("🎯 **Previsibilidade:** Fluxo de caixa com baixa volatilidade, facilitando o planejamento de estoque.")

    # Insight 3: Sazonalidade (Statsmodels)
    try:
        decomp = seasonal_decompose(df['Vendas'], model='additive', period=7)
        seasonal_range = decomp.seasonal.max() - decomp.seasonal.min()
        if seasonal_range > mean_val * 0.1:
            insights.append("🔄 **Sazonalidade Forte:** Confirmada variação cíclica semanal significativa no volume de atacado.")
    except:
        pass
        
    return insights, mean_val, current_val, growth, volatility_score

# --- INTERFACE - BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3094/3094896.png", width=100)
    st.title("Painel de Controle")
    st.markdown("---")
    
    # Botão de Reset
    if 'data_seed' not in st.session_state:
        st.session_state.data_seed = 0
    
    if st.button("🔄 Gerar Novos Dados", use_container_width=True):
        st.session_state.data_seed += 1
        st.cache_data.clear()
        st.rerun()
        
    st.markdown("### Filtros de Análise")
    start_date = st.date_input("De:", datetime(2024, 1, 1), format="DD/MM/YYYY")
    end_date = st.date_input("Até:", datetime(2025, 12, 31), format="DD/MM/YYYY")
    
    st.markdown("---")
    st.markdown("**Desenvolvido por:** TecSolutions")
    st.markdown("💡 *Dashboard de Portfólio*")

# --- CARREGAMENTO E FILTRAGEM ---
df_full = generate_data(st.session_state.data_seed)
mask = (df_full.index >= pd.Timestamp(start_date)) & (df_full.index <= pd.Timestamp(end_date))
df = df_full.loc[mask]

# --- DASHBOARD PRINCIPAL ---
st.title("🛒 Análise de Vendas Atacadistas")
st.markdown("Visualize e analise tendências de vendas em escala atacadista com métricas preditivas.")

# KPIs no Topo
insight_list, mean_val, curr_val, growth, vol_score = get_insights(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vendas Totais</div>
            <div class="metric-value">R$ {df['Vendas'].sum()/1e6:.1f}M</div>
            <div class="metric-delta">Período Selecionado</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    color_class = "delta-up" if growth >= 0 else "delta-down"
    symbol = "▲" if growth >= 0 else "▼"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Média Diária</div>
            <div class="metric-value">R$ {mean_val:,.0f}</div>
            <div class="metric-delta {color_class}">{symbol} {abs(growth):.1f}% vs Últ. Mês</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volatilidade (CV)</div>
            <div class="metric-value">{vol_score:.1f}%</div>
            <div class="metric-delta">Estabilidade de Fluxo</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    last_val = df['Vendas'].iloc[-1]
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Último Registro</div>
            <div class="metric-value">R$ {last_val:,.0f}</div>
            <div class="metric-delta">Valor de Fechamento</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Gráficos Principais
tab1, tab2, tab3 = st.tabs(["📈 Série Temporal", "🧩 Decomposição Sazonal", "📊 Distribuição"])

with tab1:
    st.markdown("<div class='section-title'>Análise de Tendência de Vendas</div>", unsafe_allow_html=True)
    fig_main = px.line(df, x=df.index, y='Vendas', 
                       color_discrete_sequence=['#4A90E2'],
                       labels={'Vendas': 'Receita (R$)', 'index': 'Data'})
    
    # Formatando Data para DD/MM/YYYY e cores de alto contraste
    fig_main.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False, 
            color='#FFFFFF', # Branco para contraste máximo
            tickformat="%d/%m/%Y",
            title_font=dict(color='#FFFFFF'),
            tickfont=dict(color='#FFFFFF')
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            color='#FFFFFF', # Branco para contraste máximo
            title_font=dict(color='#FFFFFF'),
            tickfont=dict(color='#FFFFFF')
        ),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=20, b=0)
    )
    fig_main.update_traces(hovertemplate="Data: %{x|%d/%m/%Y}<br>Receita: R$ %{y:,.2f}")
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    with st.expander("📝 Insights Automatizados"):
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        for ins in insight_list:
            st.write(ins)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-title'>Decomposição de Séries Temporais</div>", unsafe_allow_html=True)
    try:
        # Requer frequência diária e sem gaps para decomposição
        res = seasonal_decompose(df['Vendas'], model='additive', period=7)
        
        # Plotly Subplots Manual (Trend & Seasonality)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df.index, y=res.trend, name="Tendência", line=dict(color='#00E676', width=3)))
        fig_trend.update_layout(
            title="Componente de Tendência (Trend)", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#FFFFFF',
            xaxis=dict(tickformat="%d/%m/%Y", color='#FFFFFF', tickfont=dict(color='#FFFFFF')),
            yaxis=dict(color='#FFFFFF', tickfont=dict(color='#FFFFFF'), gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(x=df.index[:14], y=res.seasonal[:14], name="Sazonalidade", marker_color='#E91E63'))
        fig_season.update_layout(
            title="Efeito Sazonal Semanal (Amostra 14 dias)", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#FFFFFF',
            xaxis=dict(tickformat="%d/%m/%Y", color='#FFFFFF', tickfont=dict(color='#FFFFFF')),
            yaxis=dict(color='#FFFFFF', tickfont=dict(color='#FFFFFF'), gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
    except:
        st.warning("Selecione um intervalo maior que 7 dias para visualizar a decomposição.")

with tab3:
    st.markdown("<div class='section-title'>Distribuição de Dados</div>", unsafe_allow_html=True)
    col_hist, col_stats = st.columns([2, 1])
    
    with col_hist:
        fig_hist = px.histogram(df, x='Vendas', nbins=50, 
                               color_discrete_sequence=['#4A90E2'],
                               marginal="box", # Adiciona box plot no topo
                               title="Histograma de Vendas")
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#FFFFFF',
            xaxis=dict(color='#FFFFFF', tickfont=dict(color='#FFFFFF'), title_font=dict(color='#FFFFFF')),
            yaxis=dict(color='#FFFFFF', tickfont=dict(color='#FFFFFF'), title_font=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_stats:
        st.markdown("### Resumo Estatístico")
        stats = df['Vendas'].describe()
        st.table(stats)

# Rodapé Educativo
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #606060; font-size: 0.8rem;">
    Este dashboard utiliza algoritmos de decomposição aditiva e cálculos de média móvel exponencial para extrair insights. 
    Os dados são gerados sinteticamente para simular o comportamento de uma rede varejista com sazonalidades de abastecimento.
</div>
""", unsafe_allow_html=True)
