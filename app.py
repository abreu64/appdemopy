import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import io

st.set_page_config(
    page_title="Analytics de Vendas Atacadistas | Portfólio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

THEME_COLORS = {
    'dark': {
        'bg': '#0E1117',
        'bg_card': 'rgba(255, 255, 255, 0.05)',
        'bg_card_hover': 'rgba(255, 255, 255, 0.08)',
        'border': 'rgba(255, 255, 255, 0.1)',
        'text': '#FFFFFF',
        'text_secondary': '#CCCCCC',
        'accent': '#4A90E2',
        'success': '#00E676',
        'danger': '#FF5252',
        'warning': '#FFB74D',
        'grid': 'rgba(255,255,255,0.1)'
    },
    'light': {
        'bg': '#F8F9FA',
        'bg_card': 'rgba(255, 255, 255, 0.9)',
        'bg_card_hover': '#FFFFFF',
        'border': 'rgba(0, 0, 0, 0.1)',
        'text': '#1A1A2E',
        'text_secondary': '#666666',
        'accent': '#2563EB',
        'success': '#10B981',
        'danger': '#EF4444',
        'warning': '#F59E0B',
        'grid': 'rgba(0,0,0,0.1)'
    }
}

def get_colors():
    return THEME_COLORS[st.session_state.theme]

def get_css():
    c = get_colors()
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {c['bg']};
        color: {c['text']};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .metric-card {{
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        background: {c['bg_card_hover']};
        border-color: {c['accent']};
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: {c['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {c['text']};
        margin: 10px 0;
    }}
    .metric-delta {{
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .delta-up {{ color: {c['success']}; }}
    .delta-down {{ color: {c['danger']}; }}
    .delta-neutral {{ color: {c['warning']}; }}

    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {c['accent']};
        margin-bottom: 25px;
        border-left: 4px solid {c['accent']};
        padding-left: 15px;
    }}

    .insight-box {{
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {c['bg_card']};
        border-radius: 10px 10px 0px 0px;
        color: {c['text']} !important;
        font-weight: 600;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {c['accent']}33 !important;
        border-bottom: 3px solid {c['accent']} !important;
    }}

    [data-testid="stTable"] {{
        background-color: {c['bg_card']};
        border-radius: 10px;
        overflow: hidden;
    }}
    [data-testid="stTable"] td, [data-testid="stTable"] th {{
        color: {c['text']} !important;
        font-weight: 500;
        border-bottom: 1px solid {c['border']} !important;
    }}

    .stSelectbox label, .stMultiSelect label, .stDateInput label {{
        color: {c['text_secondary']} !important;
    }}

    .alert-success {{
        background: {c['success']}22;
        border: 1px solid {c['success']};
        color: {c['success']};
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    .alert-warning {{
        background: {c['warning']}22;
        border: 1px solid {c['warning']};
        color: {c['warning']};
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    .alert-danger {{
        background: {c['danger']}22;
        border: 1px solid {c['danger']};
        color: {c['danger']};
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}

    .comparison-card {{
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }}

    @media (max-width: 768px) {{
        .metric-value {{ font-size: 1.5rem; }}
        .section-title {{ font-size: 1.2rem; }}
    }}
</style>
"""

st.markdown(get_css(), unsafe_allow_html=True)

CATEGORIAS = ['Eletrônicos', 'Vestuário', 'Alimentos', 'Móveis', 'Cosméticos']
REGIOES = ['Norte', 'Sul', 'Leste', 'Oeste', 'Centro']
VENDEDORES = ['Ana Silva', 'Bruno Costa', 'Carla Santos', 'Daniel Oliveira', 'Eva Mendes']

@st.cache_data
def generate_data(seed_offset=0):
    np.random.seed(42 + seed_offset)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq='D')
    n = len(dates)
    
    trend = np.linspace(50000, 85000, n)
    day_seasonality = [1.0, 1.0, 1.1, 1.05, 1.3, 1.4, 0.8]
    weekly = np.array([day_seasonality[d.weekday()] for d in dates])
    month_seasonality = {1:0.9, 2:0.85, 3:1.0, 4:1.05, 5:1.1, 6:1.2, 7:1.0, 8:1.0, 9:1.1, 10:1.15, 11:1.3, 12:1.6}
    monthly = np.array([month_seasonality[d.month] for d in dates])
    noise = np.random.normal(0, 5000, n)
    sales = (trend * weekly * monthly) + noise
    
    df = pd.DataFrame({
        'Data': dates,
        'Vendas': sales,
        'Categoria': np.random.choice(CATEGORIAS, n),
        'Regiao': np.random.choice(REGIOES, n),
        'Vendedor': np.random.choice(VENDEDORES, n),
        'Quantidade': np.random.randint(10, 500, n),
        'TicketMedio': sales / np.random.randint(1, 50, n)
    })
    df.set_index('Data', inplace=True)
    return df

def forecast_sales(df, days=30):
    try:
        model = ARIMA(df['Vendas'], order=(5, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=days)
        forecast_df = pd.DataFrame({
            'Data': pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days, freq='D'),
            'Vendas_Previstas': forecast.values,
            'Intervalo_Inferior': forecast.values * 0.85,
            'Intervalo_Superior': forecast.values * 1.15
        })
        return forecast_df
    except Exception as e:
        return None

def get_insights(df):
    current_val = df['Vendas'].iloc[-1]
    mean_val = df['Vendas'].mean()
    std_val = df['Vendas'].std()
    
    last_30 = df['Vendas'].iloc[-30:].mean() if len(df) >= 30 else df['Vendas'].mean()
    prev_30 = df['Vendas'].iloc[-60:-30].mean() if len(df) >= 60 else df['Vendas'].iloc[:-30].mean()
    growth = ((last_30 - prev_30) / prev_30) * 100 if prev_30 != 0 else 0
    
    volatility_score = (std_val / mean_val) * 100
    
    insights = []
    alert_type = 'success'
    
    if growth > 5:
        insights.append(f"🚀 **Tendência de Alta:** Crescimento de {growth:.1f}% nos últimos 30 dias.")
        alert_type = 'success'
    elif growth < -5:
        insights.append(f"⚠️ **Alerta:** Queda de {abs(growth):.1f}% na média mensal recente.")
        alert_type = 'warning'
    else:
        insights.append("📊 **Estabilidade:** Vendas estáveis no curto prazo.")
        alert_type = 'success'
        
    if volatility_score > 20:
        insights.append(f"📉 **Alta Variância:** Volatilidade de {volatility_score:.1f}%.")
    else:
        insights.append("🎯 **Previsibilidade:** Baixa volatilidade, fluxo estável.")
    
    try:
        decomp = seasonal_decompose(df['Vendas'], model='additive', period=7)
        seasonal_range = decomp.seasonal.max() - decomp.seasonal.min()
        if seasonal_range > mean_val * 0.1:
            insights.append("🔄 **Sazonalidade Forte:** Variação cíclica semanal detectada.")
    except:
        pass
        
    return insights, mean_val, current_val, growth, volatility_score, alert_type

def calculate_comparison(df, period1_start, period1_end, period2_start, period2_end):
    p1 = df.loc[period1_start:period1_end]
    p2 = df.loc[period2_start:period2_end]
    
    if p1.empty or p2.empty:
        return None
    
    return {
        'period1_total': p1['Vendas'].sum(),
        'period2_total': p2['Vendas'].sum(),
        'period1_avg': p1['Vendas'].mean(),
        'period2_avg': p2['Vendas'].mean(),
        'growth': ((p2['Vendas'].sum() - p1['Vendas'].sum()) / p1['Vendas'].sum()) * 100 if p1['Vendas'].sum() != 0 else 0
    }

def create_chart_config(c):
    return {
        'plot_bgcolor': 'rgba(0,0,0,0)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,1)',
        'paper_bgcolor': 'rgba(0,0,0,0)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,1)',
        'font_color': c['text'],
        'xaxis': dict(
            showgrid=False, 
            color=c['text'], 
            tickformat="%d/%m/%Y",
            title_font=dict(color=c['text']),
            tickfont=dict(color=c['text'])
        ),
        'yaxis': dict(
            showgrid=True, 
            gridcolor=c['grid'], 
            color=c['text'],
            title_font=dict(color=c['text']),
            tickfont=dict(color=c['text'])
        )
    }

with st.sidebar:
    st.markdown("### 🎛️ Painel de Controle")
    
    if st.button(f"{'🌙' if st.session_state.theme == 'dark' else '☀️'} {'Modo Claro' if st.session_state.theme == 'dark' else 'Modo Escuro'}", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()
    
    st.markdown("---")
    
    if 'data_seed' not in st.session_state:
        st.session_state.data_seed = 0
    
    if st.button("🔄 Gerar Novos Dados", use_container_width=True):
        st.session_state.data_seed += 1
        st.cache_data.clear()
        st.rerun()
        
    st.markdown("### 📅 Filtros de Análise")
    start_date = st.date_input("De:", datetime(2024, 1, 1), format="DD/MM/YYYY")
    end_date = st.date_input("Até:", datetime(2025, 12, 31), format="DD/MM/YYYY")
    
    st.markdown("#### 🏷️ Categoria")
    selected_categories = st.multiselect("Selecione:", CATEGORIAS, default=CATEGORIAS)
    
    st.markdown("#### 📍 Região")
    selected_regions = st.multiselect("Selecione:", REGIOES, default=REGIOES)
    
    st.markdown("#### 👤 Vendedor")
    selected_vendors = st.multiselect("Selecione:", VENDEDORES, default=VENDEDORES)
    
    st.markdown("#### ⚠️ Alertas")
    st.session_state.alert_threshold = st.slider(
        "Notificar se vendas diárias abaixo de (R$)",
        min_value=0,
        max_value=int(100000),
        value=40000,
        step=5000
    )
    
    st.markdown("---")
    st.markdown("**Desenvolvido por:** TecSolutions")
    st.markdown("💡 *Dashboard de Portfólio*")

df_full = generate_data(st.session_state.data_seed)

mask = (
    (df_full.index >= pd.Timestamp(start_date)) & 
    (df_full.index <= pd.Timestamp(end_date)) &
    (df_full['Categoria'].isin(selected_categories)) &
    (df_full['Regiao'].isin(selected_regions)) &
    (df_full['Vendedor'].isin(selected_vendors))
)
df = df_full.loc[mask]

c = get_colors()
st.title("🛒 Análise de Vendas Atacadistas")
st.markdown("Visualize e analise tendências de vendas em escala atacadista com métricas preditivas.")

col_alert1, col_alert2 = st.columns([3, 1])
with col_alert1:
    alert_days = df[df['Vendas'] < st.session_state.alert_threshold]
    if len(alert_days) > 0:
        st.markdown(f"""
        <div class="alert-danger">
            ⚠️ <b>Alerta:</b> {len(alert_days)} dia(s) com vendas abaixo de R$ {st.session_state.alert_threshold:,.0f}
        </div>
        """, unsafe_allow_html=True)

with col_alert2:
    with st.expander("📋 Detalhes"):
        if len(alert_days) > 0:
            st.dataframe(alert_days[['Vendas']].head(10))

insight_list, mean_val, curr_val, growth, vol_score, alert_type = get_insights(df)

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
    vol_class = "delta-down" if vol_score > 20 else "delta-up"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volatilidade (CV)</div>
            <div class="metric-value">{vol_score:.1f}%</div>
            <div class="metric-delta {vol_class}">{"Alta" if vol_score > 20 else "Estável"}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    last_val = df['Vendas'].iloc[-1] if len(df) > 0 else 0
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Último Registro</div>
            <div class="metric-value">R$ {last_val:,.0f}</div>
            <div class="metric-delta">Valor de Fechamento</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Série Temporal", 
    "🔮 Previsão (Forecasting)",
    "🧩 Decomposição", 
    "📊 Análises",
    "💾 Exportar"
])

with tab1:
    st.markdown("<div class='section-title'>Análise de Tendência de Vendas</div>", unsafe_allow_html=True)
    
    fig_main = px.line(df, x=df.index, y='Vendas', 
                       color_discrete_sequence=[c['accent']],
                       labels={'Vendas': 'Receita (R$)', 'index': 'Data'})
    
    fig_main.update_layout(**create_chart_config(c))
    fig_main.update_traces(hovertemplate="Data: %{x|%d/%m/%Y}<br>Receita: R$ %{y:,.2f}")
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    with st.expander("📝 Insights Automatizados"):
        for ins in insight_list:
            st.write(ins)

with tab2:
    st.markdown("<div class='section-title'>Previsão de Vendas (ARIMA - 30 dias)</div>", unsafe_allow_html=True)
    
    forecast_days = st.slider("Dias para prever:", 7, 90, 30)
    
    with st.spinner("Gerando previsão..."):
        forecast_df = forecast_sales(df, forecast_days)
    
    if forecast_df is not None:
        col_forecast1, col_forecast2 = st.columns([2, 1])
        
        with col_forecast1:
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=df.index[-30:], y=df['Vendas'].iloc[-30:],
                name='Histórico', line=dict(color=c['accent'])
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Data'], y=forecast_df['Vendas_Previstas'],
                name='Previsão', line=dict(color=c['success'], dash='dash')
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Data'].tolist() + forecast_df['Data'].tolist()[::-1],
                y=forecast_df['Intervalo_Superior'].tolist() + forecast_df['Intervalo_Inferior'].tolist()[::-1],
                fill='toself', fillcolor='rgba(0, 230, 118, 0.2)' if st.session_state.theme == 'dark' else 'rgba(16, 185, 129, 0.2)',
                line=dict(width=0), name='Intervalo de Confiança'
            ))
            fig_forecast.update_layout(**create_chart_config(c))
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with col_forecast2:
            st.markdown("### 📋 Resumo da Previsão")
            forecast_mean = forecast_df['Vendas_Previstas'].mean()
            forecast_total = forecast_df['Vendas_Previstas'].sum()
            last_actual = df['Vendas'].iloc[-30:].mean()
            diff_pct = ((forecast_mean - last_actual) / last_actual) * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Média Prevista</div>
                <div class="metric-value">R$ {forecast_mean:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Forecast</div>
                <div class="metric-value">R$ {forecast_total/1e6:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">vs Histórico</div>
                <div class="metric-value {'delta-up' if diff_pct >= 0 else 'delta-down'}">{'+' if diff_pct >= 0 else ''}{diff_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Selecione um período maior para gerar a previsão.")

with tab3:
    st.markdown("<div class='section-title'>Decomposição de Séries Temporais</div>", unsafe_allow_html=True)
    try:
        res = seasonal_decompose(df['Vendas'], model='additive', period=7)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df.index, y=res.trend, name="Tendência", line=dict(color=c['success'], width=3)))
        fig_trend.update_layout(title="Componente de Tendência", **create_chart_config(c))
        st.plotly_chart(fig_trend, use_container_width=True)

        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(x=df.index[:14], y=res.seasonal[:14], name="Sazonalidade", marker_color=c['accent']))
        fig_season.update_layout(title="Efeito Sazonal Semanal (14 dias)", **create_chart_config(c))
        st.plotly_chart(fig_season, use_container_width=True)
        
    except:
        st.warning("Selecione um intervalo maior que 7 dias para visualizar a decomposição.")

with tab4:
    st.markdown("<div class='section-title'>📊 Análises Detalhadas</div>", unsafe_allow_html=True)
    
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["Por Categoria", "Por Região", "Por Vendedor", "Comparar Períodos"])
    
    with sub_tab1:
        cat_sales = df.groupby('Categoria')['Vendas'].agg(['sum', 'mean', 'count']).reset_index()
        cat_sales.columns = ['Categoria', 'Total', 'Média', 'Qtd Dias']
        cat_sales = cat_sales.sort_values('Total', ascending=False)
        
        fig_cat = px.pie(cat_sales, values='Total', names='Categoria', 
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_cat.update_layout(**{k: v for k, v in create_chart_config(c).items() if k in ['paper_bgcolor', 'font_color']})
        st.plotly_chart(fig_cat, use_container_width=True)
        st.dataframe(cat_sales, use_container_width=True)
    
    with sub_tab2:
        reg_sales = df.groupby('Regiao')['Vendas'].agg(['sum', 'mean']).reset_index()
        reg_sales.columns = ['Região', 'Total', 'Média']
        reg_sales = reg_sales.sort_values('Total', ascending=False)
        
        fig_reg = px.bar(reg_sales, x='Região', y='Total', color='Região',
                         color_discrete_sequence=[c['accent']])
        fig_reg.update_layout(**create_chart_config(c))
        st.plotly_chart(fig_reg, use_container_width=True)
        st.dataframe(reg_sales, use_container_width=True)
    
    with sub_tab3:
        vend_sales = df.groupby('Vendedor')['Vendas'].agg(['sum', 'mean']).reset_index()
        vend_sales.columns = ['Vendedor', 'Total', 'Média']
        vend_sales = vend_sales.sort_values('Total', ascending=False)
        
        fig_vend = px.bar(vend_sales, x='Vendedor', y='Total', color='Vendedor',
                         color_discrete_sequence=[c['success']])
        fig_vend.update_layout(**create_chart_config(c))
        st.plotly_chart(fig_vend, use_container_width=True)
        st.dataframe(vend_sales, use_container_width=True)
    
    with sub_tab4:
        st.markdown("#### Selecione dois períodos para comparar:")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("**Período 1**")
            p1_start = st.date_input("De (P1):", datetime(2024, 1, 1), key="p1_start", format="DD/MM/YYYY")
            p1_end = st.date_input("Até (P1):", datetime(2024, 6, 30), key="p1_end", format="DD/MM/YYYY")
        
        with col_p2:
            st.markdown("**Período 2**")
            p2_start = st.date_input("De (P2):", datetime(2024, 7, 1), key="p2_start", format="DD/MM/YYYY")
            p2_end = st.date_input("Até (P2):", datetime(2024, 12, 31), key="p2_end", format="DD/MM/YYYY")
        
        if st.button("🔍 Comparar"):
            comp = calculate_comparison(df, p1_start, p1_end, p2_start, p2_end)
            if comp:
                col_c1, col_c2, col_c3 = st.columns(3)
                
                with col_c1:
                    st.markdown(f"""
                    <div class="comparison-card">
                        <div class="metric-label">Total P1</div>
                        <div class="metric-value">R$ {comp['period1_total']/1e6:.1f}M</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c2:
                    st.markdown(f"""
                    <div class="comparison-card">
                        <div class="metric-label">Total P2</div>
                        <div class="metric-value">R$ {comp['period2_total']/1e6:.1f}M</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c3:
                    color = c['success'] if comp['growth'] >= 0 else c['danger']
                    st.markdown(f"""
                    <div class="comparison-card">
                        <div class="metric-label">Crescimento</div>
                        <div class="metric-value" style="color: {color}">{'+' if comp['growth'] >= 0 else ''}{comp['growth']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='section-title'>💾 Exportar Dados</div>", unsafe_allow_html=True)
    
    export_format = st.radio("Formato:", ["CSV", "Excel (XLSX)"], horizontal=True)
    
    if export_format == "CSV":
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"vendas_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Vendas')
            summary = pd.DataFrame({
                'Métrica': ['Total', 'Média', 'Máximo', 'Mínimo', 'Volatilidade'],
                'Valor': [df['Vendas'].sum(), df['Vendas'].mean(), df['Vendas'].max(), 
                          df['Vendas'].min(), vol_score]
            })
            summary.to_excel(writer, sheet_name='Resumo', index=False)
        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name=f"vendas_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.markdown("### 📋 Prévia dos Dados")
    page_size = 10
    if 'page' not in st.session_state:
        st.session_state.page = 0
    
    total_pages = max(1, len(df) // page_size)
    start_idx = st.session_state.page * page_size
    end_idx = start_idx + page_size
    
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Anterior") and st.session_state.page > 0:
            st.session_state.page -= 1
            st.rerun()
    with col_page:
        st.markdown(f"<div style='text-align: center; color: {c['text_secondary']}'>Página {st.session_state.page + 1} de {total_pages + 1}</div>", unsafe_allow_html=True)
    with col_next:
        if st.button("Próxima ➡️") and st.session_state.page < total_pages:
            st.session_state.page += 1
            st.rerun()

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {c['text_secondary']}; font-size: 0.8rem;">
    Dashboard utiliza ARIMA para previsão, decomposição aditiva para análise sazonal e filtros multidimensionais.
    Dados sintéticos simulando comportamento atacadista com sazonalidades de abastecimento.
</div>
""", unsafe_allow_html=True)
