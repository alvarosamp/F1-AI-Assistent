"""
F1 AI Race Insights — Dashboard Streamlit.

COMO RODAR:
    pip install streamlit plotly
    streamlit run dashboard.py

PÁGINAS:
    1. Simulação de Corrida — escolha GP e grid, rode Monte Carlo, veja probabilidades
    2. Calibração — reliability diagrams e Brier scores do modelo
    3. Análise de Modelo — feature importance, RMSE por GP, por piloto
    4. Sobre o Projeto — arquitetura, métricas, decisões técnicas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "simulation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="F1 AI Race Insights",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: 800;
        color: #E10600;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1em;
        color: #666;
        margin-top: -10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85em;
        opacity: 0.7;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

KNOWN_DRIVERS = {
    "VER": "Red Bull Racing", "PER": "Red Bull Racing",
    "LEC": "Ferrari", "SAI": "Ferrari",
    "NOR": "McLaren", "PIA": "McLaren",
    "HAM": "Mercedes", "RUS": "Mercedes",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "GAS": "Alpine", "OCO": "Alpine",
    "TSU": "RB", "RIC": "RB", "LAW": "RB",
    "BOT": "Kick Sauber", "ZHO": "Kick Sauber",
    "MAG": "Haas F1 Team", "HUL": "Haas F1 Team",
    "ALB": "Williams", "SAR": "Williams", "COL": "Williams",
    "BEA": "Ferrari",
}

KNOWN_GPS = [
    "Australian Grand Prix", "Bahrain Grand Prix", "Saudi Arabian Grand Prix",
    "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
    "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
    "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]

GP_LAPS = {
    "Bahrain Grand Prix": 57, "Saudi Arabian Grand Prix": 50,
    "Australian Grand Prix": 58, "Japanese Grand Prix": 53,
    "Chinese Grand Prix": 56, "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63, "Monaco Grand Prix": 78,
    "Canadian Grand Prix": 70, "Spanish Grand Prix": 66,
    "Austrian Grand Prix": 71, "British Grand Prix": 52,
    "Hungarian Grand Prix": 70, "Belgian Grand Prix": 44,
    "Dutch Grand Prix": 72, "Italian Grand Prix": 53,
    "Azerbaijan Grand Prix": 51, "Singapore Grand Prix": 62,
    "United States Grand Prix": 56, "Mexico City Grand Prix": 71,
    "São Paulo Grand Prix": 69, "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57, "Abu Dhabi Grand Prix": 58,
}


@st.cache_resource
def load_simulator():
    # O simulador v2 está em `src/simulation/race_simulate.py`.
    # Mantém fallback para o nome antigo caso exista em algum branch.
    try:
        from race_simulate import RaceSimulator
    except ModuleNotFoundError:
        from race_simulator import RaceSimulator

    return RaceSimulator()


def make_reliability_chart(rel_data: list[dict], title: str) -> go.Figure:
    if not rel_data:
        return go.Figure()
    pred = [b["pred_mean"] for b in rel_data]
    real = [b["real_rate"] for b in rel_data]
    n = [b["n"] for b in rel_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Calibração perfeita", showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=pred, y=real, width=0.12, name="Modelo",
        marker_color="#E10600", opacity=0.85,
        text=[f"n={ni}" for ni in n], textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="Probabilidade prevista",
        yaxis_title="Taxa real observada",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.1]),
        height=400, template="plotly_dark",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("# 🏎️ F1 AI Insights")
page = st.sidebar.radio(
    "Navegação",
    ["Simulação de Corrida", "Calibração", "Análise do Modelo", "Sobre o Projeto"],
)


# ============================================================
# PAGE 1: SIMULAÇÃO
# ============================================================
if page == "Simulação de Corrida":
    st.markdown('<p class="main-header">Simulação Monte Carlo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Selecione o GP e o grid de largada para simular a corrida</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_gp = st.selectbox("Grand Prix", KNOWN_GPS, index=0)
        total_laps = GP_LAPS.get(selected_gp, 55)
        st.caption(f"Voltas: {total_laps}")

        n_sims = st.select_slider(
            "Simulações",
            options=[100, 500, 1000, 2000, 5000, 10000],
            value=2000,
        )
        st.caption(f"Mais simulações = mais preciso, mais lento (~{n_sims/880:.0f}s)")

        st.markdown("### Grid de Largada")
        st.caption("Selecione os pilotos na ordem de largada")

        all_drivers = sorted(KNOWN_DRIVERS.keys())
        default_grid = ["VER", "SAI", "LEC", "NOR", "PIA", "RUS", "HAM", "ALO", "STR", "PER"]
        selected_drivers = st.multiselect(
            "Pilotos (ordem = posição de largada)",
            options=all_drivers,
            default=default_grid,
            max_selections=20,
        )

        run_sim = st.button("🏁 Simular Corrida", type="primary", use_container_width=True)

    with col2:
        if run_sim and len(selected_drivers) >= 2:
            grid = []
            for i, drv in enumerate(selected_drivers, 1):
                grid.append({
                    "driver": drv,
                    "team": KNOWN_DRIVERS.get(drv, "Unknown"),
                    "grid_pos": i,
                    "quali_pos": i,
                    "gap_to_pole_ms": (i - 1) * 200,
                })

            with st.spinner(f"Simulando {n_sims:,} corridas..."):
                sim = load_simulator()
                results = sim.simulate(
                    gp=selected_gp, grid=grid,
                    n_simulations=n_sims,
                    total_laps=total_laps,
                    seed=42, verbose=False,
                )

            probs = results["probabilities"]

            # Tabela principal
            rows = []
            for drv in selected_drivers:
                if drv in probs:
                    p = probs[drv]
                    rows.append({
                        "Piloto": drv,
                        "Equipe": KNOWN_DRIVERS.get(drv, "?"),
                        "Vitória": f"{p['win']*100:.1f}%",
                        "Pódio": f"{p['podium']*100:.1f}%",
                        "Top 6": f"{p['top6']*100:.1f}%",
                        "Top 10": f"{p['top10']*100:.1f}%",
                        "DNF": f"{p['DNF']*100:.1f}%",
                    })

            st.markdown(f"### Resultados — {selected_gp}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Gráfico de barras
            win_data = sorted(
                [(drv, probs[drv]["win"]) for drv in probs],
                key=lambda x: -x[1],
            )
            fig = go.Figure(go.Bar(
                x=[d[0] for d in win_data],
                y=[d[1] * 100 for d in win_data],
                marker_color=["#E10600" if i == 0 else "#333" for i in range(len(win_data))],
                text=[f"{d[1]*100:.1f}%" for d in win_data],
                textposition="outside",
            ))
            fig.update_layout(
                title="Probabilidade de Vitória",
                yaxis_title="%", height=400,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap de posições
            position_data = []
            for drv in selected_drivers[:10]:
                if drv in probs:
                    p = probs[drv]
                    row = [p.get(f"P{pos}", 0) * 100 for pos in range(1, len(selected_drivers) + 1)]
                    position_data.append(row)

            if position_data:
                fig2 = go.Figure(go.Heatmap(
                    z=position_data,
                    x=[f"P{i}" for i in range(1, len(selected_drivers) + 1)],
                    y=selected_drivers[:10],
                    colorscale="RdYlGn_r",
                    text=[[f"{v:.1f}%" for v in row] for row in position_data],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                ))
                fig2.update_layout(
                    title="Distribuição de Posição Final (%)",
                    height=400, template="plotly_dark",
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.caption(f"SC probability: {results['sc_probability']*100:.1f}% | Simulações: {n_sims:,}")

        elif run_sim:
            st.warning("Selecione pelo menos 2 pilotos.")


# ============================================================
# PAGE 2: CALIBRAÇÃO
# ============================================================
elif page == "Calibração":
    st.markdown('<p class="main-header">Calibração do Modelo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Validação contra todas as 24 corridas de 2024</p>', unsafe_allow_html=True)

    cal_file = MODELS_DIR / "calibration_report.json"
    if not cal_file.exists():
        st.warning("Rode `calibrate_simulator.py` primeiro para gerar o relatório.")
    else:
        with open(cal_file) as f:
            report = json.load(f)

        # Métricas resumo
        cols = st.columns(5)
        market_labels = {"win": "Vitória", "podium": "Pódio", "top6": "Top 6", "top10": "Top 10", "dnf": "DNF"}

        for col, (market, label) in zip(cols, market_labels.items()):
            if market in report:
                r = report[market]
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{r['improvement_pct']:+.1f}%</div>
                        <div class="metric-label">vs baseline</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Reliability diagrams
        st.markdown("### Reliability Diagrams")
        st.caption("Calibração perfeita = barras na diagonal. Acima = modelo underconfident. Abaixo = overconfident.")

        rel_cols = st.columns(3)
        for i, (market, label) in enumerate(list(market_labels.items())[:3]):
            if market in report and "reliability" in report[market]:
                with rel_cols[i]:
                    fig = make_reliability_chart(report[market]["reliability"], label)
                    st.plotly_chart(fig, use_container_width=True)

        rel_cols2 = st.columns(3)
        for i, (market, label) in enumerate(list(market_labels.items())[3:]):
            if market in report and "reliability" in report[market]:
                with rel_cols2[i]:
                    fig = make_reliability_chart(report[market]["reliability"], label)
                    st.plotly_chart(fig, use_container_width=True)

        # Tabela detalhada
        st.markdown("### Métricas Detalhadas")
        rows = []
        for market, label in market_labels.items():
            if market in report:
                r = report[market]
                rows.append({
                    "Mercado": label,
                    "Brier (Modelo)": f"{r['brier_model']:.4f}",
                    "Brier (Baseline)": f"{r['brier_baseline']:.4f}",
                    "Ganho": f"{r['improvement_pct']:+.1f}%",
                    "ECE": f"{r['ece']:.4f}",
                    "Base Rate": f"{r['base_rate']*100:.1f}%",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("""
        **Interpretação:**
        - **Brier Score**: média de (probabilidade − resultado)² — menor é melhor
        - **ECE**: Expected Calibration Error — mede se "70% previsto" realmente acerta ~70%
        - **Ganho**: melhoria percentual sobre o baseline (usar grid position como preditor)
        - O modelo **bate o baseline em todos os 5 mercados**
        """)


# ============================================================
# PAGE 3: ANÁLISE DO MODELO
# ============================================================
elif page == "Análise do Modelo":
    st.markdown('<p class="main-header">Análise do Modelo v3</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Feature importance, performance por GP e piloto</p>', unsafe_allow_html=True)

    # Feature importance
    imp_file = MODELS_DIR / "global_feature_importance_v3.csv"
    if imp_file.exists():
        imp = pd.read_csv(imp_file)
        st.markdown("### Feature Importance (Top 20)")

        fig = go.Figure(go.Bar(
            x=imp.head(20)["importance"].values[::-1],
            y=imp.head(20)["feature"].values[::-1],
            orientation="h",
            marker_color="#E10600",
        ))
        fig.update_layout(
            height=500, template="plotly_dark",
            xaxis_title="Importância",
            margin=dict(l=200),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Métricas do modelo
    st.markdown("### Métricas Walk-Forward (treina 2022+2023 → testa 2024)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", "0.722")
    with col2:
        st.metric("R²", "0.638")
    with col3:
        st.metric("MAE", "0.537")
    with col4:
        st.metric("Ganho vs Trivial", "39.9%")

    # Diagnóstico por GP
    diag_file = MODELS_DIR / "diagnose_model_v2_report.csv"
    if diag_file.exists():
        diag = pd.read_csv(diag_file, index_col=0)
        st.markdown("### RMSE por Grand Prix")

        fig = go.Figure(go.Bar(
            x=diag.index,
            y=diag["rmse"],
            marker_color="#E10600",
        ))
        fig.update_layout(
            height=400, template="plotly_dark",
            xaxis_tickangle=-45, yaxis_title="RMSE (s)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: SOBRE O PROJETO
# ============================================================
elif page == "Sobre o Projeto":
    st.markdown('<p class="main-header">F1 AI Race Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema de decision support para corridas de Fórmula 1</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Arquitetura

    O sistema é composto por 4 modelos que alimentam um simulador Monte Carlo:

    1. **Modelo de Lap Time** (XGBoost, RMSE 0.72s walk-forward)
       - 49 features incluindo telemetria, weather, race control, Ergast
       - Target encoding CV-safe para Driver/Team/GP
       - Validação walk-forward: treina 2022+2023, testa 2024

    2. **Modelo de DNF** (Logistic Regression, Brier 0.087)
       - Features históricas anti-leakage (expanding mean + shift)
       - Calibrado nativamente (ECE 0.014)

    3. **Modelo de Safety Car** (Beta-Binomial bayesiano)
       - Estimativa por pista com smoothing e pesos temporais

    4. **Modelo de Degradação de Pneu** (Regressão linear por compound)
       - Coeficientes medidos empiricamente em 59k voltas

    ### Simulador Monte Carlo

    O simulador roda a corrida **10.000 vezes** em ~12 segundos, amostrando
    eventos aleatórios (DNF, Safety Car) a cada simulação. O resultado é uma
    **distribuição de probabilidade** por piloto por posição final.

    ### Calibração

    Validado contra todas as 24 corridas de 2024. O modelo **bate o baseline
    trivial (grid position) em todos os 5 mercados**: Win (+4%), Podium (+20%),
    Top 6 (+35%), Top 10 (+12%), DNF (+1.4%).

    ### Decisões Técnicas Importantes

    - **Anti-leakage**: 10+ testes pytest provando que nenhuma feature vaza info do futuro
    - **Target encoding**: substituiu label encoding que estava PIORANDO o modelo
    - **Walk-forward**: validação temporal honesta (treina no passado, testa no futuro)
    - **Reliability diagrams**: prova formal que probabilidades são calibradas
    - **Baseline comparison**: todo modelo é comparado contra "chutar a média"

    ### Stack Tecnológico

    FastF1 · XGBoost · scikit-learn · Optuna · MLflow · NumPy · Pandas · Streamlit · Plotly

    ### Dados

    - **96.598 voltas brutas** coletadas via FastF1 (2022+2023+2024)
    - **59.362 voltas de race** após filtragem (IQR + residual)
    - **78 features** por volta
    - **Weather, race control, Ergast** integrados por volta

    ---
    *Projeto desenvolvido como sistema de decision support. Não é aconselhamento de apostas.*
    """)

    st.markdown("---")
    st.caption("F1 AI Race Insights v3.0 | Sprint 1-3 | 2024")