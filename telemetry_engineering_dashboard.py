from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from telemetry.telemetry_signals import add_derived_signals, summarize_lap_signals


st.set_page_config(
    page_title="F1 Telemetry Engineering",
    page_icon="F1",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    [data-testid="stMetric"] {background: #101114; border: 1px solid #262a31; padding: 10px 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_fastf1():
    import fastf1

    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return fastf1


@st.cache_data(show_spinner=True)
def load_session(year: int, gp: str, session_name: str):
    fastf1 = load_fastf1()
    session = fastf1.get_session(year, gp, session_name)
    session.load(laps=True, telemetry=True, weather=True)
    return session


def pick_lap(session, driver: str, mode: str, lap_number: int | None):
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        raise ValueError(f"No laps found for {driver}.")
    if mode == "Fastest":
        return laps.pick_fastest()
    selected = laps[laps["LapNumber"] == lap_number]
    if selected.empty:
        raise ValueError(f"Lap {lap_number} not found for {driver}.")
    return selected.iloc[0]


def lap_telemetry(lap) -> pd.DataFrame:
    telemetry = lap.get_telemetry().add_distance()
    telemetry = add_derived_signals(telemetry)
    telemetry["lap_seconds"] = pd.to_timedelta(telemetry["Time"]).dt.total_seconds()
    return telemetry


def signal_figure(telemetry: pd.DataFrame, title: str) -> go.Figure:
    rows = [
        ("Speed", "Speed", "km/h", "#00d2ff"),
        ("Throttle", "Throttle", "%", "#31ff45"),
        ("brake_pressure_proxy", "Brake pressure proxy", "%", "#ff2b2b"),
        ("RPM", "RPM", "rpm", "#d15cff"),
        ("nGear", "Gear", "gear", "#ffb000"),
        ("DRS", "DRS", "mode", "#00ffff"),
        ("lateral_change", "Steering proxy", "proxy", "#ffffff"),
        ("DistanceToDriverAhead", "Distance to driver ahead", "m", "#ff7f0e"),
        ("dirty_air_score", "Dirty air score", "0-1", "#f5c542"),
    ]

    fig = make_subplots(
        rows=len(rows),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        subplot_titles=[item[1] for item in rows],
    )
    x = telemetry["Distance"]
    for idx, (column, name, unit, color) in enumerate(rows, start=1):
        if column not in telemetry:
            continue
        fig.add_trace(
            go.Scatter(x=x, y=telemetry[column], mode="lines", name=name, line=dict(color=color, width=1.4)),
            row=idx,
            col=1,
        )
        fig.update_yaxes(title_text=unit, row=idx, col=1)

    fig.update_xaxes(title_text="Distance (m)", row=len(rows), col=1)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=1260,
        showlegend=False,
        margin=dict(l=60, r=20, t=70, b=35),
    )
    return fig


def track_map(telemetry: pd.DataFrame, driver: str) -> go.Figure:
    fig = go.Figure()
    color = telemetry["Speed"] if "Speed" in telemetry else None
    fig.add_trace(
        go.Scatter(
            x=telemetry["X"],
            y=telemetry["Y"],
            mode="markers+lines",
            marker=dict(size=4, color=color, colorscale="Turbo", showscale=True, colorbar=dict(title="km/h")),
            line=dict(color="#555", width=1),
            name=driver,
            text=[f"{d:.0f}m" for d in telemetry["Distance"]],
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def compare_signal_figure(tel_a: pd.DataFrame, tel_b: pd.DataFrame, driver_a: str, driver_b: str) -> go.Figure:
    max_distance = min(float(tel_a["Distance"].max()), float(tel_b["Distance"].max()))
    distance = np.linspace(0, max_distance, 700)

    def interp(tel: pd.DataFrame, col: str):
        clean = tel[["Distance", col]].dropna().sort_values("Distance")
        return np.interp(distance, clean["Distance"], clean[col]) if not clean.empty else np.zeros_like(distance)

    speed_a = interp(tel_a, "Speed")
    speed_b = interp(tel_b, "Speed")
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        subplot_titles=["Speed overlay", "Speed delta", "Throttle", "Brake proxy", "Dirty air score"],
    )
    fig.add_trace(go.Scatter(x=distance, y=speed_a, name=driver_a, line=dict(color="#e10600")), row=1, col=1)
    fig.add_trace(go.Scatter(x=distance, y=speed_b, name=driver_b, line=dict(color="#00d2be")), row=1, col=1)
    fig.add_trace(go.Scatter(x=distance, y=speed_a - speed_b, name=f"{driver_a}-{driver_b}", line=dict(color="#f5c542")), row=2, col=1)
    fig.add_trace(go.Scatter(x=distance, y=interp(tel_a, "Throttle"), name=f"{driver_a} throttle", line=dict(color="#31ff45")), row=3, col=1)
    fig.add_trace(go.Scatter(x=distance, y=interp(tel_b, "Throttle"), name=f"{driver_b} throttle", line=dict(color="#0f8f28")), row=3, col=1)
    fig.add_trace(go.Scatter(x=distance, y=interp(tel_a, "brake_pressure_proxy"), name=f"{driver_a} brake", line=dict(color="#ff2b2b")), row=4, col=1)
    fig.add_trace(go.Scatter(x=distance, y=interp(tel_b, "brake_pressure_proxy"), name=f"{driver_b} brake", line=dict(color="#ff8a8a")), row=4, col=1)
    fig.add_trace(go.Scatter(x=distance, y=interp(tel_a, "dirty_air_score"), name=f"{driver_a} dirty air", line=dict(color="#ff7f0e")), row=5, col=1)
    fig.add_trace(go.Scatter(x=distance, y=interp(tel_b, "dirty_air_score"), name=f"{driver_b} dirty air", line=dict(color="#ffd166")), row=5, col=1)
    fig.update_layout(template="plotly_dark", height=900, margin=dict(l=60, r=20, t=60, b=35))
    fig.update_xaxes(title_text="Distance (m)", row=5, col=1)
    return fig


def metrics_row(summary: dict[str, float]):
    cols = st.columns(6)
    values = [
        ("Max speed", summary.get("max_speed", 0), "km/h"),
        ("Avg speed", summary.get("avg_speed", 0), "km/h"),
        ("Full throttle", summary.get("full_throttle_pct", 0), "%"),
        ("Brake", summary.get("brake_pct", 0), "% lap"),
        ("Dirty air", summary.get("dirty_air_pct", 0), "% lap"),
        ("Cornering", summary.get("cornering_intensity", 0), "proxy"),
    ]
    for col, (label, value, suffix) in zip(cols, values):
        col.metric(label, f"{value:.1f} {suffix}")


def main():
    st.title("F1 Telemetry Engineering")

    with st.sidebar:
        year = st.number_input("Year", min_value=2018, max_value=2026, value=2024, step=1)
        gp = st.text_input("Grand Prix", value="Monaco Grand Prix")
        session_name = st.selectbox("Session", ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"], index=3)
        driver_a = st.text_input("Driver A", value="VER").upper()
        driver_b = st.text_input("Driver B", value="LEC").upper()
        mode = st.radio("Lap selection", ["Fastest", "Manual"], horizontal=True)
        lap_a = st.number_input("Lap A", min_value=1, value=1, step=1, disabled=mode == "Fastest")
        lap_b = st.number_input("Lap B", min_value=1, value=1, step=1, disabled=mode == "Fastest")
        run = st.button("Load telemetry", type="primary", use_container_width=True)

    if not run:
        st.info("Choose a session and press Load telemetry.")
        return

    session = load_session(int(year), gp, session_name)
    selected_lap_a = pick_lap(session, driver_a, mode, int(lap_a))
    selected_lap_b = pick_lap(session, driver_b, mode, int(lap_b))
    tel_a = lap_telemetry(selected_lap_a)
    tel_b = lap_telemetry(selected_lap_b)
    summary_a = summarize_lap_signals(tel_a)
    summary_b = summarize_lap_signals(tel_b)

    lap_time_a = pd.to_timedelta(selected_lap_a["LapTime"]).total_seconds()
    lap_time_b = pd.to_timedelta(selected_lap_b["LapTime"]).total_seconds()

    top = st.columns([1, 1, 1, 1])
    top[0].metric(f"{driver_a} lap", int(selected_lap_a["LapNumber"]))
    top[1].metric(f"{driver_a} time", f"{lap_time_a:.3f}s")
    top[2].metric(f"{driver_b} lap", int(selected_lap_b["LapNumber"]))
    top[3].metric(f"{driver_b} time", f"{lap_time_b:.3f}s", delta=f"{lap_time_a - lap_time_b:+.3f}s A-B")

    tab_a, tab_b, tab_compare, tab_map, tab_dirty = st.tabs([driver_a, driver_b, "Compare", "Track map", "Dirty air"])

    with tab_a:
        metrics_row(summary_a)
        st.plotly_chart(signal_figure(tel_a, f"{driver_a} telemetry"), use_container_width=True)

    with tab_b:
        metrics_row(summary_b)
        st.plotly_chart(signal_figure(tel_b, f"{driver_b} telemetry"), use_container_width=True)

    with tab_compare:
        st.plotly_chart(compare_signal_figure(tel_a, tel_b, driver_a, driver_b), use_container_width=True)

    with tab_map:
        left, right = st.columns(2)
        left.plotly_chart(track_map(tel_a, driver_a), use_container_width=True)
        right.plotly_chart(track_map(tel_b, driver_b), use_container_width=True)

    with tab_dirty:
        cols = ["Distance", "DriverAhead", "DistanceToDriverAhead", "dirty_air_score", "Speed", "Throttle", "Brake"]
        available = [col for col in cols if col in tel_a.columns]
        st.dataframe(tel_a[available].sort_values("dirty_air_score", ascending=False).head(40), use_container_width=True)


if __name__ == "__main__":
    main()
