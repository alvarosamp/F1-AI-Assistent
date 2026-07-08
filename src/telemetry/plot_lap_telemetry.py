from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from telemetry.telemetry_signals import add_derived_signals, summarize_lap_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FastF1 telemetry signals for one driver lap.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--gp", required=True, help='Example: "Monaco Grand Prix"')
    parser.add_argument("--session", default="Q", help="FP1, FP2, FP3, Q, S, SQ, R")
    parser.add_argument("--driver", required=True, help="Driver abbreviation, e.g. VER")
    parser.add_argument("--lap", type=int, default=None, help="Optional lap number. Defaults to fastest lap.")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "cache")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "telemetry")
    return parser.parse_args()


def load_lap_telemetry(year: int, gp: str, session_name: str, driver: str, lap_number: int | None, cache: Path):
    try:
        import fastf1
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: fastf1\n"
            "Install the project requirements first:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))

    session = fastf1.get_session(year, gp, session_name)
    session.load(laps=True, telemetry=True, weather=True)

    laps = session.laps.pick_drivers(driver.upper())
    if laps.empty:
        raise ValueError(f"No laps found for driver {driver}.")

    if lap_number is None:
        lap = laps.pick_fastest()
    else:
        selected = laps[laps["LapNumber"] == lap_number]
        if selected.empty:
            raise ValueError(f"Lap {lap_number} not found for driver {driver}.")
        lap = selected.iloc[0]

    telemetry = lap.get_telemetry().add_distance()
    telemetry = add_derived_signals(telemetry)
    return session, lap, telemetry


def build_figure(telemetry, title: str) -> go.Figure:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: plotly\n"
            "Install the project requirements first:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    fig = make_subplots(
        rows=10,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=[
            "Speed",
            "Throttle",
            "Brake",
            "Brake pressure proxy",
            "RPM",
            "Gear",
            "DRS",
            "Steering proxy / lateral direction",
            "Distance to driver ahead",
            "Dirty air score",
        ],
    )

    x = telemetry["Distance"]
    traces = [
        ("Speed", "Speed", "km/h", "#e10600"),
        ("Throttle", "Throttle", "%", "#1f77b4"),
        ("Brake", "Brake", "on/off", "#ff7f0e"),
        ("brake_pressure_proxy", "Brake pressure proxy", "%", "#ff2b2b"),
        ("RPM", "RPM", "rpm", "#2ca02c"),
        ("nGear", "Gear", "gear", "#9467bd"),
        ("DRS", "DRS", "mode", "#17becf"),
        ("lateral_change", "Steering proxy", "proxy", "#d62728"),
        ("DistanceToDriverAhead", "Distance to driver ahead", "m", "#f5c542"),
        ("dirty_air_score", "Dirty air score", "0-1", "#ffffff"),
    ]

    for row_idx, (column, name, unit, color) in enumerate(traces, start=1):
        if column not in telemetry:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=telemetry[column],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.8),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text=unit, row=row_idx, col=1)

    fig.update_xaxes(title_text="Distance (m)", row=10, col=1)
    fig.update_layout(
        title=title,
        height=1500,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=70, r=30, t=80, b=40),
    )
    return fig


def main() -> int:
    args = parse_args()
    session, lap, telemetry = load_lap_telemetry(
        args.year,
        args.gp,
        args.session,
        args.driver,
        args.lap,
        args.cache,
    )

    lap_number = int(lap["LapNumber"])
    lap_time = lap["LapTime"]
    title = f"{args.driver.upper()} telemetry - {args.gp} {args.year} {args.session} - lap {lap_number}"
    fig = build_figure(telemetry, title)

    args.out.mkdir(parents=True, exist_ok=True)
    safe_gp = args.gp.lower().replace(" ", "_").replace("/", "_")
    out_file = args.out / f"{args.year}_{safe_gp}_{args.session}_{args.driver.upper()}_lap_{lap_number}.html"
    fig.write_html(out_file)

    summary = summarize_lap_signals(telemetry)
    print(f"Session: {session.event['EventName']} {args.session}")
    print(f"Driver: {args.driver.upper()} | Lap: {lap_number} | LapTime: {lap_time}")
    for key, value in summary.items():
        print(f"{key}: {value:.3f}")
    print(f"Saved telemetry plot: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
