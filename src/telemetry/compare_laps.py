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
    parser = argparse.ArgumentParser(
        description="Compare FastF1 telemetry between two drivers' laps (delta time + channel overlay)."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--gp", required=True, help='Example: "Monaco Grand Prix"')
    parser.add_argument("--session", default="Q", help="FP1, FP2, FP3, Q, S, SQ, R")
    parser.add_argument("--driver-a", required=True, help="Reference driver abbreviation, e.g. VER")
    parser.add_argument("--driver-b", required=True, help="Comparison driver abbreviation, e.g. NOR")
    parser.add_argument("--lap-a", type=int, default=None, help="Optional lap number for driver A. Defaults to fastest lap.")
    parser.add_argument("--lap-b", type=int, default=None, help="Optional lap number for driver B. Defaults to fastest lap.")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "cache")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "telemetry")
    return parser.parse_args()


def _load_session(year: int, gp: str, session_name: str, cache: Path):
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
    return session


def _pick_lap(session, driver: str, lap_number: int | None):
    laps = session.laps.pick_drivers(driver.upper())
    if laps.empty:
        raise ValueError(f"No laps found for driver {driver}.")

    if lap_number is None:
        return laps.pick_fastest()

    selected = laps[laps["LapNumber"] == lap_number]
    if selected.empty:
        raise ValueError(f"Lap {lap_number} not found for driver {driver}.")
    return selected.iloc[0]


def build_comparison(session, lap_a, lap_b):
    from fastf1 import utils

    delta_time, ref_tel, compare_tel = utils.delta_time(lap_a, lap_b)

    tel_a = add_derived_signals(lap_a.get_telemetry().add_distance())
    tel_b = add_derived_signals(lap_b.get_telemetry().add_distance())

    return delta_time, ref_tel, compare_tel, tel_a, tel_b


def build_figure(driver_a: str, driver_b: str, delta_time, tel_a, tel_b, title: str):
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
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.18, 0.22, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=[
            f"Delta time ({driver_b} vs {driver_a}, negative = {driver_b} ahead)",
            "Speed",
            "Throttle",
            "Brake",
            "Gear",
            "DRS",
        ],
    )

    colors = {"a": "#e10600", "b": "#1f77b4"}

    fig.add_trace(
        go.Scatter(
            x=tel_a["Distance"],
            y=delta_time,
            mode="lines",
            name="Delta",
            line=dict(color="#888888", width=1.8),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="#444444", width=1, dash="dot"), row=1, col=1)

    channel_rows = [
        ("Speed", "km/h", 2),
        ("Throttle", "%", 3),
        ("Brake", "on/off", 4),
        ("nGear", "gear", 5),
        ("DRS", "mode", 6),
    ]

    for column, unit, row in channel_rows:
        for label, tel, color in ((driver_a, tel_a, colors["a"]), (driver_b, tel_b, colors["b"])):
            if column not in tel:
                continue
            fig.add_trace(
                go.Scatter(
                    x=tel["Distance"],
                    y=tel[column],
                    mode="lines",
                    name=f"{label} {column}",
                    line=dict(color=color, width=1.6),
                    showlegend=(row == 2),
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text=unit, row=row, col=1)

    fig.update_xaxes(title_text="Distance (m)", row=6, col=1)
    fig.update_layout(
        title=title,
        height=1350,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=30, t=100, b=40),
    )
    return fig


def main() -> int:
    args = parse_args()
    session = _load_session(args.year, args.gp, args.session, args.cache)

    lap_a = _pick_lap(session, args.driver_a, args.lap_a)
    lap_b = _pick_lap(session, args.driver_b, args.lap_b)

    delta_time, ref_tel, compare_tel, tel_a, tel_b = build_comparison(session, lap_a, lap_b)

    lap_num_a = int(lap_a["LapNumber"])
    lap_num_b = int(lap_b["LapNumber"])
    lap_time_a = lap_a["LapTime"]
    lap_time_b = lap_b["LapTime"]

    title = (
        f"{args.driver_a.upper()} vs {args.driver_b.upper()} - {args.gp} {args.year} {args.session} "
        f"- laps {lap_num_a}/{lap_num_b}"
    )
    fig = build_figure(args.driver_a.upper(), args.driver_b.upper(), delta_time, tel_a, tel_b, title)

    args.out.mkdir(parents=True, exist_ok=True)
    safe_gp = args.gp.lower().replace(" ", "_").replace("/", "_")
    out_file = (
        args.out
        / f"{args.year}_{safe_gp}_{args.session}_{args.driver_a.upper()}_vs_{args.driver_b.upper()}.html"
    )
    fig.write_html(out_file)

    summary_a = summarize_lap_signals(tel_a)
    summary_b = summarize_lap_signals(tel_b)
    final_delta = float(delta_time.iloc[-1])

    print(f"Session: {session.event['EventName']} {args.session}")
    print(f"{args.driver_a.upper()}: lap {lap_num_a} | LapTime: {lap_time_a}")
    print(f"{args.driver_b.upper()}: lap {lap_num_b} | LapTime: {lap_time_b}")
    print(f"Final delta ({args.driver_b.upper()} vs {args.driver_a.upper()}): {final_delta:+.3f}s")
    print()
    print(f"{'metric':<20}{args.driver_a.upper():>12}{args.driver_b.upper():>12}")
    for key in summary_a:
        print(f"{key:<20}{summary_a[key]:>12.3f}{summary_b.get(key, float('nan')):>12.3f}")
    print(f"\nSaved comparison plot: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
