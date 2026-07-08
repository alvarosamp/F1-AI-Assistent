from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from telemetry.telemetry_signals import add_derived_signals, summarize_lap_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare telemetry signals for two FastF1 driver laps.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--gp", required=True, help='Example: "Monaco Grand Prix"')
    parser.add_argument("--session", default="Q", help="FP1, FP2, FP3, Q, S, SQ, R")
    parser.add_argument("--driver-a", required=True)
    parser.add_argument("--driver-b", required=True)
    parser.add_argument("--lap-a", type=int, default=None, help="Optional lap number for driver A.")
    parser.add_argument("--lap-b", type=int, default=None, help="Optional lap number for driver B.")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "cache")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "telemetry")
    return parser.parse_args()


def load_session(year: int, gp: str, session_name: str, cache: Path):
    try:
        import fastf1
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: fastf1\n"
            "Install requirements first:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    session = fastf1.get_session(year, gp, session_name)
    session.load(laps=True, telemetry=True, weather=True)
    return session


def pick_lap(session, driver: str, lap_number: int | None):
    laps = session.laps.pick_driver(driver.upper())
    if laps.empty:
        raise ValueError(f"No laps found for driver {driver}.")
    if lap_number is None:
        return laps.pick_fastest()
    selected = laps[laps["LapNumber"] == lap_number]
    if selected.empty:
        raise ValueError(f"Lap {lap_number} not found for driver {driver}.")
    return selected.iloc[0]


def lap_time_seconds(lap) -> float:
    return float(pd.to_timedelta(lap["LapTime"]).total_seconds())


def telemetry_for_lap(lap) -> pd.DataFrame:
    telemetry = lap.get_telemetry().add_distance()
    telemetry = add_derived_signals(telemetry)
    telemetry["lap_seconds"] = pd.to_timedelta(telemetry["Time"]).dt.total_seconds()
    return telemetry


def interpolate_on_distance(telemetry: pd.DataFrame, distance_grid: np.ndarray, column: str) -> np.ndarray:
    clean = telemetry[["Distance", column]].dropna().sort_values("Distance")
    if clean.empty:
        return np.zeros_like(distance_grid)
    return np.interp(distance_grid, clean["Distance"].to_numpy(), clean[column].to_numpy())


def build_comparison_frame(tel_a: pd.DataFrame, tel_b: pd.DataFrame) -> pd.DataFrame:
    max_distance = float(min(tel_a["Distance"].max(), tel_b["Distance"].max()))
    distance = np.linspace(0, max_distance, 600)
    frame = pd.DataFrame({"Distance": distance})

    for column in ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS", "lateral_change", "lap_seconds"]:
        frame[f"{column}_a"] = interpolate_on_distance(tel_a, distance, column)
        frame[f"{column}_b"] = interpolate_on_distance(tel_b, distance, column)

    frame["speed_delta"] = frame["Speed_a"] - frame["Speed_b"]
    frame["time_delta"] = frame["lap_seconds_a"] - frame["lap_seconds_b"]
    return frame


def build_figure(frame: pd.DataFrame, driver_a: str, driver_b: str, title: str):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: plotly\n"
            "Install requirements first:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    fig = make_subplots(
        rows=8,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[
            "Speed",
            "Speed delta",
            "Throttle",
            "Brake",
            "RPM",
            "Gear",
            "DRS",
            "Steering proxy / lateral direction",
        ],
    )

    x = frame["Distance"]
    row_specs = [
        ("Speed", "km/h"),
        ("speed_delta", "km/h"),
        ("Throttle", "%"),
        ("Brake", "on/off"),
        ("RPM", "rpm"),
        ("nGear", "gear"),
        ("DRS", "mode"),
        ("lateral_change", "proxy"),
    ]

    for row_idx, (column, unit) in enumerate(row_specs, start=1):
        if column == "speed_delta":
            fig.add_trace(
                go.Scatter(x=x, y=frame[column], mode="lines", name=f"{driver_a}-{driver_b}", line=dict(color="#f5c542")),
                row=row_idx,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(x=x, y=frame[f"{column}_a"], mode="lines", name=driver_a, line=dict(color="#e10600")),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=frame[f"{column}_b"], mode="lines", name=driver_b, line=dict(color="#00d2be")),
                row=row_idx,
                col=1,
            )
        fig.update_yaxes(title_text=unit, row=row_idx, col=1)

    fig.update_xaxes(title_text="Distance (m)", row=8, col=1)
    fig.update_layout(
        title=title,
        height=1350,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=70, r=30, t=90, b=40),
    )
    return fig


def print_summary(driver_a: str, driver_b: str, lap_a, lap_b, tel_a: pd.DataFrame, tel_b: pd.DataFrame, frame: pd.DataFrame):
    time_a = lap_time_seconds(lap_a)
    time_b = lap_time_seconds(lap_b)
    faster = driver_a if time_a < time_b else driver_b
    gap = abs(time_a - time_b)

    summary_a = summarize_lap_signals(tel_a)
    summary_b = summarize_lap_signals(tel_b)

    print(f"{driver_a} lap {int(lap_a['LapNumber'])}: {time_a:.3f}s")
    print(f"{driver_b} lap {int(lap_b['LapNumber'])}: {time_b:.3f}s")
    print(f"Faster: {faster} by {gap:.3f}s")
    print()
    print("Signal differences:")
    for key in ["max_speed", "avg_speed", "avg_throttle", "full_throttle_pct", "brake_pct", "avg_gear", "cornering_intensity"]:
        if key in summary_a and key in summary_b:
            print(f"{key}: {driver_a}={summary_a[key]:.3f} | {driver_b}={summary_b[key]:.3f} | delta={summary_a[key]-summary_b[key]:+.3f}")

    strongest_speed_gain = frame.iloc[frame["speed_delta"].abs().idxmax()]
    print()
    print(
        "Largest speed delta: "
        f"{strongest_speed_gain['speed_delta']:+.1f} km/h at {strongest_speed_gain['Distance']:.0f}m "
        f"({driver_a} minus {driver_b})"
    )


def main() -> int:
    args = parse_args()
    driver_a = args.driver_a.upper()
    driver_b = args.driver_b.upper()
    session = load_session(args.year, args.gp, args.session, args.cache)

    lap_a = pick_lap(session, driver_a, args.lap_a)
    lap_b = pick_lap(session, driver_b, args.lap_b)
    tel_a = telemetry_for_lap(lap_a)
    tel_b = telemetry_for_lap(lap_b)
    frame = build_comparison_frame(tel_a, tel_b)

    title = f"{driver_a} vs {driver_b} telemetry - {args.gp} {args.year} {args.session}"
    fig = build_figure(frame, driver_a, driver_b, title)

    args.out.mkdir(parents=True, exist_ok=True)
    safe_gp = args.gp.lower().replace(" ", "_").replace("/", "_")
    out_file = args.out / f"{args.year}_{safe_gp}_{args.session}_{driver_a}_vs_{driver_b}.html"
    fig.write_html(out_file)

    print_summary(driver_a, driver_b, lap_a, lap_b, tel_a, tel_b, frame)
    print(f"Saved comparison plot: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
