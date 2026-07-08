"""Shared helpers for loading real FastF1 sessions and telemetry."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_YEARS = [2022, 2023, 2024, 2025, 2026]
SESSION_CODES = ["FP1", "FP2", "FP3", "Q", "R"]

_cache_enabled = False
_session_cache: dict[tuple[int, str, str], Any] = {}


def enable_cache() -> None:
    global _cache_enabled
    if _cache_enabled:
        return
    import fastf1

    fastf1.Cache.enable_cache(str(CACHE_DIR))
    _cache_enabled = True


def get_schedule(year: int):
    import fastf1

    enable_cache()
    return fastf1.get_event_schedule(year, include_testing=False)


def get_session_cached(year: int, gp: str, session_code: str):
    import fastf1

    key = (year, gp, session_code)
    if key in _session_cache:
        return _session_cache[key]

    enable_cache()
    session = fastf1.get_session(year, gp, session_code)
    session.load(laps=True, telemetry=True, weather=True)
    _session_cache[key] = session
    return session


def _clean_records(df: pd.DataFrame) -> list[dict]:
    compact = df.copy()
    if "Time" in compact:
        compact["Time"] = compact["Time"].dt.total_seconds()
    compact = compact.where(compact.notna(), None)
    return compact.to_dict(orient="records")


def driver_lap_telemetry(session, driver: str, lap_number: int | None = None) -> dict:
    from telemetry.telemetry_signals import add_derived_signals, summarize_lap_signals

    laps = session.laps.pick_drivers(driver.upper())
    if laps.empty:
        raise ValueError(f"No laps found for {driver}.")

    if lap_number is None:
        lap = laps.pick_fastest()
    else:
        selected = laps[laps["LapNumber"] == lap_number]
        if selected.empty:
            raise ValueError(f"Lap {lap_number} not found for {driver}.")
        lap = selected.iloc[0]

    telemetry = lap.get_telemetry().add_distance()
    telemetry = add_derived_signals(telemetry)

    columns = [
        "Distance",
        "Time",
        "Speed",
        "Throttle",
        "Brake",
        "brake_pressure_proxy",
        "RPM",
        "nGear",
        "DRS",
        "X",
        "Y",
        "DriverAhead",
        "DistanceToDriverAhead",
        "dirty_air_score",
        "steering_proxy",
        "lateral_change",
        "accel_proxy",
    ]
    available = [c for c in columns if c in telemetry.columns]

    lap_time = lap["LapTime"]
    lap_time_s = lap_time.total_seconds() if pd.notna(lap_time) else None

    return {
        "driver": driver.upper(),
        "lap_number": int(lap["LapNumber"]),
        "lap_time_s": lap_time_s,
        "compound": lap.get("Compound"),
        "team": lap.get("Team"),
        "summary": summarize_lap_signals(telemetry),
        "samples": _clean_records(telemetry[available]),
    }
