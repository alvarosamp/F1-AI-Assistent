"""
Coleta telemetria completa de 2026 (mesma estrutura de make_dataset_v2.py:
laps + telemetria por volta + weather + safety car/VSC/yellow flags + Ergast
quali/grid) reaproveitando `process_session` do coletor v2 — mais os sinais
de engenharia já implementados em src/telemetry/telemetry_signals.py
(dirty_air_score, DistanceToDriverAhead, steering_proxy, brake_pressure_proxy,
accel_proxy), que o coletor v2 (2022-2024) nunca calculou.

Gera data/raw/telemetry_full_2026.csv, no mesmo schema de telemetry_full_v2.csv
mais as colunas novas (NaN no histórico, que não foi recoletado), pra poder
ser concatenado e alimentar build_dnf_dataset.py / features com
regulation_era=3 (2026 = motores híbridos novos, chassis ativo-aero).

USO:
    python src/data/make_dataset_2026.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.make_dataset_v2 import (  # noqa: E402
    CACHE_DIR, RAW_DIR, build_lap_flags, merge_weather_into_lap,
    get_ergast_quali_and_grid, load_schedule_cached, _ERGAST_CACHE,
    load_ergast_cache, with_retry,
)
from telemetry.telemetry_signals import add_derived_signals  # noqa: E402

YEAR = 2026
SESSION_CODES = ["R", "Q"]
OUT_FILE = RAW_DIR / "telemetry_full_2026.csv"
PARTIAL_FILE = RAW_DIR / "telemetry_full_2026_partial.csv"
CHECKPOINT_FILE = RAW_DIR / "checkpoint_2026.json"


def session_key(gp: str, code: str) -> str:
    return f"{YEAR}|{gp}|{code}"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_checkpoint(state: dict) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def mark_failed(state: dict, key: str) -> None:
    """Adiciona ao checkpoint sem duplicar a cada rerun."""
    if key not in state["failed"]:
        state["failed"].append(key)
    save_checkpoint(state)


def load_event_dates(year: int) -> dict[str, pd.Timestamp]:
    """EventDate por GP (load_schedule_cached não guarda datas, só nome+round)."""
    full_schedule = with_retry(fastf1.get_event_schedule, f"schedule dates {year}", year, include_testing=False)
    return dict(zip(full_schedule["EventName"], full_schedule["EventDate"]))

ENGINEERING_COLUMNS = [
    "avg_distance_to_car_ahead",
    "dirty_air_pct",
    "dirty_air_score_mean",
    "cornering_intensity",
    "brake_pressure_mean",
    "accel_proxy_std",
]


def extract_lap_engineering_signals(lap) -> dict | None:
    """Mesma lógica de src/telemetry/telemetry_signals.py, agregada por volta."""
    try:
        tel = lap.get_telemetry().add_distance()
        if len(tel) < 10:
            return None
        tel = add_derived_signals(tel)
        return {
            "speed_mean": float(tel["Speed"].mean()),
            "speed_max": float(tel["Speed"].max()),
            "speed_std": float(tel["Speed"].std()),
            "throttle_mean": float(tel["Throttle"].mean()),
            "throttle_std": float(tel["Throttle"].std()),
            "brake_ratio": float((tel["Brake"] > 0).mean()),
            "rpm_mean": float(tel["RPM"].mean()),
            "gear_mean": float(tel["nGear"].mean()),
            "drs_ratio": float((tel["DRS"] >= 10).mean()),
            "avg_distance_to_car_ahead": float(tel["DistanceToDriverAhead"].mean()),
            "dirty_air_pct": float((tel["dirty_air_score"] > 0.25).mean()),
            "dirty_air_score_mean": float(tel["dirty_air_score"].mean()),
            "cornering_intensity": float(tel["steering_proxy"].abs().mean()),
            "brake_pressure_mean": float(tel["brake_pressure_proxy"].mean()),
            "accel_proxy_std": float(tel["accel_proxy"].std()),
        }
    except Exception:
        return None


def process_session_with_engineering(year: int, gp: str, session_code: str, round_number: int) -> pd.DataFrame:
    session = with_retry(fastf1.get_session, f"{year} {gp} {session_code}", year, gp, session_code)
    with_retry(session.load, f"load {year} {gp} {session_code}",
               laps=True, telemetry=True, weather=True, messages=True)
    laps = session.laps
    if laps is None or len(laps) == 0:
        return pd.DataFrame()
    weather_df = session.weather_data
    flags = build_lap_flags(session)
    ergast_data = {}
    if session_code == "R":
        ergast_data = get_ergast_quali_and_grid(year, round_number, ergast_cache=_ERGAST_CACHE)
    rows = []
    for _, lap in laps.iterlaps():
        telemetry = extract_lap_engineering_signals(lap)
        if telemetry is None:
            continue
        weather = merge_weather_into_lap(lap, weather_df)
        lap_no = int(lap["LapNumber"]) if pd.notna(lap["LapNumber"]) else -1
        lap_flags = flags.get(lap_no, {"is_sc": 0, "is_vsc": 0, "is_yellow": 0})
        driver_code = str(lap["Driver"])
        erg = ergast_data.get(driver_code, {})
        row = {
            "year": year, "gp": gp, "session_code": session_code,
            "Driver": driver_code,
            "DriverNumber": int(lap["DriverNumber"]) if pd.notna(lap["DriverNumber"]) else -1,
            "Team": str(lap["Team"]),
            "LapNumber": lap_no,
            "Stint": float(lap["Stint"]) if pd.notna(lap["Stint"]) else np.nan,
            "Compound": str(lap["Compound"]),
            "TyreLife": float(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else np.nan,
            "Position": float(lap["Position"]) if pd.notna(lap["Position"]) else np.nan,
            "TrackStatus": str(lap.get("TrackStatus", "1")),
            "LapTime": lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else np.nan,
            "PitOutTime": lap.get("PitOutTime"),
            "PitInTime": lap.get("PitInTime"),
            "Deleted": bool(lap.get("Deleted", False)) if pd.notna(lap.get("Deleted")) else False,
            **telemetry, **weather, **lap_flags, **erg,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    _ERGAST_CACHE.update(load_ergast_cache())

    state = load_checkpoint()
    completed = set(state["completed"])
    print(f"[INFO] {len(completed)} sessões já completadas no checkpoint")

    if PARTIAL_FILE.exists():
        df_all = pd.read_csv(PARTIAL_FILE)
        print(f"[INFO] Parcial carregado: {len(df_all)} linhas")
    else:
        df_all = pd.DataFrame()

    schedule = load_schedule_cached(YEAR)
    schedule = schedule[schedule["RoundNumber"] > 0]
    event_dates = load_event_dates(YEAR)
    today = pd.Timestamp.now().normalize()

    for _, event in schedule.iterrows():
        gp = event["EventName"]
        round_number = int(event.get("RoundNumber", 0))

        event_date = event_dates.get(gp)
        if event_date is not None and pd.notna(event_date) and (event_date - pd.Timedelta(days=3)) > today:
            print(f"[wait] {gp}: fim de semana ainda não aconteceu (evento em {event_date.date()}) — não conta como falha")
            continue

        for code in SESSION_CODES:
            key = session_key(gp, code)
            if key in completed:
                print(f"[skip] {key} (checkpoint)")
                continue
            print(f"\nProcessando {key} ...")
            try:
                df = process_session_with_engineering(YEAR, gp, code, round_number)
            except Exception:
                print(f"  [FAIL TOTAL] {key}")
                traceback.print_exc()
                mark_failed(state, key)
                continue
            if len(df) == 0:
                print(f"  [SKIP] {gp} {code}: 0 linhas (corrida ainda não disputada?)")
                mark_failed(state, key)
                continue
            df_all = pd.concat([df_all, df], ignore_index=True)
            df_all.to_csv(PARTIAL_FILE, index=False)
            state["completed"].append(key)
            if key in state["failed"]:
                state["failed"].remove(key)
            save_checkpoint(state)
            print(f"  [OK] {gp} {code}: {len(df)} linhas | parcial total: {len(df_all)}")

    if len(df_all) > 0:
        df_all.to_csv(OUT_FILE, index=False)
        print(f"\n[OK] {len(df_all)} linhas totais -> {OUT_FILE}")
        print(f"     GPs coletados: {df_all['gp'].nunique()}")
    else:
        print("\n[FAIL] Nenhum dado coletado.")


if __name__ == "__main__":
    main()
