"""
Coletor de dados F1 via FastF1 — v2 (Sprint 2).

MUDANÇAS vs v1:
    - Coleta 2022+2023+2024 (3x mais dados do que tinha no Sprint 1)
    - Weather por volta (AirTemp, TrackTemp, Humidity, Rainfall, Wind)
    - Race control -> flags por volta (is_sc, is_vsc, is_yellow)
    - Ergast: quali_position, grid_position, gap_to_pole_ms
    - Checkpoint retomável, rate-limit handling, cache de schedule e Ergast

CORREÇÃO: _ERGAST_CACHE agora é definido em nível de módulo, permitindo que
outros scripts (make_dataset_v2_staged.py) importem process_session sem NameError.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
import traceback
import warnings

import fastf1
from fastf1.exceptions import RateLimitExceededError
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOCAL_CACHE_DIR = CACHE_DIR / "local"
SCHEDULE_CACHE_DIR = LOCAL_CACHE_DIR / "schedules"
ERGAST_CACHE_FILE = LOCAL_CACHE_DIR / "ergast_quali_grid_cache.json"
CHECKPOINT_FILE = RAW_DIR / "checkpoint_v2.json"
PARTIAL_FILE = RAW_DIR / "telemetry_full_v2_partial.csv"
FINAL_FILE = RAW_DIR / "telemetry_full_v2.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Cache global de Ergast em nível de módulo — existe desde o import,
# funciona mesmo quando outros scripts importam process_session sem chamar main()
_ERGAST_CACHE: dict = {}

YEARS = [2022, 2023, 2024]
SESSION_CODES = ["R", "Q"]

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5
RATE_LIMIT_SLEEP_SECONDS = 60 * 60 + 10


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_checkpoint(state: dict) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def session_key(year: int, gp: str, code: str) -> str:
    return f"{year}|{gp}|{code}"


def with_retry(fn, label: str, *args, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededError as e:
            print(f"    [RATE LIMIT] {label}: {e} (aguardando {RATE_LIMIT_SLEEP_SECONDS}s)")
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)
        except Exception as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                print(f"    [FAIL] {label}: {e}")
                raise
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"    [retry {attempt}/{MAX_RETRIES}] {label}: {e} (esperando {wait}s)")
            time.sleep(wait)


def load_schedule_cached(year: int) -> pd.DataFrame:
    cache_file = SCHEDULE_CACHE_DIR / f"event_schedule_{year}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        if "RoundNumber" in df.columns:
            df["RoundNumber"] = pd.to_numeric(df["RoundNumber"], errors="coerce")
        return df
    schedule = with_retry(fastf1.get_event_schedule, f"schedule {year}", year, include_testing=False)
    keep_cols = [c for c in ["EventName", "RoundNumber"] if c in schedule.columns]
    schedule[keep_cols].copy().to_csv(cache_file, index=False)
    return schedule


def load_ergast_cache() -> dict:
    if ERGAST_CACHE_FILE.exists():
        try:
            with open(ERGAST_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_ergast_cache(cache: dict) -> None:
    try:
        with open(ERGAST_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def extract_lap_telemetry(lap) -> dict | None:
    try:
        tel = lap.get_car_data().add_distance()
        if len(tel) < 10:
            return None
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
        }
    except Exception:
        return None


def merge_weather_into_lap(lap, weather_df: pd.DataFrame) -> dict:
    if weather_df is None or len(weather_df) == 0:
        return {}
    try:
        lap_start = lap["LapStartTime"]
        if pd.isna(lap_start):
            return {}
        idx = (weather_df["Time"] - lap_start).abs().idxmin()
        row = weather_df.loc[idx]
        return {
            "AirTemp": float(row.get("AirTemp", np.nan)),
            "TrackTemp": float(row.get("TrackTemp", np.nan)),
            "Humidity": float(row.get("Humidity", np.nan)),
            "Pressure": float(row.get("Pressure", np.nan)),
            "Rainfall": float(bool(row.get("Rainfall", False))),
            "WindSpeed": float(row.get("WindSpeed", np.nan)),
            "WindDirection": float(row.get("WindDirection", np.nan)),
        }
    except Exception:
        return {}


def build_lap_flags(session) -> dict[int, dict]:
    flags: dict[int, dict] = {}
    try:
        rc = session.race_control_messages
        if rc is None or len(rc) == 0:
            return flags
        sc_active = False
        vsc_active = False
        for _, msg in rc.iterrows():
            lap_no = msg.get("Lap")
            if pd.isna(lap_no):
                continue
            lap_no = int(lap_no)
            cat = str(msg.get("Category", ""))
            status = str(msg.get("Status", ""))
            text = str(msg.get("Message", "")).upper()
            if cat == "SafetyCar":
                if "DEPLOYED" in status or "DEPLOYED" in text:
                    if "VIRTUAL" in text:
                        vsc_active = True
                    else:
                        sc_active = True
                elif "ENDING" in status or "IN THIS LAP" in text:
                    if "VIRTUAL" in text:
                        vsc_active = False
                    else:
                        sc_active = False
            if lap_no not in flags:
                flags[lap_no] = {"is_sc": 0, "is_vsc": 0, "is_yellow": 0}
            if sc_active:
                flags[lap_no]["is_sc"] = 1
            if vsc_active:
                flags[lap_no]["is_vsc"] = 1
            if "YELLOW" in text and "CLEAR" not in text:
                flags[lap_no]["is_yellow"] = 1
    except Exception as e:
        print(f"    [warn] race_control_messages: {e}")
    return flags


def get_ergast_quali_and_grid(year: int, round_number: int, ergast_cache: dict | None = None) -> dict:
    cache_key = f"{year}:{round_number}"
    if ergast_cache is not None and cache_key in ergast_cache:
        cached = ergast_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached
    try:
        ergast = fastf1.ergast.Ergast()
        quali = with_retry(ergast.get_qualifying_results,
                           f"ergast quali {year} r{round_number}",
                           season=year, round=round_number)
        results = quali.content[0] if quali.content else None
        if results is None or len(results) == 0:
            return {}
        out = {}
        best_times_ms = []
        for _, row in results.iterrows():
            best = None
            for t in [row.get("Q3"), row.get("Q2"), row.get("Q1")]:
                if pd.notna(t):
                    best = t.total_seconds() * 1000 if hasattr(t, "total_seconds") else None
                    if best:
                        break
            best_times_ms.append(best)
        pole_ms = min((t for t in best_times_ms if t is not None), default=None)
        for (_, row), best_ms in zip(results.iterrows(), best_times_ms):
            code = row.get("driverCode") or row.get("code") or ""
            out[str(code)] = {
                "quali_position": int(row.get("position", -1)) if pd.notna(row.get("position")) else -1,
                "grid_position": int(row.get("position", -1)) if pd.notna(row.get("position")) else -1,
                "gap_to_pole_ms": (best_ms - pole_ms) if (best_ms and pole_ms) else np.nan,
            }
        if ergast_cache is not None:
            ergast_cache[cache_key] = out
            save_ergast_cache(ergast_cache)
        return out
    except Exception as e:
        print(f"    [warn] ergast quali: {e}")
        return {}


def process_session(year: int, gp: str, session_code: str, round_number: int) -> pd.DataFrame:
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
        telemetry = extract_lap_telemetry(lap)
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
    state = load_checkpoint()
    completed = set(state["completed"])
    print(f"[INFO] {len(completed)} sessões já completadas no checkpoint")

    # Popula o cache do Ergast do disco (não reatribui, atualiza o mesmo dict)
    _ERGAST_CACHE.update(load_ergast_cache())

    if PARTIAL_FILE.exists():
        df_all = pd.read_csv(PARTIAL_FILE)
        print(f"[INFO] Parcial carregado: {len(df_all)} linhas")
    else:
        df_all = pd.DataFrame()

    total_sessions = 0
    success = 0

    for year in YEARS:
        try:
            schedule = load_schedule_cached(year)
        except Exception as e:
            print(f"[ERRO] schedule {year}: {e}")
            continue
        for _, event in schedule.iterrows():
            gp = event["EventName"]
            round_number = int(event.get("RoundNumber", 0))
            if round_number == 0:
                continue
            for code in SESSION_CODES:
                total_sessions += 1
                key = session_key(year, gp, code)
                if key in completed:
                    print(f"[skip] {key} (checkpoint)")
                    continue
                print(f"\n[{total_sessions}] Processando {key} ...")
                try:
                    df = process_session(year, gp, code, round_number)
                    if len(df) == 0:
                        print(f"    [warn] 0 linhas extraídas")
                        state["failed"].append(key)
                    else:
                        df_all = pd.concat([df_all, df], ignore_index=True)
                        df_all.to_csv(PARTIAL_FILE, index=False)
                        state["completed"].append(key)
                        success += 1
                        print(f"    [OK] {len(df)} voltas | parcial total: {len(df_all)}")
                    save_checkpoint(state)
                except Exception:
                    print(f"    [FAIL TOTAL] {key}")
                    traceback.print_exc()
                    state["failed"].append(key)
                    save_checkpoint(state)

    if len(df_all) > 0:
        df_all.to_csv(FINAL_FILE, index=False)
        print(f"\n[OK] FINAL salvo em {FINAL_FILE}")
        print(f"     Total linhas: {len(df_all)}")
        print(f"     Sessões OK:   {success}/{total_sessions}")


if __name__ == "__main__":
    main()