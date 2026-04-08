from __future__ import annotations

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path

YEAR = 2023

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "telemetry_full.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def ensure_dirs():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_time(x):
    try:
        return x.total_seconds()
    except:
        return None


def extract_telemetry_features(lap):
    try:
        tel = lap.get_car_data().add_distance()

        return {
            "speed_mean": tel["Speed"].mean(),
            "speed_max": tel["Speed"].max(),
            "speed_std": tel["Speed"].std(),

            "throttle_mean": tel["Throttle"].mean(),
            "throttle_std": tel["Throttle"].std(),

            "brake_ratio": (tel["Brake"] > 0).mean(),

            "rpm_mean": tel["RPM"].mean(),
            "gear_mean": tel["nGear"].mean(),

            "drs_ratio": (tel["DRS"] > 0).mean(),
        }

    except:
        return {
            "speed_mean": np.nan,
            "speed_max": np.nan,
            "speed_std": np.nan,
            "throttle_mean": np.nan,
            "throttle_std": np.nan,
            "brake_ratio": np.nan,
            "rpm_mean": np.nan,
            "gear_mean": np.nan,
            "drs_ratio": np.nan,
        }


def process_session(year, gp, session_code):
    print(f"[INFO] {gp} - {session_code}")

    session = fastf1.get_session(year, gp, session_code)
    session.load()

    laps = session.laps.dropna(subset=["LapTime", "Driver"])

    rows = []

    for _, lap in laps.iterlaps():
        base = {
            "Driver": lap["Driver"],
            "DriverNumber": lap["DriverNumber"],
            "Team": lap["Team"],
            "LapNumber": lap["LapNumber"],
            "Stint": lap["Stint"],
            "Compound": lap["Compound"],
            "TyreLife": lap["TyreLife"],
            "Position": lap["Position"],
            "TrackStatus": lap["TrackStatus"],
            "LapTime": safe_time(lap["LapTime"]),
            "Sector1Time": safe_time(lap["Sector1Time"]),
            "Sector2Time": safe_time(lap["Sector2Time"]),
            "Sector3Time": safe_time(lap["Sector3Time"]),
            "year": year,
            "gp": gp,
            "session_code": session_code,
        }

        telemetry = extract_telemetry_features(lap)

        rows.append({**base, **telemetry})

    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    schedule = fastf1.get_event_schedule(YEAR)

    all_data = []

    for _, row in schedule.iterrows():
        gp = row["EventName"]

        for session_code in ["R", "Q"]:
            try:
                df = process_session(YEAR, gp, session_code)
                all_data.append(df)
            except Exception as e:
                print(f"[ERRO] {gp} {session_code}: {e}")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== FINAL =====")
    print(final_df.shape)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()