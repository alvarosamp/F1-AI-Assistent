"""Volta ideal (teórica) — soma dos melhores tempos de setor 1/2/3 do piloto
na sessão, comparada com a melhor volta real que ele de fato cravou."""

from __future__ import annotations

import pandas as pd

SECTOR_COLUMNS = ["Sector1Time", "Sector2Time", "Sector3Time"]


def compute_ideal_lap(session, driver: str) -> dict:
    laps = session.laps.pick_drivers(driver.upper())
    if laps.empty:
        raise ValueError(f"Nenhuma volta encontrada para {driver}.")

    valid = laps.dropna(subset=SECTOR_COLUMNS)
    if valid.empty:
        return {"available": False, "driver": driver.upper(), "reason": "Sem tempos de setor completos nesta sessão."}

    best_sectors = []
    ideal_seconds = 0.0
    for i, col in enumerate(SECTOR_COLUMNS, start=1):
        best_row = valid.loc[valid[col].idxmin()]
        sector_s = best_row[col].total_seconds()
        ideal_seconds += sector_s
        best_sectors.append({
            "sector": i,
            "time_s": round(sector_s, 3),
            "lap_number": int(best_row["LapNumber"]),
        })

    actual_best = laps.pick_fastest()
    actual_seconds = None
    actual_lap_number = None
    if actual_best is not None and not actual_best.empty and pd.notna(actual_best.get("LapTime")):
        actual_seconds = actual_best["LapTime"].total_seconds()
        actual_lap_number = int(actual_best["LapNumber"])

    gap_s = round(actual_seconds - ideal_seconds, 3) if actual_seconds is not None else None

    return {
        "available": True,
        "driver": driver.upper(),
        "ideal_lap_s": round(ideal_seconds, 3),
        "actual_best_lap_s": round(actual_seconds, 3) if actual_seconds is not None else None,
        "actual_best_lap_number": actual_lap_number,
        "gap_to_ideal_s": gap_s,
        "sectors": best_sectors,
    }
