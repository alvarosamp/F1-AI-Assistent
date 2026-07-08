"""
Extrai resultados reais de uma temporada via FastF1 (leve — sem telemetria completa)
e monta um CSV no mesmo schema usado por calibrate_simulate.py:

    gp, driver, team, grid_pos, quali_pos, gap_to_pole_ms, final_position, dnf

USO:
    python src/simulation/extract_season_results.py --season 2026

OUTPUT:
    data/processed/results_<season>_real.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fastf1
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def best_time_ms(row) -> float | None:
    for col in ("Q3", "Q2", "Q1"):
        val = row.get(col)
        if pd.notna(val):
            return val.total_seconds() * 1000
    return None


def extract_round(season: int, round_number: int, event_name: str) -> list[dict]:
    race = fastf1.get_session(season, round_number, "R")
    race.load(laps=False, telemetry=False, weather=False, messages=False)
    race_results = race.results

    try:
        quali = fastf1.get_session(season, round_number, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        quali_results = quali.results
        pole_time_ms = None
        for _, r in quali_results.sort_values("Position").iterrows():
            t = best_time_ms(r)
            if t is not None:
                pole_time_ms = t
                break
    except Exception:
        quali_results = None
        pole_time_ms = None

    rows = []
    for _, r in race_results.iterrows():
        drv = r["Abbreviation"]
        grid_pos = int(r["GridPosition"]) if pd.notna(r["GridPosition"]) else 20
        classified_pos = str(r.get("ClassifiedPosition", ""))
        dnf = 0 if classified_pos.isdigit() else 1
        final_position = int(classified_pos) if dnf == 0 else 99

        quali_pos = grid_pos
        gap_to_pole_ms = grid_pos * 200.0
        if quali_results is not None and drv in quali_results["Abbreviation"].values:
            qr = quali_results[quali_results["Abbreviation"] == drv].iloc[0]
            if pd.notna(qr.get("Position")):
                quali_pos = int(qr["Position"])
            t = best_time_ms(qr)
            if t is not None and pole_time_ms is not None:
                gap_to_pole_ms = t - pole_time_ms

        rows.append({
            "gp": event_name,
            "driver": drv,
            "team": r["TeamName"],
            "grid_pos": grid_pos,
            "quali_pos": quali_pos,
            "gap_to_pole_ms": max(gap_to_pole_ms, 0.0),
            "final_position": final_position,
            "dnf": dnf,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    fastf1.Cache.enable_cache(str(CACHE_DIR))

    schedule = fastf1.get_event_schedule(args.season)
    schedule = schedule[schedule["RoundNumber"] > 0]

    all_rows = []
    for _, ev in schedule.iterrows():
        round_number = int(ev["RoundNumber"])
        event_name = ev["EventName"]
        try:
            rows = extract_round(args.season, round_number, event_name)
        except Exception as e:
            print(f"  [SKIP] {event_name}: {e}")
            continue
        if rows:
            all_rows.extend(rows)
            n_dnf = sum(r["dnf"] for r in rows)
            print(f"  [OK] {event_name}: {len(rows)} drivers, {n_dnf} DNFs")

    if not all_rows:
        print("Nenhuma corrida disputada encontrada para essa temporada.")
        return

    rdf = pd.DataFrame(all_rows)
    out = PROJECT_ROOT / "data" / "processed" / f"results_{args.season}_real.csv"
    rdf.to_csv(out, index=False)
    print(f"\n[OK] {len(rdf)} entries, {rdf['dnf'].sum()} DNFs, {rdf['gp'].nunique()} GPs -> {out}")


if __name__ == "__main__":
    main()
