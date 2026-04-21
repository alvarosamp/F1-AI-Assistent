"""
Construção do dataset de DNF (Did Not Finish) pro Sprint 3 — Entrega 1.

O QUE FAZ:
    Lê data/raw/telemetry_full_v2.csv (voltas de corrida).
    Agrega por (year, gp, Driver) e gera:
        - label `dnf`: 1 se piloto completou < 90% das voltas da corrida
        - features históricas shiftadas anti-leakage

CRITÉRIO DE DNF:
    "Completou menos de 90% das voltas do vencedor"
    Captura: abandonos mecânicos + carros gravemente atrasados (10+ voltas atrás).
    Taxa global: ~11.67% (medido empiricamente nos dados 2022-2024).

ANTI-LEAKAGE:
    Features de taxa histórica usam `expanding().mean().shift(1)`:
        - expanding: agrega todas as corridas ANTES desta dentro do ano
        - shift(1): a corrida atual NUNCA entra no cálculo da sua própria feature
    Ordem cronológica preservada via factorize por ano (que mantém ordem de
    aparição no dataset, que já vem ordenado por data).

OUTPUT:
    data/processed/dnf_dataset.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "telemetry_full_v2.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dnf_dataset.csv"

DNF_THRESHOLD_PCT = 0.90  # piloto "terminou" se completou >= 90% das voltas


def build_base_entries(df_race: pd.DataFrame) -> pd.DataFrame:
    """Agrega voltas em (race, driver) e calcula label DNF."""
    race_max_laps = df_race.groupby(["year", "gp"])["LapNumber"].max()
    driver_max_laps = df_race.groupby(["year", "gp", "Driver"])["LapNumber"].max()
    first_lap = df_race.groupby(["year", "gp", "Driver"]).first()

    entries = []
    for (year, gp, driver), first_row in first_lap.iterrows():
        race_total = race_max_laps.get((year, gp), 0)
        driver_last = driver_max_laps.get((year, gp, driver), 0)
        if race_total == 0:
            continue
        completion = driver_last / race_total
        dnf = int(completion < DNF_THRESHOLD_PCT)

        entries.append({
            "year": year,
            "gp": gp,
            "Driver": driver,
            "Team": first_row["Team"],
            "grid_position": first_row.get("grid_position", np.nan),
            "quali_position": first_row.get("quali_position", np.nan),
            "driver_last_lap": int(driver_last),
            "race_total_laps": int(race_total),
            "lap_completion_pct": completion,
            "dnf": dnf,
        })

    return pd.DataFrame(entries)


def add_historical_rates(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Features de taxa histórica (expanding mean + shift).

    IMPORTANTE: ordem cronológica correta é CRÍTICA pra não vazar.
    Usamos `factorize` por ano que preserva a ordem de primeira aparição
    (que é a ordem do calendário no raw do FastF1).
    """
    ds = ds.copy()
    ds["_gp_order"] = ds.groupby("year")["gp"].transform(lambda s: pd.factorize(s)[0] + 1)
    ds = ds.sort_values(["year", "_gp_order"]).reset_index(drop=True)

    def expanding_shifted(group: pd.Series) -> pd.Series:
        return group.expanding().mean().shift(1)

    ds["dnf_rate_driver"] = ds.groupby("Driver")["dnf"].transform(expanding_shifted)
    ds["dnf_rate_team"] = ds.groupby("Team")["dnf"].transform(expanding_shifted)
    ds["dnf_rate_gp"] = ds.groupby("gp")["dnf"].transform(expanding_shifted)

    ds["regulation_era"] = ds["year"].map({2022: 0, 2023: 1, 2024: 2})

    ds = ds.drop(columns=["_gp_order"])
    return ds


def main() -> None:
    print(f"Carregando: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    df_race = df[df["session_code"] == "R"].copy()
    print(f"Voltas de race: {len(df_race)}")

    ds = build_base_entries(df_race)
    print(f"\n(race, driver) pairs: {len(ds)}")
    print(f"DNF rate global: {ds['dnf'].mean()*100:.2f}%")

    ds = add_historical_rates(ds)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Dataset salvo em {OUTPUT_FILE}")
    print(f"Shape: {ds.shape}")

    print(f"\nDNF rate por ano:")
    print(ds.groupby("year")["dnf"].agg(["mean", "sum", "count"]).to_string())

    print(f"\nFeatures históricas (NaN counts — esperado nas primeiras corridas):")
    print(ds[["dnf_rate_driver", "dnf_rate_team", "dnf_rate_gp"]].isna().sum().to_string())


if __name__ == "__main__":
    main()