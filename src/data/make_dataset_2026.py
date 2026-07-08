"""
Coleta telemetria completa de 2026 (mesma estrutura de make_dataset_v2.py:
laps + telemetria por volta + weather + safety car/VSC/yellow flags + Ergast
quali/grid) reaproveitando `process_session` do coletor v2.

Gera data/raw/telemetry_full_2026.csv, no mesmo schema de telemetry_full_v2.csv,
pra poder ser concatenado e alimentar build_dnf_dataset.py / features com
regulation_era=3 (2026 = motores híbridos novos, chassis ativo-aero).

USO:
    python src/data/make_dataset_2026.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import fastf1
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.make_dataset_v2 import (  # noqa: E402
    CACHE_DIR, RAW_DIR, process_session, load_schedule_cached, _ERGAST_CACHE,
    load_ergast_cache,
)

YEAR = 2026
SESSION_CODES = ["R", "Q"]
OUT_FILE = RAW_DIR / "telemetry_full_2026.csv"


def main() -> None:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    _ERGAST_CACHE.update(load_ergast_cache())

    schedule = load_schedule_cached(YEAR)
    schedule = schedule[schedule["RoundNumber"] > 0]

    df_all = pd.DataFrame()
    for _, event in schedule.iterrows():
        gp = event["EventName"]
        round_number = int(event.get("RoundNumber", 0))
        gp_frames = []
        for code in SESSION_CODES:
            try:
                df = process_session(YEAR, gp, code, round_number)
            except Exception as e:
                print(f"  [SKIP] {gp} {code}: {e}")
                continue
            if len(df) == 0:
                print(f"  [SKIP] {gp} {code}: 0 linhas (corrida ainda não disputada?)")
                continue
            gp_frames.append(df)
            print(f"  [OK] {gp} {code}: {len(df)} linhas")
        if gp_frames:
            df_all = pd.concat([df_all] + gp_frames, ignore_index=True)
            df_all.to_csv(OUT_FILE, index=False)

    if len(df_all) > 0:
        print(f"\n[OK] {len(df_all)} linhas totais -> {OUT_FILE}")
        print(f"     GPs coletados: {df_all['gp'].nunique()}")
    else:
        print("\n[FAIL] Nenhum dado coletado.")


if __name__ == "__main__":
    main()
