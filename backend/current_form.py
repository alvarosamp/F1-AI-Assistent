"""Calcula 'forma atual' (pace_delta por piloto) a partir de treinos/quali reais,
em vez do CSV estático com placeholders manuais (configs/season_adaptation_2026.csv).

pace_delta = quão mais rápido/lento o piloto está NESTE fim de semana em relação
à média do grid, medido pelo gap para a volta mais rápida da sessão (Q > FP3 >
FP2 > FP1, na ordem de confiabilidade). Negativo = mais rápido que a média.
"""

from __future__ import annotations

import pandas as pd

from backend.fastf1_shared import get_session_cached

# (session_code, confidence): Q é o sinal mais limpo (sem programa de simulação
# de corrida/combustível variável misturando o ritmo); treinos livres pesam menos.
SESSION_PRIORITY = [("Q", 0.75), ("FP3", 0.55), ("FP2", 0.45), ("FP1", 0.35)]
MIN_DRIVERS_WITH_LAP = 3


def _best_lap_gaps(session) -> dict[str, float]:
    try:
        laps = session.laps
    except Exception:
        # fastf1 pode "carregar com sucesso" uma sessão futura sem levantar
        # exceção em .load(), mas ainda assim deixar .laps não populado.
        return {}
    if laps is None or laps.empty:
        return {}

    best_by_driver: dict[str, float] = {}
    for driver in laps["Driver"].dropna().unique():
        driver_laps = laps.pick_drivers(driver)
        fastest = driver_laps.pick_fastest()
        if fastest is None or fastest.empty or pd.isna(fastest.get("LapTime")):
            continue
        best_by_driver[driver] = fastest["LapTime"].total_seconds()

    if len(best_by_driver) < MIN_DRIVERS_WITH_LAP:
        return {}

    fastest_time = min(best_by_driver.values())
    return {driver: t - fastest_time for driver, t in best_by_driver.items()}


def compute_current_form(year: int, gp: str) -> tuple[dict[str, dict[str, float]], dict | None]:
    """Retorna (current_form, meta). current_form vazio se nenhuma sessão do
    fim de semana já tiver dados reais (ex: corrida futura sem treino ainda)."""

    for code, confidence in SESSION_PRIORITY:
        try:
            session = get_session_cached(year, gp, code)
        except Exception:
            continue

        gaps = _best_lap_gaps(session)
        if not gaps:
            continue

        avg_gap = sum(gaps.values()) / len(gaps)
        current_form = {
            driver: {"pace_delta": round(gap - avg_gap, 4), "confidence": confidence}
            for driver, gap in gaps.items()
        }
        meta = {
            "session": code,
            "year": year,
            "n_drivers": len(current_form),
        }
        return current_form, meta

    return {}, None
