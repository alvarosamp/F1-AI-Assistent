from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.fastf1_shared import SESSION_CODES, driver_lap_telemetry, get_session_cached
from backend.ideal_lap import compute_ideal_lap
from backend.race_engineer import find_driver_ahead, generate_briefing, recommend_strategy
from backend.routers.engineering import (
    _automatic_analysis,
    _comparison_samples,
    _select_driver_lap,
)
from telemetry.telemetry_signals import add_derived_signals

router = APIRouter(prefix="/api/race-engineer", tags=["race-engineer"])


def _build_analysis(year: int, gp: str, session_code: str, driver: str, lap: int | None) -> dict:
    session_obj = get_session_cached(year, gp, session_code)
    lap_row = _select_driver_lap(session_obj, driver, lap)
    lap_number = int(lap_row["LapNumber"])

    ideal = compute_ideal_lap(session_obj, driver)
    ahead = find_driver_ahead(session_obj, driver, lap_number)
    strategy = recommend_strategy(session_obj, driver, lap_number)

    comparison = None
    if ahead and ahead.get("driver_ahead"):
        try:
            ahead_lap_row = _select_driver_lap(session_obj, ahead["driver_ahead"], lap_number)
            tel_a = add_derived_signals(lap_row.get_telemetry().add_distance())
            tel_b = add_derived_signals(ahead_lap_row.get_telemetry().add_distance())
            telemetry_a = driver_lap_telemetry(session_obj, driver, lap_number)
            telemetry_b = driver_lap_telemetry(session_obj, ahead["driver_ahead"], int(ahead_lap_row["LapNumber"]))
            samples = _comparison_samples(tel_a, tel_b)
            analysis = _automatic_analysis(driver.upper(), ahead["driver_ahead"], telemetry_a, telemetry_b, samples)
            comparison = {
                "driver_ahead": ahead["driver_ahead"],
                "comparison_samples": samples,
                "analysis": analysis,
            }
        except Exception:
            comparison = None

    return {
        "year": year,
        "gp": gp,
        "session": session_code,
        "driver": driver.upper(),
        "lap_number": lap_number,
        "ideal_lap": ideal,
        "ahead": ahead,
        "strategy": strategy,
        "comparison_vs_ahead": comparison,
    }


@router.get("/analysis")
async def get_analysis(year: int, gp: str, session: str, driver: str, lap: int | None = None):
    if session not in SESSION_CODES:
        raise HTTPException(status_code=400, detail=f"Sessão inválida: {session}")
    try:
        return await run_in_threadpool(_build_analysis, year, gp, session, driver, lap)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao montar análise: {exc}") from exc


class BriefingRequest(BaseModel):
    year: int
    gp: str
    session: str
    driver: str
    lap: int | None = None


@router.post("/briefing")
async def post_briefing(req: BriefingRequest):
    if req.session not in SESSION_CODES:
        raise HTTPException(status_code=400, detail=f"Sessão inválida: {req.session}")
    try:
        context = await run_in_threadpool(_build_analysis, req.year, req.gp, req.session, req.driver, req.lap)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao montar análise: {exc}") from exc

    briefing = await run_in_threadpool(generate_briefing, context)
    return {**context, "briefing": briefing}
