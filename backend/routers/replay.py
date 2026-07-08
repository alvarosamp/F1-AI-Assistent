import time
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.fastf1_shared import AVAILABLE_YEARS, driver_lap_telemetry, get_session_cached

router = APIRouter(prefix="/api/replay", tags=["replay"])

REPLAY_SESSION_CODES = ["Q", "R"]
SECONDS_PER_LAP = 3.0

_replays: dict[str, dict] = {}


@router.get("/sessions")
def list_replay_sessions():
    return {"years": AVAILABLE_YEARS, "session_codes": REPLAY_SESSION_CODES}


class StartReplayRequest(BaseModel):
    year: int
    gp: str
    session: str = "R"


def _prepare_replay(year: int, gp: str, session_code: str) -> dict:
    session = get_session_cached(year, gp, session_code)
    laps = session.laps

    total_laps = int(laps["LapNumber"].max()) if not laps.empty else 0

    per_lap: dict[int, list[dict]] = {}
    for lap_number in range(1, total_laps + 1):
        lap_rows = laps[laps["LapNumber"] == lap_number].copy()
        lap_rows = lap_rows.sort_values("Position")
        rows = []
        for _, lap in lap_rows.iterrows():
            lap_time = lap["LapTime"]
            rows.append({
                "driver": lap["Driver"],
                "position": int(lap["Position"]) if pd.notna(lap["Position"]) else None,
                "last_lap_s": lap_time.total_seconds() if pd.notna(lap_time) else None,
                "compound": lap.get("Compound"),
            })
        per_lap[lap_number] = rows

    return {
        "year": year,
        "gp": gp,
        "session_code": session_code,
        "event_name": session.event["EventName"],
        "total_laps": total_laps,
        "per_lap": per_lap,
        "started_at": time.time(),
    }


@router.post("/start")
async def start_replay(req: StartReplayRequest):
    if req.session not in REPLAY_SESSION_CODES:
        raise HTTPException(status_code=400, detail=f"Sessão inválida para replay: {req.session}")

    try:
        replay_state = await run_in_threadpool(_prepare_replay, req.year, req.gp, req.session)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao carregar sessão: {exc}") from exc

    replay_id = str(uuid.uuid4())
    _replays[replay_id] = replay_state
    return {"replay_id": replay_id, "total_laps": replay_state["total_laps"], "event_name": replay_state["event_name"]}


def _current_lap(replay_state: dict) -> int:
    elapsed = time.time() - replay_state["started_at"]
    lap = int(elapsed // SECONDS_PER_LAP) + 1
    return min(max(lap, 1), replay_state["total_laps"] or 1)


@router.get("/{replay_id}/state")
def get_replay_state(replay_id: str):
    replay_state = _replays.get(replay_id)
    if replay_state is None:
        raise HTTPException(status_code=404, detail="Replay não encontrado")

    lap_number = _current_lap(replay_state)
    finished = (time.time() - replay_state["started_at"]) >= replay_state["total_laps"] * SECONDS_PER_LAP

    return {
        "lap_number": lap_number,
        "total_laps": replay_state["total_laps"],
        "finished": finished,
        "standings": replay_state["per_lap"].get(lap_number, []),
    }


@router.get("/{replay_id}/telemetry/{driver}")
async def get_replay_telemetry(replay_id: str, driver: str):
    replay_state = _replays.get(replay_id)
    if replay_state is None:
        raise HTTPException(status_code=404, detail="Replay não encontrado")

    lap_number = _current_lap(replay_state)

    def _load():
        session = get_session_cached(replay_state["year"], replay_state["gp"], replay_state["session_code"])
        return driver_lap_telemetry(session, driver, lap_number)

    try:
        return await run_in_threadpool(_load)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao carregar telemetria: {exc}") from exc
