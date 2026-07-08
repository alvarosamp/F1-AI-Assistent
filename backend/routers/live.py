from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.live_session import get_live_session, start_live_session

router = APIRouter(prefix="/api/live", tags=["live"])


class ConnectRequest(BaseModel):
    year: int
    gp: str
    session: str = "R"


@router.post("/connect")
async def connect(req: ConnectRequest):
    try:
        await run_in_threadpool(start_live_session, req.year, req.gp, req.session)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao conectar no live timing: {exc}") from exc
    return {"connecting": True}


@router.get("/status")
def status():
    live = get_live_session()
    if live is None:
        return {"connected": False, "session_active": False}
    return {"connected": live.is_connected, "session_active": live.session_active}


@router.get("/state")
def state():
    live = get_live_session()
    if live is None:
        raise HTTPException(status_code=404, detail="Nenhuma sessão ao vivo iniciada")
    return live.get_state()


@router.get("/telemetry/{driver}")
async def telemetry(driver: str):
    live = get_live_session()
    if live is None:
        raise HTTPException(status_code=404, detail="Nenhuma sessão ao vivo iniciada")
    try:
        return await run_in_threadpool(live.get_driver_telemetry, driver)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
