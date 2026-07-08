"""Conexão best-effort com o live timing real da F1 via fastf1.livetiming.

Só recebe dados quando existe uma sessão de F1 acontecendo AGORA. Fora
disso, o cliente conecta mas fica sem receber mensagens (session_active=False).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd

from backend.fastf1_shared import CACHE_DIR, driver_lap_telemetry, enable_cache

LIVE_DIR = CACHE_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

REFRESH_INTERVAL_S = 15
HEARTBEAT_TIMEOUT_S = 30


class LiveSession:
    def __init__(self, year: int, gp: str, session_code: str):
        self.year = year
        self.gp = gp
        self.session_code = session_code
        self.filename = str(LIVE_DIR / f"live_{year}_{gp.replace(' ', '_')}_{session_code}.txt")

        self._client = None
        self._client_thread: threading.Thread | None = None
        self._refresh_thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._last_session = None
        self._last_error: str | None = None

    def connect(self) -> None:
        from fastf1.livetiming.client import SignalRClient

        enable_cache()
        self._client = SignalRClient(self.filename, filemode="w", timeout=HEARTBEAT_TIMEOUT_S)

        def _run_client():
            try:
                self._client.start()
            except Exception as exc:  # pragma: no cover - best-effort live feed
                self._last_error = str(exc)

        self._client_thread = threading.Thread(target=_run_client, daemon=True)
        self._client_thread.start()

        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_loop(self) -> None:
        import fastf1
        from fastf1.livetiming.data import LiveTimingData

        while not self._stop.is_set():
            time.sleep(REFRESH_INTERVAL_S)
            if not Path(self.filename).exists():
                continue
            try:
                livedata = LiveTimingData(self.filename)
                session = fastf1.get_session(self.year, self.gp, self.session_code)
                session.load(laps=True, telemetry=True, weather=False, livedata=livedata)
                with self._lock:
                    self._last_session = session
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)

    @property
    def is_connected(self) -> bool:
        return bool(self._client and self._client._is_connected)

    @property
    def session_active(self) -> bool:
        if not self._client or self._client._t_last_message is None:
            return False
        return (time.time() - self._client._t_last_message) < HEARTBEAT_TIMEOUT_S

    def get_state(self) -> dict:
        with self._lock:
            session = self._last_session
            error = self._last_error

        if session is None:
            return {"available": False, "error": error, "standings": []}

        laps = session.laps
        if laps.empty:
            return {"available": False, "error": "Ainda sem voltas registradas", "standings": []}

        latest_lap = int(laps["LapNumber"].max())
        current = laps[laps["LapNumber"] == latest_lap].sort_values("Position")

        rows = []
        for _, lap in current.iterrows():
            lap_time = lap["LapTime"]
            rows.append({
                "driver": lap["Driver"],
                "position": int(lap["Position"]) if pd.notna(lap["Position"]) else None,
                "last_lap_s": lap_time.total_seconds() if pd.notna(lap_time) else None,
                "compound": lap.get("Compound"),
            })

        return {"available": True, "error": None, "lap_number": latest_lap, "standings": rows}

    def get_driver_telemetry(self, driver: str) -> dict:
        with self._lock:
            session = self._last_session

        if session is None:
            raise ValueError("Ainda não há dados de sessão ao vivo disponíveis.")

        return driver_lap_telemetry(session, driver)

    def stop(self) -> None:
        self._stop.set()


_live_session: LiveSession | None = None


def get_live_session() -> LiveSession | None:
    return _live_session


def start_live_session(year: int, gp: str, session_code: str) -> LiveSession:
    global _live_session
    if _live_session is not None:
        _live_session.stop()

    _live_session = LiveSession(year, gp, session_code)
    _live_session.connect()
    return _live_session
