from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from backend.fastf1_shared import (
    AVAILABLE_YEARS,
    SESSION_CODES,
    driver_lap_telemetry,
    get_schedule,
    get_session_cached,
)

router = APIRouter(prefix="/api/engineering", tags=["engineering"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@router.get("/sessions")
async def list_sessions(year: int | None = None):
    if year is None:
        return {"years": AVAILABLE_YEARS, "session_codes": SESSION_CODES}

    if year not in AVAILABLE_YEARS:
        raise HTTPException(status_code=400, detail=f"Unsupported year: {year}")

    try:
        schedule = await run_in_threadpool(get_schedule, year)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load calendar: {exc}") from exc

    gps = [row for row in schedule["EventName"].tolist() if row]
    return {"year": year, "gps": gps, "session_codes": SESSION_CODES}


def _build_summary(year: int, gp: str, session_code: str) -> dict:
    session_obj = get_session_cached(year, gp, session_code)
    laps = session_obj.laps

    rows = []
    for driver in laps["Driver"].dropna().unique():
        driver_laps = laps.pick_drivers(driver)
        if driver_laps.empty:
            continue
        best = driver_laps.pick_fastest()
        has_best = best is not None and not best.empty and pd.notna(best.get("LapTime"))
        compounds = sorted({c for c in driver_laps["Compound"].dropna().unique()})
        rows.append(
            {
                "driver": driver,
                "best_lap_s": best["LapTime"].total_seconds() if has_best else None,
                "best_lap_number": int(best["LapNumber"]) if has_best else None,
                "laps_completed": int(driver_laps["LapNumber"].max()) if not driver_laps.empty else 0,
                "compounds": compounds,
            }
        )

    rows.sort(key=lambda r: (r["best_lap_s"] is None, r["best_lap_s"]))
    return {
        "year": year,
        "gp": gp,
        "session": session_code,
        "event_name": session_obj.event["EventName"],
        "drivers": rows,
    }


@router.get("/summary")
async def get_summary(year: int, gp: str, session: str):
    if session not in SESSION_CODES:
        raise HTTPException(status_code=400, detail=f"Invalid session: {session}")
    try:
        return await run_in_threadpool(_build_summary, year, gp, session)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load session: {exc}") from exc


def _build_telemetry(year: int, gp: str, session_code: str, driver: str, lap: int | None) -> dict:
    session_obj = get_session_cached(year, gp, session_code)
    return driver_lap_telemetry(session_obj, driver, lap)


@router.get("/telemetry")
async def get_telemetry(year: int, gp: str, session: str, driver: str, lap: int | None = None):
    if session not in SESSION_CODES:
        raise HTTPException(status_code=400, detail=f"Invalid session: {session}")
    try:
        return await run_in_threadpool(_build_telemetry, year, gp, session, driver, lap)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load telemetry: {exc}") from exc


def _select_driver_lap(session_obj, driver: str, lap_number: int | None):
    laps = session_obj.laps.pick_drivers(driver.upper())
    if laps.empty:
        raise ValueError(f"No laps found for {driver}.")
    if lap_number is None:
        return laps.pick_fastest()
    selected = laps[laps["LapNumber"] == lap_number]
    if selected.empty:
        raise ValueError(f"Lap {lap_number} not found for {driver}.")
    return selected.iloc[0]


def _numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


def _interpolate_on_grid(df: pd.DataFrame, column: str, grid: np.ndarray) -> list[float | None]:
    if column not in df or df[column].dropna().empty:
        return [None] * len(grid)

    x = _numeric_array(df["Distance"])
    y = _numeric_array(df[column])
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return [None] * len(grid)
    return np.interp(grid, x[mask], y[mask]).round(4).tolist()


def _comparison_samples(tel_a: pd.DataFrame, tel_b: pd.DataFrame) -> list[dict]:
    max_distance = float(min(tel_a["Distance"].max(), tel_b["Distance"].max()))
    grid = np.linspace(0.0, max_distance, 420)

    tel_a = tel_a.assign(TimeSeconds=tel_a["Time"].dt.total_seconds())
    tel_b = tel_b.assign(TimeSeconds=tel_b["Time"].dt.total_seconds())

    speed_a = _interpolate_on_grid(tel_a, "Speed", grid)
    speed_b = _interpolate_on_grid(tel_b, "Speed", grid)
    time_a = _interpolate_on_grid(tel_a, "TimeSeconds", grid)
    time_b = _interpolate_on_grid(tel_b, "TimeSeconds", grid)
    throttle_a = _interpolate_on_grid(tel_a, "Throttle", grid)
    throttle_b = _interpolate_on_grid(tel_b, "Throttle", grid)
    brake_a = _interpolate_on_grid(tel_a, "brake_pressure_proxy", grid)
    brake_b = _interpolate_on_grid(tel_b, "brake_pressure_proxy", grid)
    dirty_a = _interpolate_on_grid(tel_a, "dirty_air_score", grid)
    dirty_b = _interpolate_on_grid(tel_b, "dirty_air_score", grid)

    samples = []
    for idx, distance in enumerate(grid):
        va = speed_a[idx]
        vb = speed_b[idx]
        ta = time_a[idx]
        tb = time_b[idx]
        samples.append(
            {
                "Distance": round(float(distance), 2),
                "speed_a": va,
                "speed_b": vb,
                "speed_delta": None if va is None or vb is None else round(va - vb, 4),
                "time_delta": None if ta is None or tb is None else round(ta - tb, 4),
                "throttle_a": throttle_a[idx],
                "throttle_b": throttle_b[idx],
                "brake_a": brake_a[idx],
                "brake_b": brake_b[idx],
                "dirty_air_a": dirty_a[idx],
                "dirty_air_b": dirty_b[idx],
            }
        )
    return samples


def _automatic_analysis(driver_a: str, driver_b: str, tel_a: dict, tel_b: dict, samples: list[dict]) -> dict:
    lap_a = tel_a.get("lap_time_s")
    lap_b = tel_b.get("lap_time_s")
    faster = None
    gap_s = None
    if lap_a is not None and lap_b is not None:
        faster = driver_a if lap_a < lap_b else driver_b
        gap_s = round(abs(lap_a - lap_b), 3)

    valid_speed_deltas = [s for s in samples if s.get("speed_delta") is not None]
    strongest_speed = max(valid_speed_deltas, key=lambda row: abs(row["speed_delta"])) if valid_speed_deltas else None

    summary_a = tel_a.get("summary", {})
    summary_b = tel_b.get("summary", {})
    dirty_delta = round(summary_a.get("dirty_air_pct", 0) - summary_b.get("dirty_air_pct", 0), 2)
    brake_delta = round(summary_a.get("brake_pct", 0) - summary_b.get("brake_pct", 0), 2)
    throttle_delta = round(summary_a.get("full_throttle_pct", 0) - summary_b.get("full_throttle_pct", 0), 2)

    findings = []
    if faster and gap_s is not None:
        findings.append(f"{faster} foi mais rapido por {gap_s:.3f}s nesta volta.")
    if strongest_speed:
        leader = driver_a if strongest_speed["speed_delta"] > 0 else driver_b
        findings.append(
            f"Maior diferenca de velocidade favoreceu {leader} em {abs(strongest_speed['speed_delta']):.1f} km/h "
            f"perto de {strongest_speed['Distance']:.0f} m."
        )
    if abs(dirty_delta) > 3:
        exposed = driver_a if dirty_delta > 0 else driver_b
        findings.append(f"{exposed} teve mais exposicao a ar sujo, possivel impacto em pneu e ritmo.")
    if abs(brake_delta) > 2:
        heavier = driver_a if brake_delta > 0 else driver_b
        findings.append(f"{heavier} passou mais tempo em frenagem, sinal de maior carga em zonas lentas.")
    if abs(throttle_delta) > 2:
        cleaner = driver_a if throttle_delta > 0 else driver_b
        findings.append(f"{cleaner} teve maior percentual de acelerador cheio.")
    if not findings:
        findings.append("As voltas estao muito proximas; a decisao depende mais de stint, trafego e pneu.")

    return {
        "faster_driver": faster,
        "gap_s": gap_s,
        "max_speed_delta": strongest_speed,
        "dirty_air_delta_pct": dirty_delta,
        "brake_delta_pct": brake_delta,
        "full_throttle_delta_pct": throttle_delta,
        "findings": findings,
    }


def _copilot_text(analysis: dict) -> dict:
    risk_flags = []
    if abs(analysis.get("dirty_air_delta_pct", 0)) > 3:
        risk_flags.append("trafego/ar sujo")
    if abs(analysis.get("brake_delta_pct", 0)) > 2:
        risk_flags.append("carga de freio")
    if analysis.get("gap_s") is not None and analysis["gap_s"] < 0.15:
        risk_flags.append("margem pequena")

    return {
        "mode": "deterministic_fallback",
        "summary": " ".join(analysis.get("findings", [])[:3]),
        "recommendation": (
            "Use esta leitura como apoio: compare ritmo limpo, trafego, degradacao e janela de pit "
            "antes de recomendar undercut ou overcut."
        ),
        "risk_flags": risk_flags,
    }


def _compare_driver_laps(
    year: int,
    gp: str,
    session_code: str,
    driver_a: str,
    driver_b: str,
    lap_a: int | None,
    lap_b: int | None,
) -> dict:
    from telemetry.telemetry_signals import add_derived_signals, summarize_lap_signals

    session_obj = get_session_cached(year, gp, session_code)
    lap_row_a = _select_driver_lap(session_obj, driver_a, lap_a)
    lap_row_b = _select_driver_lap(session_obj, driver_b, lap_b)

    tel_df_a = add_derived_signals(lap_row_a.get_telemetry().add_distance())
    tel_df_b = add_derived_signals(lap_row_b.get_telemetry().add_distance())

    telemetry_a = driver_lap_telemetry(session_obj, driver_a, int(lap_row_a["LapNumber"]))
    telemetry_b = driver_lap_telemetry(session_obj, driver_b, int(lap_row_b["LapNumber"]))
    samples = _comparison_samples(tel_df_a, tel_df_b)
    analysis = _automatic_analysis(driver_a.upper(), driver_b.upper(), telemetry_a, telemetry_b, samples)

    summary_a = summarize_lap_signals(tel_df_a)
    summary_b = summarize_lap_signals(tel_df_b)
    return {
        "year": year,
        "gp": gp,
        "session": session_code,
        "event_name": session_obj.event["EventName"],
        "driver_a": telemetry_a,
        "driver_b": telemetry_b,
        "summary_delta": {
            key: round(summary_a.get(key, 0) - summary_b.get(key, 0), 4)
            for key in sorted(set(summary_a) | set(summary_b))
        },
        "comparison_samples": samples,
        "analysis": analysis,
        "copilot": _copilot_text(analysis),
    }


@router.get("/compare")
async def compare_telemetry(
    year: int,
    gp: str,
    session: str,
    driver_a: str,
    driver_b: str,
    lap_a: int | None = None,
    lap_b: int | None = None,
):
    if session not in SESSION_CODES:
        raise HTTPException(status_code=400, detail=f"Invalid session: {session}")
    try:
        return await run_in_threadpool(_compare_driver_laps, year, gp, session, driver_a, driver_b, lap_a, lap_b)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to compare telemetry: {exc}") from exc


@router.get("/drift")
async def get_drift_monitor(year: int = 2026):
    report = PROJECT_ROOT / "reports" / f"drift_monitor_{year}.json"
    markdown = PROJECT_ROOT / "reports" / f"drift_monitor_{year}.md"
    if not report.exists():
        fallback = PROJECT_ROOT / "reports" / "drift_monitor_2024.json"
        report = fallback if fallback.exists() else report
        markdown = PROJECT_ROOT / "reports" / "drift_monitor_2024.md"
    if not report.exists():
        raise HTTPException(status_code=404, detail="Drift report not found yet.")

    data = pd.read_json(report, typ="series").to_dict()
    md_text = markdown.read_text(encoding="utf-8") if markdown.exists() else ""
    return {"source": report.name, "report": data, "markdown": md_text[:4000]}
