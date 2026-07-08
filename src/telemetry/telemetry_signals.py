from __future__ import annotations

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "Speed",
    "Throttle",
    "Brake",
    "brake_pressure_proxy",
    "RPM",
    "nGear",
    "DRS",
    "DistanceToDriverAhead",
    "dirty_air_score",
    "steering_proxy",
    "lateral_change",
    "accel_proxy",
]


def add_derived_signals(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Add useful engineering-style signals to a FastF1 telemetry frame.

    FastF1 public data usually does not include real steering angle. The
    steering_proxy signal below is derived from X/Y trajectory changes:
    positive and negative values indicate the car rotating in opposite
    directions along the lap.
    """
    df = telemetry.copy()

    if "Distance" not in df.columns:
        raise ValueError("Telemetry must include Distance. Use add_distance() first.")

    if {"X", "Y"}.issubset(df.columns):
        dx = df["X"].astype(float).diff()
        dy = df["Y"].astype(float).diff()
        raw_heading = pd.Series(np.arctan2(dy, dx), index=df.index).bfill().ffill().fillna(0.0)
        heading = np.unwrap(raw_heading.to_numpy())
        df["steering_proxy"] = pd.Series(heading, index=df.index).diff().fillna(0.0)
        df["lateral_change"] = df["steering_proxy"].rolling(5, min_periods=1).mean()
    else:
        df["steering_proxy"] = 0.0
        df["lateral_change"] = 0.0

    if "Brake" in df.columns:
        df["Brake"] = df["Brake"].astype(int)
        df["brake_pressure_proxy"] = df["Brake"] * 100

    if "Throttle" in df.columns:
        df["Throttle"] = pd.to_numeric(df["Throttle"], errors="coerce").fillna(0)

    if "Speed" in df.columns:
        df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce").ffill()
        df["accel_proxy"] = df["Speed"].diff().fillna(0.0)

    if "DistanceToDriverAhead" in df.columns:
        distance = pd.to_numeric(df["DistanceToDriverAhead"], errors="coerce")
        distance = distance.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        df["DistanceToDriverAhead"] = distance
        df["dirty_air_score"] = ((120.0 - distance) / 120.0).clip(lower=0.0, upper=1.0)
    else:
        df["DistanceToDriverAhead"] = np.nan
        df["dirty_air_score"] = 0.0

    return df


def summarize_lap_signals(telemetry: pd.DataFrame) -> dict[str, float]:
    """Return compact metrics that can later be sent to an LLM."""
    df = telemetry.copy()
    summary: dict[str, float] = {}

    if "Speed" in df:
        summary["max_speed"] = float(df["Speed"].max())
        summary["avg_speed"] = float(df["Speed"].mean())
        summary["min_speed"] = float(df["Speed"].min())

    if "Throttle" in df:
        summary["avg_throttle"] = float(df["Throttle"].mean())
        summary["full_throttle_pct"] = float((df["Throttle"] >= 95).mean() * 100)

    if "Brake" in df:
        summary["brake_pct"] = float((df["Brake"] > 0).mean() * 100)

    if "DistanceToDriverAhead" in df and df["DistanceToDriverAhead"].notna().any():
        summary["avg_distance_to_car_ahead"] = float(df["DistanceToDriverAhead"].mean())
        summary["dirty_air_pct"] = float((df["dirty_air_score"] > 0.25).mean() * 100)

    if "DRS" in df:
        summary["drs_pct"] = float((df["DRS"] > 0).mean() * 100)

    if "nGear" in df:
        summary["avg_gear"] = float(df["nGear"].mean())

    if "steering_proxy" in df:
        summary["cornering_intensity"] = float(df["steering_proxy"].abs().mean())

    return summary
