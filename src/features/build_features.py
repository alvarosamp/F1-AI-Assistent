from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "telemetry_full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "telemetry_features.csv"


def main():
    df = pd.read_csv(RAW_FILE)

    # =============================
    # SORT (CRÍTICO)
    # =============================
    df = df.sort_values(["year", "gp", "session_code", "DriverNumber", "LapNumber"])

    group_cols = ["year", "gp", "session_code", "DriverNumber"]

    # =============================
    # ENCODING DE PNEU
    # =============================
    compound_map = {
        "SOFT": 0,
        "MEDIUM": 1,
        "HARD": 2,
        "INTERMEDIATE": 3,
        "WET": 4,
    }

    df["CompoundEncoded"] = (
        df["Compound"]
        .astype(str)
        .str.upper()
        .map(compound_map)
        .fillna(-1)
    )

    # =============================
    # HISTÓRICO DE PERFORMANCE
    # =============================

    df["lap_time_mean_3"] = (
        df.groupby(group_cols)["LapTime"]
        .rolling(3)
        .mean()
        .reset_index(level=group_cols, drop=True)
    )

    df["lap_time_delta"] = df.groupby(group_cols)["LapTime"].diff()

    # =============================
    # TELEMETRIA TEMPORAL (🔥 NOVO)
    # =============================

    df["speed_delta"] = df.groupby(group_cols)["speed_mean"].diff()
    df["throttle_delta"] = df.groupby(group_cols)["throttle_mean"].diff()
    df["brake_delta"] = df.groupby(group_cols)["brake_ratio"].diff()

    # =============================
    # DEGRADAÇÃO REAL (🔥 MUITO IMPORTANTE)
    # =============================

    df["degradation_score"] = (
        df["lap_time_delta"] +
        (-df["speed_delta"]) +
        df["brake_ratio"]
    )

    # =============================
    # COMPORTAMENTO DO PILOTO (🔥 DIFERENCIAL)
    # =============================

    # agressividade
    df["aggression_score"] = (
        df["throttle_mean"] *
        (1 - df["brake_ratio"])
    )

    # consistência
    df["consistency_score"] = (
        df.groupby(group_cols)["LapTime"]
        .rolling(5)
        .std()
        .reset_index(level=group_cols, drop=True)
    )

    # eficiência
    df["efficiency_score"] = df["speed_mean"] / (df["rpm_mean"] + 1)

    # =============================
    # USO DE DRS
    # =============================

    df["drs_usage_intensity"] = df["drs_ratio"] * df["speed_max"]

    # =============================
    # DESGASTE DE PNEU
    # =============================

    df["tyre_ratio"] = (
        df["TyreLife"] /
        df.groupby(group_cols)["TyreLife"].transform("max")
    )

    # =============================
    # PROGRESSO DO STINT
    # =============================

    df["stint_progress"] = (
        df.groupby(group_cols + ["Stint"]).cumcount() + 1
    )

    df["stint_progress"] = df["stint_progress"] / df.groupby(
        group_cols + ["Stint"]
    )["stint_progress"].transform("max")

    # =============================
    # TARGET (🔥 ESSENCIAL)
    # =============================

    df["session_baseline"] = df.groupby(
        ["year", "gp", "session_code"]
    )["LapTime"].transform("median")

    df["LapTimeResidual"] = df["LapTime"] - df["session_baseline"]

    # =============================
    # CLEAN
    # =============================

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n[INFO] BUILD FEATURES COMPLETO")
    print(df.shape)
    print(df.head())


if __name__ == "__main__":
    main()