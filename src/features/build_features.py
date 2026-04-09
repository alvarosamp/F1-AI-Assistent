"""
Feature engineering para modelagem de tempo de volta em F1.

REGRA DE OURO ANTI-LEAKAGE:
    Para prever a volta N, só podemos usar informação observável ATÉ o final
    da volta N-1. Toda feature derivada de LapTime, Speed, etc. da própria
    volta N é PROIBIDA, porque ela contém o próprio target.

DECISÕES ANTI-RUÍDO (validadas via diagnose.py):
    - Pre-Season Testing é REMOVIDO (dinâmica diferente).
    - Race e Qualifying viram DATASETS SEPARADOS (telemetry_features_race.csv
      e telemetry_features_quali.csv) — eles têm dinâmica fundamentalmente
      diferente e NÃO devem treinar o mesmo modelo.
    - Filter de LapTime baseado em IQR por sessão, não em corte fixo, porque
      Mônaco e Spa têm ritmos completamente diferentes.
    - Filter agressivo de outliers no residual: |residual| > 3s sai (em race;
      em quali o critério é mais frouxo porque a variância é legítima).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "telemetry_full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_RACE = PROCESSED_DIR / "telemetry_features_race.csv"
OUTPUT_QUALI = PROCESSED_DIR / "telemetry_features_quali.csv"

GROUP_COLS = ["year", "gp", "session_code", "DriverNumber"]

TELEMETRY_COLS = [
    "speed_mean", "speed_max", "speed_std",
    "throttle_mean", "throttle_std",
    "brake_ratio", "rpm_mean", "gear_mean", "drs_ratio",
]

COMPOUND_MAP = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4,
}

# GPs que NÃO são corridas reais
EXCLUDE_GPS = {"Pre-Season Testing"}


def encode_compound(df: pd.DataFrame) -> pd.DataFrame:
    df["CompoundEncoded"] = (
        df["Compound"].astype(str).str.upper().map(COMPOUND_MAP).fillna(-1)
    )
    return df


def add_lap_history_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(GROUP_COLS)["LapTime"]
    df["lap_time_prev"] = g.shift(1)

    df["lap_time_mean_3_prev"] = df.groupby(GROUP_COLS)["lap_time_prev"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["lap_time_delta_prev"] = df.groupby(GROUP_COLS)["lap_time_prev"].transform(
        lambda s: s.diff()
    )
    df["lap_time_std_5_prev"] = df.groupby(GROUP_COLS)["lap_time_prev"].transform(
        lambda s: s.rolling(5, min_periods=2).std()
    )
    return df


def add_telemetry_history_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in TELEMETRY_COLS:
        if col not in df.columns:
            continue
        prev_col = f"{col}_prev"
        df[prev_col] = df.groupby(GROUP_COLS)[col].shift(1)
        delta_col = f"{col}_delta_prev"
        df[delta_col] = df.groupby(GROUP_COLS)[prev_col].diff()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["degradation_score_prev"] = (
        df.get("lap_time_delta_prev", 0).fillna(0)
        + (-df.get("speed_mean_delta_prev", 0).fillna(0))
        + df.get("brake_ratio_prev", 0).fillna(0)
    )
    if "throttle_mean_prev" in df.columns and "brake_ratio_prev" in df.columns:
        df["aggression_score_prev"] = (
            df["throttle_mean_prev"] * (1 - df["brake_ratio_prev"])
        )
    if "speed_mean_prev" in df.columns and "rpm_mean_prev" in df.columns:
        df["efficiency_score_prev"] = (
            df["speed_mean_prev"] / (df["rpm_mean_prev"] + 1)
        )
    if "drs_ratio_prev" in df.columns and "speed_max_prev" in df.columns:
        df["drs_usage_intensity_prev"] = (
            df["drs_ratio_prev"] * df["speed_max_prev"]
        )
    df["consistency_score_prev"] = df["lap_time_std_5_prev"]
    return df


def add_tyre_and_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    df["tyre_ratio"] = (
        df["TyreLife"]
        / df.groupby(GROUP_COLS)["TyreLife"].transform("max")
    )
    df["stint_progress"] = (
        df.groupby(GROUP_COLS + ["Stint"]).cumcount() + 1
    )
    df["stint_progress"] = (
        df["stint_progress"]
        / df.groupby(GROUP_COLS + ["Stint"])["stint_progress"].transform("max")
    )
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df["session_baseline"] = df.groupby(
        ["year", "gp", "session_code"]
    )["LapTime"].transform("median")
    df["LapTimeResidual"] = df["LapTime"] - df["session_baseline"]
    return df


def filter_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Remove out/in laps, deletadas, GPs excluídos e LapTime nulo."""
    n0 = len(df)
    if "PitOutTime" in df.columns:
        df = df[df["PitOutTime"].isna()]
    if "PitInTime" in df.columns:
        df = df[df["PitInTime"].isna()]
    if "Deleted" in df.columns:
        df = df[~df["Deleted"].fillna(False).astype(bool)]
    df = df[~df["gp"].isin(EXCLUDE_GPS)]
    df = df[df["LapTime"].notna()]
    print(f"[BASIC FILTER] {n0} -> {len(df)} ({100*len(df)/max(n0,1):.1f}%)")
    return df


def filter_iqr_per_session(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """
    Filter por IQR DENTRO de cada sessão. Mais robusto que corte fixo, porque
    Mônaco (~75s) e Spa (~106s) têm distribuições completamente diferentes.
    Remove tudo abaixo de Q1-k*IQR e acima de Q3+k*IQR. k=1.5 é o padrão de
    boxplot — pega outliers reais sem cortar voltas legítimas.
    """
    n0 = len(df)
    session_keys = ["year", "gp", "session_code"]

    # Usa transform pra calcular Q1/Q3/IQR por sessão sem dropar colunas
    q1 = df.groupby(session_keys)["LapTime"].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby(session_keys)["LapTime"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr

    df = df[(df["LapTime"] >= lo) & (df["LapTime"] <= hi)].copy()
    print(f"[IQR FILTER]   {n0} -> {len(df)} ({100*len(df)/max(n0,1):.1f}%)")
    return df


def filter_residual(df: pd.DataFrame, max_abs_residual: float) -> pd.DataFrame:
    """
    Corte final pelo residual. Em race, qualquer volta com |residual| > 3s
    é claramente uma volta com problema (tráfego severo, dano, in lap mal
    detectada, SC mascarado). Em quali o limite é mais frouxo porque
    out/cool laps em quali são parte da estratégia.
    """
    n0 = len(df)
    df = df[df["LapTimeResidual"].abs() <= max_abs_residual]
    print(f"[RESIDUAL <{max_abs_residual}s] {n0} -> {len(df)} ({100*len(df)/max(n0,1):.1f}%)")
    return df


def process(df: pd.DataFrame, label: str, max_abs_residual: float) -> pd.DataFrame:
    print(f"\n=== PROCESSANDO {label.upper()} ===")
    df = df.sort_values(GROUP_COLS + ["LapNumber"]).reset_index(drop=True)
    df = filter_iqr_per_session(df, k=1.5)
    df = encode_compound(df)
    df = add_lap_history_features(df)
    df = add_telemetry_history_features(df)
    df = add_derived_features(df)
    df = add_tyre_and_stint_features(df)
    df = build_target(df)
    df = filter_residual(df, max_abs_residual=max_abs_residual)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["LapTimeResidual"])
    return df


def main() -> None:
    df = pd.read_csv(RAW_FILE)
    df = filter_basic(df)

    df_race = df[df["session_code"] == "R"].copy()
    df_quali = df[df["session_code"] == "Q"].copy()

    print(f"\nRace raw:  {len(df_race)} voltas")
    print(f"Quali raw: {len(df_quali)} voltas")

    df_race_out = process(df_race, "race", max_abs_residual=3.0)
    df_quali_out = process(df_quali, "quali", max_abs_residual=5.0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_race_out.to_csv(OUTPUT_RACE, index=False)
    df_quali_out.to_csv(OUTPUT_QUALI, index=False)

    print(f"\n[OK] Race  -> {OUTPUT_RACE.name}  shape={df_race_out.shape}")
    print(f"[OK] Quali -> {OUTPUT_QUALI.name} shape={df_quali_out.shape}")
    print(f"\nResidual stats (race):")
    print(df_race_out["LapTimeResidual"].describe(percentiles=[.01,.5,.99]))


if __name__ == "__main__":
    main()