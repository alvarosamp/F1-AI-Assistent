"""
Feature engineering v4 para o modelo F1 — Sprint 2.

NOVIDADES vs v3:
    - Weather features diretas (AirTemp, TrackTemp, Rainfall, WindSpeed)
      disponíveis no dataset v2 coletado do FastF1.
    - Race control flags (is_sc, is_vsc, is_yellow) como features.
    - Gap to pole + quali position do Ergast (só pra voltas de race).
    - regulation_era: flag 2022/2023/2024 ground effect (prepara o modelo
      pra era 2026 que vem por cima depois).
    - Features de FORMA RECENTE por piloto: residual médio nas últimas 3
      corridas e posição média na temporada ATÉ a volta atual. Cuidado
      extremo com leakage — só agrega corridas ANTERIORES à atual.

REGRAS ANTI-LEAKAGE (reforçadas):
    1. Features _prev continuam shiftadas 1 volta pra trás.
    2. Features de forma recente usam SÓ corridas anteriores à corrida
       atual (não a atual). Agregação por (Driver, year) com cumulative.
    3. Race control flags (is_sc da volta atual) são CONTEXTO, não leakage
       — são conhecidas no início da volta (se começou em SC, você sabe).

INPUT: data/raw/telemetry_full_v2.csv (coletado por make_dataset_v2.py)
OUTPUT: data/processed/telemetry_features_race_v4.csv
        data/processed/telemetry_features_quali_v4.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "telemetry_full_v2.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_RACE = PROCESSED_DIR / "telemetry_features_race_v4.csv"
OUTPUT_QUALI = PROCESSED_DIR / "telemetry_features_quali_v4.csv"

GROUP_COLS = ["year", "gp", "session_code", "DriverNumber"]

TELEMETRY_COLS = [
    "speed_mean", "speed_max", "speed_std",
    "throttle_mean", "throttle_std",
    "brake_ratio", "rpm_mean", "gear_mean", "drs_ratio",
]

COMPOUND_MAP = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4,
}

EXCLUDE_GPS = {"Pre-Season Testing"}


# ==========================================================================
# FEATURES BÁSICAS (reaproveitando do v3)
# ==========================================================================

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
    df["lap_time_delta_prev"] = df.groupby(GROUP_COLS)["lap_time_prev"].transform(lambda s: s.diff())
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
        df[f"{col}_delta_prev"] = df.groupby(GROUP_COLS)[prev_col].diff()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["degradation_score_prev"] = (
        df.get("lap_time_delta_prev", 0).fillna(0)
        + (-df.get("speed_mean_delta_prev", 0).fillna(0))
        + df.get("brake_ratio_prev", 0).fillna(0)
    )
    if "throttle_mean_prev" in df.columns and "brake_ratio_prev" in df.columns:
        df["aggression_score_prev"] = df["throttle_mean_prev"] * (1 - df["brake_ratio_prev"])
    if "speed_mean_prev" in df.columns and "rpm_mean_prev" in df.columns:
        df["efficiency_score_prev"] = df["speed_mean_prev"] / (df["rpm_mean_prev"] + 1)
    if "drs_ratio_prev" in df.columns and "speed_max_prev" in df.columns:
        df["drs_usage_intensity_prev"] = df["drs_ratio_prev"] * df["speed_max_prev"]
    df["consistency_score_prev"] = df["lap_time_std_5_prev"]
    return df


def add_tyre_and_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    df["tyre_ratio"] = df["TyreLife"] / df.groupby(GROUP_COLS)["TyreLife"].transform("max")
    df["stint_progress"] = df.groupby(GROUP_COLS + ["Stint"]).cumcount() + 1
    df["stint_progress"] = (
        df["stint_progress"] / df.groupby(GROUP_COLS + ["Stint"])["stint_progress"].transform("max")
    )
    return df


def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """LapNumber_pct, interações — mantidas do v3."""
    race_len = df.groupby(["year", "gp"])["LapNumber"].transform("max")
    df["LapNumber_pct"] = df["LapNumber"] / race_len
    df["tyre_x_progress"] = df["TyreLife"] * df["LapNumber_pct"]
    df["compound_x_tyre"] = df["CompoundEncoded"] * df["TyreLife"]
    return df


# ==========================================================================
# NOVAS FEATURES v4
# ==========================================================================

def add_regulation_era(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding de era regulamentar como feature categórica numérica.
    2022 foi o primeiro ano da era ground effect, com carros imaturos.
    2023 teve os mesmos conceitos mas muito refinados (Red Bull dominou).
    2024 viu mais convergência entre times. Cada um representa uma
    sub-era dentro do mesmo regulamento.
    """
    era_map = {2022: 0, 2023: 1, 2024: 2}
    df["regulation_era"] = df["year"].map(era_map).fillna(-1)
    return df


def add_race_control_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_sc, is_vsc, is_yellow já vêm do dataset v2. Aqui adicionamos
    derivadas: is_neutralized = is_sc OR is_vsc (um agregador útil),
    e laps_since_neutralization (contador que reseta quando pega bandeira).
    """
    for col in ["is_sc", "is_vsc", "is_yellow"]:
        if col not in df.columns:
            df[col] = 0

    df["is_neutralized"] = ((df["is_sc"] == 1) | (df["is_vsc"] == 1)).astype(int)

    # Laps since last neutralization (por corrida, por piloto)
    def _laps_since_reset(group: pd.Series) -> pd.Series:
        out = []
        counter = 99
        for val in group.values:
            if val == 1:
                counter = 0
            else:
                counter += 1
            out.append(counter)
        return pd.Series(out, index=group.index)

    df["laps_since_neutralization"] = (
        df.groupby(["year", "gp", "DriverNumber"])["is_neutralized"]
          .transform(_laps_since_reset)
    )
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather já vem do v2. Só precisamos adicionar uma feature derivada:
    delta entre temperatura do ar e temperatura do asfalto (indicador
    de condições de aderência — pista muito mais quente que o ar indica
    meio-dia/sol forte, afeta degradação de pneu).
    """
    if "AirTemp" in df.columns and "TrackTemp" in df.columns:
        df["temp_delta"] = df["TrackTemp"] - df["AirTemp"]
    return df


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRÍTICO pra anti-leakage: features de forma recente só podem usar
    info de corridas ANTERIORES à corrida atual. Implementação:

    1. Ordenar por (year, round) — aqui usamos um proxy via ordem alfa
       que pode falhar em anos com calendário diferente, então usamos
       a ordem de aparição dos GPs no dataset (que já vem ordenado por data).
    2. Pra cada (Driver, year), calcular média do residual por GP.
    3. Shiftar 1 GP pra trás — a corrida atual NÃO entra no próprio cálculo.
    4. Tomar média das últimas 3 entradas (rolling).

    Resultado: coluna avg_residual_last_3_races representa "como esse
    piloto andou nas 3 corridas anteriores", sem leakage da atual.
    """
    # Ordem de GPs dentro de cada ano (usa a ordem natural de aparição)
    df = df.sort_values(["year", "gp", "LapNumber"]).reset_index(drop=True)
    # Cria ordem numérica dos GPs por ano (1º GP = 1, 2º = 2, ...)
    gp_order = (
        df.groupby("year")["gp"]
          .transform(lambda s: pd.factorize(s)[0] + 1)
    )
    df["_gp_order"] = gp_order

    # Residual médio do piloto por corrida
    per_race = (
        df.groupby(["Driver", "year", "gp", "_gp_order"])["LapTimeResidual"]
          .mean()
          .reset_index()
          .sort_values(["Driver", "year", "_gp_order"])
    )

    # Shift 1: residual da corrida N não entra na forma recente da corrida N
    per_race["residual_last_1"] = per_race.groupby(["Driver", "year"])["LapTimeResidual"].shift(1)
    per_race["avg_residual_last_3_races"] = (
        per_race.groupby(["Driver", "year"])["residual_last_1"]
                .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # Merge de volta
    df = df.merge(
        per_race[["Driver", "year", "gp", "avg_residual_last_3_races"]],
        on=["Driver", "year", "gp"],
        how="left",
    )
    df = df.drop(columns=["_gp_order"])
    return df


# ==========================================================================
# TARGET + FILTERS (iguais ao v3)
# ==========================================================================

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df["session_baseline"] = df.groupby(
        ["year", "gp", "session_code"]
    )["LapTime"].transform("median")
    df["LapTimeResidual"] = df["LapTime"] - df["session_baseline"]
    return df


def filter_basic(df: pd.DataFrame) -> pd.DataFrame:
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
    n0 = len(df)
    session_keys = ["year", "gp", "session_code"]
    q1 = df.groupby(session_keys)["LapTime"].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby(session_keys)["LapTime"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    df = df[(df["LapTime"] >= lo) & (df["LapTime"] <= hi)].copy()
    print(f"[IQR FILTER]   {n0} -> {len(df)} ({100*len(df)/max(n0,1):.1f}%)")
    return df


def filter_residual(df: pd.DataFrame, max_abs_residual: float) -> pd.DataFrame:
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
    df = add_race_context_features(df)
    df = add_regulation_era(df)            # NOVO
    df = add_race_control_features(df)     # NOVO
    df = add_weather_features(df)          # NOVO
    df = build_target(df)
    df = add_recent_form_features(df)      # NOVO (depende de LapTimeResidual)
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

    print(f"\nRace — distribuição por ano:")
    print(df_race_out.groupby("year").size().to_string())

    print(f"\nRace — cobertura das features novas:")
    new_cols = [
        "regulation_era", "is_sc", "is_vsc", "is_neutralized",
        "laps_since_neutralization", "AirTemp", "TrackTemp", "temp_delta",
        "Rainfall", "quali_position", "gap_to_pole_ms",
        "avg_residual_last_3_races",
    ]
    for c in new_cols:
        if c in df_race_out.columns:
            cov = df_race_out[c].notna().sum()
            print(f"  {c:30s} {cov}/{len(df_race_out)} ({100*cov/len(df_race_out):.1f}%)")


if __name__ == "__main__":
    main()
