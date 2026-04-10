"""
Testes anti-leakage do build_features v3.

Estratégia: criamos um stint sintético, rodamos o pipeline, e verificamos:
    1. Embaralhar LapTime das voltas >= K NÃO muda as features das voltas < K
    2. Features _prev são NaN na primeira volta (não há histórico)
    3. Nenhuma feature _prev é idêntica ao LapTime da volta atual
    4. lap_time_prev na volta N é exatamente o LapTime da volta N-1
    5. [NOVO v3] LapNumber_pct é calculado com base no MAX de LapNumber
       da corrida inteira — não há leakage porque max é determinístico,
       conhecido antes de cada volta começar (tamanho da corrida é público).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))

import numpy as np
import pandas as pd
import pytest

from build_features import (  # noqa: E402
    add_lap_history_features,
    add_telemetry_history_features,
    add_derived_features,
    add_tyre_and_stint_features,
    add_race_context_features,
    encode_compound,
    GROUP_COLS,
    TELEMETRY_COLS,
)


def make_synthetic_df(n_laps: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "year": 2024,
        "gp": "TestGP",
        "session_code": "R",
        "DriverNumber": 1,
        "LapNumber": np.arange(1, n_laps + 1),
        "Stint": 1,
        "Compound": "MEDIUM",
        "TyreLife": np.arange(1, n_laps + 1),
        "LapTime": 80 + rng.normal(0, 0.5, n_laps).cumsum() * 0.1,
        "speed_mean": 200 + rng.normal(0, 5, n_laps),
        "speed_max": 320 + rng.normal(0, 3, n_laps),
        "speed_std": 40 + rng.normal(0, 1, n_laps),
        "throttle_mean": 65 + rng.normal(0, 2, n_laps),
        "throttle_std": 30 + rng.normal(0, 1, n_laps),
        "brake_ratio": 0.18 + rng.normal(0, 0.01, n_laps),
        "rpm_mean": 11000 + rng.normal(0, 100, n_laps),
        "gear_mean": 5.5 + rng.normal(0, 0.1, n_laps),
        "drs_ratio": 0.25 + rng.normal(0, 0.02, n_laps),
    })


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + ["LapNumber"]).reset_index(drop=True)
    df = encode_compound(df)
    df = add_lap_history_features(df)
    df = add_telemetry_history_features(df)
    df = add_derived_features(df)
    df = add_tyre_and_stint_features(df)
    df = add_race_context_features(df)
    return df


def test_no_leakage_from_future_laps():
    """
    Hard test: embaralha voltas >= K e verifica que voltas < K ficam idênticas.
    Se qualquer feature _prev mudar, é leakage do futuro.
    """
    df_orig = make_synthetic_df(n_laps=30, seed=0)
    df_modified = df_orig.copy()
    K = 15

    rng = np.random.default_rng(999)
    mask = df_modified["LapNumber"] >= K
    df_modified.loc[mask, "LapTime"] = rng.uniform(70, 90, size=mask.sum())
    for col in TELEMETRY_COLS:
        df_modified.loc[mask, col] = rng.uniform(0, 500, size=mask.sum())

    out_orig = run_pipeline(df_orig)
    out_mod = run_pipeline(df_modified)

    past_orig = out_orig[out_orig["LapNumber"] < K].reset_index(drop=True)
    past_mod = out_mod[out_mod["LapNumber"] < K].reset_index(drop=True)

    feature_cols = [c for c in past_orig.columns if c.endswith("_prev")]
    assert len(feature_cols) > 0

    leaking = []
    for col in feature_cols:
        a = past_orig[col].fillna(-999999).to_numpy()
        b = past_mod[col].fillna(-999999).to_numpy()
        if not np.allclose(a, b, equal_nan=False):
            leaking.append(col)

    assert not leaking, (
        f"LEAKAGE em features: {leaking}. "
        f"Elas mudaram quando o futuro foi alterado."
    )


def test_first_lap_features_are_nan():
    df = make_synthetic_df(n_laps=10, seed=1)
    out = run_pipeline(df)
    first_lap = out[out["LapNumber"] == 1].iloc[0]

    must_be_nan = [
        "lap_time_prev", "lap_time_mean_3_prev", "lap_time_delta_prev",
        "speed_mean_prev", "throttle_mean_prev", "brake_ratio_prev",
    ]
    for col in must_be_nan:
        assert pd.isna(first_lap[col]), (
            f"{col} deveria ser NaN na volta 1, vale {first_lap[col]}"
        )


def test_lap_time_not_in_features():
    df = make_synthetic_df(n_laps=20, seed=2)
    out = run_pipeline(df)
    feature_cols = [c for c in out.columns if c.endswith("_prev")]
    out_clean = out.dropna(subset=feature_cols)

    for col in feature_cols:
        identical = (out_clean[col] == out_clean["LapTime"]).sum()
        assert identical == 0, (
            f"{col} é idêntica ao LapTime atual em {identical} linhas"
        )


def test_lap_time_prev_equals_previous_lap_time():
    df = make_synthetic_df(n_laps=10, seed=3)
    out = run_pipeline(df).sort_values("LapNumber").reset_index(drop=True)

    for i in range(1, len(out)):
        expected = out.loc[i - 1, "LapTime"]
        actual = out.loc[i, "lap_time_prev"]
        assert actual == pytest.approx(expected), (
            f"lap_time_prev na volta {i+1} = {actual}, esperado {expected}"
        )


def test_lap_number_pct_is_monotonic_and_bounded():
    """
    LapNumber_pct deve ir de ~1/n_laps até 1.0 monotonicamente.
    Não é leakage: o tamanho da corrida é determinístico e público.
    """
    df = make_synthetic_df(n_laps=50, seed=5)
    out = run_pipeline(df).sort_values("LapNumber").reset_index(drop=True)

    assert "LapNumber_pct" in out.columns
    pct = out["LapNumber_pct"].to_numpy()
    assert pct[0] > 0 and pct[0] < 0.1
    assert pct[-1] == pytest.approx(1.0)
    assert np.all(np.diff(pct) > 0), "LapNumber_pct deveria ser monotonic crescente"


def test_interaction_features_computed():
    df = make_synthetic_df(n_laps=20, seed=7)
    out = run_pipeline(df)
    assert "tyre_x_progress" in out.columns
    assert "compound_x_tyre" in out.columns
    # compound_x_tyre = CompoundEncoded * TyreLife, MEDIUM=1, então igual a TyreLife
    expected = out["TyreLife"] * 1.0
    assert np.allclose(out["compound_x_tyre"], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])