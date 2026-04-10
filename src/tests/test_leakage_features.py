"""
Testes que provam que build_features.py não vaza informação do futuro.

A estratégia é forte: criamos um dataset sintético onde a volta N tem
um LapTime CONHECIDO e VERIFICÁVEL, depois embaralhamos os LapTimes
das voltas >= K e checamos que TODAS as features de uma volta < K
permanecem idênticas. Se alguma feature da volta < K mudar, é porque
ela estava lendo informação de uma volta posterior == leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
# test file: F1-AI-Assistent/src/testes/test_*.py
# build_features.py lives at: F1-AI-Assistent/src/features/build_features.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))

from build_features import (  # noqa: E402
    add_lap_history_features,
    add_telemetry_history_features,
    add_derived_features,
    add_tyre_and_stint_features,
    encode_compound,
    GROUP_COLS,
    TELEMETRY_COLS,
)


def make_synthetic_df(n_laps: int = 30, seed: int = 0) -> pd.DataFrame:
    """Cria um stint sintético de um piloto numa sessão."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
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
    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + ["LapNumber"]).reset_index(drop=True)
    df = encode_compound(df)
    df = add_lap_history_features(df)
    df = add_telemetry_history_features(df)
    df = add_derived_features(df)
    df = add_tyre_and_stint_features(df)
    return df


def test_no_leakage_from_future_laps():
    """
    Hard test: pego o pipeline rodado no dataset original. Crio um clone
    onde só as voltas >= K têm LapTime e telemetria DIFERENTES. Rodo o
    pipeline no clone. Para todas as voltas i < K, TODAS as features _prev
    devem ser idênticas. Se alguma diferir, é leakage do futuro.
    """
    df_orig = make_synthetic_df(n_laps=30, seed=0)
    df_modified = df_orig.copy()

    K = 15  # ponto de corte: voltas a partir daqui são "futuro"

    # Embaralha radicalmente o futuro
    rng = np.random.default_rng(999)
    df_modified.loc[df_modified["LapNumber"] >= K, "LapTime"] = (
        rng.uniform(70, 90, size=(df_modified["LapNumber"] >= K).sum())
    )
    for col in TELEMETRY_COLS:
        df_modified.loc[df_modified["LapNumber"] >= K, col] = (
            rng.uniform(0, 500, size=(df_modified["LapNumber"] >= K).sum())
        )

    out_orig = run_pipeline(df_orig)
    out_mod = run_pipeline(df_modified)

    past_orig = out_orig[out_orig["LapNumber"] < K].reset_index(drop=True)
    past_mod = out_mod[out_mod["LapNumber"] < K].reset_index(drop=True)

    feature_cols = [c for c in past_orig.columns if c.endswith("_prev")]
    assert len(feature_cols) > 0, "pipeline não gerou nenhuma feature _prev"

    leaking = []
    for col in feature_cols:
        a = past_orig[col].fillna(-999999).to_numpy()
        b = past_mod[col].fillna(-999999).to_numpy()
        if not np.allclose(a, b, equal_nan=False):
            leaking.append(col)

    assert not leaking, (
        f"LEAKAGE DETECTADO nas features: {leaking}\n"
        f"Estas features mudaram quando o futuro foi alterado, "
        f"o que prova que elas estavam lendo informação posterior."
    )


def test_first_lap_features_are_nan():
    """
    Na primeira volta de cada stint, todas as features _prev de histórico
    devem ser NaN — não há volta anterior para olhar. Se vier 0 ou outro
    valor, o pipeline está mascarando NaN com fillna prematuro.
    """
    df = make_synthetic_df(n_laps=10, seed=1)
    out = run_pipeline(df)

    first_lap = out[out["LapNumber"] == 1].iloc[0]

    must_be_nan = [
        "lap_time_prev",
        "lap_time_mean_3_prev",
        "lap_time_delta_prev",
        "speed_mean_prev",
        "throttle_mean_prev",
        "brake_ratio_prev",
    ]

    for col in must_be_nan:
        assert pd.isna(first_lap[col]), (
            f"{col} deveria ser NaN na volta 1, mas vale {first_lap[col]}. "
            f"Isso indica fillna prematuro ou leakage."
        )


def test_lap_time_not_in_features():
    """
    Sanidade: nenhuma feature _prev pode ser exatamente igual ao LapTime
    da volta atual. Se for, é leakage trivial.
    """
    df = make_synthetic_df(n_laps=20, seed=2)
    out = run_pipeline(df)

    feature_cols = [c for c in out.columns if c.endswith("_prev")]
    out_clean = out.dropna(subset=feature_cols)

    for col in feature_cols:
        # nenhuma feature deve coincidir 1:1 com LapTime atual
        identical = (out_clean[col] == out_clean["LapTime"]).sum()
        assert identical == 0, (
            f"Feature {col} é idêntica ao LapTime atual em {identical} linhas. "
            f"Isso é leakage direto."
        )


def test_lap_time_prev_equals_previous_lap_time():
    """
    Verifica explicitamente que lap_time_prev na volta N é o LapTime da
    volta N-1. Se essa propriedade não vale, o shift está errado.
    """
    df = make_synthetic_df(n_laps=10, seed=3)
    out = run_pipeline(df).sort_values("LapNumber").reset_index(drop=True)

    for i in range(1, len(out)):
        expected = out.loc[i - 1, "LapTime"]
        actual = out.loc[i, "lap_time_prev"]
        assert actual == pytest.approx(expected), (
            f"lap_time_prev na volta {i+1} = {actual}, esperado {expected}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])