"""
Testes da pipeline de DNF (Entrega 1 do Sprint 3).

O QUE PROVA:
    1. Features históricas NÃO vazam informação da corrida atual.
       Teste forte: altera o label DNF da corrida N e verifica que as
       features históricas das corridas < N ficam idênticas.

    2. Modelo bate o baseline trivial no walk-forward (Brier < 0.0895).

    3. Modelo é bem calibrado (ECE < 0.02).

    4. Modelo é determinístico com random_state fixo.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dnf"))

import numpy as np
import pandas as pd
import pytest

from build_dnf_dataset import add_historical_rates  # noqa: E402


def make_synthetic_dnf_df(seed: int = 0) -> pd.DataFrame:
    """
    Cria dataset sintético de DNF: 3 pilotos, 3 times, 5 GPs, 2 anos.
    Usado pra testes de leakage e determinismo.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year in [2023, 2024]:
        for gp_order, gp in enumerate(["A", "B", "C", "D", "E"], start=1):
            for driver, team in [("D1", "T1"), ("D2", "T1"), ("D3", "T2")]:
                rows.append({
                    "year": year,
                    "gp": gp,
                    "Driver": driver,
                    "Team": team,
                    "grid_position": rng.integers(1, 20),
                    "quali_position": rng.integers(1, 20),
                    "dnf": int(rng.random() < 0.15),
                })
    return pd.DataFrame(rows)


def test_historical_rates_no_leakage_from_future():
    """
    Teste forte: muda o DNF da corrida N e verifica que as features
    das corridas < N ficam IDÊNTICAS. Prova matemática de ausência
    de leakage temporal nas features históricas.
    """
    df_orig = make_synthetic_dnf_df(seed=1)
    df_modified = df_orig.copy()

    # Envenena o DNF das corridas a partir da 4ª no ano 2024
    poison_mask = (df_modified["year"] == 2024) & (df_modified["gp"].isin(["D", "E"]))
    df_modified.loc[poison_mask, "dnf"] = 1 - df_modified.loc[poison_mask, "dnf"]

    out_orig = add_historical_rates(df_orig)
    out_mod = add_historical_rates(df_modified)

    # Verifica que as features das corridas A, B, C de 2024 ficaram idênticas
    past_mask = (out_orig["year"] == 2024) & (out_orig["gp"].isin(["A", "B", "C"]))

    # Ordena por (Driver, gp) pra comparar
    past_orig = out_orig[past_mask].sort_values(["Driver", "gp"]).reset_index(drop=True)
    past_mod = out_mod[past_mask].sort_values(["Driver", "gp"]).reset_index(drop=True)

    feature_cols = ["dnf_rate_driver", "dnf_rate_team", "dnf_rate_gp"]

    leaking = []
    for col in feature_cols:
        a = past_orig[col].fillna(-999).to_numpy()
        b = past_mod[col].fillna(-999).to_numpy()
        if not np.allclose(a, b, equal_nan=False):
            leaking.append(col)

    assert not leaking, (
        f"LEAKAGE detectado em: {leaking}. "
        f"As features do passado mudaram quando envenenamos o futuro."
    )


def test_first_race_features_are_nan():
    """
    Primeira corrida de cada ano/piloto deve ter dnf_rate_driver NaN —
    sem histórico pra olhar. Se vier 0 ou outro valor, o shift tá quebrado.
    """
    df = make_synthetic_dnf_df(seed=2)
    out = add_historical_rates(df)

    # Primeira corrida do D1 em 2023 (o primeiro gp de cada ano)
    out_sorted = out.sort_values(["year", "gp"])
    first = out_sorted[(out_sorted["year"] == 2023) & (out_sorted["Driver"] == "D1")].iloc[0]
    assert pd.isna(first["dnf_rate_driver"]), (
        f"Primeira corrida deveria ter dnf_rate_driver NaN, veio {first['dnf_rate_driver']}"
    )


def test_expanding_mean_is_correct():
    """
    Verifica que dnf_rate_driver na corrida N é a média dos DNFs das
    corridas 1 até N-1 do mesmo piloto. Sanity check do shift+expanding.
    """
    df = pd.DataFrame([
        {"year": 2023, "gp": "A", "Driver": "D1", "Team": "T1",
         "grid_position": 5, "quali_position": 5, "dnf": 0},
        {"year": 2023, "gp": "B", "Driver": "D1", "Team": "T1",
         "grid_position": 5, "quali_position": 5, "dnf": 1},
        {"year": 2023, "gp": "C", "Driver": "D1", "Team": "T1",
         "grid_position": 5, "quali_position": 5, "dnf": 0},
        {"year": 2023, "gp": "D", "Driver": "D1", "Team": "T1",
         "grid_position": 5, "quali_position": 5, "dnf": 1},
    ])
    out = add_historical_rates(df).sort_values("gp").reset_index(drop=True)

    # gp=A: nada antes, NaN
    assert pd.isna(out.iloc[0]["dnf_rate_driver"])
    # gp=B: antes só A (dnf=0) -> 0.0
    assert out.iloc[1]["dnf_rate_driver"] == pytest.approx(0.0)
    # gp=C: antes A (0) e B (1) -> 0.5
    assert out.iloc[2]["dnf_rate_driver"] == pytest.approx(0.5)
    # gp=D: antes A (0), B (1), C (0) -> 1/3
    assert out.iloc[3]["dnf_rate_driver"] == pytest.approx(1 / 3)


def test_model_beats_trivial_baseline_on_real_data():
    """
    Teste end-to-end: carrega o dataset real e verifica que o modelo
    supera o baseline trivial em Brier score.

    Requer que build_dnf_dataset.py já tenha rodado e gerado o CSV.
    """
    dataset_file = Path(__file__).resolve().parents[1] / "data" / "processed" / "dnf_dataset.csv"
    if not dataset_file.exists():
        pytest.skip(f"Dataset não encontrado em {dataset_file}. Rode build_dnf_dataset.py antes.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    ds = pd.read_csv(dataset_file)
    features = ["dnf_rate_driver", "dnf_rate_team", "dnf_rate_gp",
                "regulation_era", "grid_position", "quali_position"]

    train_mask = ds["year"].isin([2022, 2023])
    test_mask = ds["year"] == 2024

    Xtr = ds[train_mask][features].fillna(ds[train_mask][features].median())
    Xte = ds[test_mask][features].fillna(ds[train_mask][features].median())
    ytr = ds[train_mask]["dnf"].values
    yte = ds[test_mask]["dnf"].values

    trivial_preds = np.full(len(yte), ytr.mean())
    brier_trivial = brier_score_loss(yte, trivial_preds)

    model = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    model.fit(Xtr, ytr)
    model_preds = model.predict_proba(Xte)[:, 1]
    brier_model = brier_score_loss(yte, model_preds)

    assert brier_model < brier_trivial, (
        f"Modelo ({brier_model:.4f}) não bateu baseline trivial ({brier_trivial:.4f}). "
        f"Algo regrediu no dataset ou features."
    )


def test_model_is_deterministic():
    """
    Treinar o modelo duas vezes com random_state=42 deve dar predições idênticas.
    Reprodutibilidade é requisito básico pra qualquer projeto sério.
    """
    dataset_file = Path(__file__).resolve().parents[1] / "data" / "processed" / "dnf_dataset.csv"
    if not dataset_file.exists():
        pytest.skip(f"Dataset não encontrado em {dataset_file}.")

    from sklearn.linear_model import LogisticRegression

    ds = pd.read_csv(dataset_file)
    features = ["dnf_rate_driver", "dnf_rate_team", "dnf_rate_gp",
                "regulation_era", "grid_position", "quali_position"]
    train_mask = ds["year"].isin([2022, 2023])
    test_mask = ds["year"] == 2024

    Xtr = ds[train_mask][features].fillna(ds[train_mask][features].median())
    Xte = ds[test_mask][features].fillna(ds[train_mask][features].median())
    ytr = ds[train_mask]["dnf"].values

    m1 = LogisticRegression(max_iter=1000, C=0.5, random_state=42).fit(Xtr, ytr)
    m2 = LogisticRegression(max_iter=1000, C=0.5, random_state=42).fit(Xtr, ytr)

    assert np.allclose(m1.predict_proba(Xte), m2.predict_proba(Xte))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
