"""
Modelo de degradação de pneu por composto — Sprint 3 Entrega 3.

DECISÃO DE DESIGN:
    NÃO treina um modelo ML separado. Usa coeficientes lineares observados
    como lookup table fixa. Razões:

    1. O modelo de lap time v3 JÁ captura degradação implicitamente via
       features TyreLife, tyre_ratio, stint_progress, compound_x_tyre.
       Um modelo separado seria redundante.

    2. Os dados mostram que a "degradação" OBSERVÁVEL é NEGATIVA (pilotos
       ficam mais rápidos conforme o stint avança) porque o efeito de
       combustível (~0.03s/volta de ganho) domina sobre degradação real
       do pneu (~0.01-0.02s/volta de perda). O resultado líquido é negativo.

    3. R² do modelo linear é muito baixo (0.001 a 0.086) — a variância
       intra-stint é dominada por tráfego, clima, erros do piloto, não
       degradação. Treinar modelo sofisticado aqui seria overfitting.

COEFICIENTES MEDIDOS (por volta, relativo à 1ª volta do stint):
    SOFT:         -0.006 s/volta  (quase flat)
    MEDIUM:       -0.033 s/volta
    HARD:         -0.035 s/volta
    INTERMEDIATE: ~-0.060 s/volta (estimado, poucos dados)

COMO O SIMULADOR USA:
    O modelo de lap time v3 prevê o LapTimeResidual base. O módulo de
    degradação adiciona um ajuste fino:

        ajuste = coef_compound * (tyre_life - 1)

    Na prática, isso diz ao simulador que:
    - Em SOFT na volta 20: ajuste de -0.11s (piloto ligeiramente mais rápido)
    - Em HARD na volta 30: ajuste de -1.0s (significativamente mais rápido)

    Parece contraintuitivo, mas é correto DADO que o residual já está
    normalizado pela mediana da sessão. O "mais rápido" é relativo à
    primeira volta do stint (que inclui efeito de pneu novo + combustível).

LIMITAÇÃO HONESTA:
    Este modelo é o componente mais fraco do simulador. A degradação real
    depende de temperatura da pista, setup do carro, estilo de pilotagem
    (agressivo vs conservador) — variáveis que não capturamos a nível de
    stint individual. Com dados de F1 de nível de equipe (que não são
    públicos), seria possível fazer muito melhor.

OUTPUT:
    models/tyre_deg_model.pkl — dict com coeficientes por compound
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "telemetry_features_race_v4.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "tyre_deg_model.pkl"
METRICS_PATH = MODELS_DIR / "tyre_deg_metrics.json"

MIN_STINT_LENGTH = 5  # ignora stints curtos


def build_stint_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula degradação por volta dentro de cada stint."""
    stint_group = ["year", "gp", "DriverNumber", "Stint"]

    df = df.copy()
    df["stint_first_residual"] = df.groupby(stint_group)["LapTimeResidual"].transform("first")
    df["deg_residual"] = df["LapTimeResidual"] - df["stint_first_residual"]

    stint_len = df.groupby(stint_group)["LapNumber"].transform("count")
    df = df[stint_len >= MIN_STINT_LENGTH].copy()

    return df


def fit_compound_coefficients(df: pd.DataFrame) -> dict:
    """Ajusta coeficiente linear por compound: deg = coef * TyreLife + intercept."""
    compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]
    results = {}

    for compound in compounds:
        sub = df[(df["Compound"].str.upper() == compound) & (df["TyreLife"] >= 2)]
        if len(sub) < 50:
            print(f"  {compound}: poucos dados ({len(sub)} voltas), pulando")
            continue

        X = sub[["TyreLife"]].values
        y = sub["deg_residual"].values

        lr = LinearRegression().fit(X, y)

        results[compound] = {
            "coef_per_lap": float(lr.coef_[0]),
            "intercept": float(lr.intercept_),
            "r2": float(lr.score(X, y)),
            "n_laps": len(sub),
            "deg_at_10_laps": float(lr.predict([[10]])[0]),
            "deg_at_20_laps": float(lr.predict([[20]])[0]),
            "deg_at_30_laps": float(lr.predict([[30]])[0]),
        }

    return results


def predict_tyre_degradation(compound: str, tyre_life: int, model: dict) -> float:
    """
    Retorna o ajuste de lap time por degradação de pneu.

    Args:
        compound: "SOFT", "MEDIUM", "HARD" ou "INTERMEDIATE"
        tyre_life: número de voltas neste pneu
        model: dict carregado do pkl

    Returns:
        Ajuste em segundos (negativo = mais rápido que volta 1 do stint)
    """
    compound = compound.upper()
    if compound not in model["coefficients"]:
        # Fallback: usa MEDIUM como default
        compound = "MEDIUM"

    coef = model["coefficients"][compound]["coef_per_lap"]
    intercept = model["coefficients"][compound]["intercept"]
    return coef * tyre_life + intercept


def main() -> None:
    print(f"Carregando: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Shape: {df.shape}")

    df_stints = build_stint_degradation(df)
    print(f"Voltas em stints ≥ {MIN_STINT_LENGTH} voltas: {len(df_stints)}")

    print(f"\n{'='*50}")
    print("COEFICIENTES POR COMPOUND")
    print("="*50)

    coefficients = fit_compound_coefficients(df_stints)

    for compound, stats in sorted(coefficients.items()):
        print(f"\n  {compound}:")
        print(f"    Coef:      {stats['coef_per_lap']:+.4f} s/volta")
        print(f"    Intercept: {stats['intercept']:+.4f} s")
        print(f"    R²:        {stats['r2']:.4f}")
        print(f"    N voltas:  {stats['n_laps']}")
        print(f"    Deg @ 10v: {stats['deg_at_10_laps']:+.3f}s")
        print(f"    Deg @ 20v: {stats['deg_at_20_laps']:+.3f}s")
        print(f"    Deg @ 30v: {stats['deg_at_30_laps']:+.3f}s")

    # Curva de degradação por compound (pra visualizar)
    print(f"\n{'='*50}")
    print("CURVA DE DEGRADAÇÃO (s vs TyreLife)")
    print("="*50)
    print(f"  {'TyreLife':>8}  {'SOFT':>8}  {'MEDIUM':>8}  {'HARD':>8}")
    for tl in [1, 5, 10, 15, 20, 25, 30]:
        vals = []
        for comp in ["SOFT", "MEDIUM", "HARD"]:
            if comp in coefficients:
                v = coefficients[comp]["coef_per_lap"] * tl + coefficients[comp]["intercept"]
                vals.append(f"{v:+.3f}")
            else:
                vals.append("  N/A  ")
        print(f"  {tl:8d}  {'  '.join(vals)}")

    # Salva
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "coefficients": coefficients,
        "min_stint_length": MIN_STINT_LENGTH,
        "note": (
            "Coeficientes são NEGATIVOS porque efeito de combustível "
            "(~0.03s/volta de ganho) domina sobre degradação real do pneu. "
            "O modelo de lap time v3 já captura degradação via TyreLife, "
            "tyre_ratio, stint_progress. Este ajuste é FINO e complementar."
        ),
    }
    joblib.dump(model_data, MODEL_PATH)

    metrics = {
        "coefficients_summary": {
            compound: {
                "coef": stats["coef_per_lap"],
                "r2": stats["r2"],
                "n": stats["n_laps"],
            }
            for compound, stats in coefficients.items()
        },
        "interpretation": "Negative coef means laptimes DECREASE (faster) as stint progresses, due to fuel burn dominating tyre wear.",
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Modelo salvo em {MODEL_PATH}")
    print(f"[OK] Métricas salvas em {METRICS_PATH}")


if __name__ == "__main__":
    main()