from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_models() -> tuple:
    """
    Carrega os modelos treinados.
    Usa caminhos relativos ao projeto, evitando path absoluto.
    """
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"

    lap_time_model_path = models_dir / "model.pkl"
    pit_model_path = models_dir / "pit_model.pkl"

    if not lap_time_model_path.exists():
        raise FileNotFoundError(f"Modelo de tempo não encontrado: {lap_time_model_path}")

    if not pit_model_path.exists():
        raise FileNotFoundError(f"Modelo de pit stop não encontrado: {pit_model_path}")

    lap_time_model = joblib.load(lap_time_model_path)
    pit_model = joblib.load(pit_model_path)

    return lap_time_model, pit_model


def build_time_features(
    tyre_life: int,
    compound: int,
    lap_times: list[float],
) -> pd.DataFrame:
    """
    Monta as features do modelo de previsão de tempo de volta.
    Mantém exatamente os nomes esperados pelo treino.
    """
    if len(lap_times) >= 3:
        lap_time_mean_3 = float(np.mean(lap_times[-3:]))
    elif len(lap_times) > 0:
        lap_time_mean_3 = float(np.mean(lap_times))
    else:
        lap_time_mean_3 = 80.0  # baseline inicial

    if len(lap_times) >= 2:
        lap_time_delta = float(lap_times[-1] - lap_times[-2])
    else:
        lap_time_delta = 0.0

    tyre_ratio = tyre_life / 50.0

    return pd.DataFrame(
        [[
            tyre_life,
            compound,
            lap_time_mean_3,
            lap_time_delta,
            tyre_ratio
        ]],
        columns=[
            "TyreLife",
            "Compound",
            "lap_time_mean_3",
            "lap_time_delta",
            "tyre_ratio",
        ],
    )


def build_pit_features(
    tyre_life: int,
    lap_times: list[float],
) -> pd.DataFrame:
    """
    Monta as features do modelo de decisão de pit stop.
    """
    if len(lap_times) >= 3:
        lap_time_mean_3 = float(np.mean(lap_times[-3:]))
    elif len(lap_times) > 0:
        lap_time_mean_3 = float(np.mean(lap_times))
    else:
        lap_time_mean_3 = 80.0

    if len(lap_times) >= 2:
        lap_time_delta = float(lap_times[-1] - lap_times[-2])
    else:
        lap_time_delta = 0.0

    tyre_ratio = tyre_life / 50.0

    return pd.DataFrame(
        [[
            tyre_life,
            lap_time_mean_3,
            lap_time_delta,
            tyre_ratio
        ]],
        columns=[
            "TyreLife",
            "lap_time_mean_3",
            "lap_time_delta",
            "tyre_ratio",
        ],
    )


def choose_next_compound(current_compound: int) -> int:
    """
    Estratégia simples para troca de composto.
    0 = SOFT
    1 = MEDIUM
    2 = HARD
    """
    available = [0, 1, 2]
    if current_compound in available:
        available.remove(current_compound)

    return int(np.random.choice(available))


def compound_name(compound_code: int) -> str:
    mapping = {
        0: "SOFT",
        1: "MEDIUM",
        2: "HARD",
    }
    return mapping.get(compound_code, "UNKNOWN")


def main() -> None:
    print("Simulando corrida com IA...")

    lap_time_model, pit_model = load_models()

    total_laps = 50
    tyre_life = 1
    compound = 1  # começa de MEDIUM
    total_time = 0.0
    lap_times: list[float] = []
    pit_stops = 0

    for lap in range(1, total_laps + 1):
        # previsão do tempo da volta
        features_time = build_time_features(
            tyre_life=tyre_life,
            compound=compound,
            lap_times=lap_times,
        )

        lap_time = float(lap_time_model.predict(features_time)[0])

        lap_times.append(lap_time)
        total_time += lap_time

        print(
            f"Lap {lap:02d} | "
            f"Pneu: {compound_name(compound):6s} | "
            f"TyreLife: {tyre_life:02d} | "
            f"Tempo: {lap_time:.2f}s"
        )

        # decisão de pit
        features_pit = build_pit_features(
            tyre_life=tyre_life,
            lap_times=lap_times,
        )

        pit_decision = int(pit_model.predict(features_pit)[0])

        # evitar pit na última volta
        if pit_decision == 1 and lap < total_laps:
            pit_stops += 1
            total_time += 20.0  # perda no pit
            new_compound = choose_next_compound(compound)

            print(
                f"  -> PIT STOP (IA) | "
                f"Troca: {compound_name(compound)} -> {compound_name(new_compound)} | "
                f"+20.00s"
            )

            compound = new_compound
            tyre_life = 1
        else:
            tyre_life += 1

    avg_lap = total_time / total_laps

    print("\n===== RESULTADO FINAL =====")
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Tempo médio por volta: {avg_lap:.2f}s")
    print(f"Quantidade de pit stops: {pit_stops}")


if __name__ == "__main__":
    main()