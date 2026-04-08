import joblib
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "global_model_optuna.pkl"

# Função para gerar os dados de pista simulados
def generate_track():
    t = np.linspace(0, 2 * np.pi, 500)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    return x * 10, y * 5

# Função para simular a corrida
def simulate_race():
    # Carregar modelo treinado
    model = joblib.load(MODEL_PATH)

    expected_features = list(getattr(model, "feature_names_in_", []))
    if not expected_features:
        raise ValueError(
            "Modelo não expõe 'feature_names_in_'. Re-treine o modelo com DataFrame ou forneça a lista de features."
        )

    # Gerar a pista Suzuka
    X_track, y_track = generate_track()
    fig, ax = plt.subplots()
    ax.plot(X_track, y_track, 'gray')

    # Iniciar o carro e as variáveis da corrida
    car, = ax.plot([], [], 'ro')
    tyre_life = 1
    compound = 1
    lap_times = []
    position = 0
    pit_stop_count = 0

    def update(frame):
        nonlocal tyre_life, compound, lap_times, position, pit_stop_count
        
        # Calcular as features para a predição do tempo de volta
        lap_time_mean = np.mean(lap_times[-3:]) if len(lap_times) >= 3 else 80
        lap_time_delta = lap_times[-1] - lap_times[-2] if len(lap_times) >= 2 else 0
        
        feature_row = {name: 0.0 for name in expected_features}

        # Features diretamente disponíveis na simulação
        feature_row["TyreLife"] = float(tyre_life)
        feature_row["lap_time_mean_3"] = float(lap_time_mean)
        feature_row["lap_time_delta"] = float(lap_time_delta)

        # O modelo foi treinado com CompoundEncoded (não 'Compound')
        if "CompoundEncoded" in feature_row:
            feature_row["CompoundEncoded"] = float(compound)

        # Heurísticas/defaults para features que o modelo espera
        if "Position" in feature_row:
            feature_row["Position"] = 1.0
        if "TrackStatus" in feature_row:
            feature_row["TrackStatus"] = 1.0
        if "speed_mean" in feature_row:
            feature_row["speed_mean"] = 180.0
        if "speed_max" in feature_row:
            feature_row["speed_max"] = 300.0
        if "speed_std" in feature_row:
            feature_row["speed_std"] = 50.0
        if "throttle_mean" in feature_row:
            feature_row["throttle_mean"] = 50.0
        if "throttle_std" in feature_row:
            feature_row["throttle_std"] = 20.0
        if "brake_ratio" in feature_row:
            feature_row["brake_ratio"] = 0.2
        if "rpm_mean" in feature_row:
            feature_row["rpm_mean"] = 9000.0
        if "gear_mean" in feature_row:
            feature_row["gear_mean"] = 5.0
        if "drs_ratio" in feature_row:
            feature_row["drs_ratio"] = 0.3
        if "is_raining" in feature_row:
            feature_row["is_raining"] = 0.0
        if "session_code_encoded" in feature_row:
            feature_row["session_code_encoded"] = 0.0
        if "team_encoded" in feature_row:
            feature_row["team_encoded"] = 0.0

        # Sinais derivados
        if "tyre_ratio" in feature_row:
            feature_row["tyre_ratio"] = float(tyre_life) / 50.0
        if "stint_progress" in feature_row:
            feature_row["stint_progress"] = min(float(tyre_life) / 20.0, 1.0)
        if "speed_delta" in feature_row:
            feature_row["speed_delta"] = 0.0
        if "throttle_delta" in feature_row:
            feature_row["throttle_delta"] = 0.0
        if "brake_delta" in feature_row:
            feature_row["brake_delta"] = 0.0
        if "degradation_score" in feature_row:
            feature_row["degradation_score"] = (
                feature_row.get("lap_time_delta", 0.0)
                + (-feature_row.get("speed_delta", 0.0))
                + feature_row.get("brake_ratio", 0.0)
            )
        if "aggression_score" in feature_row:
            feature_row["aggression_score"] = feature_row.get("throttle_mean", 0.0) * (
                1.0 - feature_row.get("brake_ratio", 0.0)
            )
        if "efficiency_score" in feature_row:
            feature_row["efficiency_score"] = feature_row.get("speed_mean", 0.0) / (
                feature_row.get("rpm_mean", 0.0) + 1.0
            )
        if "drs_usage_intensity" in feature_row:
            feature_row["drs_usage_intensity"] = feature_row.get("drs_ratio", 0.0) * feature_row.get(
                "speed_max", 0.0
            )
        if "consistency_score" in feature_row:
            feature_row["consistency_score"] = 0.0

        features = pd.DataFrame([feature_row], columns=expected_features)

        # Predição do tempo de volta
        lap_time = model.predict(features)[0]
        lap_times.append(lap_time)

        # Calcular a velocidade proporcional à previsão de tempo
        speed = max(1, int(200 / lap_time))
        position = (position + speed) % len(X_track)
        # Line2D.set_data espera sequências; para um ponto, use listas
        car.set_data([X_track[position]], [y_track[position]])

        # Atualizar o desgaste do pneu
        tyre_life += 1
        
        # Simulação de pit stop
        if tyre_life >= 20 and pit_stop_count < 2:  # Exemplo de regra de pit
            pit_stop_count += 1
            print(f"Pit Stop ({pit_stop_count})!")
            tyre_life = 0  # Reset tyre life
            return car, 
        
        return car,

    ani = FuncAnimation(fig, update, frames=500, blit=True, interval=50)
    
    plt.axis('equal')
    plt.show()

# Função para rodar a simulação
def run_simulation():
    print("Simulando a corrida com IA...")
    simulate_race()

if __name__ == "__main__":
    run_simulation()