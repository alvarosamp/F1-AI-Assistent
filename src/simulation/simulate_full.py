
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
import pandas as pd

# Ensure src is in sys.path for absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pista_suzuka import get_track
from llm.engenheiro import explain_decision

#Carregando modelos
def load_models():
    root = Path(__file__).resolve().parents[2]
    model = joblib.load(root / "models" / "model.pkl")
    pit_model = joblib.load(root / "models" / "pit_model.pkl")
    return model, pit_model

# --------------------------
# SIMULAÇÃO
# --------------------------
def main():

    model, pit_model = load_models()
    x_track, y_track = get_track()

    fig, ax = plt.subplots()
    ax.plot(x_track, y_track, 'gray')
    car, = ax.plot([], [], 'ro')

    tyre_life = 1
    compound = 1
    lap_times = []
    position = 0
    total_time = 0

    results = []

    def update(frame):
        nonlocal tyre_life, compound, lap_times, position, total_time

        lap = frame + 1

        # FEATURES
        lap_time_mean = np.mean(lap_times[-3:]) if len(lap_times) >= 3 else 80
        lap_time_delta = lap_times[-1] - lap_times[-2] if len(lap_times) >= 2 else 0

        features = pd.DataFrame([[
            tyre_life,
            compound,
            lap_time_mean,
            lap_time_delta,
            tyre_life / 50
        ]], columns=[
            'TyreLife','Compound','lap_time_mean_3','lap_time_delta','tyre_ratio'
        ])

        lap_time = float(model.predict(features)[0])
        lap_times.append(lap_time)
        total_time += lap_time

        # MOVIMENTO REAL
        progress = 1 / lap_time
        position = (position + int(progress * len(x_track))) % len(x_track)

        car.set_data([x_track[position]], [y_track[position]])

        # PIT STOP
        pit_features = pd.DataFrame([[
            tyre_life,
            lap_time_mean,
            lap_time_delta,
            tyre_life / 50
        ]], columns=[
            'TyreLife','lap_time_mean_3','lap_time_delta','tyre_ratio'
        ])

        pit_decision = int(pit_model.predict(pit_features)[0])

        if pit_decision == 1 and frame > 10:
            explanation = explain_decision({
                "lap": lap,
                "tyre_life": tyre_life,
                "lap_time_delta": lap_time_delta,
                "lap_time_mean_3": lap_time_mean,
                "pit": True,
                "pit_stop_needed": pit_decision == 1,
                "predicted_pit_time": 20,  # valor estimado fixo
                "predicted_lap_time": lap_time
            })

            print("\n========================")
            print(f"PIT STOP na volta {lap}")
            print(explanation)
            print("========================\n")

            tyre_life = 1
            compound = np.random.choice([0,1,2])
            total_time += 20
        else:
            tyre_life += 1

        # SALVAR RESULTADO
        results.append({
            "lap": lap,
            "lap_time": lap_time,
            "tyre_life": tyre_life,
            "compound": compound,
            "total_time": total_time,
            "pit": pit_decision
        })

        return car,

    ani = FuncAnimation(fig, update, frames=100, interval=50)

    plt.title("F1 AI Simulator - Suzuka Real")
    plt.axis('equal')
    plt.show()

    # salvar csv
    df = pd.DataFrame(results)
    df.to_csv("data/simulation_results.csv", index=False)


if __name__ == "__main__":
    main()
