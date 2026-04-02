import joblib
import random

def main():

    print("Simulando corrida com IA...")

    # modelo de tempo de volta
    model = joblib.load("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\models\\model.pkl")

    # modelo de decisão de pit
    pit_model = joblib.load("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\models\\pit_model.pkl")

    tyre_life = 0
    total_time = 0


    import pandas as pd

    for lap in range(50):
        tyre_life += 1

        # FEATURES DO MODELO DE TEMPO
        features_time = pd.DataFrame([
            [
                tyre_life,
                random.choice([0, 1, 2]),   # Compound
                random.uniform(80, 100),    # média recente
                random.uniform(-1, 1),      # delta
                tyre_life / 50
            ]
        ], columns=['TyreLife', 'Compound', 'lap_time_mean_3', 'lap_time_delta', 'tyre_ratio'])

        lap_time = model.predict(features_time)[0]
        total_time += lap_time
        print(f"Lap {lap+1}: {lap_time:.2f}s")

        # FEATURES DO MODELO DE PIT
        features_pit = pd.DataFrame([
            [
                tyre_life,
                features_time['lap_time_mean_3'].iloc[0],
                features_time['lap_time_delta'].iloc[0],
                tyre_life / 50
            ]
        ], columns=['TyreLife', 'lap_time_mean_3', 'lap_time_delta', 'tyre_ratio'])

        pit_decision = pit_model.predict(features_pit)[0]

        # DECISÃO DE PIT
        if pit_decision == 1:
            print("PIT STOP (IA)!")
            tyre_life = 0
            total_time += 20  # tempo de pit

    print(f"\nTempo total: {total_time:.2f}s")


if __name__ == "__main__":
    main()