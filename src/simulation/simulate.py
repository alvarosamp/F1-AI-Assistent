import joblib
import random

def main():

    print("Simulando corrida...")

    model = joblib.load("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\src\\models\\modelos\\model.pkl")

    tyre_life = 0
    total_time = 0

    for lap in range(50):

        tyre_life += 1

        features = [[
            tyre_life,
            random.choice([0,1,2]),   # tipo de pneu
            random.uniform(80, 100),
            random.uniform(-1, 1),
            tyre_life / 50
        ]]

        lap_time = model.predict(features)[0]

        total_time += lap_time

        print(f"Lap {lap+1}: {lap_time:.2f}s")

        # pit stop
        if tyre_life > 20:
            print("PIT STOP!")
            tyre_life = 0
            total_time += 20

    print(f"\nTempo total: {total_time:.2f}s")


if __name__ == "__main__":
    main()