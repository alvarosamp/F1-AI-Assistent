import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def main():

    print("Treinando modelo de pit stop...")

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/processed/features.csv")

    # regra para criar label (temporário)
    df['pit_label'] = (df['TyreLife'] > 20).astype(int)

    features = [
        'TyreLife',
        'lap_time_mean_3',
        'lap_time_delta',
        'tyre_ratio'
    ]

    X = df[features]
    y = df['pit_label']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, "models/pit_model.pkl")

    print("Modelo de pit stop salvo!")


if __name__ == "__main__":
    main()