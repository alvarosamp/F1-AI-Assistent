from __future__ import annotations
from pathlib import Path
import joblib 
import mlflow
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "telemetry_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PER_TRACK_DIR = MODELS_DIR / "per_track"


def get_features(df):
    return [
        col for col in df.columns
        if col not in ["LapTime", "LapTimeResidual", "gp", "Driver", "Team"]
    ]


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a purely numeric feature matrix suitable for XGBoost.

    Converts object/categorical columns using one-hot encoding.
    """

    X = df[get_features(df)].copy()

    # Normalize dtypes
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    X = pd.get_dummies(X, dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PER_TRACK_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    mlflow.set_experiment('f1-local-models')
    results = []
    
    for gp, df_gp in df.groupby("gp"):

        if len(df_gp) < 150:
            print(f"[SKIP] {gp}")
            continue

        X = build_model_matrix(df_gp)
        y = df_gp["LapTimeResidual"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with mlflow.start_run(run_name=f"local_{gp}"):

            model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            print(f"{gp} | RMSE={rmse:.2f}")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            model_path = PER_TRACK_DIR / f"{gp.replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)

            results.append({
                "gp": gp,
                "rmse_local": rmse,
                "mae_local": mae,
                "r2_local": r2
            })

    pd.DataFrame(results).to_csv(PER_TRACK_DIR / "results_per_track.csv", index=False)


if __name__ == "__main__":
    main()