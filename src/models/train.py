from xml.parsers.expat import model

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import mlflow
import numpy as np

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\src\\data\\processed\\features.csv")
    features = [
        'TyreLife',
        'Compound',
        'lap_time_mean_3',
        'lap_time_delta',
        'tyre_ratio',
    ]
    target = "LapTime"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="f1-lap-time-model"):
        model = XGBRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"Lap Time Model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        
        #Metricas mlflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)
        
        #Salvando modelo
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

if __name__ == "__main__":
    main()