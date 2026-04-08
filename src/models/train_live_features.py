from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
import fastf1
from pathlib import Path
import time

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Cache do FastF1
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Função para coleta de telemetria
def collect_telemetry_data(session):
    session.load()
    
    laps = session.laps
    car_data = laps.get_car_data()
    car_data = car_data.add_distance()
    
    return car_data

# Função para extrair features em tempo real
def extract_features(car_data):
    features = {
        "speed_mean": car_data["Speed"].mean(),
        "throttle_mean": car_data["Throttle"].mean(),
        "brake_mean": car_data["Brake"].mean(),
        "rpm_mean": car_data["RPM"].mean(),
        "gear_mean": car_data["nGear"].mean(),
        "drs_ratio": (car_data["DRS"] > 0).mean(),
    }
    
    return features

# Função de predição em tempo real
def real_time_prediction(session, model):
    while True:
        print("Coletando dados de telemetria ao vivo...")

        # Coleta os dados ao vivo
        car_data = collect_telemetry_data(session)
        
        # Extrai as features necessárias
        features = extract_features(car_data)
        
        # Prepara o DataFrame para fazer predição
        features_df = pd.DataFrame([features])
        
        # Faz a predição (tempo de volta, por exemplo)
        prediction = model.predict(features_df)[0]
        
        print(f"Tempo estimado de volta: {prediction:.2f} segundos")
        
        time.sleep(1)

# Inicializando a sessão e o modelo
def start_live_assistant():
    year = 2023
    gp = "Monaco"  # Exemplo de GP
    session_code = "R"  # Exemplo de sessão (corrida)

    session = fastf1.get_session(year, gp, session_code)
    session.load()

    # Carregar o modelo
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    model = joblib.load(model_path)
    
    # Iniciar a predição em tempo real
    real_time_prediction(session, model)


if __name__ == "__main__":
    start_live_assistant()