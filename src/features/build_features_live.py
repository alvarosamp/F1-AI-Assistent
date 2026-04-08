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
    """
    Coleta os dados da telemetria para cada volta da sessão.
    """
    session.load()

    laps = session.laps
    car_data = laps.get_car_data()
    car_data = car_data.add_distance()  # Adiciona a distância percorrida ao dataframe

    return car_data

# Função para extrair features em tempo real
def extract_features(car_data):
    """
    Extrai as principais features da telemetria.
    """
    features = {
        "speed_mean": car_data["Speed"].mean(),               # Velocidade média
        "speed_max": car_data["Speed"].max(),                 # Velocidade máxima
        "speed_std": car_data["Speed"].std(),                 # Desvio padrão da velocidade
        "throttle_mean": car_data["Throttle"].mean(),         # Aceleração média
        "throttle_std": car_data["Throttle"].std(),           # Desvio padrão da aceleração
        "brake_mean": car_data["Brake"].mean(),               # Média de frenagem
        "brake_ratio": (car_data["Brake"] > 0).mean(),        # Proporção de tempo com freio acionado
        "rpm_mean": car_data["RPM"].mean(),                   # RPM médio
        "gear_mean": car_data["nGear"].mean(),                # Marcha média
        "drs_ratio": (car_data["DRS"] > 0).mean(),            # Proporção de tempo com DRS ativado
        "lap_time_mean_3": car_data["LapTime"].mean(),        # Tempo médio das últimas 3 voltas
        "lap_time_delta": car_data["LapTime"].diff().mean(),  # Diferença de tempo entre as voltas
    }
    return features

# Função de predição em tempo real
def real_time_prediction(session, model):
    """
    Roda uma predição em tempo real durante a corrida, coletando dados continuamente.
    """
    while True:
        print("Coletando dados de telemetria ao vivo...")

        # Coleta os dados ao vivo
        car_data = collect_telemetry_data(session)
        
        # Extrai as features necessárias para a predição
        features = extract_features(car_data)
        
        # Prepara o DataFrame para fazer predição
        features_df = pd.DataFrame([features])
        
        # Faz a predição (tempo de volta, por exemplo)
        prediction = model.predict(features_df)[0]
        
        # Exibe o tempo de volta estimado
        print(f"Tempo estimado de volta: {prediction:.2f} segundos")
        
        # Aguardar o próximo dado de telemetria (delay de 1 segundo entre os ciclos)
        time.sleep(1)

# Inicializando a sessão e o modelo
def start_live_assistant():
    """
    Inicia o assistente de predição ao vivo, carregando o modelo e os dados.
    """
    year = 2023
    gp = "Monaco"  # Exemplo de GP (você pode mudar para o GP atual)
    session_code = "R"  # Exemplo de sessão (corrida)

    session = fastf1.get_session(year, gp, session_code)
    session.load()

    # Carregar o modelo treinado
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    model = joblib.load(model_path)
    
    # Iniciar a predição em tempo real
    real_time_prediction(session, model)


if __name__ == "__main__":
    start_live_assistant()