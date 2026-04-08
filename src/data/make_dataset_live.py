import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib
# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Cache do FastF1
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Função para coletar dados de telemetria de uma corrida
def get_live_data(session):
    session.load()
    
    # Coleta de dados da telemetria
    laps = session.laps
    car_data = laps.get_car_data()
    car_data = car_data.add_distance()  # Adiciona a distância percorrida

    # Calcula a velocidade média, aceleração média, e outras métricas
    car_data["speed_mean"] = car_data["Speed"].mean()
    car_data["brake_mean"] = car_data["Brake"].mean()
    car_data["throttle_mean"] = car_data["Throttle"].mean()

    return car_data

# Função de predição em tempo real
def live_prediction(session, model, track_name):
    while True:
        print(f"Analisando dados ao vivo para {track_name}...")

        # Coleta os dados de telemetria da corrida
        car_data = get_live_data(session)
        
        # Extrai as features para o modelo de predição
        features = pd.DataFrame([{
            "speed_mean": car_data["speed_mean"].iloc[0],
            "brake_mean": car_data["brake_mean"].iloc[0],
            "throttle_mean": car_data["throttle_mean"].iloc[0],
            "lap_time_mean_3": np.mean(car_data["Speed"]),
            "lap_time_delta": car_data["Speed"].iloc[-1] - car_data["Speed"].iloc[0]
        }])
        
        # Faz a predição
        prediction = model.predict(features)[0]
        
        # Output da predição (exemplo de tempo de volta)
        print(f"Tempo estimado de volta: {prediction:.2f} segundos")
        
        # Aguardar o próximo dado de telemetria (delay de 1 segundo entre os ciclos)
        time.sleep(1)

# Inicializando a sessão e o modelo
def start_assistant():
    year = 2023
    gp = "Monaco"  # Exemplo de GP
    session_code = "R"  # Exemplo de sessão (corrida)

    session = fastf1.get_session(year, gp, session_code)
    session.load()

    # Carregar o modelo treinado
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    model = joblib.load(model_path)
    
    # Iniciar a predição em tempo real
    live_prediction(session, model, gp)


if __name__ == "__main__":
    start_assistant()