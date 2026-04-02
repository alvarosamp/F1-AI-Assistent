import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def main():
    print("[DEBUG] Criando pasta 'models' se não existir...")
    os.makedirs("models", exist_ok=True)
    print("[DEBUG] Lendo features.csv...")
    df = pd.read_csv(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\F1-AI-Assistent\data\processed\features.csv")
    print("[DEBUG] Primeiras linhas do dataframe:")
    print(df.head())
    features = [
        'TyreLife',
        'Compound',
        'lap_time_mean_3',
        'lap_time_delta',
        'tyre_ratio'
    ]
    
    target = 'LapTime'
    
    print("[DEBUG] Selecionando features e target...")
    X = df[features]
    y = df[target]
    
    print("[DEBUG] Realizando train_test_split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[DEBUG] Instanciando e treinando o modelo XGBRegressor...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print("[DEBUG] Modelo treinado.")
    
    print("[DEBUG] Realizando predições e calculando métricas...")
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    
    
    print("[DEBUG] Garantindo que a pasta 'modelos' existe...")
    os.makedirs("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\src\\models\\modelos", exist_ok=True)
    print("[DEBUG] Salvando modelo em modelos/model.pkl ...")
    joblib.dump(model, "C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\src\\models\\modelos\\model.pkl")

    print("[DEBUG] Modelo salvo com sucesso!")


if __name__ == "__main__":
    main()