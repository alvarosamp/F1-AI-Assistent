import pandas as pd
import os

def main():
    os.makedirs(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\F1-AI-Assistent\data\processed", exist_ok=True)
    df = pd.read_csv(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\F1-AI-Assistent\data\raw\data.csv")
    
    #Codificação de pneuus
    df['Compound'] = df['Compound'].map({'SOFT': 0, 'MEDIUM': 1, 'HARD': 2})
    
    #Media das ultimas voltas
    df['lap_time_mean_3'] = (
        df.groupby('Driver')['LapTime']
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
    )
    
    #Variação de tempo 
    df['lap_time_delta'] = df.groupby('Driver')['LapTime'].diff()
    
    #desgate relativo
    df['tyre_ratio'] = df['TyreLife'] / df.groupby('Driver')['TyreLife'].transform('max')
    
    df = df.fillna(0)
    
    df.to_csv(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\F1-AI-Assistent\data\processed\features.csv", index=False)
    
    
if __name__ == "__main__":
    main()