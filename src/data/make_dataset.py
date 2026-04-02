import fastf1
import pandas as pd
import os

def main():
    os.makedirs("data/raw", exist_ok=True)
    cache_dir = os.path.join("..", "..", "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(2023, 'Monaco', 'R')
    session.load()

    laps = session.laps

    df = laps[['Driver', 'LapTime', 'Compound', 'TyreLife', 'Position']].dropna()
    df['LapTime'] = df['LapTime'].dt.total_seconds()
    print(df.head())
    df.to_csv("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\data\\raw\\data.csv", index=False)

if __name__ == "__main__":
    main()