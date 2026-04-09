# diagnose.py — rodar com: python diagnose.py
import pandas as pd
from pathlib import Path

PROCESSED = Path("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\data\\processed\\telemetry_features.csv")
df = pd.read_csv(PROCESSED)

print("="*60)
print("1. SESSÕES NO DATASET")
print("="*60)
print(df.groupby("session_code").size().sort_values(ascending=False))
print()

print("="*60)
print("2. ANOS NO DATASET")
print("="*60)
print(df.groupby("year").size())
print()

print("="*60)
print("3. TRACKSTATUS — distribuição")
print("="*60)
if "TrackStatus" in df.columns:
    print(df["TrackStatus"].value_counts().head(10))
    print(f"dtype: {df['TrackStatus'].dtype}")
else:
    print("TrackStatus AUSENTE — esse é o problema do filter")
print()

print("="*60)
print("4. DISTRIBUIÇÃO DO TARGET (LapTimeResidual)")
print("="*60)
print(df["LapTimeResidual"].describe(percentiles=[.01,.05,.5,.95,.99]))
print()

print("="*60)
print("5. LapTime bruto — quantas voltas absurdas sobraram?")
print("="*60)
print(df["LapTime"].describe(percentiles=[.01,.05,.5,.95,.99]))
print(f"Voltas > 120s: {(df['LapTime']>120).sum()}")
print(f"Voltas > 150s: {(df['LapTime']>150).sum()}")
print()

print("="*60)
print("6. RESIDUAL POR SESSION_CODE — quali deveria ter residual menor")
print("="*60)
print(df.groupby("session_code")["LapTimeResidual"].agg(["mean","std","min","max"]))
print()

print("="*60)
print("7. PRE-SEASON TESTING ainda no dataset?")
print("="*60)
print(df[df["gp"].str.contains("Test", case=False, na=False)]["gp"].value_counts())