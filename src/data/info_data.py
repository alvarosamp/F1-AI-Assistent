import pandas as pd
df = pd.read_csv(r'C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\F1-AI-Assistent\data\raw\telemetry_full_v2.csv')
print('Shape:', df.shape)
print()
print('Por ano:')
print(df.groupby('year').size())
print()
print('Por ano + sessão:')
print(df.groupby(['year','session_code']).size())
print()
print('Colunas novas:')
for col in ['AirTemp','TrackTemp','Rainfall','is_sc','is_vsc','quali_position','gap_to_pole_ms']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f'  {col}: {non_null} não-nulos ({100*non_null/len(df):.1f}%)')
