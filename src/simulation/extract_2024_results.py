# src/simulation/extract_2024_results.py
import pandas as pd, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "telemetry_full_v2.csv")
df_race = df[(df['session_code']=='R') & (df['year']==2024)].copy()

race_max = df_race.groupby('gp')['LapNumber'].max()
drv_max = df_race.groupby(['gp','Driver'])['LapNumber'].max().reset_index().rename(columns={'LapNumber':'drv_last_lap'})
drv_max['race_max'] = drv_max['gp'].map(race_max)
drv_max['dnf'] = (drv_max['drv_last_lap'] < 0.9 * drv_max['race_max']).astype(int)

last = df_race.sort_values('LapNumber').groupby(['gp','Driver']).last().reset_index()
first = df_race.sort_values('LapNumber').groupby(['gp','Driver']).first().reset_index()
merged = drv_max.merge(last[['gp','Driver','Position','Team']], on=['gp','Driver'])
merged = merged.merge(first[['gp','Driver','grid_position','quali_position','gap_to_pole_ms']], on=['gp','Driver'], how='left')

results = []
for gp in sorted(merged['gp'].unique()):
    gp_data = merged[merged['gp']==gp]
    for rank, (_, r) in enumerate(gp_data[gp_data['dnf']==0].sort_values('Position').iterrows(), 1):
        results.append({'gp':gp,'driver':r['Driver'],'team':r['Team'],
            'grid_pos':int(r['grid_position']) if pd.notna(r['grid_position']) else rank,
            'quali_pos':int(r['quali_position']) if pd.notna(r['quali_position']) else rank,
            'gap_to_pole_ms':float(r['gap_to_pole_ms']) if pd.notna(r['gap_to_pole_ms']) else rank*200,
            'final_position':rank,'dnf':0})
    for _, r in gp_data[gp_data['dnf']==1].iterrows():
        results.append({'gp':gp,'driver':r['Driver'],'team':r['Team'],
            'grid_pos':int(r['grid_position']) if pd.notna(r['grid_position']) else 20,
            'quali_pos':int(r['quali_position']) if pd.notna(r['quali_position']) else 20,
            'gap_to_pole_ms':float(r['gap_to_pole_ms']) if pd.notna(r['gap_to_pole_ms']) else 5000,
            'final_position':99,'dnf':1})

rdf = pd.DataFrame(results)
out = PROJECT_ROOT / "data" / "processed" / "results_2024_real.csv"
rdf.to_csv(out, index=False)
print(f"[OK] {len(rdf)} entries, {rdf['dnf'].sum()} DNFs, {rdf['gp'].nunique()} GPs -> {out}")