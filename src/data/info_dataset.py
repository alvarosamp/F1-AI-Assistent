import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "data" / "raw" / "telemetry_full.csv"
output_path = project_root / "data" / "processed" / "telemetry_full_head.csv"

df = pd.read_csv(input_path)

head_df = df.head()
output_path.parent.mkdir(parents=True, exist_ok=True)
head_df.to_csv(output_path, index=False)

print(head_df)

print(df.shape)