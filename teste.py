import joblib

# Permite carregar artefatos pickled que referenciam o módulo `target_encoding`
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
FEATURES_DIR = SRC_DIR / "features"
for _path in (SRC_DIR, FEATURES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Import explícito para garantir que joblib/pickle consiga resolver o módulo ao deserializar
import target_encoding  # noqa: F401

enc = joblib.load(PROJECT_ROOT / "models" / "global_target_encoder_v3.pkl")

print("Driver target-encoding mappings:")
for drv, val in sorted(enc.encodings["Driver"].items(), key=lambda x: x[1]):
    print(f"  {drv}: {val:.4f}")

print("\nTeam target-encoding mappings:")
for team, val in sorted(enc.encodings["Team"].items(), key=lambda x: x[1]):
    print(f"  {team}: {val:.4f}")

print("\nGlobal means (fallback for unseen categories):")
for col, gm in enc.global_means.items():
    print(f"  {col}: {gm:.4f}")