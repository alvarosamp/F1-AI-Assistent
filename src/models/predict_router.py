from __future__ import annotations

from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_MODEL_PATH = PROJECT_ROOT / "models" / "global_model_optuna.pkl"
LOCAL_DIR = PROJECT_ROOT / "models" / "per_track_optuna"


def sanitize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


class ModelRouter:
    def __init__(self) -> None:
        if not GLOBAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo global não encontrado: {GLOBAL_MODEL_PATH}")

        self.global_model = joblib.load(GLOBAL_MODEL_PATH)
        self.local_models = {}

        if LOCAL_DIR.exists():
            for file in LOCAL_DIR.glob("*.pkl"):
                self.local_models[file.stem] = joblib.load(file)

    def predict(self, X, track_name: str):
        track_slug = sanitize_name(track_name)

        if track_slug in self.local_models:
            print(f"[ROUTER] Usando modelo local para: {track_name}")
            model = self.local_models[track_slug]
        else:
            print(f"[ROUTER] Usando modelo global (fallback) para: {track_name}")
            model = self.global_model

        return model.predict(X)