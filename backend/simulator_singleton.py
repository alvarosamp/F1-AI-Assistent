"""Carrega o RaceSimulator uma única vez e reutiliza entre requisições."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "simulation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

MODELS_DIR = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def get_simulator():
    try:
        from race_simulate import RaceSimulator
    except ModuleNotFoundError:
        from race_simulator import RaceSimulator

    return RaceSimulator()
