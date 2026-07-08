import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


@router.get("")
def get_calibration():
    cal_file = MODELS_DIR / "calibration_report.json"
    if not cal_file.exists():
        raise HTTPException(status_code=404, detail="calibration_report.json não encontrado")

    with open(cal_file, encoding="utf-8") as f:
        return json.load(f)
