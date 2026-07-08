from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from backend.reference_data import MODEL_METRICS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

router = APIRouter(prefix="/api/model-analysis", tags=["model-analysis"])


@router.get("")
def get_model_analysis():
    result = {"metrics": MODEL_METRICS, "feature_importance": [], "rmse_by_gp": []}

    imp_file = MODELS_DIR / "global_feature_importance_v3.csv"
    if imp_file.exists():
        imp = pd.read_csv(imp_file)
        result["feature_importance"] = imp.head(20).to_dict(orient="records")

    diag_file = MODELS_DIR / "diagnose_model_v2_report.csv"
    if diag_file.exists():
        diag = pd.read_csv(diag_file)
        diag_sorted = diag.sort_values(by="rmse", ascending=False)
        result["rmse_by_gp"] = diag_sorted.to_dict(orient="records")

    return result
