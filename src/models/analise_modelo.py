'''
Diagnostico pos treino do modelo global . Para entender onde ele erramais

Gera :
- Feature importance com interpretação
- Erros por GP (quais o modelo generaliza mal)
- Erros por piloto ( quais pilotos o modelo generaliza mal)
- Curva com erro por LapNumber_pct (começo vs fim da corrida)
- Sanity check : modelo bate baseline trivial ? 
'''

from __future__ import annotations
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "telemetry_features_race.csv"

# Permite carregar artefatos pickled que referenciam o módulo `features.*`
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
 
MODEL_PATH = MODELS_DIR / "global_model_v2.pkl"
ENCODER_PATH = MODELS_DIR / "global_target_encoder_v2.pkl"
FEATURES_PATH = MODELS_DIR / "global_feature_columns_v2.json"

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    with open(FEATURES_PATH, "r") as f:
        feat_meta  = json.load(f)
        
    numeric_features = feat_meta ['numeric_features']
    all_features = feat_meta ['all_features_in_order']
    df = pd.read_csv(DATA_FILE)
    # Simula o que build_features v3 produz caso o CSV seja antigo
    if "LapNumber_pct" not in df.columns:
        df["LapNumber_pct"] = df["LapNumber"] / df.groupby(["year", "gp"])["LapNumber"].transform("max")
    if "tyre_x_progress" not in df.columns:
        df["tyre_x_progress"] = df["TyreLife"] * df["LapNumber_pct"]
    if "compound_x_tyre" not in df.columns:
        df["compound_x_tyre"] = df["CompoundEncoded"] * df["TyreLife"]
 
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["LapTimeResidual"])
 
    X_num = df[numeric_features].reset_index(drop=True)
    X_cat = encoder.transform(df).reset_index(drop=True)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X[all_features]  # garante ordem igual ao treino
    y = df["LapTimeResidual"].reset_index(drop=True)
 
    print("Gerando predições no dataset inteiro...")
    preds = model.predict(X)
    errors = y - preds
    abs_errors = np.abs(errors)
    # ================================================================
    # 1. SANITY CHECK — modelo bate trivial?
    # ================================================================
    print("\n" + "=" * 60)
    print("1. SANITY CHECK — modelo vs baseline trivial")
    print("=" * 60)
    rmse = float(np.sqrt((errors ** 2).mean()))
    mae = float(abs_errors.mean())
    trivial_rmse = float(y.std())
    print(f"RMSE do modelo (in-sample)    : {rmse:.4f}")
    print(f"RMSE do baseline trivial (0)  : {trivial_rmse:.4f}")
    print(f"Ganho sobre trivial           : {(1 - rmse/trivial_rmse)*100:.1f}%")
    print("NOTA: este é o RMSE IN-SAMPLE (no treino). O número honesto é")
    print("      o do GroupKFold reportado no train_global_optuna.py.")
    
     # ================================================================
    # 2. FEATURE IMPORTANCE
    # ================================================================
    print("\n" + "=" * 60)
    print("2. FEATURE IMPORTANCE (top 20)")
    print("=" * 60)
    imp = pd.DataFrame({
        "feature": all_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(imp.head(20).to_string(index=False))
    top3 = imp.head(3)["importance"].sum() * 100
    top10 = imp.head(10)["importance"].sum() * 100
    print(f"\nTop 3 concentram  : {top3:.1f}%")
    print(f"Top 10 concentram : {top10:.1f}%")
    if top3 > 65:
        print("[AVISO] Top 3 muito concentrado — modelo pode estar dependendo")
        print("        demais de poucas features. Considere regularização maior.")
        
    # ================================================================
    # 3. ERRO POR GP
    # ================================================================
    print("\n" + "=" * 60)
    print("3. RMSE POR GP (ordenado do pior pro melhor)")
    print("=" * 60)
    df_err = df.copy()
    df_err["abs_error"] = abs_errors.values
    df_err["sq_error"] = (errors ** 2).values
    by_gp = df_err.groupby("gp").agg(
        rmse=("sq_error", lambda s: np.sqrt(s.mean())),
        mae=("abs_error", "mean"),
        n=("abs_error", "size"),
    ).sort_values("rmse", ascending=False)
    print(by_gp.to_string())
 
    # ================================================================
    # 4. ERRO POR PILOTO
    # ================================================================
    print("\n" + "=" * 60)
    print("4. ERRO MÉDIO POR PILOTO (top 5 mais difíceis)")
    print("=" * 60)
    by_drv = df_err.groupby("Driver").agg(
        mae=("abs_error", "mean"),
        n=("abs_error", "size"),
    ).sort_values("mae", ascending=False)
    print(by_drv.head(5).to_string())
    print("\n(top 5 mais fáceis)")
    print(by_drv.tail(5).to_string())
 
    # ================================================================
    # 5. ERRO POR FASE DA CORRIDA
    # ================================================================
    print("\n" + "=" * 60)
    print("5. RMSE POR FASE DA CORRIDA (LapNumber_pct)")
    print("=" * 60)
    df_err["race_phase"] = pd.cut(
        df_err["LapNumber_pct"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["Início (0-20%)", "20-40%", "40-60%", "60-80%", "Fim (80-100%)"],
    )
    by_phase = df_err.groupby("race_phase", observed=True).agg(
        rmse=("sq_error", lambda s: np.sqrt(s.mean())),
        n=("abs_error", "size"),
    )
    print(by_phase.to_string())
 
    # ================================================================
    # OUTPUT — salva um CSV de diagnóstico
    # ================================================================
    diag_path = MODELS_DIR / "diagnose_model_v2_report.csv"
    by_gp.to_csv(diag_path)
    print(f"\n[OK] Diagnóstico por GP salvo em {diag_path}")
 
 
if __name__ == "__main__":
    main()
    
 