"""
Treinamento do modelo global de lap time residual em f1 com:
    - GroupKFold (5 folds) em vez de LOGO -> mais estavel
    - Target Encoding CV-safe para Driver, team, gp
    - Features de contexto de corrida (LapNumber, lapNumber_pct, Stint)
    - Optuna para tuning de hiperparametros
    - MLflow para tracking
    - Metrics ganho sobre baseline trivial" reportada explicitamente

Por que GrupoKFold em vez de LeaveOneGroupOut? 
    LOGO com 22 folds, cada um testando 1 pista só. A variancia entre folds é altissima
    (pistas molhadas destroem o RMSE do fold), e a média fica enganosa. GroupKFold com 5 folds agrupa 4-5
    pistas por fold de teste, reduz ruido, e a ainda testa generalizaçao cross-pista

Por que target encoding em vez de label encoding ?
    Label encoding trata codigo como ordinais : - gp=17 parece estar entre 
    gp=16 e gp = 18, oque é nonsense. Com validaçao por grupo, categorias de tete
    nunca foram vistas no treino, e o modelo trata o codigo ordinal como se tivesse relaçao com o target. Target encoding captura a relaçao entre categoria e target sem assumir ordenaçao, e é CV-safe (calcula o encoding usando apenas as linhas de treino de cada fold).    
    'qual o residual medio deste piloto no treino ?"" vira a feature, e categorias novas caem no global_mean
"""


from __future__ import annotations

from pathlib import Path
import json
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\data\\processed\\telemetry_features_race.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
from src.features.target_encoding import TargetEncoderCV

TARGET_COL = "LapTimeResidual"
CATEGORICAL_COLS = ["Driver", "Team", "gp"]
N_SPLITS = 5
N_TRIALS = 60
 
# Features numéricas candidatas — só as que efetivamente existem no df serão usadas
NUMERIC_CANDIDATES = [
    # Contexto de corrida (features críticas descobertas na análise)
    "LapNumber",
    "Stint",
    "LapNumber_pct",
    "tyre_x_progress",
    "compound_x_tyre",
 
    # Estado conhecido no início da volta
    "TyreLife",
    "CompoundEncoded",
    "Position",
    "tyre_ratio",
    "stint_progress",
 
    # Clima (quando disponível)
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed",
    "is_raining",
 
    # Histórico de lap time (shiftado)
    "lap_time_prev",
    "lap_time_mean_3_prev",
    "lap_time_delta_prev",
    "lap_time_std_5_prev",
 
    # Telemetria da volta anterior
    "speed_mean_prev",
    "speed_max_prev",
    "speed_std_prev",
    "throttle_mean_prev",
    "throttle_std_prev",
    "brake_ratio_prev",
    "rpm_mean_prev",
    "gear_mean_prev",
    "drs_ratio_prev",
 
    # Deltas de telemetria (N-1 vs N-2)
    "speed_mean_delta_prev",
    "speed_max_delta_prev",
    "throttle_mean_delta_prev",
    "brake_ratio_delta_prev",
 
    # Scores derivados (todos baseados em _prev)
    "degradation_score_prev",
    "aggression_score_prev",
    "efficiency_score_prev",
    "drs_usage_intensity_prev",
    "consistency_score_prev",
]


def prepare_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    # Garante is_raining seja booleano
    if "is_raining" not in df.columns and 'Rainfall' in df.columns:
        df["is_raining"] = (
            pd.to_numeric(df["Rainfall"], errors="coerce").fillna(0) > 0
        ).astype(int)
    #Normaliza os dados
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    #Remove infinitos 
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL]).copy()
    return df

def get_feature_list(df : pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC_CANDIDATES if c in df.columns]

def build_features_for_fold(
        df: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        numeric_features: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoderCV]:
    """
    Constroi X_train e X_test para um fold do CV. O target encoder é 
    fitado só no treino do fold - sem leakage
    """
    X_tr_num = df.iloc[train_idx][numeric_features].reset_index(drop=True)
    X_te_num = df.iloc[test_idx][numeric_features].reset_index(drop=True)
    encoder = TargetEncoderCV(cols=CATEGORICAL_COLS, smoothing = 20.0)
    encoder.fit(df.iloc[train_idx], TARGET_COL)
    tr_te = encoder.transform(df.iloc[train_idx]).reset_index(drop=True)
    te_te = encoder.transform(df.iloc[test_idx]).reset_index(drop=True)
    X_tr = pd.concat([X_tr_num, tr_te], axis=1)
    X_te = pd.concat([X_te_num, te_te], axis=1)
    return X_tr, X_te, encoder

def evaluate_groupkfold(
        df:pd.DataFrame,
        numeric_features: list[str],
        params : dict,
        n_splits : int = N_SPLITS
    ) -> dict:
    y = df[TARGET_COL]
    groups = df["gp"].values 
    gfk = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gfk.split(df, y, groups = groups), start = 1):
        X_tr, X_te, _ = build_features_for_fold(df, tr_idx, te_idx, numeric_features)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)

        model = XGBRegressor(
            objective = "reg:squarederror",
            random_state = 42,
            n_jobs = -1,
            tree_method = "hist",
            **params

        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        rmse = float(mean_squared_error(y_te, preds, squared=False))
        mae = float(mean_absolute_error(y_te, preds))
        r2 = float(r2_score(y_te, preds))
        fold_results.append({"fold": fold, "rmse": rmse, "mae": mae, "r2": r2})
        test_gps = sorted(df.iloc[test_idx]["gp"].unique().tolist())
        fold_results.append({
            "fold" : fold_idx,
            "test_gps" : ",".join(test_gps),
            "rmse" : rmse,
            "mae" : mae,
            "r2" : r2
            "train_rows" : len(train_idx),
            "test_rows" : len(test_idx)
        })
        results_df = pd.DataFrame(fold_results)
        std_y = float(y.std())
        rmse_mean = float(results_df["rmse"].mean())

        return {
        "fold_results": results_df,
        "rmse_mean": rmse_mean,
        "rmse_std": float(results_df["rmse"].std(ddof=0)),
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std(ddof=0)),
        "r2_mean": float(results_df["r2"].mean()),
        "r2_std": float(results_df["r2"].std(ddof=0)),
        "baseline_trivial_rmse": std_y,
        "improvement_over_trivial": 1.0 - rmse_mean / std_y,
    }

def main() -> None :
    if not DATA_FILE.exists():
        print(f"Erro: arquivo de dados não encontrado em {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    df = prepare_dataframe(df)
    numeric_features = get_feature_list(df)
    missing_critical = [c for c in ["LapNumber", "Stint", "LapNumber_pct"] if c not in df.columns]
    if missing_critical:
        print(f"\n[AVISO] Features críticas ausentes: {missing_critical}")
        print("Re-rode o build_features.py v3 para gerá-las.")
    mlflow.set_experiment("f1-global-xgb-v2")
    def objective(trial : optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
        metrics = evaluate_groupkfold(df, numeric_features, params)
        return metrics["rmse_mean"]
    
    study = optuna.create_study(direction="minimize", study_name = "f1_global_xgb_v2")
    study.optimize(objective, n_trials = N_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    best_metrics = evaluate_groupkfold(df, numeric_features, best_params)
    fold_results = best_metrics["fold_results"]
    print("\n" + "=" * 60)
    print("GLOBAL v2 — RESULTADOS FINAIS")
    print("=" * 60)
    print(f"Best params: {best_params}")
    print()
    print(f"RMSE médio     : {best_metrics['rmse_mean']:.4f} ± {best_metrics['rmse_std']:.4f}")
    print(f"MAE médio      : {best_metrics['mae_mean']:.4f} ± {best_metrics['mae_std']:.4f}")
    print(f"R²  médio      : {best_metrics['r2_mean']:.4f} ± {best_metrics['r2_std']:.4f}")
    print()
    print(f"Baseline trivial RMSE (std target): {best_metrics['baseline_trivial_rmse']:.4f}")
    print(f"Ganho sobre trivial              : {best_metrics['improvement_over_trivial']*100:.1f}%")
    print()
    print("Resultados por fold:")
    print(fold_results.to_string(index=False))
    print("\nTreinando modelo final no dataset completo...")
    final_encoder = TargetEncoderCV(cols=CATEGORICAL_COLS, smoothing=20.0)
    final_encoder.fit(df, target=TARGET_COL)
    X_num = df[numeric_features].reset_index(drop=True)
    X_cat = final_encoder.transform(df).reset_index(drop=True)
    X_full = pd.concat([X_num, X_cat], axis=1)
    y_full = df[TARGET_COL].reset_index(drop=True)
 
    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )
    final_model.fit(X_full, y_full)
    
