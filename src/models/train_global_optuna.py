from __future__ import annotations

from pathlib import Path
import json
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\F1-AI-Assistent\\data\\processed\\telemetry_features_race.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "LapTimeResidual"


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "gp",
        "session_code",
        "Team",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    if TARGET_COL not in df.columns:
        if "LapTime" not in df.columns:
            raise ValueError(
                f"Coluna alvo '{TARGET_COL}' ausente e não foi possível calculá-la (faltando 'LapTime')."
            )

        # Normaliza LapTime (pode vir como float em segundos ou como string)
        df["LapTime"] = pd.to_numeric(df["LapTime"], errors="coerce")

        group_cols = [c for c in ["year", "gp", "session_code"] if c in df.columns]
        if not group_cols:
            # Fallback: baseline global se não houver colunas de agrupamento
            df["session_baseline"] = df["LapTime"].median()
        else:
            df["session_baseline"] = df.groupby(group_cols)["LapTime"].transform("median")

        df[TARGET_COL] = df["LapTime"] - df["session_baseline"]

    # Alguns datasets antigos usam 'Compound' já codificado (numérico)
    if "CompoundEncoded" not in df.columns and "Compound" in df.columns:
        compound_numeric = pd.to_numeric(df["Compound"], errors="coerce")
        if compound_numeric.notna().any():
            df["CompoundEncoded"] = compound_numeric

    numeric_candidates = [
        "TyreLife",
        "CompoundEncoded",
        "Position",
        "TrackStatus",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "WindSpeed",
        "speed_mean",
        "speed_max",
        "speed_std",
        "throttle_mean",
        "throttle_std",
        "brake_ratio",
        "rpm_mean",
        "gear_mean",
        "drs_ratio",
        "lap_time_mean_3",
        "lap_time_delta",
        "speed_delta",
        "throttle_delta",
        "brake_delta",
        "degradation_score",
        "aggression_score",
        "consistency_score",
        "efficiency_score",
        "drs_usage_intensity",
        "tyre_ratio",
        "stint_progress",
        "LapTimeResidual",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_raining" not in df.columns:
        if "Rainfall" in df.columns:
            df["is_raining"] = (pd.to_numeric(df["Rainfall"], errors="coerce").fillna(0) > 0).astype(int)
        else:
            df["is_raining"] = 0

    if "session_code_encoded" not in df.columns:
        df["session_code_encoded"] = df["session_code"].astype("category").cat.codes

    if "team_encoded" not in df.columns:
        df["team_encoded"] = df["Team"].astype("category").cat.codes

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL]).copy()
    return df


def get_feature_list(df: pd.DataFrame) -> list[str]:
    candidate_features = [
        # Conhecidas no início da volta (sem leakage por construção)
        "TyreLife",
        "CompoundEncoded",
        "Position",
        "TrackStatus",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "WindSpeed",
        "is_raining",
        "session_code_encoded",
        "team_encoded",
        "tyre_ratio",
        "stint_progress",

        # Histórico de lap time (todas shiftadas)
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

        # Deltas da telemetria (N-1 vs N-2)
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
    return [c for c in candidate_features if c in df.columns]


def evaluate_logo(X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict) -> dict:
    logo = LeaveOneGroupOut()
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_group = str(groups.iloc[test_idx].iloc[0])

        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        fold_results.append({
            "fold": fold_idx,
            "test_gp": test_group,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
        })

    results_df = pd.DataFrame(fold_results)
    return {
        "fold_results": results_df,
        "rmse_mean": float(results_df["rmse"].mean()),
        "rmse_std": float(results_df["rmse"].std(ddof=0)),
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std(ddof=0)),
        "r2_mean": float(results_df["r2"].mean()),
        "r2_std": float(results_df["r2"].std(ddof=0)),
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = prepare_dataframe(df)

    features = get_feature_list(df)
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COL]
    groups = df["gp"]

    mlflow.set_experiment("f1-global-optuna")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }

        metrics = evaluate_logo(X, y, groups, params)
        return metrics["rmse_mean"]

    study = optuna.create_study(direction="minimize", study_name="f1_global_xgb")
    study.optimize(objective, n_trials=75)

    best_params = study.best_params
    best_metrics = evaluate_logo(X, y, groups, best_params)
    fold_results = best_metrics["fold_results"]

    print("\n===== GLOBAL OPTUNA =====")
    print("Best params:", best_params)
    print(f"RMSE médio: {best_metrics['rmse_mean']:.2f} ± {best_metrics['rmse_std']:.2f}")
    print(f"MAE médio : {best_metrics['mae_mean']:.2f} ± {best_metrics['mae_std']:.2f}")
    print(f"R2 médio  : {best_metrics['r2_mean']:.2f} ± {best_metrics['r2_std']:.2f}")

    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X, y)

    with mlflow.start_run(run_name="global_optuna_best"):
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("validation_strategy", "LeaveOneGroupOut")
        mlflow.log_param("group_col", "gp")
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_groups", int(groups.nunique()))

        for k, v in best_params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("rmse_mean", best_metrics["rmse_mean"])
        mlflow.log_metric("rmse_std", best_metrics["rmse_std"])
        mlflow.log_metric("mae_mean", best_metrics["mae_mean"])
        mlflow.log_metric("mae_std", best_metrics["mae_std"])
        mlflow.log_metric("r2_mean", best_metrics["r2_mean"])
        mlflow.log_metric("r2_std", best_metrics["r2_std"])

        model_path = MODELS_DIR / "global_model_optuna.pkl"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")

        feature_file = MODELS_DIR / "global_feature_columns.txt"
        with open(feature_file, "w", encoding="utf-8") as f:
            f.write("\n".join(features))
        mlflow.log_artifact(str(feature_file), artifact_path="metadata")

        importance_df = pd.DataFrame({
            "feature": features,
            "importance": final_model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance_file = MODELS_DIR / "global_feature_importance_optuna.csv"
        importance_df.to_csv(importance_file, index=False)
        mlflow.log_artifact(str(importance_file), artifact_path="metadata")

        fold_results_file = MODELS_DIR / "global_fold_results_optuna.csv"
        fold_results.to_csv(fold_results_file, index=False)
        mlflow.log_artifact(str(fold_results_file), artifact_path="cv_results")

if __name__ == "__main__":
    main()