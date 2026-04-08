from __future__ import annotations

from pathlib import Path
import json
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "telemetry_features.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "per_track_optuna"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRACKS_TO_OPTIMIZE = [
    "Monaco Grand Prix",
    "Japanese Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Azerbaijan Grand Prix",
]


def sanitize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "is_raining" not in df.columns and "Rainfall" in df.columns:
        df["is_raining"] = (pd.to_numeric(df["Rainfall"], errors="coerce").fillna(0) > 0).astype(int)

    if "session_code_encoded" not in df.columns:
        df["session_code_encoded"] = df["session_code"].astype("category").cat.codes

    if "team_encoded" not in df.columns:
        df["team_encoded"] = df["Team"].astype("category").cat.codes

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["LapTimeResidual"]).copy()
    return df


def get_feature_list(df: pd.DataFrame) -> list[str]:
    candidate_features = [
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
        "is_raining",
        "session_code_encoded",
        "team_encoded",
    ]
    return [c for c in candidate_features if c in df.columns]


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = prepare_dataframe(df)

    mlflow.set_experiment("f1-local-optuna")
    all_results = []

    for gp_name in TRACKS_TO_OPTIMIZE:
        track_df = df[df["gp"] == gp_name].copy()

        if len(track_df) < 150:
            print(f"[SKIP] {gp_name}: poucas linhas ({len(track_df)})")
            continue

        features = get_feature_list(track_df)
        X = track_df[features].fillna(0)
        y = track_df["LapTimeResidual"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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

            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                **params,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            return rmse

        study = optuna.create_study(direction="minimize", study_name=f"local_{gp_name}")
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        final_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **best_params,
        )
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        print(f"[LOCAL OPTUNA] {gp_name} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.2f}")

        track_slug = sanitize_name(gp_name)

        with mlflow.start_run(run_name=f"local_optuna_{track_slug}"):
            mlflow.log_param("scope", "local_track")
            mlflow.log_param("gp", gp_name)
            mlflow.log_param("target", "LapTimeResidual")
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("rows", len(track_df))

            for k, v in best_params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            model_path = MODELS_DIR / f"{track_slug}.pkl"
            joblib.dump(final_model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")

            params_path = MODELS_DIR / f"{track_slug}_best_params.json"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2)
            mlflow.log_artifact(str(params_path), artifact_path="optuna")

        all_results.append({
            "gp": gp_name,
            "rmse_local_optuna": rmse,
            "mae_local_optuna": mae,
            "r2_local_optuna": r2,
            "rows": len(track_df),
        })

    results_df = pd.DataFrame(all_results)
    results_file = MODELS_DIR / "results_local_optuna.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n[INFO] Resultados locais Optuna salvos em: {results_file}")
    
if __name__ == "__main__":
    main()