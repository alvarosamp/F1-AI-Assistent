from __future__ import annotations

from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "telemetry_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "LapTime",
        "LapTimeResidual",
        "year",
        "gp",
        "session_code",
        "Team",
    ]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_required}")

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
        "LapTime",
        "LapTimeResidual",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Se o build já criou, só reaproveita.
    if "CompoundEncoded" not in df.columns:
        if "Compound" not in df.columns:
            raise ValueError("Coluna 'Compound' necessária para criar CompoundEncoded")

        compound_map = {
            "SOFT": 0,
            "S": 0,
            "MEDIUM": 1,
            "M": 1,
            "HARD": 2,
            "H": 2,
            "INTERMEDIATE": 3,
            "I": 3,
            "WET": 4,
            "W": 4,
        }

        compound_norm = df["Compound"].astype(str).str.strip().str.upper()
        df["CompoundEncoded"] = compound_norm.map(compound_map).fillna(-1).astype(int)

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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = prepare_dataframe(df)

    target = "LapTimeResidual"
    group_col = "gp"

    if group_col not in df.columns:
        raise ValueError(f"Coluna de grupo ausente: {group_col}")

    features = get_feature_list(df)
    if not features:
        raise ValueError("Nenhuma feature válida encontrada.")

    X = df[features].copy()
    y = df[target].copy()
    groups = df[group_col].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    logo = LeaveOneGroupOut()

    mlflow.set_experiment("f1-lap-time-residual-telemetry-advanced")

    fold_results = []
    best_rmse = float("inf")
    best_model = None
    best_group = None

    with mlflow.start_run(run_name="xgb_logo_advanced_telemetry"):
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            test_group = str(groups.iloc[test_idx].iloc[0])

            model = XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            print(
                f"[Fold {fold_idx}] GP={test_group} | "
                f"RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.2f}"
            )

            fold_results.append({
                "fold": fold_idx,
                "test_gp": test_group,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_group = test_group

        results_df = pd.DataFrame(fold_results)

        mean_rmse = float(results_df["rmse"].mean())
        std_rmse = float(results_df["rmse"].std(ddof=0))
        mean_mae = float(results_df["mae"].mean())
        std_mae = float(results_df["mae"].std(ddof=0))
        mean_r2 = float(results_df["r2"].mean())
        std_r2 = float(results_df["r2"].std(ddof=0))

        print("\n===== MÉDIAS =====")
        print(f"RMSE médio: {mean_rmse:.2f} ± {std_rmse:.2f}")
        print(f"MAE médio : {mean_mae:.2f} ± {std_mae:.2f}")
        print(f"R2 médio  : {mean_r2:.2f} ± {std_r2:.2f}")
        print(f"Melhor fold: {best_group} | RMSE={best_rmse:.2f}")

        mlflow.log_metric("rmse_mean", mean_rmse)
        mlflow.log_metric("rmse_std", std_rmse)
        mlflow.log_metric("mae_mean", mean_mae)
        mlflow.log_metric("mae_std", std_mae)
        mlflow.log_metric("r2_mean", mean_r2)
        mlflow.log_metric("r2_std", std_r2)

        mlflow.log_param("validation_strategy", "LeaveOneGroupOut")
        mlflow.log_param("group_col", group_col)
        mlflow.log_param("target", target)
        mlflow.log_param("model_name", "XGBRegressor")
        mlflow.log_param("n_estimators", 400)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("subsample", 0.9)
        mlflow.log_param("colsample_bytree", 0.9)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_groups", int(groups.nunique()))

        if best_model is None:
            raise RuntimeError("Nenhum modelo foi treinado.")

        model_path = MODELS_DIR / "model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")

        feature_file = MODELS_DIR / "feature_columns.txt"
        with open(feature_file, "w", encoding="utf-8") as f:
            f.write("\n".join(features))
        mlflow.log_artifact(str(feature_file), artifact_path="metadata")

        importance_df = pd.DataFrame({
            "feature": features,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)

        importance_file = MODELS_DIR / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        mlflow.log_artifact(str(importance_file), artifact_path="metadata")

        fold_results_file = MODELS_DIR / "fold_results.csv"
        results_df.to_csv(fold_results_file, index=False)
        mlflow.log_artifact(str(fold_results_file), artifact_path="cv_results")


if __name__ == "__main__":
    main()