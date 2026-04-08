from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

GLOBAL_FILE = MODELS_DIR / "global_fold_results_optuna.csv"
LOCAL_FILE = MODELS_DIR / "per_track_optuna" / "results_local_optuna.csv"
OUTPUT_FILE = MODELS_DIR / "global_vs_local_optuna_comparison.csv"


def main() -> None:
    if not GLOBAL_FILE.exists():
        raise FileNotFoundError(f"Arquivo global não encontrado: {GLOBAL_FILE}")

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(f"Arquivo local não encontrado: {LOCAL_FILE}")

    global_df = pd.read_csv(GLOBAL_FILE)
    local_df = pd.read_csv(LOCAL_FILE)

    global_df = global_df.rename(columns={
        "test_gp": "gp",
        "rmse": "rmse_global",
        "mae": "mae_global",
        "r2": "r2_global",
    })

    global_df = global_df[["gp", "rmse_global", "mae_global", "r2_global"]]
    merged = global_df.merge(local_df, on="gp", how="inner")

    merged["rmse_winner"] = merged.apply(
        lambda row: "local_optuna" if row["rmse_local_optuna"] < row["rmse_global"] else "global_optuna",
        axis=1,
    )
    merged["mae_winner"] = merged.apply(
        lambda row: "local_optuna" if row["mae_local_optuna"] < row["mae_global"] else "global_optuna",
        axis=1,
    )
    merged["r2_winner"] = merged.apply(
        lambda row: "local_optuna" if row["r2_local_optuna"] > row["r2_global"] else "global_optuna",
        axis=1,
    )

    merged["rmse_gain"] = merged["rmse_global"] - merged["rmse_local_optuna"]
    merged["mae_gain"] = merged["mae_global"] - merged["mae_local_optuna"]
    merged["r2_gain"] = merged["r2_local_optuna"] - merged["r2_global"]

    merged = merged.sort_values("rmse_gain", ascending=False)
    merged.to_csv(OUTPUT_FILE, index=False)

    print("\n===== COMPARAÇÃO GLOBAL OPTUNA vs LOCAL OPTUNA =====")
    print(merged[[
        "gp",
        "rmse_global", "rmse_local_optuna", "rmse_winner", "rmse_gain",
        "mae_global", "mae_local_optuna", "mae_winner", "mae_gain",
        "r2_global", "r2_local_optuna", "r2_winner", "r2_gain"
    ]])

    print(f"\n[INFO] Comparação salva em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()