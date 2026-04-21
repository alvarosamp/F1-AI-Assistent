"""
Treina o modelo de DNF (Entrega 1 do Sprint 3).

DECISÕES DE DESIGN (validadas empiricamente):
    - Logistic Regression (C=0.5) — bate Brier 0.0873 no walk-forward
    - SEM calibração pós-hoc — medimos que o modelo JÁ está calibrado
      (ECE=0.0137). Isotonic regression overfittou (ECE subiu pra 0.062),
      Platt scaling não adicionou nada (ECE 0.0160).
    - GradientBoosting e XGBoost classifier foram TESTADOS e descartados:
      overfit em dataset pequeno (876 amostras de treino).

VALIDAÇÃO:
    - Walk-forward: treina em 2022+2023, testa em 2024
    - Baselines reportados: trivial (sempre média global) e naive (taxa
      histórica do piloto sem modelo)
    - Métricas: Brier score, Log Loss, ECE (Expected Calibration Error),
      reliability diagram por bins

NÚMEROS ESPERADOS (medidos no dataset real):
    Trivial baseline:   Brier=0.0895  LogLoss=0.3253  ECE=0.0284
    LR uncalibrated:    Brier=0.0873  LogLoss=0.3129  ECE=0.0137  <- este
    LR + Platt:         Brier=0.0879  LogLoss=0.3168  ECE=0.0160
    LR + Isotonic:      Brier=0.0911  LogLoss=0.3559  ECE=0.0618  (overfit)

Se seu output divergir mais de ±0.003 nos Brier, algo mudou no dataset.

OUTPUT:
    models/dnf_model.pkl
    models/dnf_metrics.json
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dnf_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "dnf_model.pkl"
METRICS_PATH = MODELS_DIR / "dnf_metrics.json"

FEATURES = [
    "dnf_rate_driver",
    "dnf_rate_team",
    "dnf_rate_gp",
    "regulation_era",
    "grid_position",
    "quali_position",
]

TRAIN_YEARS = [2022, 2023]
TEST_YEAR = 2024


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE: média ponderada do gap entre probabilidade prevista e taxa real,
    agregado em bins. Métrica padrão pra medir calibração de probabilidade.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() > 0:
            gap = abs(p[mask].mean() - y[mask].mean())
            total_ece += (mask.sum() / len(p)) * gap
    return float(total_ece)


def reliability_diagram(p: np.ndarray, y: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Retorna os bins do reliability diagram como lista de dicts (p_pred, p_real, n)."""
    bins = np.linspace(0, max(p.max() + 0.01, 0.2), n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0:
            continue
        out.append({
            "bin_lo": float(bins[i]),
            "bin_hi": float(bins[i + 1]),
            "n": int(mask.sum()),
            "pred_mean": float(p[mask].mean()),
            "real_rate": float(y[mask].mean()),
        })
    return out


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Rode build_dnf_dataset.py primeiro. Não achei {DATA_FILE}")

    ds = pd.read_csv(DATA_FILE)
    print(f"Dataset: {ds.shape}")
    print(f"DNF rate global: {ds['dnf'].mean()*100:.2f}%")

    train_mask = ds["year"].isin(TRAIN_YEARS)
    test_mask = ds["year"] == TEST_YEAR

    Xtr_raw = ds[train_mask][FEATURES]
    Xte_raw = ds[test_mask][FEATURES]

    # Fallback de NaN pras features históricas (primeiras corridas do ano)
    median = Xtr_raw.median()
    Xtr = Xtr_raw.fillna(median)
    Xte = Xte_raw.fillna(median)

    ytr = ds[train_mask]["dnf"].values
    yte = ds[test_mask]["dnf"].values

    print(f"\nTreino: {len(Xtr)} amostras, {int(ytr.sum())} DNFs ({ytr.mean()*100:.1f}%)")
    print(f"Teste:  {len(Xte)} amostras, {int(yte.sum())} DNFs ({yte.mean()*100:.1f}%)")

    # ===== BASELINES =====
    print("\n=" * 30)
    print("BASELINES")
    print("=" * 30)
    trivial = np.full(len(yte), ytr.mean())
    brier_trivial = brier_score_loss(yte, trivial)
    ll_trivial = log_loss(yte, trivial)
    ece_trivial = expected_calibration_error(trivial, yte)
    print(f"Trivial (sempre média treino):")
    print(f"  Brier = {brier_trivial:.4f}")
    print(f"  LogLoss = {ll_trivial:.4f}")
    print(f"  ECE = {ece_trivial:.4f}")

    naive = np.clip(Xte["dnf_rate_driver"].values, 0.001, 0.999)
    brier_naive = brier_score_loss(yte, naive)
    ll_naive = log_loss(yte, naive)
    print(f"\nNaive (taxa histórica do piloto):")
    print(f"  Brier = {brier_naive:.4f}")
    print(f"  LogLoss = {ll_naive:.4f}")

    # ===== MODELO =====
    print("\n" + "=" * 30)
    print("MODELO")
    print("=" * 30)
    model = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    model.fit(Xtr, ytr)

    preds = model.predict_proba(Xte)[:, 1]
    brier = brier_score_loss(yte, preds)
    ll = log_loss(yte, preds)
    ece = expected_calibration_error(preds, yte)

    print(f"LogisticRegression (C=0.5):")
    print(f"  Brier = {brier:.4f}   (baseline trivial: {brier_trivial:.4f})")
    print(f"  LogLoss = {ll:.4f}    (baseline trivial: {ll_trivial:.4f})")
    print(f"  ECE = {ece:.4f}       (baseline trivial: {ece_trivial:.4f})")

    improvement_brier = (brier_trivial - brier) / brier_trivial * 100
    print(f"\nGanho sobre trivial (Brier): {improvement_brier:+.1f}%")

    if brier < brier_trivial:
        print("[OK] Modelo bate o baseline trivial.")
    else:
        print("[WARN] Modelo NÃO bate o baseline trivial. Revisar features.")

    if ece < 0.02:
        print("[OK] ECE < 0.02 — modelo bem calibrado, sem necessidade de Platt/Isotonic.")
    else:
        print("[WARN] ECE >= 0.02 — considerar calibração pós-hoc.")

    # ===== FEATURE IMPORTANCE =====
    print("\n" + "=" * 30)
    print("COEFICIENTES DO MODELO")
    print("=" * 30)
    for f, c in zip(FEATURES, model.coef_[0]):
        print(f"  {f:25s} {c:+.4f}")
    print(f"  {'intercept':25s} {model.intercept_[0]:+.4f}")

    # ===== RELIABILITY DIAGRAM =====
    print("\n" + "=" * 30)
    print("RELIABILITY DIAGRAM")
    print("=" * 30)
    rel = reliability_diagram(preds, yte, n_bins=5)
    print(f"  {'bin':20s} {'n':>5s} {'pred':>8s} {'real':>8s}")
    for b in rel:
        print(f"  [{b['bin_lo']:.3f}-{b['bin_hi']:.3f}]     {b['n']:5d}  {b['pred_mean']:.4f}   {b['real_rate']:.4f}")

    # ===== SAVE =====
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "median_fill": median.to_dict()}, MODEL_PATH)

    metrics = {
        "baseline_trivial": {
            "brier": brier_trivial,
            "log_loss": ll_trivial,
            "ece": ece_trivial,
        },
        "model": {
            "type": "LogisticRegression",
            "C": 0.5,
            "brier": brier,
            "log_loss": ll,
            "ece": ece,
            "improvement_over_trivial_pct": improvement_brier,
        },
        "features": FEATURES,
        "coefficients": dict(zip(FEATURES, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
        "reliability_diagram": rel,
        "train_rows": int(len(Xtr)),
        "test_rows": int(len(Xte)),
        "train_years": TRAIN_YEARS,
        "test_year": TEST_YEAR,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Modelo salvo em {MODEL_PATH}")
    print(f"[OK] Métricas salvas em {METRICS_PATH}")


if __name__ == "__main__":
    main()