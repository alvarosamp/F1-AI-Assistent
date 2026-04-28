"""
Calibração do Simulador Monte Carlo contra resultados reais de 2024.

O QUE FAZ:
    1. Pra cada uma das 24 corridas de 2024, roda o simulador com o grid
       real de largada e coleta as probabilidades previstas
    2. Compara probabilidades previstas com resultados reais:
       - Quando o modelo diz "70% de podium", acerta ~70%?
    3. Gera reliability diagram, Brier score, ECE por mercado
    4. Compara com baseline "grid position = resultado final"

MÉTRICAS:
    - Brier Score: média de (probabilidade - resultado)² — menor é melhor
    - ECE (Expected Calibration Error): gap médio entre previsão e realidade
    - Reliability diagram: gráfico bins de previsão vs taxa real
    - Comparação: modelo vs baseline trivial (grid position)

OUTPUT:
    models/calibration_report.json — métricas numéricas
    Terminal: reliability diagrams e Brier scores por mercado
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulation.race_simulate import RaceSimulator  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_FILE = PROJECT_ROOT / "data" / "processed" / "results_2024_real.csv"

# Total de voltas por GP (aproximado — varia por pista)
GP_LAPS = {
    "Bahrain Grand Prix": 57, "Saudi Arabian Grand Prix": 50,
    "Australian Grand Prix": 58, "Japanese Grand Prix": 53,
    "Chinese Grand Prix": 56, "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63, "Monaco Grand Prix": 78,
    "Canadian Grand Prix": 70, "Spanish Grand Prix": 66,
    "Austrian Grand Prix": 71, "British Grand Prix": 52,
    "Hungarian Grand Prix": 70, "Belgian Grand Prix": 44,
    "Dutch Grand Prix": 72, "Italian Grand Prix": 53,
    "Azerbaijan Grand Prix": 51, "Singapore Grand Prix": 62,
    "United States Grand Prix": 56, "Mexico City Grand Prix": 71,
    "São Paulo Grand Prix": 69, "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57, "Abu Dhabi Grand Prix": 58,
}
DEFAULT_LAPS = 55


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            gap = abs(probs[mask].mean() - outcomes[mask].mean())
            ece += (mask.sum() / len(probs)) * gap
    return float(ece)


def reliability_bins(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            result.append({
                "bin_lo": float(bins[i]),
                "bin_hi": float(bins[i + 1]),
                "n": int(mask.sum()),
                "pred_mean": float(probs[mask].mean()),
                "real_rate": float(outcomes[mask].mean()),
                "gap": float(abs(probs[mask].mean() - outcomes[mask].mean())),
            })
    return result


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Resultados reais não encontrados: {RESULTS_FILE}\n"
            f"Rode o script de extração de resultados primeiro."
        )

    results_real = pd.read_csv(RESULTS_FILE)
    gps = sorted(results_real["gp"].unique())
    print(f"Corridas a simular: {len(gps)}")
    print(f"Carregando simulador...")

    sim = RaceSimulator()

    # Simula cada corrida de 2024
    all_predictions = []
    t0 = time.perf_counter()

    for gp_idx, gp in enumerate(gps, 1):
        gp_real = results_real[results_real["gp"] == gp].sort_values("grid_pos")
        total_laps = GP_LAPS.get(gp, DEFAULT_LAPS)

        # Monta grid
        grid = []
        for _, row in gp_real.iterrows():
            grid.append({
                "driver": row["driver"],
                "team": row["team"],
                "grid_pos": int(row["grid_pos"]),
                "quali_pos": int(row["quali_pos"]),
                "gap_to_pole_ms": float(row["gap_to_pole_ms"]),
            })

        # Roda simulação (1000 sims pra ser rápido — 24 corridas × 12s = ~5 min)
        result = sim.simulate(
            gp=gp, grid=grid,
            n_simulations=2_000,
            total_laps=total_laps,
            seed=42 + gp_idx,
            verbose=False,
        )

        probs = result["probabilities"]

        # Junta previsões com resultados reais
        for _, row in gp_real.iterrows():
            drv = row["driver"]
            if drv not in probs:
                continue
            p = probs[drv]
            real_pos = int(row["final_position"]) if row["dnf"] == 0 else 99
            is_dnf = int(row["dnf"])

            all_predictions.append({
                "gp": gp,
                "driver": drv,
                "grid_pos": int(row["grid_pos"]),
                "real_position": real_pos,
                "real_dnf": is_dnf,
                "real_win": int(real_pos == 1),
                "real_podium": int(real_pos <= 3),
                "real_top6": int(real_pos <= 6),
                "real_top10": int(real_pos <= 10),
                "pred_win": p["win"],
                "pred_podium": p["podium"],
                "pred_top6": p["top6"],
                "pred_top10": p["top10"],
                "pred_dnf": p["DNF"],
            })

        elapsed = time.perf_counter() - t0
        eta = elapsed / gp_idx * (len(gps) - gp_idx)
        print(f"  [{gp_idx:2d}/{len(gps)}] {gp:35s} | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    pdf = pd.DataFrame(all_predictions)
    print(f"\nTotal predictions: {len(pdf)}")

    # ================================================================
    # CALIBRAÇÃO POR MERCADO
    # ================================================================
    markets = [
        ("win", "pred_win", "real_win"),
        ("podium", "pred_podium", "real_podium"),
        ("top6", "pred_top6", "real_top6"),
        ("top10", "pred_top10", "real_top10"),
        ("dnf", "pred_dnf", "real_dnf"),
    ]

    report = {}

    print(f"\n{'='*70}")
    print("CALIBRAÇÃO DO SIMULADOR — 2024 (24 corridas)")
    print(f"{'='*70}")

    for market_name, pred_col, real_col in markets:
        p = pdf[pred_col].values
        y = pdf[real_col].values

        bs = brier_score(p, y)
        ece = expected_calibration_error(p, y, n_bins=5)
        rel = reliability_bins(p, y, n_bins=5)

        # Baseline: grid position
        # Win baseline: P1 no grid ganha com prob ~40% historicamente
        if market_name == "win":
            baseline_p = np.where(pdf["grid_pos"].values == 1, 0.40, 0.60 / (len(grid) - 1))
        elif market_name == "podium":
            baseline_p = np.where(pdf["grid_pos"].values <= 3, 0.60, 0.10)
        elif market_name == "top6":
            baseline_p = np.where(pdf["grid_pos"].values <= 6, 0.70, 0.15)
        elif market_name == "top10":
            baseline_p = np.where(pdf["grid_pos"].values <= 10, 0.85, 0.10)
        else:  # dnf
            baseline_p = np.full(len(y), y.mean())

        bs_baseline = brier_score(baseline_p, y)

        print(f"\n--- {market_name.upper()} ---")
        print(f"  Base rate (real):    {y.mean()*100:.1f}%")
        print(f"  Brier (modelo):      {bs:.4f}")
        print(f"  Brier (baseline):    {bs_baseline:.4f}")
        improvement = (bs_baseline - bs) / bs_baseline * 100 if bs_baseline > 0 else 0
        print(f"  Ganho sobre baseline: {improvement:+.1f}%")
        print(f"  ECE:                 {ece:.4f}")

        print(f"  Reliability diagram:")
        print(f"    {'Bin':20s} {'n':>5s} {'Pred':>8s} {'Real':>8s} {'Gap':>8s}")
        for b in rel:
            print(f"    [{b['bin_lo']:.2f}-{b['bin_hi']:.2f}]     {b['n']:5d}  {b['pred_mean']:.4f}   {b['real_rate']:.4f}   {b['gap']:.4f}")

        if bs < bs_baseline:
            print(f"  [OK] Modelo BATE o baseline.")
        else:
            print(f"  [WARN] Modelo NÃO bate o baseline.")

        report[market_name] = {
            "brier_model": bs,
            "brier_baseline": bs_baseline,
            "improvement_pct": improvement,
            "ece": ece,
            "base_rate": float(y.mean()),
            "reliability": rel,
        }

    # ================================================================
    # RESUMO
    # ================================================================
    print(f"\n{'='*70}")
    print("RESUMO")
    print(f"{'='*70}")
    print(f"{'Mercado':>10s}  {'Brier Modelo':>14s}  {'Brier Baseline':>16s}  {'Ganho':>8s}  {'ECE':>8s}")
    for market_name in ["win", "podium", "top6", "top10", "dnf"]:
        r = report[market_name]
        print(
            f"{market_name:>10s}  {r['brier_model']:14.4f}  {r['brier_baseline']:16.4f}  "
            f"{r['improvement_pct']:+7.1f}%  {r['ece']:8.4f}"
        )

    # Salva
    report_path = MODELS_DIR / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Relatório salvo em {report_path}")

    # Salva predições
    pred_path = MODELS_DIR / "calibration_predictions_2024.csv"
    pdf.to_csv(pred_path, index=False)
    print(f"[OK] Predições salvas em {pred_path}")


if __name__ == "__main__":
    main()