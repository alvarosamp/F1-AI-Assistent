"""
Calibração do Simulador Monte Carlo contra resultados reais de uma temporada
arbitrária (generaliza calibrate_simulate.py, que era hardcoded pra 2024).

USO:
    python src/simulation/calibrate_season.py --season 2026

OUTPUT:
    models/calibration_report_<season>.json
    models/calibration_predictions_<season>.csv
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--sims", type=int, default=2000)
    args = parser.parse_args()
    season = args.season

    results_file = PROJECT_ROOT / "data" / "processed" / f"results_{season}_real.csv"
    if not results_file.exists():
        raise FileNotFoundError(
            f"Resultados reais não encontrados: {results_file}\n"
            f"Rode: python src/simulation/extract_season_results.py --season {season}"
        )

    results_real = pd.read_csv(results_file)
    gps = sorted(results_real["gp"].unique())
    print(f"Corridas a simular: {len(gps)}")
    print("Carregando simulador...")

    sim = RaceSimulator()

    all_predictions = []
    all_errors = []
    t0 = time.perf_counter()

    for gp_idx, gp in enumerate(gps, 1):
        gp_real = results_real[results_real["gp"] == gp].sort_values("grid_pos")

        grid = []
        for _, row in gp_real.iterrows():
            grid.append({
                "driver": row["driver"],
                "team": row["team"],
                "grid_pos": int(row["grid_pos"]),
                "quali_pos": int(row["quali_pos"]),
                "gap_to_pole_ms": float(row["gap_to_pole_ms"]),
            })

        try:
            result = sim.simulate(
                gp=gp, grid=grid,
                n_simulations=args.sims,
                total_laps=DEFAULT_LAPS,
                seed=42 + gp_idx,
                verbose=False,
            )
        except Exception as e:
            print(f"  [SKIP] {gp}: {e}")
            all_errors.append({"gp": gp, "error": str(e)})
            continue

        probs = result["probabilities"]

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

    if all_errors:
        print(f"\n[WARN] {len(all_errors)}/{len(gps)} corridas falharam na simulação:")
        for e in all_errors:
            print(f"  - {e['gp']}: {e['error']}")

    if not all_predictions:
        print("\n[FAIL] Nenhuma predição gerada.")
        return

    pdf = pd.DataFrame(all_predictions)
    print(f"\nTotal predictions: {len(pdf)}")

    markets = [
        ("win", "pred_win", "real_win"),
        ("podium", "pred_podium", "real_podium"),
        ("top6", "pred_top6", "real_top6"),
        ("top10", "pred_top10", "real_top10"),
        ("dnf", "pred_dnf", "real_dnf"),
    ]

    report = {}

    print(f"\n{'='*70}")
    print(f"CALIBRAÇÃO DO SIMULADOR — {season} ({pdf['gp'].nunique()} corridas)")
    print(f"{'='*70}")

    for market_name, pred_col, real_col in markets:
        p = pdf[pred_col].values
        y = pdf[real_col].values

        bs = brier_score(p, y)
        ece = expected_calibration_error(p, y, n_bins=5)
        rel = reliability_bins(p, y, n_bins=5)

        if market_name == "win":
            baseline_p = np.where(pdf["grid_pos"].values == 1, 0.40, 0.60 / 21)
        elif market_name == "podium":
            baseline_p = np.where(pdf["grid_pos"].values <= 3, 0.60, 0.10)
        elif market_name == "top6":
            baseline_p = np.where(pdf["grid_pos"].values <= 6, 0.70, 0.15)
        elif market_name == "top10":
            baseline_p = np.where(pdf["grid_pos"].values <= 10, 0.85, 0.10)
        else:
            baseline_p = np.full(len(y), y.mean())

        bs_baseline = brier_score(baseline_p, y)
        improvement = (bs_baseline - bs) / bs_baseline * 100 if bs_baseline > 0 else 0

        print(f"\n--- {market_name.upper()} ---")
        print(f"  Base rate (real):    {y.mean()*100:.1f}%")
        print(f"  Brier (modelo):      {bs:.4f}")
        print(f"  Brier (baseline):    {bs_baseline:.4f}")
        print(f"  Ganho sobre baseline: {improvement:+.1f}%")
        print(f"  ECE:                 {ece:.4f}")

        report[market_name] = {
            "brier_model": bs,
            "brier_baseline": bs_baseline,
            "improvement_pct": improvement,
            "ece": ece,
            "base_rate": float(y.mean()),
            "reliability": rel,
        }

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

    report_path = MODELS_DIR / f"calibration_report_{season}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Relatório salvo em {report_path}")

    pred_path = MODELS_DIR / f"calibration_predictions_{season}.csv"
    pdf.to_csv(pred_path, index=False)
    print(f"[OK] Predições salvas em {pred_path}")


if __name__ == "__main__":
    main()
