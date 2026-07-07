"""
Race-week prediction CLI.

Use this after qualifying: provide a grid CSV and get win/podium/top10
probabilities from the Monte Carlo simulator.

Example:
    python src/predict_race_week.py --gp "Australian Grand Prix" --grid examples/race_week_grid.csv --sims 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from betting_recommender import (
    RecommendationConfig,
    build_recommendations,
    format_recommendation_summary,
    load_calibration_report,
    load_odds,
    save_recommendations_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


GP_LAPS = {
    "Bahrain Grand Prix": 57,
    "Saudi Arabian Grand Prix": 50,
    "Australian Grand Prix": 58,
    "Japanese Grand Prix": 53,
    "Chinese Grand Prix": 56,
    "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63,
    "Monaco Grand Prix": 78,
    "Canadian Grand Prix": 70,
    "Spanish Grand Prix": 66,
    "Austrian Grand Prix": 71,
    "British Grand Prix": 52,
    "Hungarian Grand Prix": 70,
    "Belgian Grand Prix": 44,
    "Dutch Grand Prix": 72,
    "Italian Grand Prix": 53,
    "Azerbaijan Grand Prix": 51,
    "Singapore Grand Prix": 62,
    "United States Grand Prix": 56,
    "Mexico City Grand Prix": 71,
    "Sao Paulo Grand Prix": 69,
    "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57,
    "Abu Dhabi Grand Prix": 58,
}

REQUIRED_GRID_COLUMNS = {"driver", "team", "grid_pos"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a race week from a qualifying/grid CSV.",
    )
    parser.add_argument("--gp", required=True, help='Grand Prix name, e.g. "Monaco Grand Prix".')
    parser.add_argument("--grid", required=True, type=Path, help="CSV with driver,team,grid_pos columns.")
    parser.add_argument("--sims", type=int, default=2000, help="Number of Monte Carlo simulations.")
    parser.add_argument("--laps", type=int, default=None, help="Override race lap count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable predictions.")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models", help="Model artifact directory.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "race_week", help="Output directory.")
    parser.add_argument("--odds", type=Path, default=None, help="Optional decimal odds CSV for bet recommendations.")
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=PROJECT_ROOT / "models" / "calibration_report.json",
        help="Calibration report used to haircut model probabilities.",
    )
    parser.add_argument("--min-edge", type=float, default=0.03, help="Minimum no-vig edge for recommendations.")
    parser.add_argument("--min-ev", type=float, default=0.05, help="Minimum conservative EV for recommendations.")
    parser.add_argument("--json", action="store_true", help="Also save the raw simulator result as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Hide simulator progress logs.")
    return parser.parse_args()


def _to_int(row: dict[str, str], key: str, default: int | None = None) -> int:
    raw = row.get(key, "")
    if raw == "" and default is not None:
        return default
    try:
        return int(float(raw))
    except ValueError as exc:
        raise ValueError(f"Column '{key}' must be numeric for driver {row.get('driver', '?')!r}.") from exc


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Column '{key}' must be numeric for driver {row.get('driver', '?')!r}.") from exc


def load_grid(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Grid CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_GRID_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Grid CSV is missing required columns: {', '.join(sorted(missing))}")

        grid = []
        for row in reader:
            driver = (row.get("driver") or "").strip().upper()
            team = (row.get("team") or "").strip()
            if not driver or not team:
                raise ValueError("Every row must include driver and team.")

            grid_pos = _to_int(row, "grid_pos")
            quali_pos = _to_int(row, "quali_pos", default=grid_pos)
            grid.append(
                {
                    "driver": driver,
                    "team": team,
                    "grid_pos": grid_pos,
                    "quali_pos": quali_pos,
                    "gap_to_pole_ms": _to_float(row, "gap_to_pole_ms", default=(grid_pos - 1) * 150.0),
                    "avg_residual_recent": _to_float(row, "avg_residual_recent", default=0.0),
                }
            )

    if len(grid) < 2:
        raise ValueError("Grid must contain at least two drivers.")

    grid.sort(key=lambda item: item["grid_pos"])
    expected_positions = list(range(1, len(grid) + 1))
    actual_positions = [item["grid_pos"] for item in grid]
    if actual_positions != expected_positions:
        raise ValueError(
            "grid_pos must be consecutive starting at 1. "
            f"Got {actual_positions}, expected {expected_positions}."
        )
    return grid


def probability_rows(results: dict[str, Any], grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probs = results["probabilities"]
    team_by_driver = {entry["driver"]: entry["team"] for entry in grid}
    rows = []
    for driver, p in probs.items():
        rows.append(
            {
                "driver": driver,
                "team": team_by_driver.get(driver, ""),
                "win_pct": round(p["win"] * 100, 2),
                "podium_pct": round(p["podium"] * 100, 2),
                "top6_pct": round(p["top6"] * 100, 2),
                "top10_pct": round(p["top10"] * 100, 2),
                "dnf_pct": round(p["DNF"] * 100, 2),
            }
        )
    return sorted(rows, key=lambda row: row["win_pct"], reverse=True)


def build_friend_summary(gp: str, rows: list[dict[str, Any]], sc_probability: float) -> str:
    leader = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    delta = leader["win_pct"] - (runner_up["win_pct"] if runner_up else 0.0)

    if delta < 5:
        race_note = "previsao aberta"
    elif delta < 15:
        race_note = "favorito claro, mas com margem real"
    else:
        race_note = "favorito forte"

    lines = [
        f"Race-week prediction - {gp}",
        f"Leitura do modelo: {race_note}. Safety Car estimado: {sc_probability * 100:.1f}%.",
        "",
        "Top chances de vitoria:",
    ]

    for idx, row in enumerate(rows[:5], start=1):
        lines.append(
            f"{idx}. {row['driver']} ({row['team']}): "
            f"{row['win_pct']:.1f}% win | {row['podium_pct']:.1f}% podium | {row['top10_pct']:.1f}% top10"
        )

    surprise = max(rows, key=lambda row: row["podium_pct"] - row["win_pct"])
    lines.extend(
        [
            "",
            f"Possivel aposta de podium: {surprise['driver']} com {surprise['podium_pct']:.1f}% de podium.",
            "Use como brincadeira/probabilidade, nao como certeza: corrida tem DNF, SC, estrategia e caos.",
        ]
    )
    return "\n".join(lines)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    total_laps = args.laps or GP_LAPS.get(args.gp, 55)
    grid = load_grid(args.grid)

    try:
        from simulation.race_simulate import RaceSimulator
    except ModuleNotFoundError as exc:
        missing = exc.name or "a project dependency"
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install the project requirements first, for example:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    simulator = RaceSimulator(models_dir=args.models_dir)
    results = simulator.simulate(
        gp=args.gp,
        grid=grid,
        n_simulations=args.sims,
        total_laps=total_laps,
        seed=args.seed,
        verbose=not args.quiet,
    )

    rows = probability_rows(results, grid)
    summary = build_friend_summary(args.gp, rows, results["sc_probability"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_gp = args.gp.lower().replace(" ", "_").replace("/", "_")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / f"{safe_gp}_{run_id}.csv"
    txt_path = args.out_dir / f"{safe_gp}_{run_id}.txt"
    save_csv(csv_path, rows)
    txt_path.write_text(summary + "\n", encoding="utf-8")

    if args.json:
        json_path = args.out_dir / f"{safe_gp}_{run_id}.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.odds:
        odds_rows = load_odds(args.odds)
        calibration_report = load_calibration_report(args.calibration_report)
        recommendations = build_recommendations(
            probabilities=results["probabilities"],
            odds_rows=odds_rows,
            calibration_report=calibration_report,
            config=RecommendationConfig(min_edge=args.min_edge, min_ev=args.min_ev),
        )
        rec_path = args.out_dir / f"{safe_gp}_{run_id}_recommendations.csv"
        rec_txt_path = args.out_dir / f"{safe_gp}_{run_id}_recommendations.txt"
        save_recommendations_csv(rec_path, recommendations)
        rec_summary = format_recommendation_summary(args.gp, recommendations)
        rec_txt_path.write_text(rec_summary + "\n", encoding="utf-8")
        print()
        print(rec_summary)
        print()
        print(f"Saved recommendations: {rec_path}")
        print(f"Saved recommendation summary: {rec_txt_path}")

    print()
    print(summary)
    print()
    print(f"Saved ranking: {csv_path}")
    print(f"Saved summary: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
