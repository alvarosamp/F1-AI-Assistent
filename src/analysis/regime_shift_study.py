from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKETS = {
    "win": ("pred_win", "real_win"),
    "podium": ("pred_podium", "real_podium"),
    "top6": ("pred_top6", "real_top6"),
    "top10": ("pred_top10", "real_top10"),
    "dnf": ("pred_dnf", "real_dnf"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study model accuracy and potential regulation/regime-shift risks.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "models" / "calibration_predictions_2024.csv",
        help="CSV with pred_* and real_* columns.",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=PROJECT_ROOT / "models" / "calibration_report.json",
        help="Optional calibration_report.json.",
    )
    parser.add_argument("--season", default="2024", help="Season label for the report.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "regime_shift_study.md")
    return parser.parse_args()


def brier(pred: pd.Series, real: pd.Series) -> float:
    return float(np.mean((pred.to_numpy() - real.to_numpy()) ** 2))


def log_loss_safe(pred: pd.Series, real: pd.Series) -> float:
    p = np.clip(pred.to_numpy(), 1e-6, 1 - 1e-6)
    y = real.to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(pred: pd.Series, real: pd.Series, n_bins: int = 5) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(pred)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred >= lo) & (pred < hi)
        if mask.any():
            ece += float(mask.sum() / total * abs(pred[mask].mean() - real[mask].mean()))
    return ece


def grid_bucket(grid_pos: int) -> str:
    if grid_pos <= 3:
        return "P1-P3"
    if grid_pos <= 6:
        return "P4-P6"
    if grid_pos <= 10:
        return "P7-P10"
    return "P11+"


def market_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, (pred_col, real_col) in MARKETS.items():
        rows.append(
            {
                "market": market,
                "n": len(df),
                "base_rate": df[real_col].mean(),
                "pred_mean": df[pred_col].mean(),
                "brier": brier(df[pred_col], df[real_col]),
                "log_loss": log_loss_safe(df[pred_col], df[real_col]),
                "ece": expected_calibration_error(df[pred_col], df[real_col]),
            }
        )
    return pd.DataFrame(rows)


def by_gp_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gp, gdf in df.groupby("gp"):
        row = {"gp": gp, "n": len(gdf), "avg_grid_dnf": gdf["real_dnf"].mean()}
        for market, (pred_col, real_col) in MARKETS.items():
            row[f"{market}_brier"] = brier(gdf[pred_col], gdf[real_col])
            row[f"{market}_mean_error"] = float((gdf[pred_col] - gdf[real_col]).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("win_brier", ascending=False)


def by_grid_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["grid_bucket"] = work["grid_pos"].apply(grid_bucket)
    rows = []
    for bucket, gdf in work.groupby("grid_bucket", sort=False):
        row = {"grid_bucket": bucket, "n": len(gdf)}
        for market, (pred_col, real_col) in MARKETS.items():
            row[f"{market}_pred"] = gdf[pred_col].mean()
            row[f"{market}_real"] = gdf[real_col].mean()
            row[f"{market}_gap"] = gdf[pred_col].mean() - gdf[real_col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def top_misses(df: pd.DataFrame, market: str, n: int = 10) -> pd.DataFrame:
    pred_col, real_col = MARKETS[market]
    work = df.copy()
    work["abs_error"] = (work[pred_col] - work[real_col]).abs()
    return work.sort_values("abs_error", ascending=False)[
        ["gp", "driver", "grid_pos", pred_col, real_col, "real_position", "real_dnf", "abs_error"]
    ].head(n)


def top_pick_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gp, gdf in df.groupby("gp"):
        pick = gdf.sort_values("pred_win", ascending=False).iloc[0]
        winner = gdf[gdf["real_win"] == 1]
        rows.append(
            {
                "gp": gp,
                "model_pick": pick["driver"],
                "pick_grid": int(pick["grid_pos"]),
                "pick_win_prob": float(pick["pred_win"]),
                "pick_result": int(pick["real_position"]),
                "real_winner": winner.iloc[0]["driver"] if not winner.empty else "?",
                "hit": bool(pick["real_win"] == 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["hit", "pick_win_prob"], ascending=[True, False])


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"

    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: format(value, floatfmt) if pd.notna(value) else "")
        else:
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = list(formatted.columns)
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def load_report(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(df: pd.DataFrame, calibration: dict, season: str) -> str:
    summary = market_summary(df)
    gp = by_gp_summary(df)
    grid = by_grid_summary(df)
    picks = top_pick_accuracy(df)

    hit_rate = picks["hit"].mean()
    high_conf = df[df["pred_win"] >= 0.4]
    high_conf_win_rate = high_conf["real_win"].mean() if len(high_conf) else np.nan
    high_conf_pred = high_conf["pred_win"].mean() if len(high_conf) else np.nan

    lines = [
        f"# Regime-shift and accuracy study - {season}",
        "",
        "## Executive summary",
        "",
        f"- Rows analysed: {len(df)} driver-race predictions across {df['gp'].nunique()} races.",
        f"- Top predicted winner hit rate: {hit_rate * 100:.1f}%.",
        f"- High-confidence win picks (`pred_win >= 40%`): predicted mean {high_conf_pred * 100:.1f}% vs real {high_conf_win_rate * 100:.1f}%.",
        "- Main risk for a new regulation season: the simulator is calibrated on the previous technical era, so relationships learned from grid, tyre behaviour, DRS/dirty air and team strength can drift.",
        "",
        "## Market metrics",
        "",
        markdown_table(summary),
        "",
        "## Calibration report comparison",
        "",
    ]

    if calibration:
        cal_rows = []
        for market in MARKETS:
            if market in calibration:
                item = calibration[market]
                cal_rows.append(
                    {
                        "market": market,
                        "brier_model": item.get("brier_model"),
                        "brier_baseline": item.get("brier_baseline"),
                        "improvement_pct": item.get("improvement_pct"),
                        "ece": item.get("ece"),
                    }
                )
        lines.append(markdown_table(pd.DataFrame(cal_rows)))
    else:
        lines.append("_No calibration report found._")

    lines.extend(
        [
            "",
            "## Worst races by win Brier",
            "",
            markdown_table(gp[["gp", "n", "win_brier", "podium_brier", "top6_brier", "top10_brier", "dnf_brier"]].head(10)),
            "",
            "## Grid bucket bias",
            "",
            markdown_table(grid),
            "",
            "## Top predicted winner misses",
            "",
            markdown_table(picks[picks["hit"] == False].head(12)),
            "",
            "## Largest individual errors",
            "",
            "### Win",
            "",
            markdown_table(top_misses(df, "win")),
            "",
            "### Podium",
            "",
            markdown_table(top_misses(df, "podium")),
            "",
            "### Top 10",
            "",
            markdown_table(top_misses(df, "top10")),
            "",
            "## Regulation-change risks for the current season",
            "",
            "1. **Power-unit and energy-management drift**: if the season has a different hybrid/ERS behaviour, historical lap-time residuals and straight-line performance features can become stale.",
            "2. **Aero/dirty-air drift**: if active aero or changed car dimensions affect following distance and overtaking, the current traffic and overtake penalties may be miscalibrated.",
            "3. **Tyre degradation drift**: narrower/different tyres or new operating windows can change stint degradation; this directly affects pit-stop strategy and Monte Carlo race pace.",
            "4. **Team-strength reset**: a major regulation change can reshuffle competitive order; target encoding by driver/team can overvalue old dominant teams.",
            "5. **Track-specific bias**: the worst-GP table should be monitored first. If the same circuits stay bad across seasons, track modelling is weak; if new bad races appear, it is likely regime drift.",
            "",
            "## Recommended fixes",
            "",
            "- Add a `season_weight` or time-decay retraining mode so current-season races matter more than old regulation-era data.",
            "- Add an explicit `regulation_era` flag and avoid mixing eras without interaction features.",
            "- Retrain target encoders using only current-era data once at least 4-6 races exist.",
            "- Calibrate market probabilities per season, not just globally.",
            "- Add a drift dashboard: compare predicted vs real win/podium/top10 after every race.",
            "- For strategy, do not trust historical pit counts directly; estimate tyre degradation from current FP long runs, then simulate strategy windows.",
            "",
            "## How to use this report",
            "",
            "Use this file as the baseline diagnostic. When you have current-season real results, generate a new `calibration_predictions_<season>.csv` with the same columns and rerun this script.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {args.predictions}")

    df = pd.read_csv(args.predictions)
    missing = {col for pair in MARKETS.values() for col in pair} - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {sorted(missing)}")

    calibration = load_report(args.calibration_report)
    report = build_report(df, calibration, args.season)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Saved report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
