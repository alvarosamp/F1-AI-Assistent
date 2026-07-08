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
    parser = argparse.ArgumentParser(description="Monitor prediction drift during a new season.")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with pred_* and real_* columns.")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "models" / "calibration_report.json")
    parser.add_argument("--season", default="current")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "drift_monitor.md")
    parser.add_argument("--json-out", type=Path, default=PROJECT_ROOT / "reports" / "drift_monitor.json")
    return parser.parse_args()


def brier(pred: pd.Series, real: pd.Series) -> float:
    return float(np.mean((pred.to_numpy() - real.to_numpy()) ** 2))


def ece(pred: pd.Series, real: pd.Series, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(pred)
    score = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred >= lo) & (pred < hi)
        if mask.any():
            score += float(mask.sum() / total * abs(pred[mask].mean() - real[mask].mean()))
    return score


def status_from_change(metric: float, baseline: float, lower_is_better: bool = True) -> str:
    if baseline <= 0:
        return "UNKNOWN"
    ratio = metric / baseline if lower_is_better else baseline / metric
    if ratio <= 1.10:
        return "OK"
    if ratio <= 1.30:
        return "ALERT"
    return "CRITICAL"


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_monitor(df: pd.DataFrame, baseline: dict) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings = []
    for market, (pred_col, real_col) in MARKETS.items():
        current_brier = brier(df[pred_col], df[real_col])
        current_ece = ece(df[pred_col], df[real_col])
        base = baseline.get(market, {})
        base_brier = float(base.get("brier_model", current_brier))
        base_ece = float(base.get("ece", current_ece))
        brier_status = status_from_change(current_brier, base_brier)
        ece_status = status_from_change(current_ece, max(base_ece, 0.01))
        status = "CRITICAL" if "CRITICAL" in {brier_status, ece_status} else "ALERT" if "ALERT" in {brier_status, ece_status} else "OK"

        if status != "OK":
            warnings.append(
                f"{market}: {status} - current Brier {current_brier:.3f} vs baseline {base_brier:.3f}; "
                f"current ECE {current_ece:.3f} vs baseline {base_ece:.3f}"
            )

        rows.append(
            {
                "market": market,
                "n": len(df),
                "current_brier": current_brier,
                "baseline_brier": base_brier,
                "current_ece": current_ece,
                "baseline_ece": base_ece,
                "pred_mean": float(df[pred_col].mean()),
                "real_rate": float(df[real_col].mean()),
                "status": status,
            }
        )
    return pd.DataFrame(rows), warnings


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: f"{value:.3f}")
        else:
            work[col] = work[col].astype(str)
    headers = list(work.columns)
    rows = work.values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def row(values):
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(values)) + " |"

    return "\n".join([row(headers), "| " + " | ".join("-" * width for width in widths) + " |", *[row(r) for r in rows]])


def build_report(summary: pd.DataFrame, warnings: list[str], season: str) -> str:
    critical = any(row == "CRITICAL" for row in summary["status"])
    alert = any(row == "ALERT" for row in summary["status"])
    if critical:
        action = "Use `major_2026` adaptation, increase current-season form weight, and avoid high-stake recommendations."
    elif alert:
        action = "Use `transition` or `major_2026` adaptation and recalibrate after the next race."
    else:
        action = "Current drift is acceptable; keep monitoring race by race."

    lines = [
        f"# Drift monitor - {season}",
        "",
        "## Status by market",
        "",
        markdown_table(summary),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- No warnings."])
    lines.extend(
        [
            "",
            "## Recommended action",
            "",
            action,
            "",
            "## Rule of thumb",
            "",
            "If two or more markets are `CRITICAL`, treat the model as out-of-regime until current-season retraining/calibration is done.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    required = {col for pair in MARKETS.values() for col in pair}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing columns: {sorted(missing)}")

    baseline = load_baseline(args.baseline)
    summary, warnings = build_monitor(df, baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(summary, warnings, args.season), encoding="utf-8")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"Saved drift report: {args.out}")
    print(f"Saved drift JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
