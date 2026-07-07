"""
Bet recommendation engine for model probabilities and decimal odds.

This module does not decide whether to bet. It ranks opportunities using
expected value, market calibration, and conservative probability haircuts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_MARKETS = {"win", "podium", "top6", "top10", "dnf"}


@dataclass(frozen=True)
class RecommendationConfig:
    min_edge: float = 0.03
    min_ev: float = 0.05
    max_recommendations: int = 8
    allow_caution_markets: bool = True


def load_odds(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Odds CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = {"driver", "market", "odds"} - fields
        if missing:
            raise ValueError(f"Odds CSV is missing required columns: {', '.join(sorted(missing))}")

        odds = []
        for row in reader:
            driver = (row.get("driver") or "").strip().upper()
            market = (row.get("market") or "").strip().lower()
            sportsbook = (row.get("sportsbook") or "").strip() or "manual"
            raw_odds = (row.get("odds") or "").strip()

            if not driver:
                raise ValueError("Every odds row must include driver.")
            if market not in SUPPORTED_MARKETS:
                raise ValueError(f"Unsupported market {market!r}. Use one of: {', '.join(sorted(SUPPORTED_MARKETS))}")

            try:
                decimal_odds = float(raw_odds)
            except ValueError as exc:
                raise ValueError(f"Odds must be numeric for {driver} {market}.") from exc
            if decimal_odds <= 1.0:
                raise ValueError(f"Decimal odds must be greater than 1.0 for {driver} {market}.")

            odds.append(
                {
                    "driver": driver,
                    "market": market,
                    "odds": decimal_odds,
                    "sportsbook": sportsbook,
                }
            )
    return odds


def load_calibration_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_win_odds(odds_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    """Return no-vig implied probabilities for winner markets when possible.

    Podium/top6/top10/DNF markets are not one-winner books, so simple
    normalization to 100% would create fake edge. For those markets we keep
    the raw implied probability.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in odds_rows:
        if row["market"] != "win":
            continue
        grouped.setdefault((row["sportsbook"], row["market"]), []).append(row)

    no_vig = {}
    for (sportsbook, market), rows in grouped.items():
        total_implied = sum(1.0 / row["odds"] for row in rows)
        if total_implied <= 1.0:
            continue
        for row in rows:
            no_vig[(sportsbook, market, row["driver"])] = (1.0 / row["odds"]) / total_implied
    return no_vig


def probability_for_market(probabilities: dict[str, Any], driver: str, market: str) -> float | None:
    if driver not in probabilities:
        return None
    driver_probs = probabilities[driver]
    key = "DNF" if market == "dnf" else market
    if key not in driver_probs:
        return None
    return float(driver_probs[key])


def reliability_for_probability(market_report: dict[str, Any], model_prob: float) -> dict[str, Any] | None:
    for item in market_report.get("reliability", []):
        lo = float(item.get("bin_lo", 0.0))
        hi = float(item.get("bin_hi", 1.0))
        if lo <= model_prob < hi or (model_prob == 1.0 and hi == 1.0):
            return item
    return None


def market_quality(market_report: dict[str, Any] | None) -> str:
    if not market_report:
        return "unknown"

    improvement = float(market_report.get("improvement_pct", 0.0))
    ece = float(market_report.get("ece", 1.0))

    if improvement >= 8.0 and ece <= 0.05:
        return "strong"
    if improvement > 0.0 and ece <= 0.08:
        return "caution"
    return "weak"


def conservative_probability(model_prob: float, market_report: dict[str, Any] | None) -> tuple[float, str]:
    if not market_report:
        return max(0.0, model_prob - 0.05), "no calibration report; applied 5pp haircut"

    ece = float(market_report.get("ece", 0.05))
    bin_info = reliability_for_probability(market_report, model_prob)
    bin_gap = float(bin_info.get("gap", ece)) if bin_info else ece
    bin_n = int(bin_info.get("n", 0)) if bin_info else 0

    low_sample_penalty = 0.0
    if bin_n and bin_n < 20:
        low_sample_penalty = 0.03
    elif not bin_n:
        low_sample_penalty = 0.02

    haircut = max(ece, bin_gap) + low_sample_penalty
    adjusted = max(0.0, model_prob - haircut)
    reason = f"haircut {haircut * 100:.1f}pp from calibration"
    if low_sample_penalty:
        reason += " and low-bin sample"
    return adjusted, reason


def confidence_label(conservative_ev: float, edge: float, quality: str) -> str:
    if quality == "strong" and conservative_ev >= 0.12 and edge >= 0.06:
        return "high"
    if quality in {"strong", "caution"} and conservative_ev >= 0.05 and edge >= 0.03:
        return "medium"
    return "low"


def build_recommendations(
    probabilities: dict[str, Any],
    odds_rows: list[dict[str, Any]],
    calibration_report: dict[str, Any] | None = None,
    config: RecommendationConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or RecommendationConfig()
    calibration_report = calibration_report or {}
    no_vig_probs = normalize_win_odds(odds_rows)

    recommendations = []
    for row in odds_rows:
        market = row["market"]
        driver = row["driver"]
        model_prob = probability_for_market(probabilities, driver, market)
        if model_prob is None:
            continue

        market_report = calibration_report.get(market)
        quality = market_quality(market_report)
        adjusted_prob, adjustment_note = conservative_probability(model_prob, market_report)
        implied_prob = 1.0 / row["odds"]
        comparable_prob = no_vig_probs.get((row["sportsbook"], market, driver), implied_prob)
        raw_ev = model_prob * row["odds"] - 1.0
        conservative_ev = adjusted_prob * row["odds"] - 1.0
        edge = adjusted_prob - comparable_prob
        confidence = confidence_label(conservative_ev, edge, quality)

        if quality == "weak":
            decision = "PASS"
            reason = "market calibration is weak"
        elif quality == "caution" and not config.allow_caution_markets:
            decision = "PASS"
            reason = "caution market disabled"
        elif conservative_ev >= config.min_ev and edge >= config.min_edge:
            decision = "RECOMMEND"
            reason = "positive conservative EV and edge"
        elif raw_ev > 0 and conservative_ev <= 0:
            decision = "WATCH"
            reason = "raw EV positive, but not after calibration haircut"
        else:
            decision = "PASS"
            reason = "edge/EV below threshold"

        recommendations.append(
            {
                "decision": decision,
                "confidence": confidence,
                "driver": driver,
                "market": market,
                "sportsbook": row["sportsbook"],
                "odds": row["odds"],
                "model_prob_pct": round(model_prob * 100, 2),
                "conservative_prob_pct": round(adjusted_prob * 100, 2),
                "implied_prob_pct": round(implied_prob * 100, 2),
                "market_prob_pct": round(comparable_prob * 100, 2),
                "edge_pct": round(edge * 100, 2),
                "ev_pct": round(raw_ev * 100, 2),
                "conservative_ev_pct": round(conservative_ev * 100, 2),
                "market_quality": quality,
                "reason": reason,
                "adjustment": adjustment_note,
            }
        )

    recommendations.sort(
        key=lambda item: (
            item["decision"] != "RECOMMEND",
            -item["conservative_ev_pct"],
            -item["edge_pct"],
        )
    )

    recommended = [item for item in recommendations if item["decision"] == "RECOMMEND"]
    rest = [item for item in recommendations if item["decision"] != "RECOMMEND"]
    return recommended[: config.max_recommendations] + rest


def save_recommendations_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("decision,confidence,driver,market,sportsbook,odds\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_recommendation_summary(gp: str, rows: list[dict[str, Any]]) -> str:
    recommended = [row for row in rows if row["decision"] == "RECOMMEND"]
    watch = [row for row in rows if row["decision"] == "WATCH"]

    lines = [
        f"Bet recommendation report - {gp}",
        "This is a decision-support ranking, not a guaranteed bet.",
        "",
    ]

    if recommended:
        lines.append("Recommended value spots:")
        for idx, row in enumerate(recommended, start=1):
            lines.append(
                f"{idx}. {row['driver']} {row['market']} @ {row['odds']:.2f} "
                f"({row['sportsbook']}): EV {row['conservative_ev_pct']:+.1f}%, "
                f"edge {row['edge_pct']:+.1f}pp, confidence {row['confidence']}"
            )
    else:
        lines.append("No recommended bets passed the conservative filters.")

    if watch:
        lines.append("")
        lines.append("Watchlist:")
        for row in watch[:5]:
            lines.append(
                f"- {row['driver']} {row['market']} @ {row['odds']:.2f}: "
                f"raw EV {row['ev_pct']:+.1f}%, conservative EV {row['conservative_ev_pct']:+.1f}%"
            )

    lines.extend(
        [
            "",
            "Filters: recommendation needs positive conservative EV, market edge, and acceptable calibration.",
        ]
    )
    return "\n".join(lines)
