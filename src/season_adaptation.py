from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MARKET_KEYS = ["win", "podium", "top6", "top10"]


@dataclass(frozen=True)
class RegulationProfile:
    name: str
    historical_weight: float
    pace_sensitivity: float
    recovery_boost: float
    favorite_shrink: float
    longshot_floor: float
    dnf_base_blend: float


REGULATION_PROFILES = {
    "stable": RegulationProfile(
        name="stable",
        historical_weight=1.00,
        pace_sensitivity=1.00,
        recovery_boost=0.00,
        favorite_shrink=0.00,
        longshot_floor=0.000,
        dnf_base_blend=0.00,
    ),
    "transition": RegulationProfile(
        name="transition",
        historical_weight=0.82,
        pace_sensitivity=1.35,
        recovery_boost=0.12,
        favorite_shrink=0.08,
        longshot_floor=0.003,
        dnf_base_blend=0.12,
    ),
    "major_2026": RegulationProfile(
        name="major_2026",
        historical_weight=0.68,
        pace_sensitivity=1.70,
        recovery_boost=0.22,
        favorite_shrink=0.16,
        longshot_floor=0.006,
        dnf_base_blend=0.20,
    ),
}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _logit(p: float) -> float:
    p = _clip(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def load_current_form(path: Path | None) -> dict[str, dict[str, float]]:
    """Load optional current-season form adjustments.

    Expected columns:
        driver, pace_delta, confidence

    `pace_delta` is seconds/lap relative to expectation. Negative means the
    driver/team is faster than the historical model expects. Confidence should
    be between 0 and 1. Extra columns are ignored.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Current-season form file not found: {path}")

    form: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"driver", "pace_delta"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Current-season form CSV missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            driver = (row.get("driver") or "").strip().upper()
            if not driver:
                continue
            pace_delta = float(row.get("pace_delta") or 0.0)
            confidence = _clip(float(row.get("confidence") or 0.7), 0.0, 1.0)
            form[driver] = {"pace_delta": pace_delta, "confidence": confidence}
    return form


def _form_delta_for_driver(driver: str, form: dict[str, dict[str, float]]) -> float:
    item = form.get(driver, {})
    return float(item.get("pace_delta", 0.0)) * float(item.get("confidence", 0.0))


def _grid_recovery_bonus(grid_pos: int, profile: RegulationProfile) -> float:
    if grid_pos <= 3:
        return 0.0
    if grid_pos <= 6:
        return profile.recovery_boost * 0.35
    if grid_pos <= 10:
        return profile.recovery_boost * 0.70
    return profile.recovery_boost


def adapt_probabilities(
    results: dict[str, Any],
    grid: list[dict[str, Any]],
    profile_name: str = "major_2026",
    current_form: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Adapt simulator probabilities for a current regulation/season context.

    This is a post-model correction layer. It intentionally does not overwrite
    the trained model; it shrinks stale historical confidence and lets current
    form/qualifying context pull probabilities.
    """
    if profile_name not in REGULATION_PROFILES:
        raise ValueError(f"Unknown regulation profile: {profile_name}")

    profile = REGULATION_PROFILES[profile_name]
    current_form = current_form or {}
    adapted = dict(results)
    original_probs = results["probabilities"]
    grid_by_driver = {entry["driver"]: entry for entry in grid}
    n_drivers = len(grid)

    win_scores: dict[str, float] = {}
    diagnostics: list[dict[str, Any]] = []

    for driver, probs in original_probs.items():
        entry = grid_by_driver.get(driver, {})
        grid_pos = int(entry.get("grid_pos", 20))
        pace_delta = _form_delta_for_driver(driver, current_form)
        recovery = _grid_recovery_bonus(grid_pos, profile)
        pace_boost = -pace_delta * profile.pace_sensitivity
        favorite_penalty = profile.favorite_shrink if probs.get("win", 0.0) > 0.40 else 0.0

        raw_score = max(probs.get("win", 0.0), 1e-8) ** profile.historical_weight
        adjusted_score = raw_score * math.exp(pace_boost + recovery - favorite_penalty)
        adjusted_score = max(adjusted_score, profile.longshot_floor)
        win_scores[driver] = adjusted_score
        diagnostics.append(
            {
                "driver": driver,
                "grid_pos": grid_pos,
                "pace_delta": round(pace_delta, 4),
                "pace_boost": round(pace_boost, 4),
                "recovery_boost": round(recovery, 4),
                "favorite_penalty": round(favorite_penalty, 4),
            }
        )

    total_win_score = sum(win_scores.values()) or 1.0
    new_probs: dict[str, dict[str, float]] = {}

    for driver, probs in original_probs.items():
        entry = grid_by_driver.get(driver, {})
        grid_pos = int(entry.get("grid_pos", 20))
        pace_delta = _form_delta_for_driver(driver, current_form)
        recovery = _grid_recovery_bonus(grid_pos, profile)
        shift = -pace_delta * profile.pace_sensitivity + recovery

        p_win = _clip(win_scores[driver] / total_win_score)

        p_podium = _sigmoid(_logit(probs.get("podium", 0.0)) * profile.historical_weight + shift)
        p_top6 = _sigmoid(_logit(probs.get("top6", 0.0)) * profile.historical_weight + shift * 0.75)
        p_top10 = _sigmoid(_logit(probs.get("top10", 0.0)) * profile.historical_weight + shift * 0.45)

        dnf_original = probs.get("DNF", 0.0)
        p_dnf = (1 - profile.dnf_base_blend) * dnf_original + profile.dnf_base_blend * 0.10

        p_podium = max(p_podium, p_win)
        p_top6 = max(p_top6, p_podium)
        p_top10 = max(p_top10, p_top6)
        if n_drivers <= 10:
            p_top10 = max(p_top10, 1.0 - p_dnf)

        updated = dict(probs)
        updated["win_raw"] = probs.get("win", 0.0)
        updated["podium_raw"] = probs.get("podium", 0.0)
        updated["top6_raw"] = probs.get("top6", 0.0)
        updated["top10_raw"] = probs.get("top10", 0.0)
        updated["DNF_raw"] = dnf_original
        updated["win"] = p_win
        updated["P1"] = p_win
        updated["podium"] = _clip(p_podium)
        updated["top6"] = _clip(p_top6)
        updated["top10"] = _clip(p_top10)
        updated["points"] = updated["top10"]
        updated["DNF"] = _clip(p_dnf)
        new_probs[driver] = updated

    adapted["probabilities_raw"] = original_probs
    adapted["probabilities"] = new_probs
    adapted["season_adaptation"] = {
        "profile": profile.name,
        "historical_weight": profile.historical_weight,
        "pace_sensitivity": profile.pace_sensitivity,
        "recovery_boost": profile.recovery_boost,
        "favorite_shrink": profile.favorite_shrink,
        "longshot_floor": profile.longshot_floor,
        "dnf_base_blend": profile.dnf_base_blend,
    }
    return adapted, diagnostics


def format_adaptation_summary(diagnostics: list[dict[str, Any]], profile_name: str) -> str:
    lines = [f"Season adaptation profile: {profile_name}", "Largest adaptation pulls:"]
    ranked = sorted(
        diagnostics,
        key=lambda row: abs(row["pace_boost"]) + abs(row["recovery_boost"]) + abs(row["favorite_penalty"]),
        reverse=True,
    )
    for row in ranked[:8]:
        lines.append(
            f"- {row['driver']}: pace_boost={row['pace_boost']:+.3f}, "
            f"recovery={row['recovery_boost']:+.3f}, favorite_penalty={row['favorite_penalty']:+.3f}"
        )
    return "\n".join(lines)
