from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betting_recommender import RecommendationConfig, build_recommendations


def test_recommender_uses_conservative_probability_for_decision():
    probabilities = {
        "NOR": {"win": 0.40, "podium": 0.80, "top6": 0.92, "top10": 0.96, "DNF": 0.05},
    }
    odds_rows = [{"driver": "NOR", "market": "win", "odds": 3.20, "sportsbook": "manual"}]
    calibration = {
        "win": {
            "improvement_pct": 10.0,
            "ece": 0.02,
            "reliability": [
                {"bin_lo": 0.2, "bin_hi": 0.6, "n": 40, "gap": 0.02},
            ],
        }
    }

    rows = build_recommendations(
        probabilities,
        odds_rows,
        calibration,
        RecommendationConfig(min_edge=0.03, min_ev=0.05),
    )

    assert rows[0]["decision"] == "RECOMMEND"
    assert rows[0]["conservative_prob_pct"] == 38.0
    assert rows[0]["conservative_ev_pct"] == 21.6


def test_recommender_passes_weak_markets_even_with_positive_ev():
    probabilities = {
        "NOR": {"win": 0.40, "podium": 0.80, "top6": 0.92, "top10": 0.96, "DNF": 0.05},
    }
    odds_rows = [{"driver": "NOR", "market": "top10", "odds": 1.30, "sportsbook": "manual"}]
    calibration = {
        "top10": {
            "improvement_pct": 1.0,
            "ece": 0.12,
            "reliability": [
                {"bin_lo": 0.8, "bin_hi": 1.0, "n": 100, "gap": 0.10},
            ],
        }
    }

    rows = build_recommendations(probabilities, odds_rows, calibration)

    assert rows[0]["decision"] == "PASS"
    assert rows[0]["market_quality"] == "weak"
