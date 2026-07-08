"""Dados de referência estáticos (pilotos, equipes, GPs) — antes hardcoded no dashboard.py."""

KNOWN_DRIVERS = {
    "VER": "Red Bull Racing", "PER": "Red Bull Racing",
    "LEC": "Ferrari", "SAI": "Ferrari",
    "NOR": "McLaren", "PIA": "McLaren",
    "HAM": "Mercedes", "RUS": "Mercedes",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "GAS": "Alpine", "OCO": "Alpine",
    "TSU": "RB", "RIC": "RB", "LAW": "RB",
    "BOT": "Kick Sauber", "ZHO": "Kick Sauber",
    "MAG": "Haas F1 Team", "HUL": "Haas F1 Team",
    "ALB": "Williams", "SAR": "Williams", "COL": "Williams",
    "BEA": "Ferrari",
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E80020",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0090FF",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#FFFFFF",
    "Williams": "#64C4FF",
    "?": "#666666",
    "Unknown": "#666666",
}

KNOWN_GPS = [
    "Australian Grand Prix", "Bahrain Grand Prix", "Saudi Arabian Grand Prix",
    "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
    "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
    "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]

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

MODEL_METRICS = {
    "rmse": 0.722,
    "r2": 0.638,
    "mae": 0.537,
    "gain_vs_trivial_pct": 39.9,
}
