"""Dados de referência estáticos (pilotos, equipes, GPs) — antes hardcoded no dashboard.py."""

KNOWN_DRIVERS = {
    "VER": "Red Bull Racing", "HAD": "Red Bull Racing",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "NOR": "McLaren", "PIA": "McLaren",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "GAS": "Alpine", "COL": "Alpine",
    "LAW": "Racing Bulls", "LIN": "Racing Bulls",
    "BOT": "Cadillac", "PER": "Cadillac",
    "HUL": "Audi", "BOR": "Audi",
    "BEA": "Haas F1 Team", "OCO": "Haas F1 Team",
    "ALB": "Williams", "SAI": "Williams",
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E80020",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0090FF",
    "Racing Bulls": "#6692FF",
    "Cadillac": "#8A8D8F",
    "Audi": "#BB0A30",
    "Haas F1 Team": "#B6BABD",
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
