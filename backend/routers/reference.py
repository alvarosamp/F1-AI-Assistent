from fastapi import APIRouter

from backend.reference_data import GP_LAPS, KNOWN_DRIVERS, KNOWN_GPS, TEAM_COLORS

router = APIRouter(prefix="/api/reference", tags=["reference"])


@router.get("")
def get_reference():
    return {
        "gps": KNOWN_GPS,
        "gp_laps": GP_LAPS,
        "drivers": KNOWN_DRIVERS,
        "team_colors": TEAM_COLORS,
    }
