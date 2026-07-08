from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.reference_data import GP_LAPS, KNOWN_DRIVERS
from backend.simulator_singleton import get_simulator

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


class SimulateRequest(BaseModel):
    gp: str
    drivers: list[str] = Field(..., min_length=2, max_length=20)
    n_simulations: int = 2000
    seed: int = 42


@router.post("")
def simulate(req: SimulateRequest):
    if req.gp not in GP_LAPS:
        raise HTTPException(status_code=400, detail=f"GP desconhecido: {req.gp}")

    total_laps = GP_LAPS.get(req.gp, 55)

    grid = []
    for i, drv in enumerate(req.drivers, 1):
        grid.append({
            "driver": drv,
            "team": KNOWN_DRIVERS.get(drv, "Unknown"),
            "grid_pos": i,
            "quali_pos": i,
            "gap_to_pole_ms": (i - 1) * 200,
        })

    simulator = get_simulator()
    results = simulator.simulate(
        gp=req.gp,
        grid=grid,
        n_simulations=req.n_simulations,
        total_laps=total_laps,
        seed=req.seed,
        verbose=False,
    )

    return {
        "gp": req.gp,
        "total_laps": total_laps,
        "n_simulations": req.n_simulations,
        "sc_probability": results["sc_probability"],
        "probabilities": results["probabilities"],
    }
