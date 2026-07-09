from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.current_form import compute_current_form
from backend.reference_data import GP_LAPS, KNOWN_DRIVERS
from backend.simulator_singleton import get_simulator

router = APIRouter(prefix="/api/simulate", tags=["simulate"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CURRENT_FORM_FILE = PROJECT_ROOT / "configs" / "season_adaptation_2026.csv"
DEFAULT_REGULATION_PROFILE = "major_2026"
CURRENT_FORM_YEAR = 2026


class SimulateRequest(BaseModel):
    gp: str
    drivers: list[str] = Field(..., min_length=2, max_length=20)
    n_simulations: int = 2000
    seed: int = 42
    apply_season_adaptation: bool = True


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

    season_adaptation = None
    probabilities_raw = None
    current_form_source = None

    if req.apply_season_adaptation:
        # O drift monitor de 2026 (reports/drift_monitor_2026.md) mostra 4/5
        # mercados em CRITICAL contra o modelo treinado só em 2022-2024 —
        # aplicamos a mesma correção pós-modelo do CLI (predict_race_week.py)
        # até o retrain com dados da temporada atual estar pronto.
        from season_adaptation import adapt_probabilities, load_current_form

        # Forma atual real (gap pro fastest lap em Q/FP3/FP2/FP1 desse fim de
        # semana) tem prioridade sobre o CSV estático com placeholders manuais.
        # O CSV só preenche pilotos sem sessão real ainda (corrida futura).
        static_form = load_current_form(CURRENT_FORM_FILE) if CURRENT_FORM_FILE.exists() else {}
        real_form, form_meta = compute_current_form(CURRENT_FORM_YEAR, req.gp)
        current_form = {**static_form, **real_form}
        if form_meta:
            current_form_source = {**form_meta, "used_static_fallback_for": sorted(set(static_form) - set(real_form))}

        adapted, _diagnostics = adapt_probabilities(
            results=results,
            grid=grid,
            profile_name=DEFAULT_REGULATION_PROFILE,
            current_form=current_form,
        )
        probabilities_raw = adapted.get("probabilities_raw")
        results = adapted
        season_adaptation = adapted.get("season_adaptation")

    return {
        "gp": req.gp,
        "total_laps": total_laps,
        "n_simulations": req.n_simulations,
        "sc_probability": results["sc_probability"],
        "probabilities": results["probabilities"],
        "probabilities_raw": probabilities_raw,
        "season_adaptation": season_adaptation,
        "current_form_source": current_form_source,
    }
