import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import calibration, engineering, live, model_analysis, reference, replay, simulate

app = FastAPI(title="F1 AI Race Insights API")

allowed_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reference.router)
app.include_router(simulate.router)
app.include_router(calibration.router)
app.include_router(model_analysis.router)
app.include_router(engineering.router)
app.include_router(replay.router)
app.include_router(live.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
