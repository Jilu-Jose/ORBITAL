"""ORBITAL API. Serves /api/* and the static frontend at /."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import demo, risk

BACKEND_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
PROCESSED = BACKEND_DIR / "data" / "processed"
MODELS = BACKEND_DIR / "models"

app = FastAPI(title="ORBITAL", version="4.2")

_cache: dict[str, tuple[float, object]] = {}


def _load_json(path: Path):
    """Read a JSON file, re-reading only when its mtime changes. None if missing/invalid."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _cache.pop(str(path), None)
        return None
    hit = _cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    _cache[str(path)] = (mtime, data)
    return data


class PredictIn(BaseModel):
    miss_km: float = Field(..., ge=0.01, le=10)
    rel_vel: float = Field(..., ge=0.5, le=16)
    inc_div: float = Field(..., ge=0, le=15)
    bstar: float = Field(..., ge=0.1, le=5)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    real = _load_json(PROCESSED / "stats.json")
    if real:
        return {**real, "demo": False}
    return demo.get_stats()


@app.get("/api/objects")
def objects():
    real = _load_json(PROCESSED / "objects.json")
    if real:
        return real
    return demo.get_objects()


@app.get("/api/metrics")
def metrics():
    real = _load_json(MODELS / "metrics.json")
    if isinstance(real, dict) and real.get("models"):
        out = {**real, "demo": False}
        if "scalability" not in out:  # benchmark.py output saved before training
            scal = _load_json(PROCESSED / "scalability.json")
            if scal:
                out["scalability"] = scal
        return out
    return demo.get_metrics()


@app.post("/api/predict")
def predict(req: PredictIn):
    return JSONResponse(risk.predict(req.miss_km, req.rel_vel, req.inc_div, req.bstar))


# Static frontend, mounted last so /api/* wins.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
