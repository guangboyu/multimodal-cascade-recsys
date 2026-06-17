"""FastAPI app exposing the recommendation cascade. Loads the registry once at startup.

Run: ``uv run uvicorn vlmrec.serving.app:app --port 8000``
Then: ``GET /recommend?user_id=0&k=20&diversity=mmr``
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .registry import load_registry
from .service import recommend

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["registry"] = load_registry()
    yield
    _state.clear()


app = FastAPI(title="VLM-Rec serving", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    reg = _state.get("registry")
    if reg is None:
        raise HTTPException(503, "registry not loaded")
    return {"status": "ok", "n_items": int(reg.d.n_items), "n_users": int(reg.d.n_users)}


@app.get("/recommend")
def rec(
    user_id: int = Query(..., ge=0, description="contiguous user_idx"),
    k: int = Query(20, ge=1, le=100),
    diversity: str = Query("mmr", pattern="^(mmr|dpp|none)$"),
):
    reg = _state["registry"]
    if user_id >= reg.d.n_users:
        raise HTTPException(404, f"user_id out of range (0..{reg.d.n_users - 1})")
    return recommend(reg, user_id, k_final=k, diversity=diversity)
