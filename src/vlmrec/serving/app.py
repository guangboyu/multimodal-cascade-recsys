"""FastAPI app exposing the recommendation cascade. Loads the registry once at startup.

Run: ``uv run uvicorn vlmrec.serving.app:app --port 8000``
Then: ``GET /recommend?user_id=0&k=20&diversity=mmr``
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Histogram, make_asgi_app

from .registry import load_registry
from .service import recommend

_state: dict = {}
REQUESTS = Counter("vlmrec_requests_total", "recommend requests", ["diversity"])
LATENCY = Histogram(
    "vlmrec_latency_ms",
    "cascade stage latency (ms)",
    ["stage"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["registry"] = load_registry()
    yield
    _state.clear()


app = FastAPI(title="VLM-Rec serving", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())  # Prometheus scrape endpoint


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
    res = recommend(
        reg,
        user_id,
        k_retrieve=int(reg.cfg.serving.k_retrieve),
        k_prerank=int(reg.cfg.serving.k_prerank),
        k_final=k,
        diversity=diversity,
        mmr_lambda=float(reg.cfg.rerank.mmr_lambda),
    )
    REQUESTS.labels(diversity=diversity).inc()
    for stage, ms in res["latency_ms"].items():
        LATENCY.labels(stage=stage).observe(ms)
    return res
