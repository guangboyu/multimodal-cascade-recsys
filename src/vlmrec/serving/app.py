"""FastAPI app exposing the recommendation cascade. Loads the registry once at startup.

Run: ``uv run uvicorn vlmrec.serving.app:app --port 8000``
Then: ``GET /recommend?user_id=0&k=20&diversity=mmr``

Beyond the scoring endpoint, a set of read-only catalog endpoints (/item, /user, /similar,
/users/sample, static /images) back the Streamlit demo — they serve human-readable metadata
and never enter the latency-tracked cascade path.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, make_asgi_app

from ..paths import Paths
from .catalog import load_catalog
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
    reg = load_registry()
    _state["registry"] = reg
    _state["catalog"] = load_catalog(Paths(reg.cfg))
    # product photos, keyed by parent_asin (mounted here so the demo needs only the API origin)
    app.mount(
        "/images",
        StaticFiles(directory=Paths(reg.cfg).images, check_dir=False),
        name="images",
    )
    yield
    _state.clear()


app = FastAPI(title="VLM-Rec serving", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())  # Prometheus scrape endpoint


def _check_item(item_idx: int) -> None:
    n = _state["catalog"].n_items
    if not 0 <= item_idx < n:
        raise HTTPException(404, f"item_idx out of range (0..{n - 1})")


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
    explain: bool = Query(False, description="include per-stage candidates + rank journey"),
    enrich: bool = Query(False, description="include title/image metadata for final items"),
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
        explain=explain,
    )
    REQUESTS.labels(diversity=diversity).inc()
    for stage, ms in res["latency_ms"].items():
        LATENCY.labels(stage=stage).observe(ms)
    if enrich:
        cat = _state["catalog"]
        res["results"] = [
            {**cat.summary(i), "score": s} for i, s in zip(res["items"], res["scores"], strict=True)
        ]
    return res


@app.get("/item/{item_idx}")
def item(item_idx: int):
    """Full item card: metadata + the parsed VLM profile."""
    _check_item(item_idx)
    return _state["catalog"].detail(item_idx)


@app.get("/items/brief")
def items_brief(ids: str = Query(..., description="comma-separated item_idx list")):
    """Batch summaries so the UI enriches whole grids in one round-trip."""
    try:
        idxs = [int(x) for x in ids.split(",") if x.strip() != ""]
    except ValueError as e:
        raise HTTPException(422, "ids must be comma-separated integers") from e
    if len(idxs) > 500:
        raise HTTPException(422, "at most 500 ids per call")
    for i in idxs:
        _check_item(i)
    cat = _state["catalog"]
    return [cat.summary(i) for i in idxs]


@app.get("/similar/{item_idx}")
def similar(item_idx: int, k: int = Query(8, ge=1, le=50)):
    """Item-to-item neighbours from the two-tower item embeddings (same FAISS index)."""
    _check_item(item_idx)
    reg, cat = _state["registry"], _state["catalog"]
    q = reg.item_e[item_idx : item_idx + 1]
    sc, idx = reg.index.search(q, k + 8)  # headroom for self + ANN -1 slots
    out = []
    for score, j in zip(sc[0], idx[0], strict=True):
        if j < 0 or int(j) == item_idx:
            continue
        out.append({**cat.summary(int(j)), "score": round(float(score), 4)})
        if len(out) == k:
            break
    return {"anchor": cat.summary(item_idx), "similar": out}


@app.get("/user/{user_idx}")
def user(user_idx: int, n: int = Query(12, ge=1, le=50)):
    """Interaction history (newest first) + the held-out test item, both enriched."""
    reg, cat = _state["registry"], _state["catalog"]
    d = reg.d
    if not 0 <= user_idx < d.n_users:
        raise HTTPException(404, f"user_idx out of range (0..{d.n_users - 1})")
    hist = d.seen_items[d.seen_indptr[user_idx] : d.seen_indptr[user_idx + 1]]
    test = int(d.test_item[user_idx])
    return {
        "user_idx": int(user_idx),
        "n_history": int(len(hist)),
        "history": [cat.summary(int(i)) for i in hist[::-1][:n]],
        "test_item": cat.summary(test) if test >= 0 else None,
    }


@app.get("/users/sample")
def users_sample(
    n: int = Query(30, ge=1, le=200),
    min_history: int = Query(5, ge=1),
    seed: int = Query(42),
):
    """Random users with enough history to make an interesting demo page."""
    d = _state["registry"].d
    hist_len = np.diff(d.seen_indptr)
    eligible = np.where(hist_len >= min_history)[0]
    if len(eligible) == 0:
        raise HTTPException(404, f"no users with history >= {min_history}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    return [
        {
            "user_idx": int(u),
            "n_history": int(hist_len[u]),
            "has_test": bool(d.test_item[u] >= 0),
        }
        for u in sorted(pick.tolist())
    ]
