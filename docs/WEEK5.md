# Week 5 — Serving

## Goal
Put the whole cascade behind an HTTP API with a real latency budget — the part that turns a notebook
into a system.

## What was built (`src/vlmrec/serving/`)
- **`registry.py`** — loads every artifact once at startup (CPU): retrieval two-tower + **FAISS** index,
  the heavy ranker, the distilled pre-ranker, and the item/user feature tensors.
- **`service.py`** — the request-time cascade with **per-stage latency**:
  FAISS retrieve (mask seen) → pre-rank cut (200→50) → ranker (DIN/DCN/MMoE) → fuse + MMR/DPP diversity.
- **`app.py`** — **FastAPI** app: `GET /health`, `GET /recommend?user_id=&k=&diversity=mmr|dpp|none`.
- **`export_onnx.py`** — exports the ranker to **ONNX** (feature tables baked in) and verifies parity.
- **`Dockerfile`** + **`docker-compose.yml`** — containerized serving (data mounted read-only).
- **`catalog.py` + `demo_app.py`** *(added later)* — read-only metadata endpoints (`/item`, `/user`,
  `/similar`, `/search`, `/users/sample`, static `/images`, `/recommend?explain=true` stage traces)
  plus a **Streamlit** demo UI over them (`make demo`). The scoring path is untouched — enrichment
  happens only when a caller asks for it.
- **`POST /recommend/session`** — the identical cascade for an ad-hoc item history ("build your own
  taste" in the demo). Possible because serving is user-id-free: the content-mode user tower pools
  item content and the ranker reads the behaviour sequence, so unseen sessions need no retraining.

## Latency (CPU, warm, 200-user sample)

| stage | avg ms |
|---|---|
| retrieve (FAISS) | 1.1 |
| pre-rank | 5.9 |
| rank (DIN+DCN+MMoE) | 3.8 |
| post-process (fuse + MMR/DPP) | 0.8 |
| **total** | **p50 11 · p95 14 · p99 16** |

The full multimodal cascade serves in **~11 ms p50 / ~16 ms p99 on CPU** — comfortably under a 100 ms
budget. (Over HTTP via uvicorn: ~10–19 ms end-to-end.)

## ONNX export
Ranker → `data/serving/ranker.onnx` (100 MB, tables baked in), ONNX Runtime vs torch parity
**max|diff| = 3.8e-6** — ready for an ONNX Runtime / Triton deployment.

## Run it
```bash
make serve        # uvicorn at http://localhost:8000
curl "http://localhost:8000/recommend?user_id=42&k=10&diversity=dpp"
make demo         # Streamlit UI at http://localhost:8501 (talks to the API over HTTP)
make export-onnx  # export + verify the ONNX ranker
docker compose up --build   # containerized: api on :8000 + demo on :8501 (mounts ./data)
```

## Notes / next (Week 6)
- Serving runs on CPU; the cu128 torch wheel runs CPU-only fine (swap to a CPU wheel to slim the image).
- Week 6 adds Prometheus monitoring, MLflow tracking, and CI (Redis/Grafana noted as follow-ups).
