# Results at a glance

End-to-end multimodal recommender on **Amazon Reviews 2023 — Video_Games** (94,762 users · 25,612
items · 814,586 interactions, temporal leave-last-out). Full multi-stage cascade: candidate
generation → pre-ranking → ranking → post-processing → serving.

## Headline numbers

| stage | what | result |
|---|---|---|
| **Retrieval** (Week 2) | two-tower vs ID-only, test R@100 | **0.210 vs 0.129 (+63%)**; **+82%** on cold-start |
| | item2item vs popularity on cold-start | i2i 0.065 vs popularity 0.000 (complementary sources) |
| **Ranking** (Week 3) | DIN+DCN-v2+MMoE, click GAUC | **0.858**; −multimodal **0.761** (cold-start collapses 0.71→0.35) |
| **Pre-rank** (Week 4) | distilled, consistency | top-50 keeps **79%** of the ranker's top-10 |
| **Post-process** (Week 4) | MMR/DPP diversity | DPP **+22% diversity / −42% NDCG** vs pure relevance |
| **Serving** (Week 5) | FastAPI cascade latency (CPU) | **p50 11 ms · p99 16 ms**; ONNX parity 3.8e-6 |

**The throughline:** multimodal item content (CLIP image + text) is the dominant driver at *both*
retrieval and ranking, and the gap is largest on cold-start / long-tail items — where pure-ID and
popularity baselines fail.

## What the project demonstrates
- **Retrieval:** two-tower / EBR, in-batch sampled softmax + logQ correction, FAISS ANN, item2item.
- **Ranking:** DIN target-attention, DCN-v2 crosses, MMoE multi-task, calibration, GAUC.
- **Features:** CLIP+text multimodal fusion, quantile-bucket + embedding for structured features.
- **Cascade:** pre-rank distillation/consistency, MMR + DPP diversity, and a documented cross-stage
  consistency (sample-selection-bias) finding.
- **Serving/MLOps:** FastAPI + FAISS + ONNX, Docker, Prometheus `/metrics`, MLflow tracking, CI.

## Resume bullets (earned)
- Built an **end-to-end multimodal recommender** (Amazon Reviews 2023, 815K interactions) with a
  retrieval → pre-rank → rank → post-process cascade served behind FastAPI at **~16 ms p99 on CPU**.
- Showed **multimodal (CLIP+text) item features beat pure-ID by +63% Recall@100 (+82% cold-start)** in
  a two-tower retriever, and lift ranking **GAUC from 0.76 → 0.86**.
- Implemented **DIN + DCN-v2 + MMoE** multi-task ranking, **FAISS** ANN retrieval, **item2item**
  co-visitation, **knowledge-distilled pre-ranking**, and **DPP/MMR** diversity re-ranking.
- Diagnosed a **cross-stage cascade-consistency** failure (sample-selection bias) and documented the
  principled fix (retrieval-score feature + hard negatives).
- Productionized with **ONNX** export (parity 4e-6), **Docker**, **Prometheus** monitoring, **MLflow**
  experiment tracking, and **GitHub Actions** CI.

See per-stage write-ups in [`docs/`](.) and the honest bug log in [`PITFALLS.md`](PITFALLS.md).
