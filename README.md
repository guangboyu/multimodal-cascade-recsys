# VLM-Rec — Multimodal Multi-Stage Recommender

An end-to-end **multimodal** recommendation system on **Amazon Reviews 2023**, built to exercise the
canonical + SOTA industrial RecSys stack: **candidate generation → pre-ranking → ranking →
post-processing → serving**. Engineering-sophisticated but deliberately understandable.

> Full design, rationale, and the 6–8 week roadmap: [`idea-stage/IDEA_REPORT.md`](idea-stage/IDEA_REPORT.md).

## Architecture

```
                       ┌──────────────── OFFLINE (precompute + train) ────────────────┐
 product image ─CLIP──▶│   Multimodal Fusion ──▶ item content embedding (shared)       │──┐
 title/desc/feat ─ST──▶│   (late fusion + modality dropout + contrastive align)        │  │
 price/cat/store ─────▶│                                                                │  │
                       └────────────────────────────────────────────────────────────────┘  ▼
 request(user) ─▶[Feast]─▶ ┌ Retrieval ┐  ┌ Pre-rank ┐  ┌  Rank  ┐  ┌ Re-rank ┐
                           │2-tower+FAISS│─▶│ distilled │─▶│DCN-v2+DIN│─▶│ DPP/MMR │─▶ top-N
                           │ + i2i (lean)│  │   (~100)  │  │  + MMoE  │  │ + rules │
                           └─────────────┘  └──────────┘  └──────────┘  └─────────┘
                          ◀──────────── ONLINE (FastAPI cascade, <100ms p99) ───────────▶
```

**Emphasis:** *deep* ranking + multimodal feature representation; *lean* retrieval — mirroring how
production systems allocate complexity.

## Status

- ✅ **Week 1 — Data + feature foundation**: download, 5-core filtering, temporal leave-last-out
  splits, image download, **precomputed CLIP image + text embeddings** (full Video_Games:
  94,762 users · 25,612 items · 814,586 interactions; 99.98% image coverage).
- ✅ **Week 2 — Retrieval** (two-tower + FAISS + i2i): content/hybrid/ID two-towers trained with
  in-batch negatives + logQ, FAISS ANN, item2item co-visitation, and a Recall@K / cold-start ablation.
- ⬜ Week 3 — Ranking (DCN-v2 + DIN + MMoE)
- ⬜ Week 4 — Pre-rank + post-process
- ⬜ Week 5 — Serving (FastAPI + ONNX + Feast)
- ⬜ Week 6 — MLOps + polish

### Week 2 retrieval — test ablation (temporal leave-last-out)

| source | R@10 | R@100 | R@500 | R@100 (cold-start) |
|---|---|---|---|---|
| two-tower (multimodal) | 0.062 | **0.210** | **0.404** | 0.044 |
| two-tower (hybrid) | 0.062 | 0.211 | 0.406 | 0.036 |
| two-tower (ID-only) | 0.036 | 0.129 | 0.263 | 0.024 |
| item2item co-visitation | 0.038 | 0.116 | 0.170 | **0.065** |
| popularity | 0.025 | 0.097 | 0.233 | 0.000 |

Multimodal-content retrieval beats pure-ID by **+63% R@100** (**+82%** on the cold-start slice)
with 30× fewer params; i2i is complementary (best on cold-start) — motivating a blended multi-source
design. FAISS HNSW recovers 81% of exact top-100. Regenerate via `make week2`.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and (optionally) an NVIDIA GPU.

```bash
uv sync                     # create .venv and install the Week-1 stack

make week1-dev              # fast capped run (~200k reviews, 500 images) — proves the pipeline
make week1                  # full Video_Games build (all reviews + items)
make week2                  # train two-towers + retrieval ablation (needs Week-1 artifacts)
```

Run individual stages:

```bash
uv run vlmrec download           # pull reviews + item metadata from HuggingFace
uv run vlmrec build-interactions # dedup, 5-core filter, temporal splits, id maps
uv run vlmrec download-images    # fetch a product-image subset
uv run vlmrec encode-text        # sentence-transformer text embeddings
uv run vlmrec encode-image       # CLIP image embeddings
uv run vlmrec eda                # dataset card + summary stats
```

Override any config key inline (OmegaConf dotlist):

```bash
uv run vlmrec week1 -o dataset.category=Baby_Products dataset.max_reviews=500000
```

## Repo layout

```
configs/                 # YAML config (dataset, filtering, splits, embeddings)
src/vlmrec/
  data/                  # download, build_interactions, download_images
  features/              # encode_text (sentence-transformers), encode_image (CLIP)
  retrieval/ ranking/ rerank/ serving/ eval/   # Weeks 2–5 (stubs for now)
  config.py paths.py utils.py cli.py
tests/                   # pure-logic smoke tests (no network/GPU needed)
data/                    # artifacts (gitignored): raw/ processed/ images/ embeddings/
idea-stage/IDEA_REPORT.md  # the design doc
```

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD) —
`McAuley-Lab/Amazon-Reviews-2023` on HuggingFace. Default category **`Video_Games`**
(~4.6M reviews, ~137K items) — visually distinctive box art + rich review text, sized to iterate
on a single GPU.
