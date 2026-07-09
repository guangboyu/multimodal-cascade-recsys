# VLM-Rec — Multimodal Multi-Stage Recommender

An end-to-end **multimodal** recommendation system on **Amazon Reviews 2023**, built to exercise the
canonical + SOTA industrial RecSys stack: **candidate generation → pre-ranking → ranking →
post-processing → serving**. Engineering-sophisticated but deliberately understandable.

## Architecture

```
                  ┌───────────────────── OFFLINE (precompute + train) ─────────────────────┐
 product image ──▶│ CLIP emb ─┐                                                             │
        └─────────│ Qwen2.5-VL ─▶ structured item profile ─▶ MiniLM emb ─┐                  │
 title/desc/feat ─│ MiniLM emb ─┴──────────────────────────────────┬─────┴─▶ fused item     │
                  │                                                │        content (1281d) │
                  │                              RQ-VAE ◀──────────┘                        │
                  │                                └──▶ semantic IDs (3×256 codes)          │
                  └──────────────────────────────────────────────────────────────┬──────────┘
                                                                                 ▼
 request(user) ──────────▶ ┌ Retrieval ┐  ┌ Pre-rank ┐  ┌  Rank   ┐  ┌ Re-rank ┐
                           │2-tower+FAISS│─▶│ distilled │─▶│DCN-v2+DIN│─▶│ DPP/MMR │─▶ top-N
                           │ + i2i (lean)│  │  (~50)    │  │+MMoE+SID │  │ + rules │
                           └─────────────┘  └──────────┘  └──────────┘  └─────────┘
                          ◀──── ONLINE (FastAPI cascade, 16ms p99 CPU; retrieval score ────▶
                                        rides through as a cross-stage ranker feature)
```

**Emphasis:** *deep* ranking + multimodal feature representation; *lean* retrieval — mirroring how
production systems allocate complexity.

**📊 Headline results & resume bullets → [`docs/RESULTS.md`](docs/RESULTS.md).** Per-stage write-ups
([`docs/WEEK1..6.md`](docs/)) and an honest bug log ([`docs/PITFALLS.md`](docs/PITFALLS.md)) included.

## Status

- ✅ **Week 1 — Data + feature foundation**: download, 5-core filtering, temporal leave-last-out
  splits, image download, **precomputed CLIP image + text embeddings** (full Video_Games:
  94,762 users · 25,612 items · 814,586 interactions; 99.98% image coverage).
- ✅ **Week 2 — Retrieval** (two-tower + FAISS + i2i): content/hybrid/ID two-towers trained with
  in-batch negatives + logQ, FAISS ANN, item2item co-visitation, and a Recall@K / cold-start ablation.
- ✅ **Week 3 — Ranking** (DCN-v2 + DIN + MMoE multi-task): re-scores candidates; ablation shows
  multimodal features dominate GAUC and cold-start collapses without them.
- ✅ **Week 4 — Pre-rank + post-process**: distilled pre-ranker (top-50 keeps 79% of the ranker's
  top-10), MMR/DPP diversity trade-off, and a documented cross-stage cascade-consistency finding.
- ✅ **Week 5 — Serving**: FastAPI cascade (retrieve→pre-rank→rank→post-process) at **~16 ms p99 on
  CPU**, FAISS index, ONNX export (parity 4e-6), Dockerfile + docker-compose.
- ✅ **Week 6 — MLOps**: GitHub Actions CI (lint + tests), Prometheus `/metrics` in the serving app,
  MLflow experiment tracking.
- ✅ **Week 7 — Cascade consistency**: hard-negative pool hygiene (held-out positives excluded),
  the retrieval score as a cross-stage ranker feature, listwise-loss variant grid selected on the
  valid split, score fusion — **+48% cascade NDCG@10** over the poisoned baseline, with the
  failure modes (reverse label leakage, residual selection bias) documented.
- ✅ **Week 8 — VLM item understanding**: Qwen2.5-VL structured profiles for every item (100% valid
  JSON, 4 h/25.6K items), profile embeddings as a third content block; the ablation honestly shows
  +1.5% overall recall and a cold-start null on this text-rich category.
- ✅ **Week 9 — Semantic IDs**: RQ-VAE codes (100% codebook utilization) as an item-ID replacement —
  **+46% cold-start retrieval** — plus a TIGER-style generative-retrieval demo with constrained
  trie decoding.
- ▶ **Week 10 — Scale run**: the same configs over Beauty_and_Personal_Care (729K users · 208K
  items · 6.6M interactions), category-scoped paths, per-stage bottleneck notes.

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

### Week 3 ranking — ablation (GAUC = per-user AUC)

| variant | GAUC | GAUC (cold-start) |
|---|---|---|
| **full (DIN + DCN-v2 + MMoE, multimodal)** | **0.858** | **0.706** |
| − multimodal | 0.761 | 0.347 |
| − DIN | 0.853 | 0.692 |
| − DCN-v2 cross | 0.850 | 0.697 |
| single-task (no MMoE) | 0.858 | 0.708 |

Multimodal features are the dominant driver — **−0.098 GAUC** overall and cold-start **collapses
below random** without them; DIN and DCN-v2 each add ~0.006–0.009; multi-task is neutral on click
while adding the satisfaction objective for free. (Honest cascade finding: the random-negative ranker
*degrades* the retriever's hard candidates — sample-selection bias, fixed in Week 4.) `make week3`.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and (optionally) an NVIDIA GPU.

```bash
uv sync --all-extras        # create .venv and install all stages (faiss, mlflow live in extras)

make week1-dev              # fast capped run (~200k reviews, 500 images) — proves the pipeline
make week1                  # full Video_Games build (all reviews + items)
make week2                  # train two-towers + retrieval ablation (needs Week-1 artifacts)
make week3                  # train ranker (DIN+DCN-v2+MMoE) + ablation
make week4                  # cascade fix: variant grid + fusion + pre-ranker distill + diversity
make week8                  # VLM item profiles (Qwen2.5-VL) + encode + feature-source ablation
make week9                  # RQ-VAE semantic IDs + SID-vs-ID ablation (+ `vlmrec tiger-demo`)
make serve                  # run the FastAPI cascade (http://localhost:8000, ~16ms p99)

# scale profile (Beauty_and_Personal_Care, category-scoped paths):
uv run vlmrec <stage> --config configs/scale.yaml
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
  retrieval/ ranking/ rerank/ serving/ mlops/  # Weeks 2–7 + MLOps
  vlm/ sid/              # Week 8 (VLM item profiles) · Week 9 (RQ-VAE semantic IDs + TIGER demo)
  config.py paths.py utils.py cli.py
tests/                   # pure-logic smoke tests (no network/GPU needed)
data/                    # artifacts (gitignored): raw/ processed/ images/ embeddings/
```

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD) —
`McAuley-Lab/Amazon-Reviews-2023` on HuggingFace. Default category **`Video_Games`**
(~4.6M reviews, ~137K items) — visually distinctive box art + rich review text, sized to iterate
on a single GPU.
