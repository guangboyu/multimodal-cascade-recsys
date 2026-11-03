# VLM-Rec

A multimodal recommender built end to end on [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/):
two-tower retrieval over FAISS, a distilled pre-ranker, a DIN + DCN-v2 + MMoE ranker, MMR/DPP
re-ranking and a FastAPI service. Items are represented by frozen text and image embeddings plus
structured profiles written by Qwen2.5-VL, and each design choice is backed by an ablation. The
system also serves as a test bed for measuring where semantic-ID generative retrieval loses recall
compared with dense retrieval.

Technical report: [web version](https://guangboyu.github.io/multimodal-cascade-recsys/tech-report.html)
· [PDF](docs/tech-report.pdf)

![Replaying a user through the live cascade](docs/demo.gif)

## Results

Video_Games (94,762 users, 25,612 items), temporal leave-last-out, test split. "Cold" means the
target item is in the bottom quartile of training popularity.

| Comparison | Result |
|---|---|
| Content two-tower vs. ID-only two-tower, Recall@100 | 0.214 vs. 0.129 (+66%; +81% on cold items) |
| Ranker GAUC, full vs. without content features | 0.858 vs. 0.761 |
| Cascade NDCG@10, poisoned hard negatives vs. clean pool + retrieval-score feature | 0.051 vs. 0.075 |
| Semantic IDs vs. learned item IDs, cold Recall@100 | +46% |
| Serving latency on CPU, p50 / p99 | 12 / 16 ms |

The same code runs on Beauty_and_Personal_Care (207,649 items) with a different config file. There
the content tower beats ID-only by 75%, removing content costs the ranker 0.06 GAUC, semantic IDs
lift cold recall by 47%, and serving holds 10 ms p99 with HNSW.

In the generative-retrieval study, quantizing item content into codes accounts for 58–92% of the
Recall@10 gap to the dense retriever, while beam search accounts for 5–19%. The report has the full
tables and the method.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/); training needs an NVIDIA GPU.

```bash
uv sync --all-extras
make data        # download, 5-core filter, temporal split, text and image embeddings
make retrieval   # two-tower variants and the retrieval ablation
make ranking     # ranker and its ablation
make rerank      # cascade consistency, pre-ranker distillation, diversity
make vlm         # Qwen2.5-VL item profiles
make sid         # RQ-VAE semantic IDs
make serve       # API on :8000 and the Streamlit demo on :8501
```

Each stage reads `configs/config.yaml`. Override any key with `-o`, for example
`uv run vlmrec retrieval-train -o retrieval.epochs=5`, or pass `--config configs/scale.yaml` to run a
stage on the Beauty catalog.

## API

```bash
curl "localhost:8000/recommend?user_id=42&k=10"               # known user
curl "localhost:8000/recommend?user_id=42&k=10&explain=true"   # candidates after each stage
curl -X POST localhost:8000/recommend/session \
     -H 'content-type: application/json' \
     -d '{"items": [18327, 442, 9032], "k": 10}'               # new session, no user id
```

Responses include per-stage latency. Unknown ids return 404 and invalid parameters return 422. The
same cascade can be called in-process with `vlmrec.serving.service.recommend`.

## Repository

| Path | Contents |
|---|---|
| `src/vlmrec/data`, `features`, `vlm` | download, splits, text, image and VLM encoders |
| `src/vlmrec/retrieval` | two-tower model, FAISS index, co-visitation, evaluation |
| `src/vlmrec/ranking`, `rerank` | ranker, pre-ranker, negative sampling, MMR and DPP |
| `src/vlmrec/sid` | RQ-VAE semantic IDs and a TIGER-style generative retriever |
| `src/vlmrec/serving`, `mlops` | FastAPI app, ONNX export, Streamlit demo, MLflow logging |
| `configs/` | `config.yaml` for Video_Games, `scale.yaml` for Beauty |
| `docs/` | technical report, per-stage notes `01`–`10`, `RESULTS.md`, `PITFALLS.md` |

`make test` runs the unit tests (no network or GPU needed) and `make lint` runs ruff.

## Data

Amazon Reviews 2023 (McAuley Lab, UCSD), read from `McAuley-Lab/Amazon-Reviews-2023` on Hugging
Face.
