# Candidate generation (lean retrieval)

## Goal
Reduce ~25k items to a few hundred good candidates per user, cheaply. Retrieval is a *recall* problem
kept deliberately simple; the ML depth is saved for ranking (03-ranking.md).

## What was built (`src/vlmrec/retrieval/`)
- **`data.py`** — assembles the item content matrix (text ⊕ image ⊕ has_image = 897-d), structured
  features via **quantile-bucket + embedding** (price, category leaf) and an item-id, per-user pooled
  train histories (with leave-one-out at train time), held-out targets, a CSR of seen items for masking,
  and item popularity.
- **`model.py`** — `TwoTower` with three `feature_mode`s for ablation:
  - `content` — pure multimodal (item = content MLP; user = pooled-history MLP),
  - `hybrid` — content ⊕ id/price/category embeddings,
  - `id` — pure collaborative (learned user/item id vectors).
  Trained with **temperature-scaled in-batch sampled softmax** + **logQ** popularity correction;
  embeddings L2-normalized (cosine = dot product).
- **`train.py`** — training loop + masked **Recall@K / NDCG@K** eval (exact, brute-force for correctness).
- **`index.py`** — **FAISS** ANN (flat / HNSW / IVF) + an ANN-vs-exact recall check.
- **`i2i.py`** — item2item **co-visitation** (popularity-damped) as a complementary source.
- **`eval.py`** — the ablation: every source, overall vs a **cold-start** (long-tail) slice, + a
  popularity baseline.

## Results — test ablation (temporal leave-last-out)
| source | R@10 | R@100 | R@500 | R@100 (cold-start) |
|---|---|---|---|---|
| **two-tower (multimodal)** | 0.062 | **0.210** | **0.404** | 0.044 |
| two-tower (hybrid) | 0.062 | 0.211 | 0.406 | 0.036 |
| two-tower (ID-only) | 0.036 | 0.129 | 0.263 | 0.024 |
| item2item co-visitation | 0.038 | 0.116 | 0.170 | **0.065** |
| popularity | 0.025 | 0.097 | 0.233 | 0.000 |

FAISS HNSW recovers **81%** of the exact top-100.

## Takeaways
- **Multimodal content beats pure-ID by +63% R@100 (+82% on cold-start) with 30× fewer params** — the
  headline result, and the strongest argument for the multimodal item tower.
- **Sources are complementary:** i2i is best on cold-start; popularity is useless there. This is why a
  real system blends a learned retriever with a co-visitation source — exactly the "lean: one two-tower
  + one i2i source" design.
- **Lean retrieval is the right call:** adding structured/id features to the tower didn't help and hurt
  cold-start, so that work moves to the ranker.

## Run it
```bash
make retrieval          # train content/hybrid/id + run the ablation
```
