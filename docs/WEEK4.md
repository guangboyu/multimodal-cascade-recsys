# Week 4 — Pre-ranking + post-processing

## Goal
Two stages around the heavy ranker: a **pre-ranker** that cheaply cuts the retriever's ~200 candidates
to ~50, and **post-processing** that turns scores into a final, *diverse* list.

## What was built (`src/vlmrec/rerank/`)
- **`candidates.py`** — precompute the two-tower's top-200 per user (the cascade candidate set + the
  source of hard negatives).
- **`prerank.py`** — a lightweight pre-ranker (small Ranker: no DIN/cross) **distilled** from the ranker's
  click logits; measures how much of the ranker's top-10 it preserves.
- **`postprocess.py`** — score fusion `P(click)·P(sat)`, **MMR** and greedy-MAP **DPP** diversity over
  item content embeddings, a per-category cap, and an intra-list diversity metric.
- **`cascade.py`** — orchestrates the hard-negative experiment, the pre-rank distillation, and the
  relevance↔diversity sweep.

## Results

**Pre-ranking (distillation):** the lightweight pre-ranker's top-50 recovers **79.3%** of the ranker's
top-10 — a cheap, consistent first cut (distillation MSE 0.22 → 0.13).

**Post-processing — relevance vs diversity** (on the pre-ranked top-50):

| method | NDCG | intra-list diversity |
|---|---|---|
| MMR λ=1.0 (pure relevance) | 0.0314 | 0.316 |
| MMR λ=0.7 | 0.0282 | 0.352 |
| MMR λ=0.5 | 0.0236 | 0.367 |
| DPP | 0.0181 | 0.385 |

A clean, tunable trade-off: lower λ (and DPP) buy diversity for a little relevance. **DPP** gives the
most diverse list (+22% diversity vs pure relevance, −42% NDCG) — the knob a product would tune.

## Cascade diagnosis (honest negative result)
Re-ranking the retriever's candidates with the ranker **lowers** NDCG@10 (retrieval order 0.109 →
ranker order 0.081), and training the ranker on **hard negatives made it worse** (0.051), not better.

Why — three compounding reasons, all worth being able to explain:
1. **"Retrieved ≈ negative" leakage.** Sampling retrieved items as negatives teaches the model that
   retrieved items are negative — but the held-out positive *is* a retrieved item, so it gets penalized too.
2. **Missing cross-stage feature.** The ranker scores by predicted click probability and has **no
   retrieval/pre-rank score feature**, so it can't *refine* the retriever's ordering — it replaces it.
3. **Retrieval-favoring metric.** The candidates were selected by the retriever's own similarity, so
   "rank the held-out positive" is a metric the retriever is already optimized for.

**Principled fix (next refinement):** feed the retrieval / pre-rank score into the ranker as a feature
(stage-score fusion) and sample hard negatives *excluding* near-positives. This is the standard way
real cascades stay consistent — and exactly the kind of systems insight the cascade was built to surface.

## Run it
```bash
make week4
```
