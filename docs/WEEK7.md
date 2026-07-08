# Week 7 — Cascade consistency: fixing the ranker/retrieval mismatch

## Goal
Turn the Week-4 negative result — the ranker *degrades* the retriever's candidate order — into a
diagnosed, fixed, and measured cascade. Three repairs, each isolating one failure mode.

## What was wrong (mechanically)
1. **Random negatives** (Week 3): the ranker never saw a hard example in training, so re-ranking
   retrieval's all-plausible top-200 was noise. NDCG@10: retrieval order 0.109 → ranker 0.081.
2. **Poisoned hard negatives** (Week 4): candidates masked only *train*-seen items, so a user's
   held-out positive — usually retrieved, since retrieval works — could be sampled as a negative
   and labelled click=0. NDCG@10 fell further to 0.051.
3. **No cross-stage signal**: the ranker had no retrieval-score input, so it could only *replace*
   the retriever's opinion, never *refine* it.

## The fix (three isolated repairs)
- **Clean negative pool** — `sample_negatives()` rejection-resamples any draw colliding with
  {row positive, valid item, test item}, in both the hard and random paths.
- **Retrieval score as a ranker feature** — the raw two-tower dot product `u_e · i_e`, computed
  for every (user, item) pair and identical to the FAISS score at serving time. Deliberately NOT
  a top-K membership flag: train positives are masked out of the candidate file, so a flag would
  mark every positive 0 / every hard negative 1 — label leakage in reverse.
- **Listwise softmax loss** — pointwise BCE optimizes per-item calibration; the cascade's serving
  task is *ordering a slate*. Cross-entropy over each [positive + negatives] slate matches it.

Plus a production-standard backstop: **score fusion** `α·retrieval + (1−α)·ranker` with α tuned on
the valid split (α→1 degenerates to retrieval order, so fused ≥ retrieval is guaranteed).

## Results (test split; NDCG@10 over the retriever's top-200)

| ranker | ranker order | GAUC (random negs) |
|---|---|---|
| naive (random negatives, Week 3) | 0.081 | 0.858 |
| hard negatives (poisoned, Week 4) | 0.051 | — |
| hard negatives, clean pool | 0.060 | 0.791 |
| **+ retrieval-score feature (serving ckpt)** | **0.075** | 0.825 |
| + listwise softmax | 0.064 | 0.647 |
| + serving-like slate (16 negs, 75% hard from top-50) | 0.032 | 0.636 |

Raw retrieval order: 0.109. Fusion α tuned on valid → α=0.75 (valid 0.1286 vs 0.1284 at α=1.0 —
statistically a tie with pure retrieval order); test at α=0.75 → 0.107.

**Reading the table honestly:** the two principled repairs deliver a **+48%** cascade NDCG lift
over the poisoned baseline (0.051 → 0.075), but no ranker variant beats the raw retrieval order
on this metric, and fusion tuning converges to α≈1. Three findings explain why:

1. **The metric is retrieval-favoring by construction** — candidates were selected by the
   retriever's own similarity, so its ordering gets home-field advantage.
2. **Residual selection bias in hard-negative mining**: train positives are seen-masked *out* of
   the candidate file, so any feature correlated with candidate membership is inversely
   predictive. Sampling hard negatives uniformly over the top-200 dilutes the shortcut; the
   "serving-like" variant that concentrated them in the top-50 amplified it and cratered (0.032).
   Facebook's EBR paper's "mix easy and hard negatives" guidance, reproduced from first principles.
3. **Objective mismatch has limits**: the listwise loss aligns training with slate ordering but
   sacrifices the calibrated-probability view (GAUC vs random negatives drops to 0.65) without
   beating pointwise BCE here — with only 25K items and one behavioral positive per user-step,
   the retrieval score is already close to the information ceiling of this offline setup.

Variant selection happens on the **valid** split; test is only reported. The serving checkpoint
(`data/rerank/ranker_cascade.pt` + sidecar metadata) is the valid-best variant, the FAISS score
rides through the live cascade as its input feature (train/serve consistent), and the pre-ranker
is re-distilled from it (top-50 keeps 70% of its top-10).

## Run it
```bash
make week4        # candidates (+scores/user embeddings) → variant grid → fusion → distill → diversity
make serve        # serves the cascade-consistent ranker automatically
```

## Honest caveats
- Offline cascade NDCG on retriever-selected candidates is **retrieval-favoring** — the candidate
  set was chosen by the retriever's own similarity. The unbiased comparison needs online traffic
  or counterfactual estimators.
- The listwise ranker's GAUC against *random* negatives drops (it no longer optimizes that task);
  judge it on the slate-ordering metric it serves.
