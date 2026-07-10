# Week 9 — Semantic IDs (RQ-VAE) + generative retrieval demo

## Goal
Replace opaque item-ID pointers with **semantic IDs**: RQ-VAE quantizes each item's fused content
embedding into 3 hierarchical codes (256 per level) — similar items share prefixes, new items get
meaningful IDs from content alone (TIGER, Rajput et al. 2023; the representation behind
YouTube/Google's SID line).

## What was built
- **`vlmrec sid-train`** — RQ-VAE over the fused item vectors with the full anti-collapse kit:
  k-means codebook init from first-batch residuals, EMA updates, dead-code re-seeding, per-level
  utilization/perplexity gates (fail below 60%), collision-rate reporting.
- **Feature modes** — two-tower `sid` (item pointer → composed code embeddings; user side stays a
  learned id vector, isolating the swap) and `content_sid` (content ⊕ codes); ranker `use_sid`.
- **`vlmrec tiger-demo`** — TIGER-lite generative retrieval: a small decoder-only transformer over
  the user's history as SID tokens generates the next item's codes; beam search constrained to the
  trie of real items — **no ANN index**. Deliberately demo-scale (the two-tower stays the
  production retriever); collisions resolve by within-group popularity.

## Codebook health (the classic VQ failure, instrumented)
On the fused 1281-d item vectors: **100% utilization** on all 3 levels, perplexity **208/249/252**
of 256 (near-uniform code usage — no collapse), collision rate 9.6% (resolved by within-group
popularity at decode; harmless for the feature use-case). Training: ~16 s.

## Results (test split; cold = bottom-quartile target train-popularity)

| item representation (two-tower) | R@100 | R@100 (cold) |
|---|---|---|
| id (raw pointer) | 0.1288 | 0.0241 |
| sid (semantic codes) | 0.0835 | **0.0353 (+46%)** |
| content | **0.2135** | 0.0436 |
| content_sid | 0.2109 | 0.0416 |

| ranker | GAUC | GAUC (cold) |
|---|---|---|
| base | 0.8580 | 0.6993 |
| + SID feature | **0.8610** | **0.7076** |

**Reading it:** semantic IDs do exactly what the papers promise on the slice they target — the
`sid` tower loses to raw IDs overall (warm items prefer a free per-item vector) but lifts
**cold-start +46%**, because a new item's codes are already trained by every item sharing them.
As ranker features they're additive even on top of full content (+0.003 GAUC, +0.008 cold).
`content_sid` ≈ `content`: codes derived from content add nothing when the content is present —
their value is as a compact *ID replacement*, not extra signal.

**TIGER-lite demo:** Recall@10 **0.0309** / NDCG@10 0.0158 on 10,000 sampled users — about half
the two-tower's 0.0625, from a 1M-parameter model with a 769-token vocabulary, no ANN index, and
48 s of training. It trails as expected at demo scale; the story is the paradigm (constant vocab,
index-free, generation-native retrieval) and the constrained-trie decoding that guarantees every
generated sequence is a real item.

`make week9` regenerates everything.
