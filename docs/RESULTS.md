# Results at a glance

End-to-end multimodal recommender on **Amazon Reviews 2023** — prototyped on `Video_Games`
(94,762 users · 25,612 items · 814,586 interactions), scale-validated on
`Beauty_and_Personal_Care` (729,576 users · 207,649 items · 6.62M interactions). Temporal
leave-last-out. Full multi-stage cascade: candidate generation → pre-ranking → ranking →
post-processing → serving, plus VLM item understanding and RQ-VAE semantic IDs.

## Headline numbers (Video_Games; test split)

| stage | what | result |
|---|---|---|
| **Retrieval** (W2) | two-tower multimodal vs ID-only, R@100 | **0.214 vs 0.129 (+66%)**; **+81%** cold-start |
| **Ranking** (W3) | DIN+DCN-v2+MMoE, click GAUC | **0.858**; −multimodal 0.761 (cold collapses 0.70→0.35) |
| **Cascade fix** (W7) | poisoned hard-negs → clean pool + retrieval-score feature, NDCG@10 | **0.051 → 0.075 (+48%)** on the original run (fused re-run: 0.063, different winner — see WEEK7) |
| **VLM profiles** (W8) | Qwen2.5-VL structured item profiles, 25.6K items | **100% JSON validity in 4.0 h**; +1.5% R@100; cold-start neutral (honestly reported) |
| **Semantic IDs** (W9) | RQ-VAE codes vs raw item IDs, cold-start R@100 | **+46%** (0.0241 → 0.0353); ranker +SID GAUC 0.858→0.861 |
| **Generative retrieval** (W9) | TIGER-lite demo, R@10 | 0.031 vs two-tower 0.063 — 1M params, no ANN index |
| **Serving** (W5/7) | FastAPI cascade p50/p99 CPU (fused 1281-d stack) | **12 / 16 ms**; ONNX Runtime path item-identical (parity 2.4e-06) |
| **Scale run** (W10) | 8.1x items through the same configs | multimodal-vs-ID **+75%** and SID cold-start **+47%** replicate; VLM lift grows to **+3.1%**; serving p99 **10 ms** (HNSW, 11 GB registry) |

**The throughline:** multimodal item content is the dominant driver at both retrieval and ranking,
and *content-derived representations* (CLIP, VLM profiles, semantic IDs) pay exactly where
collaborative signal is absent — the cold-start / long-tail slice.

## The two stories interviewers should ask about

**1. The cascade-consistency arc (W3→W7).** A 0.858-GAUC ranker *degraded* the retriever's
candidate order (NDCG@10 0.106 → 0.082): sample-selection bias — it trained on random negatives
and served on retrieval's hard ones. Naive hard-negative mining made it *worse* (0.051): the
candidate pool masked only train-seen items, so held-out positives were being labelled as
negatives. The fix — a clean negative pool + the retrieval score as a ranker input — recovered
+48% on that run (0.051 → 0.075; the fused-feature re-run flips the winning variant and lands at
0.063 — variant ranking is feature-set-dependent, which is why selection is automated on the
valid split). Two second-order findings: a "was-retrieved" membership flag would have been reverse
label leakage (train positives are masked out of the candidate file), and concentrating hard
negatives in the top-50 cratered NDCG (0.032, and 0.015 on the re-run), demonstrating *residual
selection bias* — Facebook EBR's "mix easy and hard negatives" guidance re-derived from a
controlled failure. Fusion-α tuning converging to α≈0.75–1.0, statistically tied with pure
retrieval order, quantifies how retrieval-favoring the offline metric is; the unbiased comparison
needs online A/B.

**2. The honest VLM ablation (W8 → W10).** Structured Qwen2.5-VL item profiles (100% valid JSON,
batched offline inference) added +1.5% overall recall but *nothing* on cold-start — the opposite
of the hypothesis. Root cause: Video_Games listings are text-rich, so the profile mostly
re-encodes what the raw-text embedding already had. The 385-d profile embedding alone nearly
matches the 513-d CLIP block — the VLM is a powerful *compressor* even when not additive. The
scale run then tested category-dependence: on the 7.7x-sparser Beauty catalog the overall lift
**doubled to +3.1%** while the extreme-cold null replicated — the refined claim is "VLM profile
value grows with sparsity, but the extreme cold tail needs ID-pathway fixes (semantic IDs), not
more content."

**3. Cross-category replication (W10).** Every headline finding was re-measured on
Beauty_and_Personal_Care (729,576 users · 207,649 items · 6.62M interactions) via a config-only
category swap: multimodal-vs-ID grew to **+75%**, SID cold-start lift replicated at **+47%**
(0.0056→0.0083), the multimodal-removal collapse and the retrieval-favoring cascade metric all
reproduced, and the cascade variant winner *changed again* (listwise softmax won at scale) —
validating automated valid-split selection over fixed model choices. Ops findings: HNSW's −1
sentinels crashed serving until `efSearch`/filtering were fixed; the 3B VLM needed batch re-tuning
to saturate the GPU (16.8 h for 207K profiles at 99.76% validity); serving holds **p50 8.6 /
p99 10.1 ms** with an 11 GB registry.

## What the project demonstrates
- **Retrieval:** two-tower/EBR, in-batch sampled softmax + logQ correction, FAISS (flat/HNSW/IVF), item2item.
- **Ranking:** DIN target-attention, DCN-v2, MMoE multi-task, GAUC with midrank ties, listwise vs pointwise losses.
- **Cascade:** hard-negative hygiene, cross-stage score features, valid-split variant selection,
  distilled pre-ranking, MMR/DPP diversity.
- **VLM/LLM engineering:** batched VLM inference (left-padding, shard-resume, atomic checkpoints,
  guided/defensive JSON), profile→embedding feature pipeline, throughput tuning (batch sizing at 100% GPU).
- **Semantic IDs:** RQ-VAE with the full anti-collapse kit (k-means init, EMA, dead-code re-seeding,
  perplexity gates), SID feature modes, TIGER-style constrained-trie generative retrieval.
- **Serving/MLOps:** FastAPI + FAISS + ONNX Runtime (verified parity), checkpoint sidecar metadata,
  Docker, Prometheus, MLflow, CI, config-driven category scaling.

## Resume bullets (earned)
- Built an **end-to-end multimodal recommender** (Amazon Reviews 2023; 815K→6.6M interactions) with
  a retrieval → pre-rank → rank → post-process cascade served at **16 ms p99 on CPU** (FastAPI +
  FAISS + ONNX Runtime, verified torch parity).
- **Diagnosed and fixed a cascade sample-selection bias**: held-out positives poisoning the
  hard-negative pool and a missing cross-stage score feature; **up to +48% cascade NDCG@10**
  (+23% on the fused-feature re-run under automated valid-split variant selection), with
  residual-selection-bias and reverse-label-leakage failure modes documented from controlled
  experiments.
- Ran **Qwen2.5-VL batch inference over the full catalog** (100% valid structured JSON, 25.6K items
  in 4 h / 207K items with tuned batching) and measured its feature value with a 5-combo ablation —
  including the honest cold-start null result.
- Implemented **RQ-VAE semantic IDs** (100% codebook utilization, perplexity-gated) delivering
  **+46% cold-start retrieval** as an item-ID replacement, plus a TIGER-style generative-retrieval
  demo with constrained trie decoding.
- Scaled the whole pipeline **8.1x by config only** (category-scoped paths) and **replicated every
  headline finding cross-category** (+75% multimodal-vs-ID, +47% SID cold-start), while capturing
  per-stage bottlenecks — GPU-underfill at small model sizes, single-thread image decode, ANN −1
  sentinels — and their fixes.

Per-stage write-ups in [`docs/`](.) and the honest bug log in [`PITFALLS.md`](PITFALLS.md).
