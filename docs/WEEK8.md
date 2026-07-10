# Week 8 — VLM item understanding

## Goal
Earn the "VLM" in the project name: a vision-language model (Qwen2.5-VL) reads every product's
image + listing text and writes a **structured item profile** (refined category, sub-genre, visual
style, key attributes, audience, tone, quality cues, one-line summary). Profiles become a third
content block — `features.sources: [text, image, vlm]` — feeding the same two-tower and ranker.

## Why (the design argument)
CLIP embeddings encode what an item *looks like*; a VLM extracts what it *is* — normalized,
human-readable attributes pulled jointly from image and text. The cost amortizes offline per item
(25K, once), not per request. Cold-start is the target: content signal is all a new item has, and
Weeks 2–3 showed content features dominate exactly there.

## What was built
- **`vlmrec vlm-profile`** — shard-checkpointed batch inference (resumable; atomic shard writes).
  Fixed-schema JSON output, stored as JSON strings (stable parquet schema), defensive parsing with
  an `ok` flag and validity-rate tracking in `profile_meta.json`.
- **`vlmrec encode-profile`** — deterministic profile→text template → the same MiniLM encoder as
  the raw-text block (isolates *what the VLM said* from encoder strength) → `profile_emb.npy`.
- **`vlmrec vlm-ablation`** — five source combos through identical two-tower + ranker recipes,
  overall + cold-start metrics.
- **`features.sources`** — one config key drives the content assembly for every stage
  (retrieval, ranking, candidates, serving).

## Engineering notes (the honest bits)
- vLLM's PyPI wheel links `libcudart.so.13` — unusable on the CUDA 12.8 driver (the torch-cu130
  pitfall repeating one layer up), and its torchaudio dependency broke transformers imports for
  the whole venv. Dropped from the extra; the **transformers backend** runs Qwen2.5-VL-7B at
  ≈1.75 items/s on the 4090 (243 min for the full catalog) after fixing **left-padding** for batched
  decoder-only generation (right padding produced garbage for every non-longest row: 3/24 validity
  → 24/24).
- Profiles are versioned artifacts: model id, validity rate, and wall-clock live in the metadata.

## Results (test split; cold = bottom-quartile target train-popularity)

| sources | dim | R@100 | R@100 (cold) | GAUC | GAUC (cold) |
|---|---|---|---|---|---|
| text+image (baseline) | 897 | 0.2104 | **0.0440** | 0.8586 | 0.7033 |
| image | 513 | 0.1924 | 0.0302 | 0.8495 | 0.6820 |
| vlm | 385 | 0.1877 | 0.0287 | 0.8444 | 0.6640 |
| image+vlm | 897 | 0.2126 | 0.0393 | 0.8586 | 0.7046 |
| **text+image+vlm** | 1281 | **0.2135** | 0.0436 | 0.8577 | 0.6994 |

Profile generation: **100% validity** over 25,612 items, 243 min on the 4090 (7B, transformers).

**Honest reading.** The full combo wins overall recall (+1.5%) and is adopted as the production
feature set, but the cold-start hypothesis is NOT supported here: profiles add nothing on the
cold slice. Two findings worth more than a bigger number:
- **`vlm` alone (385-d) nearly matches `image` alone (513-d)** — one small text embedding of the
  structured profile compresses most of the multimodal signal.
- **`image+vlm` matches `text+image` at identical dims** — the profile is an adequate *substitute*
  for raw listing text, but not additive beyond it on this category: Video_Games listings are
  text-rich, so the VLM verbalizes little the raw text didn't already say. The value case for
  profiles is sparse/noisy listings — exactly what the Week-10 category swap tests.

`make week8` regenerates everything.
