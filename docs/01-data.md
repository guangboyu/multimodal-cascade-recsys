# Data + multimodal feature foundation

## Goal
Turn raw Amazon Reviews 2023 (`Video_Games`) into a clean, split, multimodal dataset that every later
stage builds on: interactions with leakage-free temporal splits, plus precomputed image + text
embeddings for each item.

## What was built
- **Download** (`data/download.py`) — reads the repo's raw `*.jsonl` directly via `huggingface_hub`
  (the config loader is gone in `datasets` 5.x; see PITFALLS #2). Lean parquet output; nested/messy
  metadata fields normalized to JSON strings for a stable schema.
- **Interactions** (`data/build_interactions.py`) — dedup (user, item), iterative **k-core** filtering,
  **temporal leave-last-out** split (per user: last → test, 2nd-last → valid), contiguous id remap. The
  k-core and split logic are pure functions with unit tests (no net/GPU).
- **Images** (`data/download_images.py`) — concurrent, resumable, Pillow-validated download of one image
  per item; shape-agnostic URL extraction; a manifest with a `has_image` flag.
- **Embeddings** — `features/encode_text.py` (Sentence-Transformers `all-MiniLM-L6-v2`, 384-d) and
  `features/encode_image.py` (open_clip **CLIP ViT-B/32**, 512-d), L2-normalized, aligned to `item_idx`,
  with a zero-vector + flag for missing images.

## Result (full corpus)
| metric | value |
|---|---|
| reviews → after 5-core | 4,624,615 → 814,586 |
| users · items | 94,762 · 25,612 |
| splits (train / valid / test) | 625,062 / 94,762 / 94,762 |
| image coverage | 99.98% (25,606 / 25,612) |
| text emb / image emb | (25612, 384) / (25612, 512) |

## Design choices
- **Temporal leave-last-out**, not random split — prevents future leakage, the standard for sequential rec.
- **Frozen encoders, precomputed once** — CLIP/MiniLM are not fine-tuned in v1; embeddings are cached so
  every downstream experiment is fast.
- **`item_idx`-aligned arrays everywhere** — row `i` of every matrix is the same item; no join needed at
  train time.

## Run it
```bash
make data          # full build
make data-dev      # fast capped subsample (proves the pipeline)
```
