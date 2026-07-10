# Week 10 — Scale run: Beauty_and_Personal_Care (8.1x)

## Goal
Re-run the entire pipeline — data → features → VLM profiles → retrieval → cascade → semantic IDs →
serving — on a category ~8x bigger, **by config only** (`configs/scale.yaml`, category-scoped
paths; the Video_Games artifacts stay untouched). Two questions: what breaks at scale, and which
findings replicate across categories.

## Dataset
23,911,390 raw reviews → 5-core → **729,576 users · 207,649 items · 6,624,441 interactions**
(density 0.0044% — 7.7x sparser than Video_Games, so absolute metrics are much lower;
within-category comparisons are the signal).

## Wall-clocks (RTX 4090, WSL2)

| stage | prototype (25.6K items) | scale (207.6K items) |
|---|---|---|
| raw download → parquet | ~1 min | 6 min (23.9M rows) |
| image download | ~1 h @ 16 workers | **25.7 min @ 32 workers** (207,641/207,648, 7.4 GB) |
| CLIP encode | minutes (single-thread) | **4 min** (DataLoader workers) |
| VLM profiles | 243 min (7B) | **16.8 h (3B, 99.76% validity)** — see incidents |
| two-tower epoch | 0.5 s | ~4 s |
| serving registry | ~5 s / ~1 GB RSS | **60 s / 11.0 GB RSS** |
| serving latency | p50 12 / p99 16 ms (flat) | **p50 8.6 / p99 10.1 ms (HNSW)** |

## What replicated across categories (the strongest evidence in the project)

| finding | Video_Games | Beauty |
|---|---|---|
| multimodal tower vs ID-only, R@100 | +63% | **+75%** (0.086 vs 0.049) |
| ranking −multimodal | GAUC −0.10, cold collapses | GAUC −0.06, cold 0.667→0.396 |
| VLM profiles overall lift | +1.5% | **+3.1%** (0.0859 vs 0.0833) — grows with sparsity |
| VLM profiles on extreme cold | neutral | neutral (null replicates) |
| SID vs raw-ID, cold-start R@100 | **+46%** | **+47%** (0.0083 vs 0.0056) |
| SID vs raw-ID, overall | −35% | −65% (768 code embs vs 207K items is harsher compression) |
| ranker +SID | +0.003 GAUC / +0.008 cold | +0.002 / +0.005 |
| i2i best cold source, popularity = 0 cold | yes | yes (i2i cold 0.019) |
| RQ-VAE codebook health | 100% util, perp 208-252 | 100% util, perp 219-253 |
| offline cascade metric retrieval-favoring | fusion α→1.0 | fusion α=0.25, test 0.089 ≈ retrieval 0.090 |

**Cascade variant selection flipped again** — Beauty's winner is `scorefeat_softmax`
(valid 0.087, test 0.059), after `scorefeat_bce` won on Video_Games text+image and
`hardneg_clean` on the fused features. Three runs, three winners: variant ranking is
dataset/feature dependent, which is precisely why selection is automated on the valid split
and the checkpoint ships with its sidecar metadata.

Other scale-specific observations: RQ-VAE collision rate rose 9.6% → 22.9% (the trigger for a 4th
disambiguation level at bigger catalogs); pre-rank distill consistency dropped 0.70 → 0.43
(a 6.8M-param student mimicking over 200K items needs more capacity/epochs — noted, not tuned);
TIGER-lite R@10 0.008 vs two-tower 0.023 (demo scale shows its limits on a 207K catalog);
HNSW-vs-exact recall@100 measured 0.69 at default `efSearch` (serving raises it to 256).

## Incidents (the bottleneck stories — details in PITFALLS)
1. **Smaller model ≠ faster**: the 3B at batch 8 left the GPU at 61% (~34 h projected). Batch 24 →
   100% util, 2.5x throughput, 16.8 h. Throughput must be re-measured per model size.
2. **HNSW returns −1 sentinels**: the ANN index pads unfilled slots with −1 (default `efSearch=16`
   can't satisfy k=264), which crashed the embedding lookup — a bug the exact flat index could
   never trigger. Fix: `efSearch≥k` + a `c >= 0` filter in the serving path.

## Run it
```bash
uv run vlmrec <stage> --config configs/scale.yaml     # every stage, same code path
```
