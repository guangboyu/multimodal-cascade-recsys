# Ranking (the deep stage)

## Goal
Re-score the few hundred retrieved candidates with a heavy model. Ranking is a *precision* problem
tied to business value, so this is where the ML investment goes: behavior modeling, feature crosses,
multi-task, and the full structured-feature treatment.

## What was built (`src/vlmrec/ranking/`)
- **`model.py`** — `Ranker` combining:
  - **DIN** (Deep Interest Network) target-attention over the user's behavior sequence (leave-one-out
    masked so a positive can't leak via its own history),
  - **DCN-v2** explicit feature crosses,
  - **MMoE** multi-gate mixture-of-experts → two task heads: **P(click)** and **P(satisfaction, rating≥4)**,
  - full **bucket + embedding** structured features (quantile price buckets, category leaf, item-id).
  Ablation switches: `use_multimodal / use_din / use_cross / use_mmoe`.
- **`data.py`** — padded per-user behavior sequences, positives with satisfaction labels; negatives
  sampled on the fly.
- **`train.py`** — multi-task BCE with sampled negatives; eval = **GAUC** (per-user AUC), AUC, LogLoss.
- **`eval.py`** — the ablation + a **cascade** eval (rank the two-tower's retrieved candidates).

## Ablation (test split, GAUC = per-user AUC)
| variant | click GAUC | AUC | LogLoss | sat AUC | GAUC (cold-start) |
|---|---|---|---|---|---|
| **full** | **0.8584** | 0.8595 | 0.1522 | 0.8590 | **0.7057** |
| − multimodal | 0.7607 | 0.7607 | 0.1968 | 0.7608 | 0.3470 |
| − DIN | 0.8526 | 0.8551 | 0.1560 | 0.8547 | 0.6917 |
| − DCN-v2 cross | 0.8496 | 0.8505 | 0.1446 | 0.8508 | 0.6972 |
| single-task (no MMoE) | 0.8575 | 0.8586 | 0.1534 | — | 0.7076 |

**Takeaways**
- **Multimodal features are the dominant contributor:** −0.098 GAUC overall, and cold-start GAUC
  *collapses below random* (0.71 → 0.35) without item content. The multimodal thesis holds at the
  ranking stage, not just retrieval.
- **DIN** (+0.006) and **DCN-v2** (+0.009) each add real, smaller gains. (Note: `−cross` has the best
  LogLoss but worse GAUC — crosses help ordering more than calibration.)
- **Multi-task is ~neutral** on click GAUC while adding the satisfaction objective — a free second head.

## Cascade finding (honest negative result → rerank motivation)
Re-ranking the retriever's top-200 with this ranker **lowered** NDCG@10 (0.109 → 0.081). The ranker
was trained on **random** negatives, which are far easier than the **hard** negatives retrieval
surfaces, so it isn't calibrated to separate already-good candidates — classic **sample-selection
bias / train-serve mismatch**. Fix: train the ranker on hard negatives sampled from retrieval
(addressed at the pre-ranking / consistency stage; see 04-rerank.md).

## Run it
```bash
make ranking          # train ranker + ablation
```
