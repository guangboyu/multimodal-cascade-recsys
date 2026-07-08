# Engineering pitfalls & lessons

A running log of the non-obvious bugs and gotchas hit while building this system, with
symptom → root cause → fix → takeaway. Kept deliberately honest — these are the things that cost
real debugging time.

---

## 1. Silent CPU fallback from a CUDA-version mismatch
- **Symptom:** training ran, but on CPU; `torch.cuda.is_available()` returned `False` with a warning
  *"The NVIDIA driver on your system is too old (found version 12080)"*.
- **Root cause:** the default PyPI wheel resolved to `torch 2.12+cu130` (CUDA **13.0** runtime), but the
  WSL2 driver only supports CUDA **12.8**. Torch couldn't initialize CUDA and silently fell back to CPU.
- **Fix:** pin `torch`/`torchvision` to the **cu128** build via a dedicated index in `pyproject.toml`
  (`[[tool.uv.index]]` + `[tool.uv.sources]`).
- **Takeaway:** a *too-new* CUDA wheel degrades silently to CPU — it doesn't error. Always assert
  `torch.cuda.is_available()` right after env setup, and match the wheel's CUDA build to the driver
  (`nvidia-smi` CUDA version is the ceiling).

## 2. `datasets` 5.0 dropped dataset loading scripts
- **Symptom:** `RuntimeError: Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py`;
  passing `trust_remote_code=True` printed *"not supported anymore"*.
- **Root cause:** the dataset ships a Python loader script; HuggingFace `datasets` ≥ 4 removed
  script-based datasets entirely.
- **Fix:** stop using `load_dataset(repo, config)`. Download the repo's raw `*.jsonl` directly with
  `huggingface_hub.hf_hub_download` and parse with polars; for capped dev runs, stream the file over
  HTTP and take the first N lines (avoids a multi-GB download).
- **Takeaway:** `load_dataset(name, config)` is not stable across major versions. For big public
  datasets, reading the raw files is more robust and often faster.

## 3. Wrong assumption about the raw file format
- **Symptom:** my first fallback looked for `*.jsonl.gz`.
- **Root cause:** the repo actually stores plain `raw/review_categories/<Cat>.jsonl` (uncompressed).
- **Fix:** list the repo files first (`HfApi().list_repo_files`) and key off reality, not assumption.
- **Takeaway:** verify the on-disk layout before writing a loader; one `list_repo_files` call saves a
  wrong-path debugging loop.

## 4. polars `with_row_index` → UInt32 → torch indexing error
- **Symptom:** `IndexError: tensors used as indices must be long, int, byte or bool tensors`.
- **Root cause:** id columns built via `with_row_index` / joins came out as **UInt32**; PyTorch index
  tensors must be int64 (`long`). UInt32 is rejected.
- **Fix:** cast id arrays to `int64` at the data-layer boundary, once.
- **Takeaway:** mind dtypes at the dataframe↔tensor boundary. Unsigned ints are *not* valid index dtypes
  in torch even though they look numeric.

## 5. Stale, mis-keyed cache when rebuilding at a different data scale
- **Symptom:** after switching from a 500k-row dev subsample to the full corpus, the image cache would
  have silently paired the *wrong* pictures with items.
- **Root cause:** images were cached as `data/images/{item_idx}.jpg`, but `item_idx` is re-assigned on
  every build (contiguous remap). The resumable "skip if file exists" logic would keep the subsample's
  files under indices that now mean different products.
- **Fix:** wipe `data/{raw,processed,images,embeddings}` before a full rebuild. (Better long-term: key
  the cache by the stable `parent_asin`, not the volatile `item_idx`.)
- **Takeaway:** never key a cache on an id that gets remapped between runs. Content-addressed or
  natural-key caching avoids silent cross-contamination.

## 6. Heterogeneous nested metadata broke the parquet schema
- **Symptom:** building a DataFrame straight from the raw metadata risked schema-inference failures /
  garbled nested columns.
- **Root cause:** Amazon metadata is messy and inconsistent — `images` can be a list-of-dicts *or* a
  dict-of-lists (HF `datasets` converts `Sequence(struct)` → struct-of-sequences), `description`/
  `features` are sometimes a list and sometimes a bare string, `price` is `"$9.99"` / `""` / `null`.
- **Fix:** normalize at ingestion — store nested/variable fields as **JSON strings** for a stable parquet
  schema, then decode defensively at the point of use (shape-agnostic URL extraction; a `_field_text`
  helper that handles str/list/dict/JSON-string).
- **Takeaway:** pin a stable schema at the ingestion boundary; push the messiness into one normalization
  layer instead of letting it leak into every consumer.

## 7. k-core filtering collapses a small subsample to empty
- **Symptom:** an early dev smoke (50k reviews) died with *"No interactions left after k-core"*.
- **Root cause:** 5-core over a tiny, sparse subsample prunes almost everything — most items had < 5
  interactions in the sample.
- **Fix:** dev profile uses a larger cap + a gentler `k_core=3`; the full build uses the real `k_core=5`.
- **Takeaway:** densification thresholds interact with sample size. Parameterize filtering so dev and
  full runs aren't governed by the same constants.

## 8. tqdm progress bars flooded the logs
- **Symptom:** reading a stage's log produced ~185 KB of progress-bar carriage returns and buried the
  result line.
- **Fix:** validate stages by inspecting the **artifacts** (parquet/npy shapes & stats) rather than
  tailing logs; reserve progress bars for interactive use.
- **Takeaway:** keep the human progress UI separate from the machine-readable validation surface.

## 9. torch 2.11's default ONNX exporter (dynamo) failed on baked constant tables
- **Symptom:** `torch.onnx.export(...)` first raised `ModuleNotFoundError: onnxscript`, then a
  `SerdeError` in the version converter, when exporting the ranker with the (~100 MB) item/feature
  tables baked in as buffers.
- **Root cause:** torch 2.11 defaults to the new dynamo-based ONNX exporter (needs `onnxscript`),
  which choked on the large constant graph.
- **Fix:** `torch.onnx.export(..., dynamo=False)` (the legacy TorchScript exporter) — parity to torch
  was `3.8e-6`.
- **Takeaway:** the ONNX export path is version-sensitive; the legacy exporter is still the robust
  default for models that bake in large constants.

## 10. Held-out positives poisoning the hard-negative pool
- **Symptom:** training the ranker on retrieval-mined hard negatives made the cascade *worse*
  (NDCG@10 0.081 → 0.051) — the opposite of what hard negatives are for.
- **Root cause:** the candidate precompute masked only **train**-seen items, but retrieval is good
  — a user's held-out valid/test positive usually IS retrieved. Sampling negatives from candidates
  therefore labelled future positives `click=0`.
- **Fix:** rejection-resample any negative colliding with {row positive, valid item, test item}
  in both the hard and random paths (`sample_negatives`). Clean pool alone: 0.051 → 0.060.
- **Takeaway:** hard-negative mining's classic false-negative trap. Ask of every mined negative:
  *could this be a positive I just haven't observed?*

## 11. A "was-retrieved" membership flag would have been reverse label leakage
- **Symptom (avoided):** the obvious cross-stage feature — a binary "item ∈ retrieval top-K" —
  looked reasonable but would have poisoned training.
- **Root cause:** train positives are seen-masked *out* of the candidate file, so at train time
  every positive carries flag=0 and most hard negatives flag=1 — the flag encodes the label,
  inverted. The same residual bias appeared empirically: concentrating hard negatives in the
  top-50 (most candidate-membership-correlated slate) cratered cascade NDCG to 0.032.
- **Fix:** use the **continuous retrieval score** `u_e·i_e`, defined for every (user, item) pair,
  positives included — no membership discontinuity, and byte-identical to the FAISS score at
  serving time. Score feature: 0.060 → 0.075 (+48% cumulative over the poisoned run).
- **Takeaway:** judge a feature by *how it is constructed at training time*, not what it means at
  serving time. Anything derived from a seen-masked artifact inherits the mask.

## 12. Offline cascade NDCG is retrieval-favoring — and fusion tuning proves it
- **Symptom:** after every fix, no ranker variant beat raw retrieval order (0.075 vs 0.109), and
  tuning score-fusion α on valid converged to α≈1 (pure retrieval).
- **Root cause:** the candidate set was *selected by the retriever's own similarity*, so metrics
  computed on it give the retriever home-field advantage; the eval can't observe items the
  retriever ranks low but the user would love.
- **Fix (framing, not code):** report the ablation ladder honestly, ship the valid-tuned fusion
  (guaranteed ≥ retrieval order by construction), and note that the unbiased comparison needs
  online A/B or counterfactual estimators (IPS/DR).
- **Takeaway:** when the eval candidates come from the model you're comparing against, treat
  "challenger loses" as expected-bias first, model-failure second.

---

## Modeling insights (not bugs, but worth being able to explain)

- **ID-embedding overfitting on sparse data.** The pure-ID two-tower drove train loss far lower
  (2.84 vs 6.74 for the content tower) yet generalized *worse* (test R@100 0.129 vs 0.210) and was
  worst on cold-start — a clean illustration of why content/multimodal features matter for the long tail.
- **More features ≠ better in retrieval.** Adding item-ID/price/category embeddings to the content tower
  (`hybrid`) matched `content` overall and *hurt* cold-start (undertrained ID embeddings for rare items).
  This is why the heavy bucket+embedding feature work belongs in the **ranker**, not the retriever.
- **Sources are complementary, not redundant.** item2item co-visitation was weakest overall but **best**
  on the cold-start slice; popularity scored exactly 0 there (it can never surface long-tail items).
  This is the concrete argument for blending multiple candidate sources.
- **In-batch sampled softmax needs a logQ correction.** Without subtracting `log P(item)`, popular items
  dominate the in-batch negatives and get unfairly penalized; the correction keeps the objective unbiased.
- **Train-serve negative mismatch (sample-selection bias).** The ranker trained on *random* negatives
  *degraded* the retriever's top-200 when re-ranking (NDCG@10 0.109 → 0.081): random negatives are too
  easy, so the model never learned to separate the *hard* candidates retrieval actually surfaces. The
  fix is to train the ranker on hard negatives sampled from retrieval — the consistency requirement
  between the retrieval and ranking stages.
- **Hard negatives aren't a free lunch — the ranker needs a cross-stage score.** Naively training the
  ranker on retrieved items as negatives *worsened* the cascade (NDCG@10 0.081 → 0.051): the held-out
  positive is itself a retrieved item, so "retrieved = negative" penalizes it, and the ranker has no
  retrieval / pre-rank score feature to *refine* (rather than replace) the retriever's order. The
  standard fix is to feed the retrieval score into the ranker as a feature and exclude near-positives
  when sampling hard negatives. (Offline cascade NDCG is also retrieval-favoring — the candidates were
  chosen by the retriever's own similarity.)
