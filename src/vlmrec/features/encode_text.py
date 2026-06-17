"""Item text embeddings via sentence-transformers.

Item text = configured metadata fields (title + description + features) concatenated and
truncated. description/features are stored JSON-encoded (see download.py), so we decode them
defensively here. Embeddings are L2-normalized and saved as float32 aligned to item_idx.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from omegaconf import DictConfig

from ..paths import Paths
from ..utils import get_logger, pick_device, timer

log = get_logger("vlmrec.encode_text")


def _field_text(v) -> str:
    """Coerce a metadata field (plain str, list, dict, or JSON-encoded str) to flat text."""
    if v is None:
        return ""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in ("[", "{"):
            try:
                v = json.loads(s)
            except json.JSONDecodeError:
                return s
        else:
            return s
    if isinstance(v, (list, tuple)):
        return " ".join(_field_text(x) for x in v if x is not None)
    if isinstance(v, dict):
        return " ".join(_field_text(x) for x in v.values() if x is not None)
    return str(v)


def build_item_texts(paths: Paths, fields: list[str], max_chars: int) -> list[str]:
    """Build one text string per item, ordered by item_idx (0..N-1)."""
    item_map = pl.read_parquet(paths.item_map_parquet)
    meta = pl.read_parquet(paths.meta_parquet)
    merged = item_map.join(meta, on="parent_asin", how="left").sort("item_idx")
    texts = []
    for row in merged.iter_rows(named=True):
        parts = [_field_text(row.get(f)) for f in fields]
        text = " ".join(p for p in parts if p).strip()
        texts.append(text[:max_chars] if max_chars else text)
    return texts


def run(cfg: DictConfig, paths: Paths) -> dict:
    paths.ensure()
    device = pick_device(str(cfg.device))
    fields = list(cfg.text.fields)
    with timer(log, f"encode text [{cfg.text.model}] on {device}"):
        texts = build_item_texts(paths, fields, int(cfg.text.max_chars))
        n_empty = sum(1 for t in texts if not t)
        log.info("items=%s  empty_text=%s", f"{len(texts):,}", f"{n_empty:,}")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg.text.model, device=device)
        emb = model.encode(
            texts,
            batch_size=int(cfg.text.batch_size),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        np.save(paths.text_emb_npy, emb)
        meta_out = {
            "model": str(cfg.text.model),
            "dim": int(emb.shape[1]),
            "n_items": int(emb.shape[0]),
            "fields": fields,
            "normalized": True,
        }
        (paths.embeddings / "text_emb.json").write_text(json.dumps(meta_out, indent=2))
    log.info("text emb -> %s  shape=%s", paths.text_emb_npy, tuple(emb.shape))
    return meta_out
