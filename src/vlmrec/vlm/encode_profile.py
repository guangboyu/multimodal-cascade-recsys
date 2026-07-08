"""Encode VLM item profiles to dense embeddings (the ``vlm`` feature-source block).

A deterministic template flattens each structured profile to one line of text, which the same
sentence-transformer used for raw item text (config ``text.model``) embeds and L2-normalizes.
Keeping the encoder identical to the ``text`` block means the ablation isolates *what the VLM
said*, not encoder strength. Items whose profile failed validation (ok=0) fall back to their
title so no row is a zero vector.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from ..features.encode_text import _field_text
from ..paths import Paths
from ..utils import get_logger, pick_device, timer

log = get_logger("vlmrec.vlm.encode_profile")

_TEXT_KEYS = ("category_refined", "sub_genre", "target_audience", "tone", "quality_cues")


def profile_to_text(profile: dict, title: str = "") -> str:
    """Deterministic profile -> text template (the exact string that gets embedded)."""
    has_content = any(profile.get(k) for k in _TEXT_KEYS) or profile.get("visual_style")
    if not has_content:
        return title
    style = ", ".join(profile.get("visual_style") or [])
    attrs = ", ".join(profile.get("key_attributes") or [])
    return (
        f"category: {profile.get('category_refined', '')} | {profile.get('sub_genre', '')}; "
        f"style: {style}; attributes: {attrs}; "
        f"audience: {profile.get('target_audience', '')}; tone: {profile.get('tone', '')}; "
        f"quality: {profile.get('quality_cues', '')}. {profile.get('one_line_summary', '')}"
    )


def run(cfg, paths: Paths) -> dict:
    df = pl.read_parquet(paths.profiles_parquet).sort("item_idx")
    item_map = pl.read_parquet(paths.item_map_parquet)
    meta = pl.read_parquet(paths.meta_parquet)
    titles = [
        _field_text(t)
        for t in (
            item_map.join(meta, on="parent_asin", how="left")
            .sort("item_idx")
            .get_column("title")
            .to_list()
        )
    ]
    assert df.height == len(titles), f"profiles/items misaligned: {df.height} vs {len(titles)}"

    texts = [
        profile_to_text(json.loads(pj), titles[i])
        for i, pj in enumerate(df.get_column("profile_json").to_list())
    ]
    device = pick_device(str(cfg.device))
    with timer(log, f"encode profiles [{cfg.text.model}] on {device}"):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(cfg.text.model), device=device)
        emb = model.encode(
            texts,
            batch_size=int(cfg.text.batch_size),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    np.save(paths.profile_emb_npy, emb)
    out = {
        "model": str(cfg.text.model),
        "dim": int(emb.shape[1]),
        "n_items": int(emb.shape[0]),
        "ok_rate": round(float(df.get_column("ok").mean()), 4),
        "nonzero_rows": int((np.abs(emb).sum(axis=1) > 0).sum()),
    }
    (paths.vlm / "profile_emb.json").write_text(json.dumps(out, indent=2))
    log.info("profile embeddings -> %s | %s", paths.profile_emb_npy, out)
    return out
