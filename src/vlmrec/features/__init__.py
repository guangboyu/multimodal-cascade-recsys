"""Multimodal feature extraction: text (sentence-transformers) + image (CLIP).

Both produce per-item embedding matrices aligned to ``item_idx`` (row i == item_idx i),
the shared representation consumed by the retrieval item-tower and the ranker.
"""
