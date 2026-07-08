"""Central, typed resolution of every artifact path, derived from ``cfg.paths``.

Keeping all filesystem locations in one place means no stage hard-codes a path, and the
whole data layout can be relocated by editing the config.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


class Paths:
    def __init__(self, cfg: DictConfig):
        root = Path(cfg.paths.root).resolve()
        self.root = root
        self.data = root / cfg.paths.data
        self.raw = root / cfg.paths.raw
        self.processed = root / cfg.paths.processed
        self.images = root / cfg.paths.images
        self.embeddings = root / cfg.paths.embeddings

    def ensure(self) -> Paths:
        for p in (self.data, self.raw, self.processed, self.images, self.embeddings):
            p.mkdir(parents=True, exist_ok=True)
        return self

    # raw ---------------------------------------------------------------
    @property
    def reviews_parquet(self) -> Path:
        return self.raw / "reviews.parquet"

    @property
    def meta_parquet(self) -> Path:
        return self.raw / "meta.parquet"

    # processed ---------------------------------------------------------
    @property
    def interactions_parquet(self) -> Path:
        return self.processed / "interactions.parquet"

    @property
    def user_map_parquet(self) -> Path:
        return self.processed / "user_map.parquet"

    @property
    def item_map_parquet(self) -> Path:
        return self.processed / "item_map.parquet"

    @property
    def stats_json(self) -> Path:
        return self.processed / "stats.json"

    @property
    def image_manifest_parquet(self) -> Path:
        return self.processed / "image_manifest.parquet"

    @property
    def dataset_card(self) -> Path:
        return self.processed / "dataset_card.md"

    # vlm profiles ------------------------------------------------------
    @property
    def vlm(self) -> Path:
        return self.data / "vlm"

    @property
    def profiles_parquet(self) -> Path:
        return self.vlm / "profiles.parquet"

    @property
    def profile_emb_npy(self) -> Path:
        return self.vlm / "profile_emb.npy"

    # embeddings --------------------------------------------------------
    @property
    def text_emb_npy(self) -> Path:
        return self.embeddings / "text_emb.npy"

    @property
    def image_emb_npy(self) -> Path:
        return self.embeddings / "image_emb.npy"

    @property
    def has_image_npy(self) -> Path:
        return self.embeddings / "has_image.npy"

    def image_path(self, item_idx: int) -> Path:
        return self.images / f"{item_idx}.jpg"
