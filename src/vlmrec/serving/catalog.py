"""Human-readable catalog for the demo endpoints: titles, prices, images, VLM profiles.

Loaded once at startup next to the model registry, and deliberately kept OUT of the scoring
path — ``/recommend`` itself never touches it, only the enrichment/inspection endpoints do.
Everything is keyed by ``item_idx`` (the contiguous id the models speak) but images stay keyed
by ``parent_asin`` on disk (docs/PITFALLS.md #5 — item_idx is re-assigned on every build).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..paths import Paths
from ..utils import get_logger

log = get_logger("vlmrec.serving.catalog")


def parse_profile(raw: str | None) -> dict:
    """VLM profile JSON -> dict, tolerating the few rows the generator flagged invalid."""
    if not raw:
        return {}
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return out if isinstance(out, dict) else {}


def display_title(title: str | None, parent_asin: str, max_chars: int = 90) -> str:
    """Trimmed title with the stable catalog id as fallback for blank metadata."""
    t = (title or "").strip() or f"[{parent_asin}]"
    return t if len(t) <= max_chars else t[: max_chars - 1] + "…"


def search_titles(titles_lc: list[str], query: str) -> list[int]:
    """Indices of titles containing every whitespace-separated query token (AND semantics).

    A linear scan is deliberate: 25K–200K lowercase titles cost single-digit ms per request,
    which is nowhere near worth an inverted index for a demo endpoint.
    """
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return []
    return [i for i, t in enumerate(titles_lc) if all(tok in t for tok in tokens)]


@dataclass
class Catalog:
    paths: Paths
    asin: list[str]  # item_idx -> parent_asin
    title: list[str]
    price: list[float]  # NaN when unparseable
    rating: list[float]
    rating_n: list[int]
    store: list[str]
    profile_raw: list[str]  # VLM profile JSON per item ("" when absent)
    _title_lc: list[str] | None = None  # built on first search

    @property
    def n_items(self) -> int:
        return len(self.asin)

    def image_rel(self, item_idx: int) -> str | None:
        """Path under the static /images mount, or None when the download never landed."""
        a = self.asin[item_idx]
        return f"{a}.jpg" if self.paths.image_file(a).exists() else None

    def summary(self, item_idx: int) -> dict:
        """The lean per-item payload embedded in list responses (grids stay lightweight)."""
        i = int(item_idx)
        price = self.price[i]
        return {
            "item_idx": i,
            "parent_asin": self.asin[i],
            "title": display_title(self.title[i], self.asin[i]),
            "price": None if np.isnan(price) else round(float(price), 2),
            "rating": round(float(self.rating[i]), 2),
            "rating_n": int(self.rating_n[i]),
            "image": self.image_rel(i),
        }

    def detail(self, item_idx: int) -> dict:
        """summary + store + the parsed VLM profile, for the item-inspector endpoint."""
        i = int(item_idx)
        return {
            **self.summary(i),
            "title_full": (self.title[i] or "").strip() or f"[{self.asin[i]}]",
            "store": self.store[i],
            "profile": parse_profile(self.profile_raw[i]),
        }

    def search(self, query: str, k: int = 12) -> list[dict]:
        """Title search, most-reviewed first (popularity is the sanest demo-facing tiebreak)."""
        if self._title_lc is None:
            self._title_lc = [t.lower() for t in self.title]
        hits = search_titles(self._title_lc, query)
        hits.sort(key=lambda i: -self.rating_n[i])
        return [self.summary(i) for i in hits[:k]]


def load_catalog(paths: Paths) -> Catalog:
    items = pl.read_parquet(paths.item_map_parquet).sort("item_idx")
    meta = pl.read_parquet(
        paths.meta_parquet,
        columns=["parent_asin", "title", "price", "average_rating", "rating_number", "store"],
    ).unique(subset=["parent_asin"], keep="first")
    df = items.join(meta, on="parent_asin", how="left")

    n = df.height
    profile_raw = [""] * n
    if paths.profiles_parquet.exists():
        prof = pl.read_parquet(paths.profiles_parquet)
        idx_pj = zip(prof["item_idx"].to_list(), prof["profile_json"].to_list(), strict=True)
        for idx, pj in idx_pj:
            if 0 <= idx < n:
                profile_raw[idx] = pj or ""

    def _f(col: str, default: float) -> list[float]:
        return df[col].cast(pl.Float64, strict=False).fill_null(default).to_list()

    cat = Catalog(
        paths=paths,
        asin=df["parent_asin"].to_list(),
        title=df["title"].fill_null("").to_list(),
        price=_f("price", float("nan")),
        rating=_f("average_rating", 0.0),
        rating_n=[int(x) for x in _f("rating_number", 0.0)],
        store=df["store"].fill_null("").to_list(),
        profile_raw=profile_raw,
    )
    log.info("catalog loaded: %d items, profiles=%s", cat.n_items, paths.profiles_parquet.exists())
    return cat
