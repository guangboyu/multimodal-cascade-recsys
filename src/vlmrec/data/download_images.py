"""Download a subset of product images for items in the interaction set.

Robust to both on-disk shapes of the Amazon `images` field:
  * list-of-dicts:   ``[{"large": url, "hi_res": url, "thumb": url, "variant": ...}, ...]``
  * struct-of-lists: ``{"large": [url, ...], "hi_res": [...], "thumb": [...]}``
    (HuggingFace `datasets` converts Sequence-of-struct into struct-of-sequences).

Downloads run concurrently, are resumable (skip files already on disk), validate bytes
through Pillow, and re-encode to RGB JPEG so the image encoder can load them reliably.
A manifest (item_idx, url, status) is written for the dataset card / coverage stats.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import requests
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from ..paths import Paths
from ..utils import get_logger, timer

log = get_logger("vlmrec.images")

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; vlmrec/0.1; +academic-portfolio)"}
_SIZES = ["large", "hi_res", "thumb"]


def _size_order(preferred: str) -> list[str]:
    return [preferred] + [s for s in _SIZES if s != preferred]


def extract_url(images, size_order: list[str]) -> str | None:
    """Return the first usable image URL, trying each size in order. Shape-agnostic."""
    if images is None:
        return None
    if isinstance(images, dict):  # struct-of-lists
        for s in size_order:
            vals = images.get(s)
            if vals:
                for v in vals:
                    if v:
                        return v
        return None
    if isinstance(images, (list, tuple)):  # list-of-dicts
        for img in images:
            if isinstance(img, dict):
                for s in size_order:
                    if img.get(s):
                        return img[s]
        return None
    return None


def _download_one(item_idx: int, url: str | None, dest, timeout: int, retries: int):
    if dest.exists():
        return item_idx, "cached"
    if not url:
        return item_idx, "no_url"
    for _ in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=_HEADERS)
            r.raise_for_status()
            Image.open(io.BytesIO(r.content)).convert("RGB").save(dest, format="JPEG", quality=90)
            return item_idx, "ok"
        except Exception:  # noqa: BLE001 - transient network/decoding errors are expected
            continue
    return item_idx, "fail"


def run(cfg: DictConfig, paths: Paths) -> dict:
    if not bool(cfg.images.enabled):
        log.info("images.enabled=false -> skipping image download")
        return {"status": "skipped"}
    paths.ensure()
    with timer(log, "download images"):
        item_map = pl.read_parquet(paths.item_map_parquet)  # parent_asin, item_idx
        meta = pl.read_parquet(paths.meta_parquet).join(item_map, on="parent_asin", how="inner")

        order = _size_order(str(cfg.images.size))
        recs = []
        for row in meta.select(["item_idx", "parent_asin", "images"]).iter_rows(named=True):
            images = row["images"]
            if isinstance(images, str):  # stored JSON-encoded by download.py
                try:
                    images = json.loads(images)
                except json.JSONDecodeError:
                    images = None
            recs.append(
                {
                    "item_idx": row["item_idx"],
                    "parent_asin": row["parent_asin"],
                    "url": extract_url(images, order),
                }
            )
        url_df = pl.DataFrame(recs).filter(pl.col("url").is_not_null()).sort("item_idx")

        cap = cfg.images.max_images
        if cap:
            url_df = url_df.head(int(cap))
        log.info("items with a usable image URL: %s (downloading)", f"{url_df.height:,}")

        results: list[tuple[int, str]] = []
        jobs = list(url_df.iter_rows(named=True))
        with ThreadPoolExecutor(max_workers=int(cfg.images.num_workers)) as ex:
            futs = [
                ex.submit(
                    _download_one,
                    j["item_idx"],
                    j["url"],
                    paths.image_file(j["parent_asin"]),
                    int(cfg.images.timeout),
                    int(cfg.images.retries),
                )
                for j in jobs
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="images", unit="img"):
                results.append(fut.result())

        manifest = url_df.join(
            pl.DataFrame(results, schema=["item_idx", "status"], orient="row"),
            on="item_idx",
            how="left",
        )
        manifest.write_parquet(paths.image_manifest_parquet)
        counts = dict(manifest.group_by("status").len().sort("status").iter_rows())
        n_ok = counts.get("ok", 0) + counts.get("cached", 0)
    log.info("image status counts: %s", counts)
    log.info("images on disk: %s -> %s", f"{n_ok:,}", paths.images)
    return {"status_counts": counts, "n_images": n_ok}
